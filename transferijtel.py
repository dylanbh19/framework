#!/usr/bin/env python
"""
transferanalysis.py  –  discover sub-intents inside “Transfer” calls

Outputs (folder: transfer_subintent_results)
    • transfer_clusters.csv           – original rows + sub_intent_k
    • transfer_cluster_summary.csv    – size + sample + top-features
    • cluster_size_hist.png           – PNG quick-look
    • cluster_scatter.png             – 2-D PCA scatter     (if ≥2 comps)
    • elbow.png                       – elbow curve
    • cluster_feature_bars.png        – bar charts of distinctive features
    • transfer_subintent.log          – UTF-8 log file
"""

from __future__ import annotations
import argparse, logging, sys, warnings, json, platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder

import matplotlib
matplotlib.use("Agg")  # headless back-end
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- optional Sentence-BERT -------------------------------- #
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
# ------------------------------------------------------------------ #

# ------------------------------ I/O ------------------------------- #
OUTDIR = Path("transfer_subintent_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

log_file = OUTDIR / "transfer_subintent.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ],
    format="%(asctime)s %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
RAND = 42
# ------------------------------------------------------------------ #

# ------------------------ helper functions ------------------------ #
def autodetect_csv() -> Path | None:
    cands = list(Path.cwd().rglob("best_augmented_data.csv"))
    if cands:
        return max(cands, key=lambda p: p.stat().st_mtime)
    return None


def build_embeddings(texts: List[str]) -> np.ndarray:
    """Sentence-BERT if available, else TF-IDF."""
    if SBERT_AVAILABLE:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        log.info("Embeddings: Sentence-BERT (all-MiniLM-L6-v2)")
        emb = model.encode(texts,
                           batch_size=256,
                           show_progress_bar=True,
                           normalize_embeddings=True)
        return emb.astype("float32")

    log.warning("Sentence-BERT unavailable – using TF-IDF fallback")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95)
    mat = tfidf.fit_transform(texts)
    return normalize(mat).toarray().astype("float32")


def one_hot_encoder() -> OneHotEncoder:
    """Return a OneHotEncoder with the right kwarg for this sklearn version."""
    import sklearn
    major, minor, *_ = sklearn.__version__.split(".")
    major, minor = int(major), int(minor)
    if (major, minor) >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_py(obj: Any):
    """Cast numpy / pandas scalars to native Python types for JSON."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]
    return obj
# ------------------------------------------------------------------ #


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Embeddings + numeric + categorical → dense feature matrix."""
    # ========= sentence embeddings =================================
    emb = build_embeddings(df["activity_sequence"].fillna("").tolist())
    emb_names = [f"emb_{i}" for i in range(emb.shape[1])]

    # ========= engineered numeric ==================================
    if "seq_length" not in df:
        df["seq_length"] = df["activity_sequence"].str.count(r"\|").fillna(0) + 1
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns
                    if c not in ("first_activity", "last_activity")]
    num_mat = StandardScaler().fit_transform(df[numeric_cols].fillna(0))
    numeric_names = numeric_cols

    # ========= categorical =========================================
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                if c not in ("activity_sequence", "intent_augmented", "intent_base")
                and df[c].nunique() <= 30]
    if cat_cols:
        ohe = one_hot_encoder()
        cat_mat = ohe.fit_transform(df[cat_cols])
        if not isinstance(cat_mat, np.ndarray):
            cat_mat = cat_mat.toarray()
        cat_names: List[str] = []
        for col, cats in zip(cat_cols, ohe.categories_):
            cat_names.extend([f"{col}={v}" for v in cats])
    else:
        cat_mat = np.empty((len(df), 0), dtype="float32")
        cat_names = []

    # ========= concat ==============================================
    feat_mat = np.hstack([emb, num_mat.astype("float32"), cat_mat.astype("float32")])
    feat_names = emb_names + numeric_names + cat_names
    return feat_mat, feat_names


def pca_reduce(mat: np.ndarray, var: float = 0.95, max_comps: int = 50) -> np.ndarray:
    pca = PCA(n_components=min(max_comps, mat.shape[1]), random_state=RAND)
    red = pca.fit_transform(mat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = np.searchsorted(cum, var) + 1
    keep = min(keep, red.shape[1])
    log.info(f"PCA kept {keep} comps – {cum[keep-1]*100:.2f} % variance")
    return red[:, :keep]


def evaluate_k(mat: np.ndarray, k_values: List[int]) -> Tuple[int, Dict[int, Tuple[float, np.ndarray]]]:
    """Return best_k & dict{k: (silhouette, labels)}; guard against collapse."""
    results: Dict[int, Tuple[float, np.ndarray]] = {}
    for k in k_values:
        km = KMeans(k, random_state=RAND, n_init="auto").fit(mat)
        labs = km.labels_
        if len(set(labs)) < 2:
            log.warning(f"k={k}: collapsed to single cluster – skipped")
            continue
        sil = silhouette_score(mat, labs, metric="cosine",
                               sample_size=min(10_000, len(mat)), random_state=RAND)
        results[k] = (sil, labs)
        log.info(f"k={k} silhouette={sil:.3f}")
    if not results:
        raise RuntimeError("All k-values collapsed; widen search range.")
    best_k = max(results, key=lambda k: results[k][0])
    log.info(f"Selected k={best_k} (silhouette {results[best_k][0]:.3f})")
    return best_k, results


def distinctive_features(mat: np.ndarray,
                         names: List[str],
                         labels: np.ndarray,
                         top_n: int = 12) -> Dict[int, List[Tuple[str, float]]]:
    """Return list of top_n feature diffs for each cluster."""
    overall = mat.mean(axis=0)
    out: Dict[int, List[Tuple[str, float]]] = {}
    for lab in np.unique(labels):
        mean_vec = mat[labels == lab].mean(axis=0)
        diff = mean_vec - overall
        idx = np.argsort(diff)[-top_n:][::-1]
        out[int(lab)] = [(names[i], diff[i]) for i in idx]
    return out


def plot_feature_bars(clust2feat: Dict[int, List[Tuple[str, float]]]):
    k = len(clust2feat)
    fig, axes = plt.subplots(k, 1, figsize=(7, k * 2.4))
    if k == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        feats, vals = zip(*clust2feat[i])
        ax.barh(range(len(feats))[::-1], vals[::-1])
        ax.set_yticks(range(len(feats))[::-1])
        ax.set_yticklabels(feats[::-1], fontsize=6)
        ax.set_title(f"Cluster {i} – distinctive features")
    plt.tight_layout()
    plt.savefig(OUTDIR / "cluster_feature_bars.png", dpi=300)
    plt.close()
# ------------------------------------------------------------------ #


def safe_sample(series: pd.Series) -> str:
    for v in series:
        if isinstance(v, str) and v.strip():
            return v[:150]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="augmented data CSV")
    args = parser.parse_args()

    csv_path = Path(args.input) if args.input else autodetect_csv()
    if not csv_path or not csv_path.exists():
        log.error("Could not find best_augmented_data.csv – supply with --input")
        sys.exit(1)

    log.info(f"CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["intent_augmented"] == "Transfer"].reset_index(drop=True)
    df = df[df["activity_sequence"].notna()]

    if df.empty:
        log.error("No Transfer rows with activity_sequence present.")
        sys.exit(1)
    log.info(f"Rows for clustering: {len(df):,}")

    # -------- feature matrix ---------------------------------------
    feat_mat, feat_names = build_feature_matrix(df)

    # -------- PCA ---------------------------------------------------
    red = pca_reduce(feat_mat)

    # -------- evaluate k (3-5) -------------------------------------
    best_k, k_results = evaluate_k(red, [3, 4, 5])
    labels = k_results[best_k][1]

    # -------- elbow plot -------------------------------------------
    inertias = {k: KMeans(k, random_state=RAND, n_init="auto").fit(red).inertia_
                for k in range(2, 16)}
    plt.figure(figsize=(6, 4))
    ks, vals = zip(*sorted(inertias.items()))
    plt.plot(ks, vals, marker="o")
    plt.axvline(best_k, ls="--", c="red")
    plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow curve")
    plt.tight_layout(); plt.savefig(OUTDIR / "elbow.png", dpi=300); plt.close()

    # -------- scatter & size plot ----------------------------------
    if red.shape[1] >= 2:
        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 5))
        palette = sns.color_palette("husl", len(set(labels)))
        sns.scatterplot(x=red[:, 0], y=red[:, 1], hue=labels,
                        palette=palette, s=8, legend=False)
        plt.title("Sub-intent clusters (PCA dims 1-2)")
        plt.tight_layout()
        plt.savefig(OUTDIR / "cluster_scatter.png", dpi=300)
        plt.close()

    size_ser = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(6, 3))
    size_ser.plot(kind="bar")
    plt.title("Cluster size distribution")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(OUTDIR / "cluster_size_hist.png", dpi=300)
    plt.close()

    # -------- add labels & save CSV --------------------------------
    df_out = df.copy()
    df_out["sub_intent_k"] = labels
    df_out.to_csv(OUTDIR / "transfer_clusters.csv", index=False)

    # -------- distinctive features ---------------------------------
    clust_feats = distinctive_features(feat_mat, feat_names, labels)
    plot_feature_bars(clust_feats)

    # -------- summary CSV ------------------------------------------
    summary_rows = []
    for c, size in size_ser.items():
        feats = "; ".join(f"{f}:{v:+.2f}" for f, v in clust_feats[c][:6])
        sample = safe_sample(df_out[df_out["sub_intent_k"] == c]["activity_sequence"])
        summary_rows.append({"cluster": c, "size": int(size),
                             "top_features": feats, "sample_activity": sample})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "transfer_cluster_summary.csv", index=False)

    # -------- JSON snapshot ----------------------------------------
    snap = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": platform.python_version(),
        "sklearn": __import__("sklearn").__version__,
        "rows": int(len(df)),
        "selected_k": int(best_k),
        "silhouette": float(k_results[best_k][0]),
        "cluster_sizes": to_py(size_ser.to_dict())
    }
    with open(OUTDIR / "transfer_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2)

    log.info(f"Outputs written to {OUTDIR.resolve()}")
    log.info("DONE %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()