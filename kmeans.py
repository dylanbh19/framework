#!/usr/bin/env python
"""
transferanalysis.py
-------------------
Clustering “Transfer” calls into sub-intents.

* Sentence-BERT embeddings on the *activity_sequence*
* Optional engineered features (numeric + categorical)
* PCA  ->  K-Means  ->  choose k with silhouette
* Visual diagnostics + JSON summary

Requires:  pandas, numpy, scikit-learn ≥ 0.24, matplotlib, seaborn,
           sentence-transformers (optional but recommended)
"""

from __future__ import annotations
import json, logging, os, sys, platform, warnings, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")       # headless back-end
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- configuration ------------------------
DATA_PATH       = Path("augmentation_results") / "best_augmented_data.csv"
OUT_DIR         = Path("transfer_subintent_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTENT_FILTER   = "Transfer"
RAND_SEED       = 42
K_VALUES        = [3, 4, 5]          # you can change this freely
PCA_VAR_KEEP    = 0.95               # keep 95 % variance
SAMPLE_SIZE_SIL = 10_000             # sample for silhouette (speed)
MAX_ROWS        = None               # set to an int for quick debugging
# --------------------------------------------------------------

log = logging.getLogger("transfer-analysis")
log.setLevel(logging.INFO)
_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=_log_fmt, datefmt="%Y-%m-%d %H:%M:%S")

# ----------------------- helpers ------------------------------
def load_sentence_model():
    """Try to load Sentence-BERT; fall back to dummy encoder if missing."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Using Sentence-BERT embeddings.")
        return model.encode
    except (ImportError, RuntimeError):
        log.warning("sentence-transformers unavailable – falling back to TF-IDF")
        tfidf = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1,2))
        return lambda texts: tfidf.fit_transform(texts).toarray()

def to_py(obj: Any):
    """Recursively cast numpy / pandas scalars to native Python types."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj

def one_hot_encoder():
    """Return a OneHotEncoder with correct kwarg for the installed sklearn."""
    import sklearn
    skl_ver = tuple(map(int, sklearn.__version__.split(".")[:2]))
    if skl_ver >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)
# --------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Embeddings + numeric + categorical -> single dense matrix."""
    # --- 1) sentence embeddings on activity_sequence ------------------
    encoder = load_sentence_model()
    embeds  = encoder(df["activity_sequence"].fillna("").tolist())
    if isinstance(embeds, list):      # TF-IDF fall-back
        embeds = np.array(embeds, dtype=np.float32)

    # --- 2) numeric engineered features ------------------------------
    num_cols = []
    if "seq_length" not in df:
        df["seq_length"] = df["activity_sequence"].str.count(r"\|").fillna(0) + 1
    num_cols.append("seq_length")

    numeric = df[num_cols].values.astype(np.float32)
    numeric = StandardScaler().fit_transform(numeric)

    # --- 3) categorical low-cardinality features ---------------------
    cat_cols = []
    for col in df.columns:
        if col in ("activity_sequence", "intent_augmented", "intent_base"):
            continue
        if df[col].dtype == object and df[col].nunique() <= 20:
            cat_cols.append(col)

    if cat_cols:
        ohe = one_hot_encoder()
        cat_mat = ohe.fit_transform(df[cat_cols])
        if not isinstance(cat_mat, np.ndarray):
            cat_mat = cat_mat.toarray()
    else:
        cat_mat = np.empty((len(df), 0), dtype=np.float32)

    # --- 4) concat ----------------------------------------------------
    features = np.hstack([embeds, numeric, cat_mat])
    feat_names = (
        [f"emb_{i}" for i in range(embeds.shape[1])]
        + num_cols
        + ([f"{c}={v}" for c in cat_cols for v in df[c].unique()
           if str(v) in df[c].unique()] if cat_cols else [])
    )
    return features.astype(np.float32), feat_names

def pca_reduce(mat: np.ndarray, var_keep=PCA_VAR_KEEP) -> np.ndarray:
    pca = PCA(n_components=var_keep, random_state=RAND_SEED)
    red = pca.fit_transform(mat)
    kept = red.shape[1]
    log.info(f"PCA kept {kept} comps – {pca.explained_variance_ratio_.sum()*100:.2f} % variance")
    return red

def evaluate_kmeans(mat: np.ndarray, k_values: List[int]) -> Tuple[int, Dict[int, Tuple[float, np.ndarray]]]:
    results: Dict[int, Tuple[float, np.ndarray]] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RAND_SEED, n_init="auto")
        labels = km.fit_predict(mat)
        uniq = np.unique(labels)
        if len(uniq) <= 1:
            log.warning(f"k={k} collapsed to a single cluster – skipped")
            continue
        sil = silhouette_score(mat, labels,
                               sample_size=min(SAMPLE_SIZE_SIL, len(mat)),
                               random_state=RAND_SEED)
        results[k] = (sil, labels)
        log.info(f"k = {k} -> silhouette = {sil:.3f}")
    if not results:
        raise RuntimeError("All k-values collapsed to one cluster; try a different range.")
    best_k = max(results, key=lambda k: results[k][0])
    log.info(f"Selected k = {best_k} (silhouette {results[best_k][0]:.3f})")
    return best_k, results

def plot_elbow(inertias: Dict[int, float]):
    ks, vals = zip(*sorted(inertias.items()))
    plt.figure(figsize=(6,4))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow curve")
    best_k = ks[np.argmin(vals)]
    plt.axvline(best_k, ls="--", c="red")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "elbow.png", dpi=300)
    plt.close()

def plot_scatter(pca_mat: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(6,6))
    palette = sns.color_palette("bright", np.unique(labels).size)
    for lab in np.unique(labels):
        sel = labels == lab
        plt.scatter(pca_mat[sel, 0], pca_mat[sel, 1],
                    s=4, alpha=0.7, label=f"C{lab}", color=palette[lab])
    plt.title("Sub-intent clusters (PCA dims 1-2)")
    plt.legend(markerscale=3, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_scatter.png", dpi=300)
    plt.close()

def plot_top_terms(df: pd.DataFrame, labels: np.ndarray, top_n=12):
    tfidf = TfidfVectorizer(min_df=5, max_df=0.8)
    mat   = tfidf.fit_transform(df["activity_sequence"].fillna(""))
    vocab = np.array(tfidf.get_feature_names_out())
    k     = np.unique(labels).size
    fig, axes = plt.subplots(k, 1, figsize=(6, k*2.2))
    if k == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        sel = labels == i
        mean_vec = mat[sel].mean(axis=0).A1
        idx = np.argsort(mean_vec)[-top_n:][::-1]
        ax.barh(np.arange(top_n)[::-1], mean_vec[idx][::-1])
        ax.set_yticks(np.arange(top_n)[::-1])
        ax.set_yticklabels(vocab[idx][::-1], fontsize=6)
        ax.set_title(f"Cluster {i} – distinctive terms")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_feature_bars.png", dpi=300)
    plt.close()

# ---------------------------- main -----------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    log.info(f"Using data file: {DATA_PATH}")
    df_all = pd.read_csv(DATA_PATH, low_memory=False)
    if MAX_ROWS: df_all = df_all.head(MAX_ROWS)

    df = df_all[df_all["intent_augmented"] == INTENT_FILTER].reset_index(drop=True)
    log.info(f"Loaded {len(df):,} rows; {df_all.shape[0]-len(df):,} others filtered out")

    # build features
    feat_mat, feat_names = build_feature_matrix(df)

    # PCA
    red_mat = pca_reduce(feat_mat)

    # evaluate K
    inertias = {}
    k_results = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=RAND_SEED, n_init="auto")
        labels = km.fit_predict(red_mat)
        inertias[k] = km.inertia_
        k_results[k] = labels
    plot_elbow(inertias)

    # choose with silhouette (with guard)
    best_k, sil_results = evaluate_kmeans(red_mat, K_VALUES)
    labels = sil_results[best_k][1]

    # visualisations
    plot_scatter(red_mat, labels)
    plot_top_terms(df, labels)

    # cluster sizes
    size_series = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(6,3))
    size_series.plot(kind="bar")
    plt.title("Cluster size distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_size_hist.png", dpi=300)
    plt.close()

    # save clustered CSV
    out_csv = OUT_DIR / "transfer_clusters.csv"
    df_out  = df.copy()
    df_out["sub_intent_cluster"] = labels
    df_out.to_csv(out_csv, index=False)
    log.info(f"Wrote clustered rows ➜ {out_csv}")

    # JSON summary
    summary = {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "total_rows": int(len(df)),
        "k_tested": K_VALUES,
        "selected_k": int(best_k),
        "silhouette": float(sil_results[best_k][0]),
        "cluster_sizes": to_py(size_series.to_dict())
    }
    with open(OUT_DIR / "cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Finished ✓")

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()