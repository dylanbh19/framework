#!/usr/bin/env python
"""
transferanalysis.py  –  discover sub-intents inside “Transfer”

Run with no arguments: script auto-detects best_augmented_data.csv
Run with --input path/to/file.csv  to use a custom file.

Outputs (folder: transfer_subintent_results):
  • transfer_clusters.csv           – original rows + sub_intent_k
  • transfer_cluster_summary.csv    – size & sample activity per cluster
  • cluster_size_hist.png           – PNG quick-look
  • cluster_scatter.png             – 2-D PCA scatter (if ≥2 comps)
  • elbow.png                       – elbow curve
  • transfer_subintent.log          – UTF-8 log file
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import seaborn as sns

# ---------- optional Sentence-BERT -------------------------------- #
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
# ------------------------------------------------------------------ #

# ---------- logging (UTF-8 safe on Windows) ----------------------- #
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
# ------------------------------------------------------------------ #


def autodetect_csv() -> Path | None:
    candidates = list(Path.cwd().rglob("best_augmented_data.csv"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def build_embeddings(texts: List[str]) -> np.ndarray:
    """Sentence-BERT if available, else normalised TF-IDF vectors."""
    if SBERT_AVAILABLE:
        model_name = "all-MiniLM-L6-v2"
        log.info(f"Embeddings: Sentence-BERT ({model_name})")
        model = SentenceTransformer(model_name, device="cpu")
        emb = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        return emb.astype("float32")

    log.warning("Sentence-BERT not installed – using TF-IDF fallback")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95).fit_transform(texts)
    return normalize(tfidf).toarray()


def pca_reduce(mat: np.ndarray, target_variance: float = .95, max_comps: int = 50) -> np.ndarray:
    """Return PCA-reduced matrix (retain ≥target_variance, ≤max_comps)."""
    pca = PCA(n_components=min(mat.shape[1], max_comps), random_state=42)
    red = pca.fit_transform(mat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = np.searchsorted(cum, target_variance) + 1
    if keep > red.shape[1]:   # clamp edge-case
        keep = red.shape[1]
    log.info(f"PCA: kept {keep} components – {cum[keep-1]*100:.2f}% variance")
    return red[:, :keep]


def choose_k(mat: np.ndarray, k_min=2, k_max=15) -> int:
    best_k, best_score = k_min, -1
    for k in range(k_min, min(k_max, len(mat) - 1) + 1):
        km = KMeans(k, random_state=42, n_init="auto").fit(mat)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(mat, km.labels_, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    log.info(f"Auto-selected k={best_k} (silhouette={best_score:.3f})")
    return best_k


def safe_sample(series: pd.Series) -> str:
    for v in series:
        if isinstance(v, str) and v.strip():
            return v[:150]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="augmented data CSV")
    args = ap.parse_args()

    csv_path = Path(args.input) if args.input else autodetect_csv()
    if not csv_path or not csv_path.exists():
        log.error("Could not find best_augmented_data.csv – supply with --input")
        sys.exit(1)

    log.info(f"CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    transfer_df = (df[df["intent_augmented"] == "Transfer"]
                   .loc[df["activity_sequence"].notna()]
                   .reset_index(drop=True))

    if transfer_df.empty:
        log.error("No rows with intent=Transfer and a non-null activity_sequence.")
        sys.exit(1)

    log.info(f"Rows for clustering: {len(transfer_df):,}")

    # ---------- embeddings ---------------------------------------- #
    texts = transfer_df["activity_sequence"].astype(str).tolist()
    emb   = build_embeddings(texts)

    # ---------- PCA ------------------------------------------------ #
    red = pca_reduce(emb)

    # ---------- Cluster ------------------------------------------- #
    k = choose_k(red)
    km = KMeans(k, random_state=42, n_init="auto").fit(red)
    transfer_df["sub_intent_k"] = km.labels_

    # ---------- Outputs ------------------------------------------- #
    OUTDIR.mkdir(exist_ok=True)
    clusters_csv = OUTDIR / "transfer_clusters.csv"
    transfer_df.to_csv(clusters_csv, index=False)
    log.info(f"Written: {clusters_csv}")

    summary = (transfer_df.groupby("sub_intent_k")
                         .agg(size=("sub_intent_k", "size"),
                              sample_activity=("activity_sequence", safe_sample))
                         .reset_index()
                         .sort_values("size", ascending=False))
    summary_csv = OUTDIR / "transfer_cluster_summary.csv"
    summary.to_csv(summary_csv, index=False)
    log.info(f"Written: {summary_csv}")

    sns.set_style("whitegrid")

    # cluster size histogram
    plt.figure(figsize=(6,4))
    plt.hist(summary["size"], bins=20, edgecolor="black")
    plt.title("Cluster size distribution")
    plt.xlabel("size"); plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(OUTDIR/"cluster_size_hist.png", dpi=300); plt.close()

    # 2-D scatter if possible
    if red.shape[1] >= 2:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=red[:,0], y=red[:,1], hue=km.labels_,
                        palette="husl", s=10, legend=False)
        plt.title("Sub-intent clusters (PCA first 2 comps)")
        plt.tight_layout()
        plt.savefig(OUTDIR/"cluster_scatter.png", dpi=300); plt.close()

    # elbow
    inertias=[]
    for kk in range(2, 16):
        inertias.append(KMeans(kk, random_state=42, n_init="auto").fit(red).inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(range(2,16), inertias, marker="o"); plt.axvline(k, ls="--", c="red")
    plt.title("Elbow curve"); plt.xlabel("k"); plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(OUTDIR/"elbow.png", dpi=300); plt.close()

    log.info("PNG visuals saved.")
    log.info("="*60)
    log.info("DONE %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("="*60)


if __name__ == "__main__":
    main()