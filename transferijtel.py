#!/usr/bin/env python
"""
transferanalysis.py
===================

Identify data-driven *sub-intents* inside the broad “Transfer” category.

Main steps
----------
1. Load augmented-intent CSV (auto-discovery or --input)
2. Keep rows where intent_augmented == "Transfer" and activity_sequence present
3. Vectorise sequences  – Sentence-BERT if available, else TF-IDF
4. Optional PCA (retain ≥95 % variance, cap 50 comps)
5. Auto-choose k (2-15) with cosine-silhouette
6. K-Means cluster
7. Write:
   • transfer_clusters.csv  – original rows + `sub_intent_k`
   • transfer_cluster_summary.csv – size & sample_activity per cluster
   • three PNG plots in same output folder
8. INFO-level logging streamed to console + UTF-8 file.

Requires
--------
pandas numpy scikit-learn matplotlib seaborn
sentence-transformers (optional, improves embeddings)

"""

from __future__ import annotations
import argparse, logging, sys, json, os
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

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# ------------------------------------------------------------------#
# Logging (UTF-8 safe on Windows)                                   #
# ------------------------------------------------------------------#
LOGDIR = Path("transfer_subintent_results")
LOGDIR.mkdir(exist_ok=True, parents=True)
log_file = LOGDIR / "transfer_subintent.log"

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch  = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
root_logger.addHandler(ch)

fh  = logging.FileHandler(log_file, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
root_logger.addHandler(fh)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------#
# Utility functions                                                 #
# ------------------------------------------------------------------#
def autodetect_csv() -> Path | None:
    """Return the most recent best_augmented_data.csv under CWD."""
    cdir = Path.cwd()
    candidates = list(cdir.rglob("best_augmented_data.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def choose_k_by_silhouette(mat: np.ndarray, k_min=2, k_max=15) -> int:
    """Pick k that maximises cosine-silhouette."""
    best_k, best_score = k_min, -1
    for k in range(k_min, min(k_max, len(mat) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(mat)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(mat, labels, metric="cosine")
        if score > best_score:
            best_k, best_score = k, score
    log.info(f"[AUTO-K] Chose k={best_k} (silhouette={best_score:.3f})")
    return best_k


def build_embeddings(texts: List[str]) -> np.ndarray:
    """Return 2-D array of sentence embeddings."""
    if SBERT_AVAILABLE:
        model_name = "all-MiniLM-L6-v2"
        log.info(f"Using Sentence-BERT embeddings model={model_name}")
        model = SentenceTransformer(model_name, device="cpu")
        emb = model.encode(texts, show_progress_bar=True, batch_size=256, normalize_embeddings=True)
        return emb.astype("float32")
    log.warning("Sentence-BERT not available – falling back to TF-IDF")
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)
    tfidf = vec.fit_transform(texts)
    return normalize(tfidf).toarray()


def pca_reduce(mat: np.ndarray, var_threshold=0.95, max_comps=50) -> np.ndarray:
    pca = PCA(n_components=min(mat.shape[1], max_comps), random_state=42)
    mat_r = pca.fit_transform(mat)
    # keep enough comps for required variance
    cum = np.cumsum(pca.explained_variance_ratio_)
    keep = np.searchsorted(cum, var_threshold) + 1
    mat_r = mat_r[:, :keep]
    log.info(f"PCA: kept {keep} comps – {cum[keep-1]*100:.2f} % variance")
    return mat_r


def safe_sample(series: pd.Series) -> str:
    for v in series:
        if isinstance(v, str) and v.strip():
            return v[:150]
    return ""


# ------------------------------------------------------------------#
# Core                                                               #
# ------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description="Cluster Transfer calls into sub-intents")
    parser.add_argument("--input", help="Path to best_augmented_data.csv")
    args = parser.parse_args()

    csv_path = Path(args.input) if args.input else autodetect_csv()
    if not csv_path or not csv_path.exists():
        log.error("Could not locate best_augmented_data.csv – provide with --input")
        sys.exit(1)

    log.info(f"Using data file: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows; {df['intent_augmented'].eq('Transfer').sum():,} with intent=Transfer")

    # ------------------------------------------------------------------#
    # Prep subset                                                       #
    # ------------------------------------------------------------------#
    transfer_df = (
        df[(df["intent_augmented"] == "Transfer") &
           (df["activity_sequence"].notna()) &
           (df["activity_sequence"].str.len() > 0)]
          .reset_index(drop=True)
    )
    if transfer_df.empty:
        log.error("No usable Transfer rows with activity_sequence; aborting.")
        sys.exit(1)

    texts = transfer_df["activity_sequence"].astype(str).tolist()

    # ------------------------------------------------------------------#
    # Embeddings & Dim-red                                              #
    # ------------------------------------------------------------------#
    emb   = build_embeddings(texts)
    red   = pca_reduce(emb)

    # ------------------------------------------------------------------#
    # Cluster                                                           #
    # ------------------------------------------------------------------#
    k_opt   = choose_k_by_silhouette(red, k_min=2, k_max=15)
    kmeans  = KMeans(n_clusters=k_opt, random_state=42, n_init="auto")
    labels  = kmeans.fit_predict(red)
    transfer_df["sub_intent_k"] = labels

    # ------------------------------------------------------------------#
    # Outputs                                                           #
    # ------------------------------------------------------------------#
    LOGDIR.mkdir(exist_ok=True, parents=True)
    clusters_csv = LOGDIR / "transfer_clusters.csv"
    transfer_df.to_csv(clusters_csv, index=False)
    log.info(f"Wrote clustered rows -> {clusters_csv}")

    # summary CSV
    summary = (
        transfer_df.groupby("sub_intent_k")
                   .agg(size=("sub_intent_k", "size"),
                        sample_activity=("activity_sequence", safe_sample))
                   .reset_index()
                   .sort_values("size", ascending=False)
    )
    summary_csv = LOGDIR / "transfer_cluster_summary.csv"
    summary.to_csv(summary_csv, index=False)
    log.info(f"Wrote summary        -> {summary_csv}")

    # ------------------------------------------------------------------#
    # Quick visuals                                                     #
    # ------------------------------------------------------------------#
    sns.set_style("whitegrid")

    # (a) cluster size hist
    plt.figure(figsize=(6,4))
    plt.hist(summary["size"], bins=20, edgecolor="black")
    plt.title("Distribution of cluster sizes")
    plt.xlabel("Cluster size")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(LOGDIR/"cluster_size_hist.png", dpi=300)
    plt.close()

    # (b) 2-D scatter (first 2 PCA comps)
    if red.shape[1] >= 2:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=red[:,0], y=red[:,1], hue=labels, palette="husl", s=10, legend=False)
        plt.title("Transfer sub-intent clusters (PCA-2D)")
        plt.xlabel("PC-1"); plt.ylabel("PC-2")
        plt.tight_layout()
        plt.savefig(LOGDIR/"cluster_scatter.png", dpi=300)
        plt.close()

    # (c) elbow plot (inertia for k=2..15)
    inertias = []
    for k in range(2, 16):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(red)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(range(2,16), inertias, marker="o")
    plt.axvline(k_opt, ls="--", c="red", label=f"k*={k_opt}")
    plt.title("Elbow curve")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOGDIR/"elbow.png", dpi=300)
    plt.close()

    log.info("PNG plots saved.")

    # ------------------------------------------------------------------#
    # Finish                                                            #
    # ------------------------------------------------------------------#
    log.info("="*60)
    log.info(f"Finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Clusters  : {k_opt}")
    log.info(f"Row count : {len(transfer_df):,}")
    log.info("="*60)


if __name__ == "__main__":
    main()