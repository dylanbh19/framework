"""
transferanalysis.py  –  sub-intent discovery inside “Transfer”
==============================================================
• Filters rows where intent_augmented == "Transfer"
• Sentence-BERT embeddings  (all-MiniLM-L6-v2, CPU-safe)
• PCA → 95 % variance  (≈30 comps)
• Runs KMeans for k = 3 **and** k = 4, keeps the one with the
  higher mean silhouette score
• Outputs
   ├─ transfer_clusters.csv           (all “Transfer” rows + cluster id)
   ├─ cluster_scatter.png             (2-D scatter, PCA dims 1–2)
   ├─ cluster_feature_bars.png        (top 15 TF-IDF terms per cluster)
   ├─ cluster_top_features.csv        (cluster, term, score)
   └─ cluster_summary.json            (metadata & scores)
"""

import argparse, logging, json, sys, warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("pip install sentence-transformers")

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --------------------------------------------------------------------------- #
#                            helpers                                          #
# --------------------------------------------------------------------------- #
def setup_logging(outdir: Path):
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "logs" / "transfer_analysis.log",
                                encoding="utf-8"),
        ],
    )

def embed(text: pd.Series, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    logging.info("Sentence-BERT model %s  (device=%s)",
                 model_name, model._target_device)
    return model.encode(text.tolist(),
                        batch_size=256,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                        device="cpu")

def pca95(x: np.ndarray):
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    x_red = pca.fit_transform(x)
    logging.info("PCA kept %d comps – %.2f %% variance",
                 x_red.shape[1], pca.explained_variance_ratio_.sum()*100)
    return x_red

def fit_kmeans(data: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(data)
    sil = silhouette_score(data, labels)
    logging.info("k = %d  →  silhouette = %.3f", k, sil)
    return labels, sil

def tfidf_top(text: pd.Series, labels, top_n=15):
    vec = TfidfVectorizer(max_features=10_000,
                          ngram_range=(1,3),
                          token_pattern=r"(?u)\b\w+\b")
    tfidf = vec.fit_transform(text)
    vocab = np.array(vec.get_feature_names_out())

    tops, rows = {}, []
    for c in np.unique(labels):
        mask = labels == c
        scores = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        idx = scores.argsort()[::-1][:top_n]
        tops[int(c)] = list(zip(vocab[idx], scores[idx].round(3)))
        for t, s in tops[int(c)]:
            rows.append({"cluster": int(c), "term": t, "score": s})
    return tops, pd.DataFrame(rows)

def plot_scatter(pca2, labels, path):
    plt.figure(figsize=(8,7))
    palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
    for lab, col in zip(np.unique(labels), palette):
        pts = pca2[labels == lab]
        plt.scatter(pts[:,0], pts[:,1], s=6, c=[col], label=f"C{lab}")
    plt.title("Sub-intent clusters (PCA first 2 comps)")
    plt.legend(markerscale=2, frameon=False, ncol=len(np.unique(labels)))
    plt.tight_layout();  plt.savefig(path, dpi=300);  plt.close()

def plot_bars(tops, path):
    n = len(tops)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2*n), sharex=False)
    if n == 1: axes = [axes]
    for ax, (cl, pairs) in zip(axes, tops.items()):
        pairs = list(reversed(pairs))  # smallest→largest for horiz bar
        ax.barh([t for t,_ in pairs], [s for _,s in pairs], color="steelblue")
        ax.set_title(f"Cluster {cl} – distinctive terms")
    plt.tight_layout();  plt.savefig(path, dpi=300);  plt.close()

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="augmentation_results/best_augmented_data.csv")
    ap.add_argument("--outdir", default="transfer_subintent_results")
    args = ap.parse_args()

    outdir = Path(args.outdir);  outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)

    df = pd.read_csv(args.input, low_memory=False)
    mask = df["intent_augmented"] == "Transfer"
    df_t = df.loc[mask].reset_index(drop=True)
    logging.info("Loaded %d rows,  %d tagged as Transfer",
                 len(df), len(df_t))

    # text for embedding
    text = df_t["activity_sequence"].fillna(
           df_t.get("first_activity", "")).astype(str)

    emb   = embed(text)
    red   = pca95(emb)

    # try k = 3 and 4
    results = {}
    for k in (3,4):
        labels, sil = fit_kmeans(red, k)
        results[k]  = {"labels": labels, "sil": sil}

    best_k = max(results, key=lambda k: results[k]["sil"])
    labels = results[best_k]["labels"]
    df_t["sub_cluster"] = labels
    logging.info("Selected k = %d (silhouette %.3f)",
                 best_k, results[best_k]["sil"])

    # top terms
    tops, df_terms = tfidf_top(text, labels, top_n=15)

    # outputs -----------------------------------------------------------------
    df_t.to_csv(outdir / "transfer_clusters.csv", index=False)
    plot_scatter(red[:,:2], labels, outdir / "cluster_scatter.png")
    plot_bars(tops, outdir / "cluster_feature_bars.png")
    df_terms.to_csv(outdir / "cluster_top_features.csv", index=False)

    summary = {
        "generated":   datetime.now().isoformat(timespec="seconds"),
        "total_rows":  len(df),
        "transfer_rows": len(df_t),
        "silhouette": {k: round(v["sil"],3) for k,v in results.items()},
        "chosen_k":    best_k,
        "cluster_sizes": df_t["sub_cluster"].value_counts().to_dict(),
    }
    with open(outdir / "cluster_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Outputs written to %s", outdir.absolute())

if __name__ == "__main__":
    main()