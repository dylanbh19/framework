#!/usr/bin/env python
"""
transfer_subintent_analysis.py  –  v2  (no first/last_activity columns)

Identify meaningful sub-clusters within the giant “Transfer” intent bucket.
The script auto-detects your CSV, builds a composite feature space using
*all* available columns except first/last_activity (they're not present),
evaluates k-means with k={3,4}, keeps the one with the better silhouette
score, trims outliers, then produces plots + a CSV you can re-merge later.

Requires:  pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn
"""

from __future__ import annotations
import json, logging, os, sys, re, warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

warnings.filterwarnings('ignore')
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
log = logging.getLogger('transfer-subint')

OUTDIR = Path('transfer_subintent_results')
OUTDIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Locate & load data                                                       #
# --------------------------------------------------------------------------- #
def _find_augmented_csv() -> Path:
    here = Path.cwd()
    for pat in ['augmentation_results*/best_augmented_data.csv',
                '**/best_augmented_data.csv', 'best_augmented_data.csv']:
        for p in here.glob(pat):
            return p
    raise FileNotFoundError("best_augmented_data.csv not found")

CSV_PATH = _find_augmented_csv()
log.info(f"Using data file: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, low_memory=False)
log.info(f"Loaded {len(df):,} rows; "
         f"{(df['intent_augmented']=='Transfer').sum():,} with intent=Transfer")

df = df[df['intent_augmented'] == 'Transfer'].copy()
df.reset_index(drop=True, inplace=True)

# --------------------------------------------------------------------------- #
# 2. Feature engineering                                                      #
# --------------------------------------------------------------------------- #
# -- 2a  sentence embedding of full sequence -------------------------------- #
if _SBERT_OK:
    device = 'cpu'   # force CPU; avoids surprises on older machines
    model_name = 'all-MiniLM-L6-v2'
    log.info(f"Using Sentence-BERT embeddings model={model_name}.")
    sbert = SentenceTransformer(model_name, device=device)

    seq_text = df['activity_sequence'].fillna('').astype(str).tolist()
    batches, emb_list = 0, []
    BATCH = 512
    for i in range(0, len(seq_text), BATCH):
        emb = sbert.encode(seq_text[i:i+BATCH], show_progress_bar=False,
                           normalize_embeddings=True)
        emb_list.append(emb)
        batches += 1
    embed_matrix = np.vstack(emb_list)
else:
    log.warning("sentence-transformers not installed -- falling back to TF-IDF only")
    embed_matrix = np.empty((len(df), 0), dtype='float32')

# -- 2b  TF-IDF bag of activities ------------------------------------------- #
act_seqs = df['activity_sequence'].fillna('').str.replace('|', ' ')
tfidf = TfidfVectorizer(ngram_range=(1,1), min_df=2, max_features=4000)
tfidf_mat = tfidf.fit_transform(act_seqs).astype('float32')

# -- 2c  simple numeric features -------------------------------------------- #
df['seq_length'] = df['activity_sequence'].fillna('').str.count(r'\|') + 1
numeric = df[['seq_length']]
if 'intent_confidence' in df.columns:
    numeric['intent_confidence'] = df['intent_confidence'].fillna(0)

num_mat = StandardScaler().fit_transform(numeric).astype('float32')

# -- 2d  categorical (auto detect low-cardinality non-numeric cols) ---------- #
cat_cols = []
for col in df.columns:
    if col in ['activity_sequence', 'intent_augmented', 'intent_base']: continue
    if df[col].dtype == object and df[col].nunique() <= 20:
        cat_cols.append(col)

if cat_cols:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
    cat_mat = ohe.fit_transform(df[cat_cols])
else:
    cat_mat = np.empty((len(df), 0))

# -- 2e  concatenate all                                                      #
from scipy import sparse
X_sparse = sparse.hstack([tfidf_mat, cat_mat], format='csr')
X_dense  = np.hstack([embed_matrix, num_mat])

from scipy.sparse import csr_matrix
X_all = sparse.hstack([csr_matrix(X_dense), X_sparse], format='csr')
log.info(f"Final feature matrix: {X_all.shape}")

# --------------------------------------------------------------------------- #
# 3. Dimensionality reduction (PCA for scatter + speed)                       #
# --------------------------------------------------------------------------- #
def _pca_reduce(mat, variance_keep=0.95, max_comps=50) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=min(max_comps, mat.shape[1]), svd_solver='randomized')
    reduced = pca.fit_transform(mat.toarray())
    var_cum = pca.explained_variance_ratio_.cumsum()
    k = np.searchsorted(var_cum, variance_keep) + 1
    log.info(f"PCA kept {k} comps – {var_cum[k-1]*100:5.2f} % variance")
    return reduced[:,:k], pca

X_pca2d, _ = _pca_reduce(X_all, variance_keep=0.95)

# outlier removal (3× MAD)
med = np.median(X_pca2d, axis=0)
mad = np.median(np.abs(X_pca2d - med), axis=0)
mask = (np.abs(X_pca2d - med) < 3*mad).all(axis=1)
X_pca = X_pca2d[mask]
X_cluster = X_all[mask]
df_clean = df.loc[mask].reset_index(drop=True)
log.info(f"Trimmed {len(df)-len(df_clean):,} outliers")

# --------------------------------------------------------------------------- #
# 4. KMeans for k in {3,4,5}                                                  #
# --------------------------------------------------------------------------- #
best_k, best_score, best_labels = None, -1, None
for k in (3,4,5):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
    labels = km.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels, sample_size=10000, random_state=42)
    log.info(f"k = {k:>2} ▸ silhouette = {score:0.3f}")
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

log.info(f"Selected k = {best_k} (silhouette {best_score:0.3f})")

df_clean['subintent_id'] = best_labels

# --------------------------------------------------------------------------- #
# 5.  Plots                                                                   #
# --------------------------------------------------------------------------- #
palette = sns.color_palette('tab10', best_k)
plt.figure(figsize=(6,6))
for cl in range(best_k):
    idx = df_clean['subintent_id'] == cl
    plt.scatter(X_pca[idx,0], X_pca[idx,1], s=4, alpha=0.6,
                label=f'C{cl}', color=palette[cl])
plt.legend(markerscale=2, fontsize=8, title='cluster')
plt.title('Sub-intent clusters (PCA dims 1-2)')
plt.tight_layout()
plt.savefig(OUTDIR/'cluster_scatter.png', dpi=300)
plt.close()

# ------------ distinguishing words per cluster (simple ΔTF-IDF) ------------ #
from sklearn.feature_extraction.text import TfidfVectorizer
texts = act_seqs[mask].tolist()
tf_all = TfidfVectorizer(ngram_range=(1,1), min_df=5).fit(texts)
M = tf_all.transform(texts)

fig, axs = plt.subplots(best_k, 1, figsize=(5, 2*best_k))
if best_k == 1: axs = [axs]
feature_names = np.array(tf_all.get_feature_names_out())

for cl in range(best_k):
    row_bool = (df_clean['subintent_id']==cl).values
    mean_tfidf = np.asarray(M[row_bool].mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[::-1][:15]
    sns.barplot(x=mean_tfidf[top_idx],
                y=feature_names[top_idx],
                ax=axs[cl], color=palette[cl])
    axs[cl].set_title(f'Cluster {cl} – distinctive terms')
    axs[cl].set_xlabel('')
    axs[cl].set_ylabel('')
plt.tight_layout()
plt.savefig(OUTDIR/'cluster_feature_bars.png', dpi=300)
plt.close()

# cluster size histogram
plt.figure(figsize=(5,3))
sns.histplot(df_clean['subintent_id'].value_counts(), bins=best_k, discrete=True)
plt.title('Cluster size distribution')
plt.xlabel('size')
plt.tight_layout()
plt.savefig(OUTDIR/'cluster_size_hist.png', dpi=300)
plt.close()

# --------------------------------------------------------------------------- #
# 6.  Save artefacts                                                          #
# --------------------------------------------------------------------------- #
(df_clean
   .assign(original_index=df_clean.index)
   .to_csv(OUTDIR/'transfer_clusters.csv', index=False))

# JSON summary – convert float32 -> float to avoid serialisation errors
summary = {
    'generated_at'  : datetime.now().isoformat(timespec='seconds'),
    'best_k'        : int(best_k),
    'silhouette'    : float(best_score),
    'cluster_sizes' : df_clean['subintent_id'].value_counts().to_dict(),
    'feature_counts': int(X_all.shape[1])
}
with open(OUTDIR/'clustering_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

log.info('✔ All artefacts written to %s', OUTDIR.resolve())
# --------------------------------------------------------------------------- #


if __name__ == '__main__':
    # nothing special to do; importing this file runs everything.
    pass