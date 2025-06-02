#!/usr/bin/env python
"""
transfer_subintent_pipeline.py
==============================
Discover & visualise sub-intents inside the broad "Transfer" bucket.

• No parameters needed – the script auto-detects your augmented data file.
• Generates:
    ├─ transfer_subintent_results/
    │   ├─ transfer_clusters.csv               (rows + cluster_id)
    │   ├─ cluster_summary.csv / .html         (explainability)
    │   ├─ model.pkl                           (PCA + Clustering + Embed info)
    │   ├─ plots/
    │   │   ├─ cluster_sizes.png
    │   │   ├─ tsne_scatter.png
    │   │   └─ activity_heatmap.png
    │   └─ dashboards/
    │       └─ tsne_scatter.html               (interactive)
    └─ logs/*.log

Requires
--------
pip install pandas numpy matplotlib seaborn plotly scikit-learn
# optional but recommended
pip install sentence-transformers hdbscan

Run
---
python transfer_subintent_pipeline.py
"""
import warnings, logging, sys, re, json, pickle
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("transfer_subintent.log"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("transfer_subintent")

# --------------------------------------------------------------------- #
# 0.  Helper – auto-detect the augmented data file
# --------------------------------------------------------------------- #
def _find_augmented_file() -> Path|None:
    patterns = [
        "augmentation_results*/best_augmented_data.csv",
        "**/best_augmented_data.csv",
        "best_augmented_data.csv",
        "*.csv"
    ]
    here = Path.cwd()
    candidates = []
    for pat in patterns:
        for p in here.glob(pat):
            if p.is_file():
                try:
                    header = p.open("r", encoding="utf-8").readline().lower()
                    if "intent_augmented" in header:
                        candidates.append(p)
                except Exception:
                    continue
    if not candidates:
        return None
    # newest first
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

# --------------------------------------------------------------------- #
# 1.  Load & preprocess
# --------------------------------------------------------------------- #
def _clean_text(t:str)->str:
    return re.sub(r"[^A-Za-z0-9 ]+"," ",str(t)).lower().strip()

def load_transfer_rows(path:Path):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    # include any row whose *final* intent is "Transfer" (whatever the base was)
    trans = df[df["intent_augmented"].str.lower()=="transfer"].copy()
    if trans.empty:
        log.error("No rows with intent 'Transfer' found – aborting.")
        sys.exit(1)
    log.info(f"Loaded {len(df):,} rows; {len(trans):,} with intent=Transfer.")
    # build text field (activities + misc)
    text = (
        trans["activity_sequence"].fillna("")+" "+
        trans.get("first_activity","").fillna("")+" "+
        trans.get("last_activity","").fillna("")
    ).apply(_clean_text)
    trans["__text"] = text
    return trans

# --------------------------------------------------------------------- #
# 2.  Embedding – SBERT  → fallback TF-IDF
# --------------------------------------------------------------------- #
def embed(text_series):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Using Sentence-BERT embeddings.")
        emb = model.encode(text_series.tolist(), batch_size=256, show_progress_bar=True)
        return emb, {"type":"sbert","model":"all-MiniLM-L6-v2"}
    except Exception as e:
        log.warning(f"Sentence-BERT unavailable – falling back to TF-IDF ({e})")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=12000, ngram_range=(1,2))
        emb = vec.fit_transform(text_series).toarray()
        return emb, {"type":"tfidf","vocab":len(vec.vocabulary_)}

# --------------------------------------------------------------------- #
# 3.  Dimensionality reduction
# --------------------------------------------------------------------- #
def reduce_dims(embed_matrix, n_components=30):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    red = pca.fit_transform(embed_matrix)
    log.info(f"PCA: kept {n_components} comps  –  {pca.explained_variance_ratio_.sum():.2%} var.")
    return red, pca

# --------------------------------------------------------------------- #
# 4.  Clustering – HDBSCAN  → fallback   K-Means
# --------------------------------------------------------------------- #
def cluster(mat, min_cluster_size=150):
    try:
        import hdbscan
        clus = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric="euclidean",
                               cluster_selection_method="eom")
        labels = clus.fit_predict(mat)
        algo = "hdbscan"
        noise = (labels==-1).sum()
        log.info(f"HDBSCAN produced {labels.max()+1} clusters  (+ noise={noise}).")
    except Exception as e:
        log.warning(f"HDBSCAN unavailable – falling back to KMeans ({e})")
        from sklearn.cluster import KMeans
        k = max(2,mat.shape[0]//min_cluster_size)
        clus = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = clus.fit_predict(mat)
        algo = "kmeans"
        log.info(f"KMeans with k={k} produced {k} clusters.")
    return labels, clus, algo

# --------------------------------------------------------------------- #
# 5.  Explainability helpers
# --------------------------------------------------------------------- #
def summarise_clusters(df, labels, topn=12):
    df["cluster_id"]=labels
    summary=[]
    for cid,g in df.groupby("cluster_id"):
        if cid==-1: continue  # skip noise
        words = " ".join(g["__text"]).split()
        common = [w for w,_ in Counter(words).most_common(topn)]
        summary.append({
            "cluster_id":cid,
            "rows":len(g),
            "top_terms":" ".join(common),
            "sample_activity":g["activity_sequence"].iloc[0][:150]+"…"
        })
    import pandas as pd
    summ = pd.DataFrame(summary).sort_values("rows",ascending=False)
    return summ

# --------------------------------------------------------------------- #
# 6.  Visualisations
# --------------------------------------------------------------------- #
def viz_cluster_sizes(summary, outdir:Path):
    import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
    plt.figure(figsize=(8,4))
    sns.barplot(x="cluster_id",y="rows",data=summary)
    plt.title("Cluster size (Transfer sub-intents)")
    plt.ylabel("Rows")
    plt.savefig(outdir/"plots"/"cluster_sizes.png",dpi=300,bbox_inches="tight")
    plt.close()

def viz_tsne(mat, labels, outdir:Path):
    import matplotlib.pyplot as plt, seaborn as sns
    from sklearn.manifold import TSNE
    tsne=TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    pts=tsne.fit_transform(mat)
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=pts[:,0],y=pts[:,1],hue=labels,legend=False,palette="tab10",s=14)
    plt.title("t-SNE of Transfer calls (colour = cluster)")
    plt.savefig(outdir/"plots"/"tsne_scatter.png",dpi=300,bbox_inches="tight")
    plt.close()
    # interactive
    try:
        import plotly.express as px
        fig=px.scatter(x=pts[:,0],y=pts[:,1],color=labels,
                       title="t-SNE Transfer sub-intents",opacity=0.7)
        fig.write_html(outdir/"dashboards"/"tsne_scatter.html")
    except Exception:
        pass

def viz_activity_heatmap(df, outdir:Path, top_act=20):
    import matplotlib.pyplot as plt, seaborn as sns, pandas as pd
    act_counts=defaultdict(Counter)
    for cid,g in df.groupby("cluster_id"):
        if cid==-1: continue
        for seq in g["activity_sequence"].fillna(""):
            for a in seq.split("|"):
                act_counts[cid][a]+=1
    # pick global top activities
    total_counts=Counter()
    for c in act_counts.values():
        total_counts.update(c)
    acts=[a for a,_ in total_counts.most_common(top_act)]
    clusters=sorted(k for k in act_counts.keys())
    import numpy as np
    mat=np.zeros((len(clusters),len(acts)))
    for i,c in enumerate(clusters):
        total=sum(act_counts[c].values())
        for j,a in enumerate(acts):
            mat[i,j]=act_counts[c][a]/total if total else 0
    plt.figure(figsize=(12,6))
    sns.heatmap(mat,xticklabels=acts,yticklabels=[f"cl{c}" for c in clusters],
                cmap="YlOrBr",cbar_kws={"label":"Rel. frequency"})
    plt.title("Top activities per Transfer cluster")
    plt.ylabel("Cluster")
    plt.xticks(rotation=45,ha="right")
    plt.savefig(outdir/"plots"/"activity_heatmap.png",dpi=300,bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------- #
# 7.  Main orchestration
# --------------------------------------------------------------------- #
def main():
    data_path=_find_augmented_file()
    if not data_path:
        log.error("Cannot find augmented data CSV with 'intent_augmented' column.")
        sys.exit(1)
    log.info(f"Using data file: {data_path}")

    # folders
    outdir=Path("transfer_subintent_results")
    (outdir/"plots").mkdir(parents=True, exist_ok=True)
    (outdir/"dashboards").mkdir(exist_ok=True)

    df=load_transfer_rows(data_path)
    emb,em_meta=embed(df["__text"])
    red,pca=reduce_dims(emb, n_components=30)
    labels,clusterer,algo=cluster(red)

    # attach clusters & create tag column (not yet named)
    df["cluster_id"]=labels
    df["intent_sub"] = df["cluster_id"].apply(lambda c: f"Transfer::cluster_{c}" if c!=-1 else "Transfer::noise")

    # write main CSV
    df.to_csv(outdir/"transfer_clusters.csv",index=False)
    log.info(f"Wrote clustered rows → {outdir/'transfer_clusters.csv'}")

    # explainability
    summary=summarise_clusters(df,labels)
    summary.to_csv(outdir/"cluster_summary.csv",index=False)
    summary.to_html(outdir/"cluster_summary.html",index=False)
    log.info("Top clusters:\n"+summary.head(10).to_string(index=False))

    # visuals
    viz_cluster_sizes(summary,outdir)
    viz_tsne(red,labels,outdir)
    viz_activity_heatmap(df,outdir)

    # persist model artefacts
    with open(outdir/"model.pkl","wb") as f:
        pickle.dump({"pca":pca,
                     "clusterer":clusterer,
                     "embed_meta":em_meta,
                     "algo":algo},f)
    # metadata json
    meta={
        "created":datetime.now().isoformat(timespec="seconds"),
        "source_file":str(data_path),
        "rows_transfer":int(len(df)),
        "clusters":int(summary.shape[0]),
        "embedding":em_meta,
        "clustering":algo
    }
    (outdir/"metadata.json").write_text(json.dumps(meta,indent=2))
    log.info(f"Pipeline complete – results in {outdir.resolve()}")

if __name__=="__main__":
    main()