from sklearn.metrics import silhouette_score, calinski_harabasz_score,\
                            davies_bouldin_score

def evaluate_clusters(mat, labels) -> dict:
    """Return a dict of common clustering metrics."""
    # cosine for silhouette because we used normalised vectors
    sil  = silhouette_score(mat, labels, metric="cosine")
    ch   = calinski_harabasz_score(mat, labels)
    db   = davies_bouldin_score(mat, labels)
    return {"silhouette_cos": sil, "calinski_harabasz": ch, "davies_bouldin": db}
    
    
    
    metrics = evaluate_clusters(red, km.labels_)
log.info("Cluster quality  | "
         f"silhouette={metrics['silhouette_cos']:.3f}  "
         f"CH={metrics['calinski_harabasz']:.1f}  "
         f"DB={metrics['davies_bouldin']:.3f}")
        
        
        
    all_k_stats = []
for kk in range(2, 9):
    km_tmp = KMeans(kk, random_state=42, n_init="auto").fit(red)
    m      = evaluate_clusters(red, km_tmp.labels_)
    all_k_stats.append(dict(k=kk, **m))
pd.DataFrame(all_k_stats).to_csv(OUTDIR/"k_sweep_metrics.csv", index=False)




# after transfer_df['sub_intent_k'] is ready …
df = pd.read_csv(csv_path, low_memory=False)          # full set
df["transfer_sub_intent"] = pd.NA                    # default NA

df.loc[transfer_df.index, "transfer_sub_intent"] = transfer_df["sub_intent_k"]
tagged_path = OUTDIR / "full_dataset_with_transfer_tags.csv"
df.to_csv(tagged_path, index=False)
log.info(f"Full dataset with tags ➜ {tagged_path}")


















# ------------------------------------------------------------------ #
#  (Paste from here ↓↓↓ – keep identical indentation)                #
# ------------------------------------------------------------------ #
### === FEATURE-IMPORTANCE BY CLUSTER ============================ ###
#
#  We work directly with TF-IDF weights on activity_sequence.
#  For every cluster k we compute the mean TF-IDF vector and pull
#  the top «N» terms that make that cluster distinctive.
#  Two artefacts are generated:
#     • transfer_cluster_features.csv   (wide table, top terms per k)
#     • cluster_features_k{n}.png       (one horizontal bar-plot / k)
#
########################################################################

TOP_N = 10  # how many keywords per cluster to surface

log.info("Calculating TF-IDF feature importance …")
tfidf_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b"
).fit(texts)

X_tfidf = tfidf_vec.transform(texts)          # (rows = transfer_df order)
terms   = np.array(tfidf_vec.get_feature_names_out())

feature_rows = []          # for CSV export

for cid in sorted(summary["sub_intent_k"].unique()):
    row_mask   = transfer_df["sub_intent_k"].values == cid
    if row_mask.sum() == 0:
        continue

    # average TF-IDF weight across rows in this cluster
    mean_vec   = np.asarray(X_tfidf[row_mask].mean(axis=0)).ravel()
    top_idx    = mean_vec.argsort()[::-1][:TOP_N]
    top_terms  = terms[top_idx]
    top_scores = mean_vec[top_idx]

    # store for CSV
    feature_rows.append({
        "cluster": cid,
        **{f"kw_{i+1}": kw for i, kw in enumerate(top_terms)}
    })

    # ------- bar-plot per cluster ---------------------------------- #
    plt.figure(figsize=(6, 3))
    sns.barplot(x=top_scores, y=top_terms, palette="viridis")
    plt.title(f"Cluster {cid} – top {TOP_N} keywords")
    plt.xlabel("mean TF-IDF weight"); plt.ylabel("")
    plt.tight_layout()
    png_path = OUTDIR / f"cluster_features_k{cid}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    log.info(f"   feature plot → {png_path.name}")

# ---------- CSV with keywords per cluster ------------------------- #
feat_csv = OUTDIR / "transfer_cluster_features.csv"
pd.DataFrame(feature_rows).to_csv(feat_csv, index=False)
log.info(f"Written: {feat_csv}")
### === END FEATURE-IMPORTANCE SECTION =========================== ###
# ------------------------------------------------------------------







# ---------- 2-D scatter (one colour per cluster) ------------------ #
if red.shape[1] >= 2:
    plt.figure(figsize=(6, 5))

    # Build a discrete palette containing exactly k distinct hues
    palette = sns.color_palette("husl", n_colors=k)

    sns.scatterplot(
        x=red[:, 0],
        y=red[:, 1],
        hue=transfer_df["sub_intent_k"].astype(str),  # treat labels as categorical
        palette=palette,
        s=12,
        edgecolor="none",
        legend="full",
    )

    plt.title("Sub-intent clusters (PCA first 2 comps)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "cluster_scatter.png", dpi=300)
    plt.close()
