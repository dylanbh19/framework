# explain_intents.py  ----------------------------------------------------------
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

sns.set_palette("husl"); plt.style.use("seaborn-v0_8-darkgrid")

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True, help="best_augmented_data.csv")
ap.add_argument("--rf_tokens", default="rf_top_tokens.csv",
                help="feature-importance csv from ML stage")
ap.add_argument("--top", type=int, default=10, help="top-N intents to plot")
args = ap.parse_args()

df = pd.read_csv(args.file)
out = Path("explainability_outputs"); out.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 1. feature importance
if Path(args.rf_tokens).exists():
    tok = pd.read_csv(args.rf_tokens).head(30)
    plt.figure(figsize=(6,6))
    sns.barplot(x="importance", y="token", data=tok, color="#4c72b0")
    plt.title("Random-Forest – top tokens overall")
    plt.tight_layout(); plt.savefig(out/"exp_feature_importance.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------ 2. confidence boxplot
plt.figure(figsize=(10,5))
top_ints = df["intent_augmented"].value_counts().head(args.top).index
sns.boxplot(x="intent_augmented", y="intent_confidence",
            data=df[df["intent_augmented"].isin(top_ints)],
            showfliers=False)
plt.xticks(rotation=45, ha="right"); plt.ylabel("Confidence")
plt.title(f"Confidence distribution – top {args.top} intents")
plt.tight_layout(); plt.savefig(out/"exp_confidence_boxplot.png", dpi=300); plt.close()

# ------------------------------------------------------------------ 3.  sample rows for QA
sample_cols = ["intent_augmented","intent_confidence","aug_method",
               "intent_source"      if "intent_source"      in df.columns else None,
               "explain_zeroshot"   if "explain_zeroshot"   in df.columns else None,
               "first_activity"     if "first_activity"     in df.columns else None,
               "last_activity"      if "last_activity"      in df.columns else None,
               "activity_sequence"  if "activity_sequence"  in df.columns else None,
              ]
sample_cols = [c for c in sample_cols if c]
df.sample(200, random_state=1)[sample_cols].to_excel(out/"exp_sample.xlsx", index=False)

print("✓ Explainability artefacts written to", out)



row = df.loc[df['intent_augmented'] == 'Transfer'].iloc[0]

print("Predicted intent :", row['intent_augmented'])
print("Confidence       :", row['intent_confidence'])
print("Came from method :", row['aug_method'])
print("Zero-shot tokens :", row.get('explain_zeroshot', '—'))
print("\nFirst / Last activity:")
print("  ➜", row.get('first_activity', '—'), "→", row.get('last_activity', '—'))
print("\nFull sequence:")
print(row.get('activity_sequence', '—')[:300], "…")



import shap, joblib
rf    = joblib.load("augmentation_results_pro/models/rf.pkl")
tfidf = joblib.load("augmentation_results_pro/models/tfidf.pkl")

# pick 2k random rows to keep SHAP quick
mask = df["intent_augmented"].isin(["Transfer","Repeat Caller"])
X    = tfidf.transform(df.loc[mask,"activity_sequence"].fillna(""))
expl = shap.TreeExplainer(rf)
shap_values = expl.shap_values(X[:2000])

shap.summary_plot(shap_values, tfidf.inverse_transform(X[:2000]),
                  feature_names=tfidf.get_feature_names_out(), show=False)
plt.savefig("explainability_outputs/shap_summary.png", dpi=300)





