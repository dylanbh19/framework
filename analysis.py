"""
augmentation_explorer.py – robust visual audit
----------------------------------------------
Usage:  python augmentation_explorer.py  [--outdir results_folder]  [--top 20]
Produces:
  • unknown-rate & improved bar charts
  • Top-N intent bars for every method
  • baseline→method heat-maps (if baseline present)
  • optional Excel exports  (add --excel)
"""

import argparse, sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ── CLI ─────────────────────────────────────────────────────────
ap=argparse.ArgumentParser()
ap.add_argument("--outdir",default="augmentation_results",
                help="folder with method_comparison.csv")
ap.add_argument("--top",type=int,default=20)
ap.add_argument("--excel",action="store_true")
ap.add_argument("--sample",type=int,default=50)
args=ap.parse_args()

OUT=Path(args.outdir).resolve(); PLOT=OUT/"plots"; PLOT.mkdir(exist_ok=True)

summary_path = next(OUT.glob("method_comparison*.csv"),None)
best_path    = next(OUT.glob("best_augmented_data*.csv"),None)
if not summary_path or not best_path:
    sys.exit("files missing in "+str(OUT))

summary = pd.read_csv(summary_path)
# rescue method column
if "method" not in summary.columns and "Method" in summary.columns:
    summary.rename(columns={"Method":"method"},inplace=True)
if "method" not in summary.columns and "Unnamed: 0" in summary.columns:
    summary.rename(columns={"Unnamed: 0":"method"},inplace=True)
if "method" not in summary.columns:
    summary.reset_index(inplace=True); summary.rename(columns={"index":"method"},inplace=True)
summary.columns=[c.lower() for c in summary.columns]

best = pd.read_csv(best_path)
intent_cols=sorted([c for c in best.columns if c.startswith("intent_")])
baseline_col=next((c for c in best.columns if c.lower()=="intent_baseline"),None)

print("\nOverview:\n", summary.to_string(index=False))

# Unknown bar
plt.figure(figsize=(8,4))
sns.barplot(x="method",y=summary["unknown_rate"]*100,data=summary,color="#d95c5c")
plt.ylabel("Unknown (%)"); plt.title("Unknown rate"); plt.xticks(rotation=45,ha="right")
plt.tight_layout(); plt.savefig(PLOT/"ex_unknown.png",dpi=300); plt.show()

# Improved bar
if "improved" in summary.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(x="method",y="improved",data=summary,color="#5c9ad9")
    plt.title("Improved from baseline"); plt.xticks(rotation=45,ha="right")
    plt.tight_layout(); plt.savefig(PLOT/"ex_improved.png",dpi=300); plt.show()

# Top-N intents per method
rows=len(intent_cols)
fig,axes=plt.subplots(rows,1,figsize=(11,3*rows))
for ax,col in zip(axes,intent_cols):
    vc=best[col].value_counts().head(args.top)
    sns.barplot(x=vc.values,y=vc.index,ax=ax)
    ax.set_title(f"Top {args.top} – {col}"); ax.set_xlabel("Count"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(PLOT/"ex_top_intents.png",dpi=300); plt.show()

# Heat-maps
if baseline_col:
    for col in intent_cols:
        if col==baseline_col: continue
        ct=pd.crosstab(best[baseline_col],best[col])
        if ct.empty: continue
        plt.figure(figsize=(max(6,0.4*ct.shape[1]),max(6,0.4*ct.shape[0])))
        sns.heatmap(np.log1p(ct),cmap="viridis")
        plt.title(f"{baseline_col} → {col} (log1p)"); plt.tight_layout()
        plt.savefig(PLOT/f"ex_heat_{col}.png",dpi=300); plt.close()
        if args.excel: ct.to_excel(OUT/f"heat_{col}.xlsx")

# sample
cols=[baseline_col] if baseline_col else []
cols+=intent_cols
if "intent_augmented" in best.columns: cols.append("intent_augmented")
print("\nSample rows:\n", best[cols].head(args.sample).to_string(index=False))
if args.excel:
    summary.to_excel(OUT/"ex_overview.xlsx",index=False)
    best[cols].head(args.sample).to_excel(OUT/"ex_sample.xlsx",index=False)
print("\nPlots saved to", PLOT)