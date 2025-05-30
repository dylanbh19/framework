# ╔══════════════════════════════════════════════════════════════╗
# ║  augmentation_analysis.py                                   ║
# ║  Explore & plot results of intent-augmentation comparison   ║
# ╚══════════════════════════════════════════════════════════════╝
"""
Run from the command-line **after** intent_augmentation_comparison has
finished:

    python augmentation_analysis.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

# ── CLI ─────────────────────────────────────────────────────────
ap = argparse.ArgumentParser("Explore augmentation results")
ap.add_argument("--outdir", default="augmentation_results",
                help="folder that contains method_comparison.csv")
ap.add_argument("--top", type=int, default=15,
                help="how many top intents to plot per method")
args = ap.parse_args()

OUT = Path(args.outdir).resolve()

comp_path  = OUT / "method_comparison.csv"
best_path  = OUT / "best_augmented_data.csv"

if not comp_path.exists() or not best_path.exists():
    raise SystemExit(f"❌  Expected {comp_path} and {best_path} – "
                     "check --outdir value.")

print(f"📂  Using results in: {OUT}")

# ── load summary & best dataset ────────────────────────────────
summary = pd.read_csv(comp_path)
best_df = pd.read_csv(best_path)

# detect which per-method columns exist
intent_cols = [c for c in best_df.columns if c.startswith("intent_")]
print(f"🛈  Detected intent columns: {intent_cols}")

# ── PLOT 1 : Unknown-rate bar ──────────────────────────────────
plt.figure(figsize=(8,4))
sns.barplot(data=summary, x="method", y=summary["unknown_rate"]*100, color="#d95c5c")
plt.ylabel("Unknown (%)")
plt.title("Unknown rate by method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
png1 = OUT / "plots" / "analysis_unknown_rates.png"
plt.savefig(png1, dpi=300)
plt.show()

# ── PLOT 2 : Improved count bar ───────────────────────────────
if "improved" in summary.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(data=summary, x="method", y="improved", color="#5c9ad9")
    plt.ylabel("Records fixed")
    plt.title("Improved-from-baseline by method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    png2 = OUT / "plots" / "analysis_improved.png"
    plt.savefig(png2, dpi=300)
    plt.show()

# ── PLOT 3 : Top-N intents per method ─────────────────────────
top_n = args.top
n_methods = len(intent_cols)
fig, axes = plt.subplots(
    n_methods, 1, figsize=(10, 3*n_methods), sharex=False)

for ax, col in zip(axes, intent_cols):
    vc = best_df[col].value_counts().head(top_n)
    sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="husl")
    ax.set_title(f"Top {top_n} intents – {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")

plt.tight_layout()
png3 = OUT / "plots" / "analysis_top_intents.png"
plt.savefig(png3, dpi=300)
plt.show()

# ── VIEW SAMPLE ROWS ───────────────────────────────────────────
sample_cols = ["intent_baseline"] + intent_cols
if "intent_augmented" in best_df.columns:
    sample_cols.append("intent_augmented")

print("\n🖥️  Showing first 50 rows (baseline vs each method):")
display_df = best_df[sample_cols].head(50)
print(display_df.to_string())

# optional: save to Excel for manual QA
excel_path = OUT / "analysis_sample.xlsx"
display_df.to_excel(excel_path, index=False)
print(f"\n💾  Sample saved to {excel_path}")

print("\n✅  Analysis complete.  Plots saved to:")
for f in (png1, png2, png3):
    print("   ", f)