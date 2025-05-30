# augmentation_analysis.py  (fixed column-name case)
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

# ── CLI ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="augmentation_results",
                help="folder that contains method_comparison.csv")
ap.add_argument("--top", type=int, default=15,
                help="top-N intents to plot for each method")
args = ap.parse_args()

OUT = Path(args.outdir).resolve()
summary_path = OUT / "method_comparison.csv"
best_path    = OUT / "best_augmented_data.csv"

if not summary_path.exists() or not best_path.exists():
    raise SystemExit("❌  Could not find method_comparison.csv or best_augmented_data.csv; "
                     "check --outdir path.")

print(f"📂  Using results in: {OUT}")

# ── load & tidy summary ─────────────────────────────────────────
summary = pd.read_csv(summary_path)
summary.columns = [c.lower() for c in summary.columns]   # <<< NEW
if "method" not in summary.columns:
    # fall back to original Capitalised spelling
    summary = summary.rename(columns={"Method": "method"})

best_df = pd.read_csv(best_path)

intent_cols = [c for c in best_df.columns if c.startswith("intent_")]
print(f"🛈  detected intent columns: {intent_cols}")

# ── Plot 1: Unknown-rate % ──────────────────────────────────────
plt.figure(figsize=(8,4))
sns.barplot(data=summary, x="method", y=summary["unknown_rate"]*100,
            color="#d95c5c")
plt.ylabel("Unknown (%)")
plt.title("Unknown rate by method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
(OUT/"plots").mkdir(exist_ok=True, parents=True)
plt.savefig(OUT/"plots"/"analysis_unknown_rates.png", dpi=300)
plt.show()

# ── Plot 2: Improved count ─────────────────────────────────────
if "improved" in summary.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(data=summary, x="method", y="improved", color="#5c9ad9")
    plt.ylabel("Improved records")
    plt.title("Improved-from-baseline")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT/"plots"/"analysis_improved.png", dpi=300)
    plt.show()

# ── Plot 3: Top-N intents per method ───────────────────────────
top_n = args.top
rows = len(intent_cols)
fig, axes = plt.subplots(rows, 1, figsize=(10, 3*rows), sharex=False)

for ax, col in zip(axes, intent_cols):
    vc = best_df[col].value_counts().head(top_n)
    sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="husl")
    ax.set_title(f"Top {top_n} intents – {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(OUT/"plots"/"analysis_top_intents.png", dpi=300)
plt.show()

# ── Preview first 50 rows ──────────────────────────────────────
sample_cols = ["intent_baseline"] + intent_cols
if "intent_augmented" in best_df.columns:
    sample_cols.append("intent_augmented")

print("\n🖥️  First 50 rows (baseline vs methods)\n")
print(best_df[sample_cols].head(50).to_string(index=False))

best_df[sample_cols].head(50).to_excel(
    OUT/"analysis_sample.xlsx", index=False)

print("\n✅  Analysis complete.  Plots saved to augmentation_results/plots/")