# ╔══════════════════════════════════════════════════════════════╗
# ║  augmentation_explorer.py  –  deep-dive utility             ║
# ║  • Auto-detect every “intent_*” column                       ║
# ║  • Handles METHOD vs method vs Method header mismatches      ║
# ║  • Gracefully skips missing fields / plots                   ║
# ║  • Generates:                                                ║
# ║      – Overview table                                        ║
# ║      – Unknown-rate & improved charts                        ║
# ║      – Top-N intent bars for *each* method                   ║
# ║      – Crosstab heat-maps baseline→method                    ║
# ║      – Optional Excel and parquet exports for QA             ║
# ╚══════════════════════════════════════════════════════════════╝
import argparse, sys, itertools
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ── CLI ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Explore intent-augmentation outputs in depth")
parser.add_argument("--outdir", default="augmentation_results",
                    help="directory containing method_comparison.csv etc.")
parser.add_argument("--top", type=int, default=15,
                    help="Top-N intents to plot for each method")
parser.add_argument("--excel", action="store_true",
                    help="also write analysis_*.xlsx files for manual QA")
parser.add_argument("--sample", type=int, default=50,
                    help="rows to sample for the preview table")
args = parser.parse_args()

OUT = Path(args.outdir).resolve()
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True, parents=True)

summary_path = next(OUT.glob("method_comparison*.csv"), None)
aug_path     = next(OUT.glob("best_augmented_data*.csv"), None)

if not summary_path or not aug_path:
    sys.exit("❌  Could not locate method_comparison.csv or "
             "best_augmented_data.csv in {}".format(OUT))

print(f"📂  Using results in {OUT}")

# ── load summary & baseline/augmented dataset ───────────────────
summary = pd.read_csv(summary_path)
summary.columns = [c.lower() for c in summary.columns]
if "method" not in summary.columns:
    summary.rename(columns=lambda c: c.lower(), inplace=True)   # second attempt
best_df = pd.read_csv(aug_path)

# detect all per-method columns
intent_cols = sorted([c for c in best_df.columns if c.startswith("intent_")])
print("🛈  Detected intent columns:", intent_cols)

# baseline column (if present)
baseline_col = next((c for c in best_df.columns
                     if c.lower() == "intent_baseline"), None)
if baseline_col is None:
    print("⚠️  No intent_baseline column – baseline→method heat-map will be skipped")

# ── 1. Overview table (print & optional Excel) ──────────────────
print("\n=== Method overview =============================")
print(summary.to_string(index=False))

if args.excel:
    summary.to_excel(OUT / "analysis_overview.xlsx", index=False)

# ── 2. Unknown-rate bar chart ───────────────────────────────────
plt.figure(figsize=(8,4))
sns.barplot(x=summary["method"], y=summary["unknown_rate"]*100, color="#d95c5c")
plt.ylabel("Unknown (%)")
plt.title("Unknown rate by method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS / "explorer_unknown_rates.png", dpi=300)
plt.show()

# ── 3. Improved count chart (if available) ──────────────────────
if "improved" in summary.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(x=summary["method"], y=summary["improved"], color="#5c9ad9")
    plt.ylabel("Improved records")
    plt.title("Improved-from-baseline")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "explorer_improved.png", dpi=300)
    plt.show()

# ── 4. Top-N intent distribution per method  ────────────────────
top_n = args.top
rows = len(intent_cols)
fig, axes = plt.subplots(rows, 1, figsize=(10, 3*rows), sharex=False)

for ax, col in zip(axes, intent_cols):
    counts = best_df[col].value_counts().head(top_n)
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(f"Top {top_n} intents – {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(PLOTS / "explorer_top_intents.png", dpi=300)
plt.show()

# ── 5. Crosstab heat-map baseline → each method (optional) ──────
if baseline_col:
    for col in intent_cols:
        if col == baseline_col:          # skip baseline vs baseline
            continue
        ct = pd.crosstab(best_df[baseline_col], best_df[col])
        # show only rows/cols that changed something
        changed = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
        fig = plt.figure(figsize=(10,8))
        sns.heatmap(np.log1p(changed), cmap="viridis")
        plt.title(f"Baseline → {col} (log-scaled counts)")
        plt.xlabel(col)
        plt.ylabel("Baseline")
        plt.tight_layout()
        fname = f"explorer_heatmap_{col}.png"
        plt.savefig(PLOTS / fname, dpi=300)
        plt.show()

        if args.excel:
            changed.to_excel(OUT / f"heatmap_{col}.xlsx")

# ── 6. Preview sample rows ──────────────────────────────────────
print(f"\n=== Preview {args.sample} rows =============================")
preview_cols = [c for c in ["intent_baseline"] if c in best_df.columns] + intent_cols
if "intent_augmented" in best_df.columns:
    preview_cols.append("intent_augmented")

sample_df = best_df[preview_cols].head(args.sample)
print(sample_df.to_string(index=False))

if args.excel:
    sample_df.to_excel(OUT / "analysis_sample.xlsx", index=False)

print("\n✅  Explorer finished.  New plots saved into", PLOTS)
if args.excel:
    print("💾  Excel exports written to", OUT)