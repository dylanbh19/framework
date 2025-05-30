# augmentation_explorer.py  –  full robust version
import argparse, sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ── CLI ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(
    description="Deep-dive visualisation of intent-augmentation results")
ap.add_argument("--outdir", default="augmentation_results",
                help="folder containing method_comparison.csv")
ap.add_argument("--top", type=int, default=15,
                help="Top-N intents to show per method")
ap.add_argument("--excel", action="store_true",
                help="write .xlsx copies of key tables")
ap.add_argument("--sample", type=int, default=50,
                help="sample rows to print / export")
args = ap.parse_args()

OUT   = Path(args.outdir).resolve()
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── locate required files ───────────────────────────────────────
summary_path = next(OUT.glob("method_comparison*.csv"), None)
best_path    = next(OUT.glob("best_augmented_data*.csv"), None)
if not summary_path or not best_path:
    sys.exit("❌  Could not find method_comparison.csv or "
             "best_augmented_data.csv in {}".format(OUT))

print("📂  Using results in", OUT)

# ── read summary & fix header/index issues ──────────────────────
summary = pd.read_csv(summary_path)

# 1. move index to column if necessary
if "method" not in summary.columns.str.lower().tolist():
    # check if there is an unnamed first column (common when index=True)
    if "Unnamed: 0" in summary.columns:
        summary.rename(columns={"Unnamed: 0": "method"}, inplace=True)
    else:
        summary.reset_index(inplace=True)
        summary.rename(columns={"index": "method"}, inplace=True)

# 2. normalise headers to lower-case
summary.columns = [c.lower() for c in summary.columns]

# 3. safety check
if "method" not in summary.columns:
    raise SystemExit("Still cannot locate a 'method' column after repair.\n"
                     f"Current columns: {summary.columns}")

best_df = pd.read_csv(best_path)

intent_cols = sorted([c for c in best_df.columns if c.startswith("intent_")])
baseline_col = next((c for c in best_df.columns if c.lower()=="intent_baseline"), None)

print("🛈  intent columns found:", intent_cols)
if not baseline_col:
    print("⚠️  No explicit intent_baseline column – baseline→method heat-maps skipped")

# ────────────────────────────────────────────────────────────────
# 1) OVERVIEW TABLE
print("\n=== Method overview ===")
print(summary.to_string(index=False))

if args.excel:
    summary.to_excel(OUT/"analysis_overview.xlsx", index=False)

# ────────────────────────────────────────────────────────────────
# 2) UNKNOWN-RATE BAR
plt.figure(figsize=(8,4))
sns.barplot(x="method", y=summary["unknown_rate"]*100, data=summary,
            palette="Reds_r")
plt.ylabel("Unknown (%)")
plt.title("Unknown rate by method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS/"explorer_unknown_rates.png", dpi=300)
plt.show()

# 3) IMPROVED BAR (if present) -----------------------------------
if "improved" in summary.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(x="method", y="improved", data=summary, palette="Blues_r")
    plt.ylabel("Improved records")
    plt.title("Improved-from-baseline")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS/"explorer_improved.png", dpi=300)
    plt.show()

# ────────────────────────────────────────────────────────────────
# 4) TOP-N INTENTS PER METHOD
top_n = args.top
fig, axes = plt.subplots(len(intent_cols), 1,
                         figsize=(11, 3*len(intent_cols)),
                         sharex=False)
for ax, col in zip(axes, intent_cols):
    vc = best_df[col].value_counts().head(top_n)
    sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="husl")
    ax.set_title(f"Top {top_n} intents – {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(PLOTS/"explorer_top_intents.png", dpi=300)
plt.show()

# ────────────────────────────────────────────────────────────────
# 5) BASELINE → METHOD HEAT-MAPS (if baseline present)
if baseline_col:
    for col in intent_cols:
        if col == baseline_col:
            continue
        ct = pd.crosstab(best_df[baseline_col], best_df[col])
        if ct.shape[0] > 60 or ct.shape[1] > 60:
            # keep it readable: filter to rows / cols that changed
            ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
        plt.figure(figsize=(10, max(6, 0.3*ct.shape[0])))
        sns.heatmap(np.log1p(ct), cmap="viridis")
        plt.title(f"Baseline → {col} (log1p scale)")
        plt.ylabel("Baseline")
        plt.xlabel(col)
        plt.tight_layout()
        fn = PLOTS/f"explorer_heatmap_{col}.png"
        plt.savefig(fn, dpi=300)
        plt.show()
        if args.excel:
            ct.to_excel(OUT/f"heatmap_{col}.xlsx")

# ────────────────────────────────────────────────────────────────
# 6) SAMPLE ROWS
sample_n = min(args.sample, len(best_df))
preview_cols = ([baseline_col] if baseline_col else []) + intent_cols
if "intent_augmented" in best_df.columns:
    preview_cols.append("intent_augmented")
sample_df = best_df[preview_cols].head(sample_n)

print(f"\n=== First {sample_n} rows ==================================================================")
print(sample_df.to_string(index=False))

if args.excel:
    sample_df.to_excel(OUT/"analysis_sample.xlsx", index=False)
    print("\n💾  Excel exports written to", OUT)

print("\n✅  Explorer finished – all plots in", PLOTS)