# ╔══════════════════════════════════════════════════════════════╗
# ║  augmentation_explorer.py – bullet-proof deep-dive utility   ║
# ║  • auto-repairs method_comparison.csv quirks                 ║
# ║  • recalculates unknown & improved if missing                ║
# ║  • produces bar charts, heat-maps, top-N bars, sample table  ║
# ╚══════════════════════════════════════════════════════════════╝
import argparse, sys, itertools, math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ── CLI ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Explore augmentation results")
ap.add_argument("--outdir", default="augmentation_results",
                help="folder containing method_comparison.csv etc.")
ap.add_argument("--top", type=int, default=20,
                help="Top-N intents to show per method")
ap.add_argument("--sample", type=int, default=50,
                help="rows to print from augmented dataset")
ap.add_argument("--excel", action="store_true",
                help="also export XLSX files for manual QA")
args = ap.parse_args()

OUT   = Path(args.outdir).resolve()
PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)

summary_path = next(OUT.glob("method_comparison*.csv"), None)
best_path    = next(OUT.glob("best_augmented_data*.csv"), None)
if not summary_path or not best_path:
    sys.exit("❌  Could not locate both method_comparison*.csv "
             "and best_augmented_data*.csv in {}".format(OUT))

print(f"📂  Reading   {summary_path.name}")
summary_raw = pd.read_csv(summary_path)
best_df     = pd.read_csv(best_path)     # main augmented dataset

# ─────────────────────────────────────────────────────────────────
# HELPER: ensure a canonical `method` column exists
# ─────────────────────────────────────────────────────────────────
def fix_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1) move index to column if header missing
    if "method" not in df.columns.str.lower().tolist():
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "method"}, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "method"}, inplace=True)
    # 2) lower-case all headers
    df.columns = [c.lower() for c in df.columns]
    # 3) final assurance
    if "method" not in df.columns:
        raise ValueError("Could not coerce a 'method' column; "
                         f"current columns: {df.columns}")
    return df

summary = fix_summary(summary_raw)

# ─────────────────────────────────────────────────────────────────
# If unknown_rate / improved missing, recompute from best_df
# ─────────────────────────────────────────────────────────────────
baseline_col = next((c for c in best_df.columns
                     if c.lower() == "intent_baseline"), None)
intent_cols   = sorted([c for c in best_df.columns if c.startswith("intent_")])

if "unknown_rate" not in summary.columns or \
   ("improved" not in summary.columns and baseline_col):

    print("🔄  Recomputing unknown_rate / improved from augmented dataset …")
    rows = []
    for col in intent_cols:
        rate = (best_df[col] == "Unknown").mean()
        row  = {"method": col.replace("intent_", ""),
                "unknown_rate": rate}
        if baseline_col:
            improved = ((best_df[baseline_col] == "Unknown") &
                        (best_df[col]       != "Unknown")).sum()
            row["improved"] = improved
        rows.append(row)
    summary_calc = pd.DataFrame(rows)

    # merge with original runtime if present
    summary = summary.drop(
        columns=[c for c in ("unknown_rate", "improved") if c in summary.columns],
        errors="ignore"
    ).merge(summary_calc, on="method", how="right")

# sort for nice plotting
summary.sort_values("unknown_rate", inplace=True)

print("\n=== Method overview =============================")
print(summary.to_string(index=False))

if args.excel:
    summary.to_excel(OUT / "analysis_overview.xlsx", index=False)

# ─────────────────────────────────────────────────────────────────
# PLOT 1 – Unknown-rate %
# ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
sns.barplot(x="method", y=summary["unknown_rate"]*100, data=summary,
            color="#d95c5c")
plt.ylabel("Unknown (%)")
plt.title("Unknown rate by method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOTS / "explorer_unknown_rate.png", dpi=300)
plt.show()

# PLOT 2 – Improved count (if we have it)
if "improved" in summary.columns:
    plt.figure(figsize=(8,4))
    sns.barplot(x="method", y="improved", data=summary, color="#5c9ad9")
    plt.ylabel("Improved records")
    plt.title("Improved-from-baseline")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "explorer_improved.png", dpi=300)
    plt.show()

# ─────────────────────────────────────────────────────────────────
# PLOT 3 – Top-N intents for every method
# ─────────────────────────────────────────────────────────────────
top_n = args.top
rows  = len(intent_cols)
fig, axes = plt.subplots(rows, 1, figsize=(11, 3*rows), sharex=False)

for ax, col in zip(axes, intent_cols):
    top = best_df[col].value_counts().head(top_n)
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="husl")
    ax.set_title(f"Top {top_n} intents – {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(PLOTS / f"explorer_top{top_n}_intents.png", dpi=300)
plt.show()

# ─────────────────────────────────────────────────────────────────
# PLOT 4 – Baseline → Method log heat-maps  (if baseline present)
# ─────────────────────────────────────────────────────────────────
if baseline_col:
    for col in intent_cols:
        if col == baseline_col:     # skip self
            continue
        ct = pd.crosstab(best_df[baseline_col], best_df[col])
        # keep it readable – strip all-zero rows/cols
        ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
        if ct.empty:
            continue
        plt.figure(figsize=(max(8, 0.5*ct.shape[1]),
                            max(6, 0.5*ct.shape[0])))
        sns.heatmap(np.log1p(ct), cmap="viridis")
        plt.title(f"{baseline_col} → {col} (log1p counts)")
        plt.xlabel(col)
        plt.ylabel("baseline")
        plt.tight_layout()
        plt.savefig(PLOTS / f"explorer_heatmap_{col}.png", dpi=300)
        plt.close()

        if args.excel:
            ct.to_excel(OUT / f"heatmap_{col}.xlsx")

# ─────────────────────────────────────────────────────────────────
# SAMPLE ROWS
# ─────────────────────────────────────────────────────────────────
preview_cols = ([baseline_col] if baseline_col else []) + intent_cols
if "intent_augmented" in best_df.columns:
    preview_cols.append("intent_augmented")

sample_n = min(args.sample, len(best_df))
print(f"\n=== First {sample_n} rows (truncated columns) ===============")
print(best_df[preview_cols].head(sample_n).to_string(index=False))

if args.excel:
    best_df[preview_cols].head(sample_n).to_excel(
        OUT / "analysis_sample.xlsx", index=False)

print("\n✅  Explorer done – new plots in", PLOTS)
if args.excel:
    print("💾  XLSX exports in", OUT)