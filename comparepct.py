# compare_activity_share.py  ────────────────────────────────────────────
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("husl"); plt.style.use("seaborn-v0_8-darkgrid")

# ── Locate newest augmentation_results* folder ─────────────────────────
cwd = Path.cwd()
res_dirs = sorted([p for p in cwd.glob("augmentation_results*") if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
if not res_dirs:
    sys.exit("❌  no augmentation_results* folder found")

res   = res_dirs[0]
best  = next(res.glob("best_augmented_data*.csv"), None)
if not best:
    sys.exit("❌  best_augmented_data*.csv not found in " + str(res))

df = pd.read_csv(best, low_memory=False)
print("✔  Loaded", best.name)

# ── choose source column with original intents ─────────────────────────
src_col = next((c for c in ["Intent", "intent_source", "intent_base"]
                if c in df.columns), None)
if not src_col:
    sys.exit("❌  could not find original intent column (Intent / intent_source)")

# ── Top-3 real intents (exclude Unknown) ───────────────────────────────
top3 = (df[src_col]
        .value_counts()
        .loc[lambda s: s.index.str.lower() != "unknown"]
        .head(3)
        .index.tolist())
print("Top-3 baseline intents:", top3)

# helper to explode sequence field
def explode(seq_series):
    return [a.strip() for seq in seq_series.fillna("")
                          for a in str(seq).split("|") if a.strip()]

def pct_table(mask):
    acts = explode(df.loc[mask, "activity_sequence"])
    if not acts:
        return pd.Series(dtype=float)
    freq = pd.Series(acts).value_counts()
    return (freq / freq.sum() * 100).round(2)  # %

# ── build baseline & predicted tables ─────────────────────────────────
baseline_tabs  = {}
predicted_tabs = {}

for intent in top3:
    baseline_tabs[intent]  = pct_table(df[src_col] == intent)
    predicted_tabs[intent] = pct_table(
        (df[src_col].str.lower() == "unknown") &
        (df["intent_augmented"] == intent)
    )

# join into DataFrames
baseline_df  = pd.concat(baseline_tabs, axis=1).fillna(0).sort_index()
predicted_df = pd.concat(predicted_tabs, axis=1).fillna(0).sort_index()

# ── export to Excel ───────────────────────────────────────────────────
out_dir = Path("activity_share_outputs"); out_dir.mkdir(exist_ok=True)
baseline_df.to_excel(out_dir / "activity_share_baseline.xlsx")
predicted_df.to_excel(out_dir / "activity_share_predicted.xlsx")
print("✓  Excel sheets written to", out_dir)

# ── optional stacked bar plot for quick glance ────────────────────────
plt.figure(figsize=(10, 6))
baseline_df[top3].head(15).T.plot(kind="bar", stacked=True, ax=plt.gca())
plt.title("Baseline – activity share (top 15 activities)")
plt.ylabel("% within intent")
plt.legend(title="Activity", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(out_dir / "activity_share_top3.png", dpi=300)
plt.close()
print("✓  Stacked-bar PNG saved")

print("\nDone – compare the two Excel files to verify similarity " 
      "between baseline and model-predicted activity mixes.")