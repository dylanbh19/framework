# compare_unknown_vs_intent.py  ────────────────────────────────────────────
"""
Visual compare: Unknown vs <chosen intent> contact activities
============================================================

• Automatically detects newest augmentation_results* folder.
• Asks which intent to compare (defaults to top volume intent).
• Produces scatter plot:
      X-axis = % share of each activity inside UNKNOWN calls
      Y-axis = % share of same activity inside CHOSEN intent calls
      green △ = share inside 'Unknown ➜ predicted as CHOSEN'
• Saves PNG in scatter_compare_outputs/.
"""

from pathlib import Path
import sys, textwrap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ── 1. find newest augmentation_results folder ───────────────────────────
base = Path.cwd()
folders = sorted([p for p in base.glob("augmentation_results*") if p.is_dir()],
                 key=lambda p: p.stat().st_mtime,
                 reverse=True)
if not folders:
    sys.exit("❌  No augmentation_results* folder found")

out_dir = folders[0]
csv = next(out_dir.glob("best_augmented_data*.csv"), None)
if not csv:
    sys.exit("❌  best_augmented_data*.csv not found in " + str(out_dir))

df = pd.read_csv(csv, low_memory=False)
print("ℹ️  Loaded", csv.name, "from", out_dir)

# ── 2. choose intent ──────────────────────────────────────────────────────
top_intent = (df["intent_augmented"]
              .value_counts()
              .loc[lambda s: s.index != "Unknown"]
              .idxmax())
print("\nTop-volume intent is:", top_intent)
chosen = input("Enter intent to compare (blank = use top): ").strip() or top_intent
if chosen not in df["intent_augmented"].unique():
    sys.exit(f"❌  Intent '{chosen}' not present in data")

print("Comparing UNKNOWN ↔", chosen)

# ── 3. helpers ────────────────────────────────────────────────────────────
def explode(series):
    """'Act1|Act2' -> ['Act1','Act2', …]"""
    return [a.strip() for s in series.fillna("") for a in str(s).split("|") if a.strip()]

def pct_table(mask):
    acts = explode(df.loc[mask, "activity_sequence"])
    if not acts:
        return pd.Series(dtype=float)
    s = pd.Series(acts).value_counts()
    return s / s.sum() * 100   # percent

pct_unknown          = pct_table(df["intent_augmented"] == "Unknown")
pct_intent           = pct_table(df["intent_augmented"] == chosen)
pct_predicted_group  = pct_table(
    (df["intent_base"] == "Unknown") & (df["intent_augmented"] == chosen)
)

# make aligned dataframe
all_acts = pct_unknown.index.union(pct_intent.index).union(pct_predicted_group.index)
data = pd.DataFrame({
    "pct_unknown":          pct_unknown.reindex(all_acts, fill_value=0),
    "pct_intent":           pct_intent.reindex(all_acts, fill_value=0),
    "pct_predicted_group":  pct_predicted_group.reindex(all_acts, fill_value=0)
}).reset_index().rename(columns={"index":"activity"})

# ── 4. scatter plot ───────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))
plt.scatter(data["pct_unknown"], data["pct_intent"],
            s=40, alpha=0.65, label="All " + chosen, color="#ff7f0e")
plt.scatter(data["pct_unknown"], data["pct_predicted_group"],
            s=45, marker="^", alpha=0.75, label="Unknown ➜ " + chosen,
            color="#2ca02c")
plt.axline((0,0), slope=1, linestyle="--", color="#999", lw=1)

# label biggest outliers
for _, row in data.sort_values("pct_intent", ascending=False).head(10).iterrows():
    plt.text(row["pct_unknown"]+0.1, row["pct_intent"]+0.1,
             row["activity"], fontsize=8)

plt.xlabel("% share inside UNKNOWN")
plt.ylabel(f"% share inside {chosen}")
plt.title(f"Activity distribution – Unknown vs {chosen}")
plt.legend()
plt.tight_layout()

save_dir = Path("scatter_compare_outputs"); save_dir.mkdir(exist_ok=True)
fname = save_dir / f"scatter_unknown_vs_{chosen.replace(' ','_')}.png"
plt.savefig(fname, dpi=300)
plt.show()

print("\n✓ Plot saved to", fname)