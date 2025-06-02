# intent_explain_runner.py  ── fixed 2025-05-30 ────────────────────────────
"""
• Auto-detect newest augmentation_results* folder
• Load best_augmented_data.csv
• Build overlay plot comparing activity distributions
• Write PNG + Excel sample to explainability_outputs/
"""

import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("husl"); plt.style.use("seaborn-v0_8-darkgrid")

root = Path.cwd()
folders = sorted([p for p in root.glob("augmentation_results*") if p.is_dir()],
                 key=lambda p: p.stat().st_mtime, reverse=True)
if not folders:
    sys.exit("❌  No augmentation_results* folder here")
OUT = folders[0]
print("ℹ️  Using", OUT)

csv = next(OUT.glob("best_augmented_data*.csv"), None)
if not csv:
    sys.exit("❌  best_augmented_data*.csv missing in " + str(OUT))
df = pd.read_csv(csv, low_memory=False)

# ── intents to compare ────────────────────────────────────────────────────
top5 = (df["intent_augmented"]
        .value_counts()
        .loc[lambda s: s.index != "Unknown"]
        .head(5)
        .index.tolist())
groups = ["Unknown"] + top5
print("👉  Groups:", groups)

# ── explode activity_sequence ─────────────────────────────────────────────
def explode(seq):
    return [a.strip() for a in str(seq).split("|") if a.strip()]

exp_rows = []
for intent in groups:
    acts = sum((explode(s) for s in df.loc[df["intent_augmented"] == intent,
                                           "activity_sequence"].fillna("")), [])
    cnt = pd.Series(acts).value_counts()
    exp_rows.append(pd.DataFrame({"intent": intent,
                                  "activity": cnt.index,
                                  "count": cnt.values}))
freq = pd.concat(exp_rows, ignore_index=True)

# ── keep activities that appear ≥1 % in ANY group ────────────────────────
pivot = (freq.pivot_table(index="activity", columns="intent",
                          values="count", fill_value=0))
pct   = pivot.div(pivot.sum()) * 100
keep  = pct.max(axis=1).ge(1.0)      # ≥1 %
freq  = freq[freq["activity"].isin(keep.index[keep])]

# convert to percentage within intent
freq["pct"] = (freq.groupby("intent")["count"]
                     .transform(lambda x: x / x.sum() * 100))

# ── plot overlay ─────────────────────────────────────────────────────────
plt.figure(figsize=(11, 0.5 + 1.8*len(keep)))
for idx, intent in enumerate(groups, start=1):
    sub = freq[freq["intent"] == intent]
    sns.barplot(x="pct", y="activity", data=sub,
                label=intent, alpha=0.55)
plt.title("Activity share per intent (overlay)")
plt.xlabel("Share within intent (%)")
plt.legend()
plt.tight_layout()

exp_dir = root / "explainability_outputs"
exp_dir.mkdir(exist_ok=True)
plt.savefig(exp_dir / "activity_profile_overlay.png", dpi=300)
plt.close()
print("✓  activity_profile_overlay.png written to", exp_dir)

# ── Excel QA sample (unchanged) ──────────────────────────────────────────
cols = ["intent_augmented","intent_confidence","aug_method",
        *(c for c in ["explain_zeroshot","intent_source",
                      "first_activity","last_activity","activity_sequence"]
          if c in df.columns)]
df.sample(min(200, len(df)), random_state=2)[cols] \
  .to_excel(exp_dir / "exp_sample.xlsx", index=False)
print("✓  exp_sample.xlsx written to", exp_dir)