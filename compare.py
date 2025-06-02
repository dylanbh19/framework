# compare_activity_profiles.py  ─────────────────────────────────────────
import re, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

CSV = Path("best_augmented_data.csv")
if not CSV.exists():
    sys.exit(f"❌ {CSV} not found. Place this script in the same folder.")

df = pd.read_csv(CSV, low_memory=False)

# ── 1. which 5 intents to compare?  ────────────────────────────────────
top5 = (df["intent_augmented"]
        .value_counts()
        .loc[lambda s: s.index != "Unknown"]
        .head(5)
        .index.tolist())

intents_of_interest = ["Unknown"] + top5
print("📊  Comparing activity profiles for:", intents_of_interest)

# ── 2. explode the activity_sequence into one row per activity ─────────
def split_seq(seq):
    return [a.strip() for a in str(seq).split("|") if a.strip()]

rows = []
for intent in intents_of_interest:
    sub = df[df["intent_augmented"] == intent]
    activities = sum((split_seq(seq) for seq in sub["activity_sequence"].fillna("")), [])
    rows.append(pd.DataFrame({"intent": intent,
                              "activity": list(Counter(activities).keys()),
                              "count":   list(Counter(activities).values())}))

freq = pd.concat(rows, ignore_index=True)

# keep activities that appear in ≥1 % of any intent group (declutter plot)
min_thresh = 0.01
keep = (freq.groupby("intent")
              .apply(lambda g: g.set_index("activity")["count"] /
                                 g["count"].sum())
              .max()
              .loc[lambda s: s >= min_thresh]
              .index)

freq = freq[freq["activity"].isin(keep)]

# ── 3. convert to % within each intent ────────────────────────────────
freq["pct"] = (freq.groupby("intent")["count"]
                     .transform(lambda x: x / x.sum() * 100))

# ── 4. ridge / density style overlay plot ─────────────────────────────
plt.figure(figsize=(11, 0.5 + 1.6*len(intents_of_interest)))

for i, intent in enumerate(intents_of_interest, start=1):
    sub = freq[freq["intent"] == intent]
    sns.barplot(x="pct", y="activity", data=sub,
                label=intent, alpha=0.5)

    plt.axhline(y=len(keep)-0.5, color="#ccc", lw=0.3)  # grid line
    plt.title("Activity profile overlay: Unknown vs top-5 intents")
    plt.xlabel("Share of activities within intent (%)")

plt.legend(title="Intent")
plt.tight_layout()
plt.savefig("activity_profile_overlay.png", dpi=300)
plt.show()

print("\n✓  activity_profile_overlay.png written to", Path.cwd())