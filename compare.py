# compare_top3_intents.py  ────────────────────────────────────────────────
"""
Visual comparison of Unknown vs Top-3 predicted intents.

 • Auto-detect newest augmentation_results* folder
 • No user arguments / prompts
 • Makes one 3-panel scatter plot:
       x-axis = % of activity in UNKNOWN calls
       y-axis = % of activity in INTENT calls
       green △ = share within “Unknown ➜ INTENT” predictions
 • Saves PNG to scatter_compare_outputs/top3_scatter_unknown_vs_intents.png
"""

from pathlib import Path
import sys, itertools

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ── locate latest results folder ─────────────────────────────────────────
root = Path.cwd()
folders = sorted([p for p in root.glob("augmentation_results*") if p.is_dir()],
                 key=lambda p: p.stat().st_mtime, reverse=True)
if not folders:
    sys.exit("❌  No augmentation_results* folder found.")

res_dir = folders[0]
csv_file = next(res_dir.glob("best_augmented_data*.csv"), None)
if not csv_file:
    sys.exit("❌  best_augmented_data*.csv missing in " + str(res_dir))

df = pd.read_csv(csv_file, low_memory=False)
print("ℹ️  Loaded", csv_file)

# ── identify Unknown + top-3 intents ──────────────────────────────────────
if "intent_augmented" not in df.columns:
    sys.exit("❌  Column 'intent_augmented' not found in CSV")

vc        = df["intent_augmented"].value_counts()
top3_ints = vc.loc[lambda s: s.index != "Unknown"].head(3).index.tolist()
print("Top-3 predicted intents:", top3_ints)

# helper: explode activity_sequence --------------------------------------
def explode(seq_series):
    return [a.strip() for seq in seq_series.fillna("")
                          for a in str(seq).split("|") if a.strip()]

def pct_table(row_mask):
    acts = explode(df.loc[row_mask, "activity_sequence"])
    if not acts:                      # empty safeguard
        return pd.Series(dtype=float)
    freq = pd.Series(acts).value_counts()
    return freq / freq.sum() * 100

pct_unknown = pct_table(df["intent_augmented"] == "Unknown")

# ── prep dataframe for each intent ───────────────────────────────────────
frames = []
for intent in top3_ints:
    pct_int   = pct_table(df["intent_augmented"] == intent)
    pct_pred  = pct_table(
        (df["intent_augmented"] == intent) & (df["Intent"].fillna("Unknown") == "Unknown")
        if "Intent" in df.columns else
        (df["intent_augmented"] == intent)  # if no baseline, same mask
    )
    acts = pct_unknown.index.union(pct_int.index).union(pct_pred.index)
    frames.append(pd.DataFrame({
        "activity": acts,
        "pct_unknown": pct_unknown.reindex(acts, fill_value=0),
        "pct_intent":  pct_int.reindex(acts,   fill_value=0),
        "pct_pred":    pct_pred.reindex(acts,  fill_value=0),
        "intent": intent
    }))

plot_df = pd.concat(frames, ignore_index=True)

# ── create 3-panel scatter ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
fig.suptitle("Unknown vs Top-3 Intent Activity Distributions", fontsize=14)

for ax, intent in zip(axes, top3_ints):
    sub = plot_df[plot_df["intent"] == intent]
    ax.scatter(sub["pct_unknown"], sub["pct_intent"],
               s=35, alpha=0.6, label=intent, color="#ff7f0e")
    ax.scatter(sub["pct_unknown"], sub["pct_pred"],
               s=40, alpha=0.8, marker="^", label="Unknown→"+intent,
               color="#2ca02c")
    ax.axline((0, 0), slope=1, linestyle="--", color="#888", lw=1)
    ax.set_title(intent)
    # label top 5 outliers
    lab = sub.sort_values("pct_intent", ascending=False).head(5)
    for _, r in lab.iterrows():
        ax.text(r["pct_unknown"]+0.1, r["pct_intent"]+0.1,
                r["activity"], fontsize=7)

axes[0].set_ylabel("% share in INTENT")
for ax in axes:
    ax.set_xlabel("% share in UNKNOWN")
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_dir = Path("scatter_compare_outputs"); out_dir.mkdir(exist_ok=True)
out_path = out_dir / "top3_scatter_unknown_vs_intents.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("✓  plot saved to", out_path)