#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────────────────────
# additional.py  –  “extra-insight” toolkit to run on *best_augmented_data.csv*
#
# What it does (all optional – will skip-gracefully if a library / column is
# missing):
#   1.  Low-confidence review list  (<0.35)                      → low_confidence.csv
#   2.  Isolation-Forest sequence anomalies                     → seq_anomalies.csv
#   3.  Sankey / flow diagram of activity sequences             → flow_sankey.html
#   4.  Reliability / calibration curve                         → confidence_calib.png
#   5.  Daily volume + Prophet forecast (if call_date)          → volume_forecast.png
#   6.  BERTopic surface topics inside Transfer cluster         → topic_summary.csv
#   7.  Synthetic “uplift” placeholder (mail_flag mock)         → uplift_eval.csv
#   8.  Market indicators join & correlation heat-map           → market_corr.png
#
# Run with *no* args  → script auto-detects best_augmented_data.csv
# Run with  --input  path/to/file.csv  to use another file
# ────────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, logging, sys, json
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML / TS
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
# Prophet optional
try:
    from prophet import Prophet               # pip install prophet
    PROPHET = True
except Exception:
    PROPHET = False
# BERTopic optional
try:
    from bertopic import BERTopic             # pip install bertopic
    from umap import UMAP                     # needed by BERTopic
    BERTOPIC = True
except Exception:
    BERTOPIC = False
# Sankey
import plotly.graph_objects as go
# Market data optional
try:
    import yfinance as yf                     # pip install yfinance
    YF = True
except Exception:
    YF = False

# ── config ─────────────────────────────────────────────────────────────────────
OUTDIR = Path("additional_results")
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG_F = OUTDIR / "additional.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_F, mode="w", encoding="utf-8")],
    format="%(asctime)s %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)
sns.set_style("whitegrid")

# ── helpers ────────────────────────────────────────────────────────────────────
def autodetect_csv() -> Path | None:
    cans = list(Path.cwd().rglob("best_augmented_data.csv"))
    return max(cans, key=lambda p: p.stat().st_mtime) if cans else None

def build_tfidf(texts: List[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95)
    mat   = tfidf.fit_transform(texts)
    return normalize(mat).toarray()

def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ── main pipeline ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="augmented CSV (defaults to best_augmented_data.csv)")
    args = ap.parse_args()

    csv_p = Path(args.input) if args.input else autodetect_csv()
    if not csv_p or not csv_p.exists():
        log.error("Could not locate augmented CSV – pass with --input")
        sys.exit(1)

    log.info(f"Reading {csv_p}")
    df = pd.read_csv(csv_p, low_memory=False)
    if "call_id" not in df.columns:            # fabricate call_id if absent
        df["call_id"] = np.arange(len(df))

    # 1 ─ Low-confidence review list
    if "intent_confidence" in df.columns:
        low = df[df["intent_confidence"] < 0.35]
        if not low.empty:
            out = OUTDIR/"low_confidence_for_review.csv"
            (low.sample(min(500, len(low)), random_state=1)
                [["call_id","activity_sequence","intent_augmented","intent_confidence"]]
                .to_csv(out, index=False))
            log.info(f"Low-confidence review list → {out}")
    else:
        log.warning("intent_confidence missing – skip low-confidence list")

    # 2 ─ Isolation-Forest anomalies on activity_sequence
    if df["activity_sequence"].notna().any():
        emb = build_tfidf(df["activity_sequence"].fillna("").tolist())
        iso = IsolationForest(contamination=0.01, random_state=42)
        df["seq_anomaly"] = (iso.fit_predict(emb) == -1)
        out = OUTDIR/"seq_anomalies.csv"
        df[df["seq_anomaly"]].to_csv(out, index=False)
        log.info(f"Sequence anomalies → {out}")
    else:
        log.warning("No activity_sequence – skip anomalies")

    # 3 ─ Sankey / flow diagram (sample 1k)
    samp = df["activity_sequence"].dropna().sample(min(1000, df["activity_sequence"].notna().sum()))
    links = {}
    for seq in samp:
        acts = seq.split("|")
        for a,b in zip(acts, acts[1:]):
            links[(a,b)] = links.get((a,b),0)+1
    if links:
        s,t,v = zip(*[(k[0],k[1],c) for k,c in links.items()])
        nodes = list(set(s)|set(t)); idx={n:i for i,n in enumerate(nodes)}
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=nodes, pad=15, thickness=20),
            link=dict(source=[idx[x] for x in s],
                      target=[idx[y] for y in t],
                      value=list(v))
        )])
        fn = OUTDIR/"flow_sankey.html"
        fig.write_html(fn); log.info(f"Sankey diagram → {fn}")
    else:
        log.warning("Could not build Sankey (no sequences)")

    # 4 ─ Confidence calibration curve
    if "intent_confidence" in df.columns and "intent_augmented" in df.columns:
        y_true = (df["intent_augmented"] != "Unknown").astype(int)
        prob_true, prob_pred = calibration_curve(y_true, df["intent_confidence"], n_bins=15, strategy="quantile")
        plt.figure(figsize=(4,4))
        plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0,1],[0,1],'--',c='grey')
        plt.xlabel("Predicted confidence"); plt.ylabel("Observed accuracy")
        plt.title("Reliability curve")
        save_fig(OUTDIR/"confidence_calib.png")
        log.info("Saved reliability curve")
    else:
        log.warning("Skipping calibration curve")

    # 5 ─ Daily volume forecast (Prophet)
    if PROPHET and "call_date" in df.columns:
        vol = df.groupby("call_date").size().reset_index(name="y")
        vol["ds"] = pd.to_datetime(vol["call_date"])
        m = Prophet(daily_seasonality=True)
        m.fit(vol[["ds","y"]])
        fut = m.make_future_dataframe(7)
        fc  = m.predict(fut)
        fig = m.plot(fc); save_fig(OUTDIR/"volume_forecast.png")
        log.info("Volume forecast saved")
    else:
        log.warning("Prophet or call_date missing – skip forecast")

    # 6 ─ BERTopic topic surfacing inside Transfer
    if BERTOPIC and not df[df["intent_augmented"]=="Transfer"].empty:
        trans = df[df["intent_augmented"]=="Transfer"]["activity_sequence"].fillna("").tolist()
        topic_model = BERTopic(n_gram_range=(1,2), calculate_probabilities=False,
                               umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                                               metric='cosine', random_state=42))
        topics, _ = topic_model.fit_transform(trans)
        freq = topic_model.get_topic_freq().head(20)
        freq.to_csv(OUTDIR/"topic_summary.csv", index=False)
        log.info("BERTopic summary → topic_summary.csv")
    else:
        log.warning("BERTopic unavailable or no Transfer rows")

    # 7 ─ Synthetic uplift placeholder
    df["mail_flag"] = np.random.randint(0,2,len(df))
    if "intent_augmented" in df.columns:
        before = (df["intent_augmented"]=="Unknown").mean()
        after  = (df[df["mail_flag"]==1]["intent_augmented"]=="Unknown").mean()
        uplift = before - after
        Path(OUTDIR/"uplift_eval.csv").write_text(
            "overall_unknown_rate,mail_flag_unknown_rate,uplift\n"
            f"{before:.4f},{after:.4f},{uplift:.4f}\n")
        log.info("Synthetic uplift placeholder written")

    # 8 ─ Market indicators correlation
    if YF and "call_date" in df.columns:
        start,end = df["call_date"].min(), df["call_date"].max()
        mkt = (yf.download("^GSPC ^VIX", start=start, end=end, progress=False)["Close"]
                 .rename(columns={"^GSPC":"sp500_close","^VIX":"vix"})
                 .reset_index()
                 .assign(call_date=lambda x: x["Date"].dt.date)
                 .drop("Date",axis=1))
        daily = (df.groupby("call_date")
                   .agg(calls=("call_id","size"))
                   .reset_index())
        panel = daily.merge(mkt, on="call_date", how="left").fillna(method="ffill")
        corr = panel[["calls","sp500_close","vix"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Market indicators vs. call volume")
        save_fig(OUTDIR/"market_corr.png")
        corr.to_csv(OUTDIR/"market_corr_matrix.csv")
        log.info("Market correlation heat-map saved")
    else:
        log.warning("No yfinance or call_date – skip market indicators")

    # ── final ────────────────────────────────────────────────────────────────
    meta = {"generated": datetime.now().isoformat(), "source_csv": str(csv_p),
            "modules":{"prophet":PROPHET,"bertopic":BERTOPIC,"yfinance":YF}}
    Path(OUTDIR/"run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("All tasks complete – outputs under %s", OUTDIR.resolve())

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()