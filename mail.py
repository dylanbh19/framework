#!/usr/bin/env python
"""
mail_call_forecast_scaffold.py
==============================

*unchanged header …*
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ▼ NEW: Plotly for the interactive dashboard
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

###############################################################################
# -------------------------- USER CONFIG  (unchanged) ----------------------- #
###############################################################################
MAIL_FILES = ["MAIL_FILE_1.csv", "MAIL_FILE_2.csv", "MAIL_FILE_3.csv"]  # TODO
CALL_FILE  = "CALL_FILE.csv"                                             # TODO
MAIL_COLUMN_MAPS = [                 # TODO – edit as before
    {"DATE_SENT": "mail_date", "PIECES": "mail_count", "CAMPAIGN": "mail_type"},
    {"MailDate": "mail_date", "Qty": "mail_count", "Type": "mail_type"},
    {"sent_on": "mail_date", "count": "mail_count", "letter": "mail_type"},
]
CALL_COL_MAP = {"CALL_DATE": "call_date", "CALLS": "call_count"}         # TODO
OUTDIR = Path("output"); OUTDIR.mkdir(exist_ok=True)
###############################################################################


def load_and_standardise(csv_path: Path, col_map: dict[str, str],
                         date_col_std: str, count_col_std: str) -> pd.DataFrame:
    # … unchanged helper …
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        print(f"⚠️  {csv_path.name}: missing {missing}")
    df = df.rename(columns=col_map)
    if date_col_std in df: df[date_col_std] = pd.to_datetime(df[date_col_std], errors="coerce")
    if count_col_std in df: df[count_col_std] = pd.to_numeric(df[count_col_std], errors="coerce").fillna(0).astype(int)
    return df[[date_col_std, count_col_std] + [c for c in ["mail_type"] if c in df]]


def main() -> None:
    # ---------------- 1 & 2: LOAD / AGGREGATE  (unchanged) ---------------- #
    mailing_frames = []
    for p, cmap in zip(MAIL_FILES, MAIL_COLUMN_MAPS):
        path = Path(p);  mailing_frames.append(load_and_standardise(path, cmap, "mail_date", "mail_count"))
    mail_df = pd.concat(mailing_frames, ignore_index=True).dropna(subset=["mail_date"])
    mail_df["mail_date"] = mail_df["mail_date"].dt.normalize()
    daily_mail = mail_df.groupby("mail_date", as_index=False)["mail_count"].sum()

    call_df = (pd.read_csv(CALL_FILE, low_memory=False)
                 .rename(columns=CALL_COL_MAP))
    call_df["call_date"] = pd.to_datetime(call_df["call_date"], errors="coerce").dt.normalize()
    call_df["call_count"] = pd.to_numeric(call_df["call_count"], errors="coerce").fillna(0).astype(int)
    daily_call = call_df.groupby("call_date", as_index=False)["call_count"].sum()

    # ---------------- 3: MERGE (unchanged) -------------------------------- #
    timeline = pd.DataFrame({"date": pd.date_range(daily_mail["mail_date"].min(),
                                                   daily_call["call_date"].max(), freq="D")})
    merged = (timeline
              .merge(daily_mail.rename(columns={"mail_date":"date"}), on="date", how="left")
              .merge(daily_call.rename(columns={"call_date":"date"}), on="date", how="left")
              .fillna(0).astype({"mail_count":int,"call_count":int}))
    merged.to_csv(OUTDIR/"daily_mail_call.csv", index=False)

    # ---------------- 4: STATIC PNGs (unchanged) --------------------------- #
    sns.set(style="whitegrid")
    plt.figure(figsize=(12,5))
    plt.plot(merged["date"],merged["mail_count"],label="Mail")
    plt.plot(merged["date"],merged["call_count"],label="Calls")
    plt.legend(); plt.title("Mail vs Calls (daily)"); plt.tight_layout()
    plt.savefig(OUTDIR/"mail_vs_calls.png",dpi=300); plt.close()

    merged["mail_lag1"] = merged["mail_count"].shift(1)
    plt.figure(figsize=(6,5))
    sns.regplot(x="mail_lag1",y="call_count",data=merged,scatter_kws=dict(s=30,alpha=.6))
    plt.title("Next-day Calls vs Previous-day Mail"); plt.tight_layout()
    plt.savefig(OUTDIR/"lag_scatter.png",dpi=300); plt.close()

    # stacked bar only if mail_type is present
    if "mail_type" in mail_df:
        t_daily = (mail_df.groupby(["mail_date","mail_type"])["mail_count"].sum().unstack(fill_value=0))
        t_daily.plot(kind="bar",stacked=True,figsize=(12,6),width=1.0)
        plt.title("Mail mix by type – daily"); plt.tight_layout()
        plt.savefig(OUTDIR/"mail_stack_by_type.png",dpi=300); plt.close()

    # ====================================================================== #
    # === NEW 5: INTERACTIVE DASHBOARD  ==================================== #
    # ====================================================================== #
    print("🛠  Building interactive dashboard …")
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"type":"xy","colspan":2}, None],
                               [{"type":"xy"},{"type":"xy"}]],
                        subplot_titles=("Mail vs Calls (daily)",
                                        "Lag-1 Scatter (calls vs mail-1)",
                                        "7-day Rolling Means"))
    # a) line chart
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["mail_count"],
                             mode="lines", name="Mail pieces"), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["call_count"],
                             mode="lines", name="Calls"), row=1, col=1)

    # b) lag scatter
    fig.add_trace(go.Scatter(x=merged["mail_lag1"], y=merged["call_count"],
                             mode="markers", marker=dict(size=4, opacity=.6),
                             name="lag-scatter"),
                  row=2,col=1)

    # c) rolling means
    merged["mail_roll7"]  = merged["mail_count"].rolling(7).mean()
    merged["call_roll7"]  = merged["call_count"].rolling(7).mean()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["mail_roll7"],
                             mode="lines", name="Mail 7-day MA"), row=2, col=2)
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["call_roll7"],
                             mode="lines", name="Call 7-day MA"), row=2, col=2)

    fig.update_layout(height=800, width=1150,
                      title_text="Mail & Call Volume – Interactive Dashboard",
                      hovermode="x unified")
    dashboard_path = OUTDIR / "interactive_dashboard.html"
    fig.write_html(dashboard_path, include_plotlyjs="cdn")
    print(f"✅ Interactive dashboard saved → {dashboard_path}")

    # ---------------------------------------------------------------------- #
    print("🎉  All outputs in:", OUTDIR.absolute())


if __name__ == "__main__":
    main()