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


# -------------------- helper  --------------------------------------------
def load_and_standardise(path: Path,
                         cmap: dict[str, str],
                         date_col_std: str = "mail_date",
                         count_col_std: str = "mail_count") -> pd.DataFrame:
    """Read one mailing CSV and force standard column names."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=cmap)

    # 🔽 NEW: sanity-check after renaming
    missing = [c for c in (date_col_std, count_col_std) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name}: after rename the dataframe is still missing "
            f"{missing}.  Available columns → {list(df.columns)}\n"
            f"Hint: update MAIL_COLUMN_MAPS so the raw names exactly match "
            f"the CSV headers (including spaces & case)."
        )

    # keep only what we need
    keep = [date_col_std, count_col_std] + [c for c in ("mail_type",) if c in df]
    return df[keep]
# -------------------------------------------------------------------------

def main() -> None:
    # ---------------- 1 & 2: LOAD / AGGREGATE  (unchanged) ---------------- #
    mailing_frames = []
    for p, cmap in zip(MAIL_FILES, MAIL_COLUMN_MAPS):
        path = Path(p);  mailing_frames.append(load_and_standardise(path, cmap, "mail_date", "mail_count"))
    mail_df = pd.concat(mailing_frames, ignore_index=True).dropna(subset=["mail_date"])
    mail_df["mail_date"] = mail_df["mail_date"].dt.normalize()
    daily_mail = mail_df.groupby("mail_date", as_index=False)["mail_count"].sum()

    # --------------- LOAD & AGGREGATE  CALLS ------------------------------- #
    call_frames = []
    for path_str, cmap in zip(CALL_FILES, CALL_COLUMN_MAPS):
        path = Path(path_str)
        df_tmp = pd.read_csv(path, low_memory=False).rename(columns=cmap)

        if "call_date" not in df_tmp:
            raise KeyError(f"{path.name}: mapping must provide date column → 'call_date'")

        # keep ONLY the date; 1 row == 1 call  ➜   let groupby .size() count them
        call_frames.append(df_tmp[["call_date"]])

    call_df = (pd.concat(call_frames, ignore_index=True)
                 .assign(call_date=lambda d: pd.to_datetime(d["call_date"],
                                                            errors="coerce").dt.normalize())
                 .dropna(subset=["call_date"]))

    # ← daily counts are now simple row counts
    daily_call = (call_df.groupby("call_date", as_index=False)
                          .size()
                          .rename(columns={"size": "call_count"}))
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