#!/usr/bin/env python
"""
run_mail_call_analysis.py
=========================
Pre-process Radar / Product / Meridian mail files + call data,
then launch comprehensive_mail_call_analysis.py.

▶  Place this script *next to* comprehensive_mail_call_analysis.py
▶  Adjust the MAIL_GROUPS mapping dictionaries below
▶  Run:  python run_mail_call_analysis.py
"""

from __future__ import annotations
import importlib.util
import sys, glob, json
from pathlib import Path
from datetime import datetime

import pandas as pd

# ------------------------------------------------------------------------------
# 1. USER CONFIG ––– Update only the mapping dictionaries
# ------------------------------------------------------------------------------

MAIL_GROUPS = [
    # ───────────── Radar ─────────────
    {
        "name":      "radar",
        "pattern":   "radar/mail_*.csv",        # Where the monthly files live
        "mappings": {                          # raw column  ➜  target column
            "BillingDate":        "mail_date",
            "Units":              "mail_volume",
            "Description":        "mail_type",
        },
        "date_format": "%Y-%m-%d",
    },

    # ───────────── Product ───────────
    {
        "name":      "product",
        "pattern":   "product/mail_*.csv",
        "mappings": {
            "Production Complete Date": "mail_date",
            "Product Count":            "mail_volume",
            "Product Type":             "mail_type",
        },
        "date_format": "%m/%d/%Y",      # example American format
    },

    # ───────────── Meridian ──────────
    {
        "name":      "meridian",
        "pattern":   "meridian/mail_*.csv",
        "mappings": {
            "Required Mailing Date":    "mail_date",
            "Estimated Mail Volume":    "mail_volume",
            "Work Order Type":          "mail_type",
        },
        "date_format": "%d/%m/%Y",      # example European format
    },
]

# Call file (raw rows)
RAW_CALL_FILE  = Path("call_center_raw.csv")   # update path
CALL_DATE_COL  = "call_date"                   # column holding a date (or datetime)
CALL_DT_FORMAT = "%Y-%m-%d"

# ------------------------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------------------------

def normalise_mail_group(cfg: dict) -> pd.DataFrame:
    """Load every monthly CSV in one group → unified dataframe."""
    files = sorted(glob.glob(cfg["pattern"]))
    if not files:
        print(f"⚠️  No files found for {cfg['name']} pattern {cfg['pattern']}")
        return pd.DataFrame()

    frames = []
    for fp in files:
        df_raw = pd.read_csv(fp, low_memory=False)
        df = pd.DataFrame()

        for raw_col, target_col in cfg["mappings"].items():
            if raw_col not in df_raw.columns:
                print(f"   · {fp}: column '{raw_col}' missing – row ignored")
                continue
            if target_col == "mail_date":
                df[target_col] = pd.to_datetime(df_raw[raw_col],
                                                format=cfg["date_format"],
                                                errors="coerce")
            elif target_col == "mail_volume":
                df[target_col] = pd.to_numeric(df_raw[raw_col], errors="coerce")
            else:  # mail_type or any other passthrough
                df[target_col] = df_raw[raw_col]

        df["mail_family"] = cfg["name"]
        df["source_file"] = Path(fp).name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["mail_date"])              # keep valid dates only
    out["mail_volume"] = out["mail_volume"].fillna(1)   # default volume = 1
    return out


def aggregate_calls(raw_file: Path, date_col: str, dt_fmt: str) -> pd.DataFrame:
    """Convert raw call rows to daily totals."""
    df = pd.read_csv(raw_file, low_memory=False)
    if date_col not in df.columns:
        raise ValueError(f"Call file missing column '{date_col}'")

    dates = pd.to_datetime(df[date_col], format=dt_fmt, errors="coerce")
    daily = (
        pd.DataFrame({"call_date": dates})
        .dropna()
        .groupby("call_date")
        .size()
        .rename("call_count")
        .reset_index()
    )
    return daily


# ------------------------------------------------------------------------------
# 3. Build unified mail & call datasets
# ------------------------------------------------------------------------------

print("\n━━━ 1. Normalising mail families ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
mail_frames = [normalise_mail_group(cfg) for cfg in MAIL_GROUPS]
mail_df = pd.concat([f for f in mail_frames if not f.empty], ignore_index=True)
if mail_df.empty:
    print("❌ No mail data loaded – aborting")
    sys.exit(1)

# save a copy
out_dir = Path("mail_call_analysis_results") / "data"
out_dir.mkdir(parents=True, exist_ok=True)
mail_df.to_csv(out_dir / "combined_mail_data.csv", index=False)
print(f"✓ Combined mail dataset saved: {len(mail_df):,} rows")

print("\n━━━ 2. Aggregating raw call data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
if not RAW_CALL_FILE.exists():
    print(f"❌ Call file {RAW_CALL_FILE} not found – aborting")
    sys.exit(1)

call_df = aggregate_calls(RAW_CALL_FILE, CALL_DATE_COL, CALL_DT_FORMAT)
call_df.to_csv(out_dir / "call_center_daily.csv", index=False)
print(f"✓ Daily call counts saved: {len(call_df):,} days")

# ------------------------------------------------------------------------------
# 4. Patch the big analyzer’s configuration on-the-fly
# ------------------------------------------------------------------------------

# We import the long file as a module, then overwrite MAIL_FILES/CALL_FILE.
spec = importlib.util.spec_from_file_location(
    "analysis_mod", Path("comprehensive_mail_call_analysis.py"))
analysis_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis_mod)

# Update its global config objects
analysis_mod.MAIL_FILES.update({
    "pattern": str(out_dir / "combined_mail_data.csv"),
    "date_column": "mail_date",
    "volume_column": "mail_volume",
    "date_format": "%Y-%m-%d",       # already converted
    "customer_id_column": None,
    "campaign_column": "mail_type",  # treat type as campaign if desired
    "region_column": None,
    "segment_column": None,
})
analysis_mod.CALL_FILE.update({
    "path": str(out_dir / "call_center_daily.csv"),
    "date_column": "call_date",
    "is_aggregated": True,
    "count_column": "call_count",
    "date_format": "%Y-%m-%d",
})

# ------------------------------------------------------------------------------
# 5. Run the analysis
# ------------------------------------------------------------------------------

print("\n━━━ 3. Running comprehensive analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
analysis_mod.main()

print("\n🎉  Finished.  See mail_call_analysis_results/ for artefacts.")
