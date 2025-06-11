#!/usr/bin/env python
"""
merge_mail_files.py
-------------------
Normalises *all* monthly mail files (RADAR, Product, Meridian) into a single
CSV called `all_mail_data.csv` with columns:

    mail_date   |   mail_volume   |   mail_type   |   source_file

Run:
    python merge_mail_files.py
"""
from __future__ import annotations

import sys, glob, logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd

##############################################################################
# Logging – console + file
##############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("merge_mail_files.log", mode="w", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

##############################################################################
# Configuration – adjust paths / mappings here only
##############################################################################

MAIL_GROUPS: List[Dict] = [
    # ─────────────────── RADAR ───────────────────
    {
        "name":     "RADAR",
        "pattern":  "RADAR/*.xlsx",                     # folder pattern
        # raw column name  ➜   wanted column
        "mappings": {
            "Required Mailing Date": "mail_date",
            "Estimated Mail Volume": "mail_volume",
            "Work Order Type":       "mail_type",
        },
    },
    # ─────────────────── PRODUCT ───────────────────
    {
        "name":     "Product",
        "pattern":  "Product/*.xlsx",
        "mappings": {
            "Production Complete Date": "mail_date",
            "Product Count":            "mail_volume",
            "Product Type":             "mail_type",
        },
    },
    # ─────────────────── MERIDIAN ──────────────────
    {
        "name":     "Meridian",
        "pattern":  "Meridian/*.xlsx",
        "mappings": {
            "BillingDate": "mail_date",
            "Units":       "mail_volume",
            "Description": "mail_type",
        },
    },
]

# Output location
OUT_DIR   = Path("data")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH  = OUT_DIR / "all_mail_data.csv"

##############################################################################
# Helpers
##############################################################################
def _parse_any_date(x) -> pd.Timestamp | pd.NaT:
    """
    Convert many messy date inputs to pandas Timestamp (or NaT):
        • '01/04/2025'
        • '1/3/2025 10:00:51 PM'
        • 20250430
        • already-datetime objects
    """
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x, errors="coerce")

    # integer like 20250430
    if isinstance(x, (int, float)) and x > 10_000_000:
        return pd.to_datetime(str(int(x)), format="%Y%m%d", errors="coerce")

    # let pandas infer (US-first) then fallback to day-first
    dt = pd.to_datetime(str(x), errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(str(x), errors="coerce", dayfirst=True)
    return dt


def _read_file(fp: Path, mapping: Dict[str, str]) -> pd.DataFrame:
    """Read a single Excel/CSV file and return a normalised DataFrame."""
    log.info(f"  → {fp.name}")
    if fp.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(fp, engine="openpyxl")
    else:
        df = pd.read_csv(fp)

    # pick & rename columns
    keep_cols = {raw: tgt for raw, tgt in mapping.items() if raw in df.columns}
    if len(keep_cols) < 2:          # need at least date + volume
        log.warning(f"    ⚠️  Skipped – missing required columns")
        return pd.DataFrame()

    mail_df = df[list(keep_cols)].rename(columns=keep_cols)

    # normalize date & volume
    mail_df["mail_date"]   = mail_df["mail_date"].apply(_parse_any_date)
    mail_df["mail_volume"] = pd.to_numeric(mail_df["mail_volume"], errors="coerce").fillna(0)

    # some files have NaN mail_type – ensure column exists
    if "mail_type" not in mail_df.columns:
        mail_df["mail_type"] = "unknown"

    mail_df["source_file"] = fp.name
    mail_df = mail_df.dropna(subset=["mail_date"])
    return mail_df[["mail_date", "mail_volume", "mail_type", "source_file"]]


##############################################################################
# Main
##############################################################################
def main() -> None:
    log.info("\n— 1. Normalising mail families —")
    all_frames: List[pd.DataFrame] = []

    for grp in MAIL_GROUPS:
        log.info(f"\n▶ {grp['name']} files")
        files = [Path(p) for p in glob.glob(grp["pattern"])]
        if not files:
            log.warning(f"  ⚠️  None found for pattern {grp['pattern']}")
            continue

        total_rows = 0
        for fp in files:
            df_norm = _read_file(fp, grp["mappings"])
            if not df_norm.empty:
                total_rows += len(df_norm)
                all_frames.append(df_norm)

        log.info(f"  ✓ Total rows loaded for {grp['name']}: {total_rows:,}")

    if not all_frames:
        log.error("No valid rows found across all families – nothing written")
        sys.exit(1)

    all_mail = pd.concat(all_frames, ignore_index=True)
    all_mail.sort_values("mail_date", inplace=True)
    all_mail.to_csv(OUT_PATH, index=False)

    log.info(f"\n🎉 Written {len(all_mail):,} rows  →  {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()