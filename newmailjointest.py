#!/usr/bin/env python
"""
merge_mail_files.py
────────────────────
Normalises *all* monthly mail files (RADAR, Product, Meridian) into one CSV:

    mail_date | mail_volume | mail_type | source_file

Run:
    python merge_mail_files.py
"""
from __future__ import annotations

import sys, glob, logging, re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd

# ───────────────────────────── Logging ────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("merge_mail_files.log", "w", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────── Configuration ────────────────────────── #

MAIL_GROUPS: List[Dict] = [

    # ───── RADAR ────────────────────────────────────────────────────
    {
        "name": "RADAR",
        "pattern": "RADAR/*.xlsx",             # adjust if csv
        "mappings": {
            "Required Mailing Date": "mail_date",
            "Estimated Mail Volume": "mail_volume",
            "Work Order Type":       "mail_type",
        },
        "date_format": "%d/%m/%Y",             # _try_ this first, else fallback to auto
        "volume_multiplier": 1,                # leave 1 unless units need scaling
    },

    # ───── Product ─────────────────────────────────────────────────
    {
        "name": "Product",
        "pattern": "Product/*.xlsx",
        "mappings": {
            "Production Complete Date": "mail_date",
            "Product Count":            "mail_volume",
            "Product Type":             "mail_type",
        },
        "date_format": "%m/%d/%Y",
        "volume_multiplier": 1,
    },

    # ───── Meridian ────────────────────────────────────────────────
    {
        "name": "Meridian",
        "pattern": "Meridian/*.xlsx",
        "mappings": {
            "BillingDate": "mail_date",        # 20250430 / 20250430 10:00
            "Units":       "mail_volume",
            "Description": "mail_type",
        },
        # Meridian dates are numeric → handled by _parse_any_date()
        "date_format": None,
        "volume_multiplier": 1,
    },
]

OUT_DIR  = Path("data")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "all_mail_data.csv"

# ──────────────────────────── Helpers ─────────────────────────────── #

def _parse_any_date(x, pref_fmt: str | None = None) -> pd.Timestamp | pd.NaT:
    """
    Flexible date-parser used for all groups:

    • accepts strings, ints, floats, pandas Timestamps
    • accepts '20250430', '20250430 230000', '1/3/2025', '01/03/2025 10:00 PM'
    • tries an optional preferred format first (if supplied) for speed
    """
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x, errors="coerce")

    s = str(x).strip()

    # Quick pass for yyyymmdd, yyyymmddhhmmss etc.
    if re.fullmatch(r"\d{8}(\d{6})?", s):
        return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")

    # Preferred explicit format first
    if pref_fmt:
        dt = pd.to_datetime(s, format=pref_fmt, errors="coerce")
        if not pd.isna(dt):
            return dt

    # Fallback: let pandas try MM/DD/YYYY then DD/MM/YYYY
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt


def _read_file(fp: Path, group: Dict) -> pd.DataFrame:
    """Load one Excel/CSV file and convert to unified schema."""
    log.info(f"  • {fp.name}")
    try:
        if fp.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(fp, engine="openpyxl")
        else:
            df = pd.read_csv(fp)
    except Exception as exc:
        log.error(f"    ✗ Failed to read ({exc})")
        return pd.DataFrame()

    mapping = group["mappings"]
    missing = [src for src in mapping if src not in df.columns]
    if missing:
        log.warning(f"    ⚠ Missing cols {missing} – skipped")
        return pd.DataFrame()

    df = df[list(mapping)].rename(columns=mapping)

    # Date + volume parsing
    df["mail_date"] = df["mail_date"].apply(lambda x: _parse_any_date(x, group.get("date_format")))
    df["mail_volume"] = (
        pd.to_numeric(df["mail_volume"], errors="coerce").fillna(0) * group.get("volume_multiplier", 1)
    )

    # Ensure mail_type exists
    if "mail_type" not in df.columns:
        df["mail_type"] = "unknown"

    # Drop invalid
    df = df[(df["mail_date"].notna()) & (df["mail_volume"] > 0)]
    df["source_file"] = fp.name
    if df.empty:
        log.warning("    ⚠ 0 valid rows")
    else:
        log.info(f"    ✓ {len(df):,} rows kept")
    return df[["mail_date", "mail_volume", "mail_type", "source_file"]]


# ──────────────────────────────── Main ──────────────────────────────── #

def main() -> None:
    log.info("\n──────────── MERGING MAIL FILES ────────────")

    frames: List[pd.DataFrame] = []

    for grp in MAIL_GROUPS:
        log.info(f"\n▶ {grp['name']} (pattern: {grp['pattern']})")
        files = sorted(Path().glob(grp["pattern"]))
        if not files:
            log.warning("  ⚠ No files found")
            continue

        for fp in files:
            df_norm = _read_file(fp, grp)
            if not df_norm.empty:
                frames.append(df_norm)

    if not frames:
        log.error("No valid rows across all families – nothing written")
        sys.exit(1)

    all_mail = pd.concat(frames, ignore_index=True).sort_values("mail_date")
    all_mail.to_csv(OUT_PATH, index=False)

    log.info(f"\n✅ Written {len(all_mail):,} rows → {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()