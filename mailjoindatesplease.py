#!/usr/bin/env python
"""
merge_mail_files.py
────────────────────
Normalises every monthly mail file (RADAR, Product, Meridian) into one CSV:

    mail_date | mail_volume | mail_type | source_file

Run:
    python merge_mail_files.py
"""
from __future__ import annotations

import sys, glob, logging, re
from pathlib import Path
from datetime import datetime, timedelta
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
        "pattern": "RADAR/*.xlsx",
        "mappings": {
            "Required Mailing Date": "mail_date",
            "Estimated Mail Volume": "mail_volume",
            "Work Order Type":       "mail_type",
        },
        "pref_date_fmt": "%d/%m/%Y",
        "volume_multiplier": 1,
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
        "pref_date_fmt": "%m/%d/%Y",
        "volume_multiplier": 1,
    },

    # ───── Meridian ────────────────────────────────────────────────
    {
        "name": "Meridian",
        "pattern": "Meridian/*.xlsx",
        "mappings": {
            "BillingDate": "mail_date",
            "Units":       "mail_volume",
            "Description": "mail_type",
        },
        # Meridian dates like 20250430 or Excel-serial 45576
        "pref_date_fmt": None,
        "volume_multiplier": 1,
    },
]

OUT_DIR  = Path("data")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "all_mail_data.csv"

# ──────────────────────────── Helpers ─────────────────────────────── #

_EXCEL_START = pd.Timestamp("1899-12-30")  # Excel’s day-0 (windows)


def _excel_serial_to_ts(n: int | float) -> pd.Timestamp | pd.NaT:
    """Convert 20 000–60 000 style serials to Timestamp."""
    if 20000 <= n <= 60000:
        try:
            return _EXCEL_START + timedelta(days=float(n))
        except Exception:
            return pd.NaT
    return pd.NaT


def _parse_any_date(val, pref_fmt: str | None = None) -> pd.Timestamp | pd.NaT:
    """
    Very tolerant date parser:
    • Excel serials 20000-60000
    • yyyymmdd strings / ints
    • tries a preferred strptime pattern first (if provided)
    • then pandas with US then EU day-first
    """
    if pd.isna(val):
        return pd.NaT

    # Excel serial shortcut
    if isinstance(val, (int, float)):
        serial_try = _excel_serial_to_ts(val)
        if not pd.isna(serial_try):
            return serial_try

    s = str(val).strip()

    # numeric yyyymmdd
    if re.fullmatch(r"\d{8}$", s):
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    # pref-format first
    if pref_fmt:
        ts = pd.to_datetime(s, format=pref_fmt, errors="coerce")
        if not pd.isna(ts):
            return ts

    # generic tries
    ts = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(ts):
        ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return ts


def _read_file(fp: Path, group: Dict) -> pd.DataFrame:
    """Load a file and normalise its columns."""
    log.info(f"  • {fp.name}")
    try:
        if fp.suffix.lower() in {".xlsx", ".xls"}:
            # Meridian – force BillingDate as str so 20250430 isn’t cast to float
            dtype = {"BillingDate": str} if group["name"] == "Meridian" else None
            df = pd.read_excel(fp, engine="openpyxl", dtype=dtype)
        else:
            df = pd.read_csv(fp)
    except Exception as exc:
        log.error(f"    ✗ read error: {exc}")
        return pd.DataFrame()

    mapping = group["mappings"]
    missing = [raw for raw in mapping if raw not in df.columns]
    if missing:
        log.warning(f"    ⚠ missing cols {missing} – skipped")
        return pd.DataFrame()

    df = df[list(mapping)].rename(columns=mapping)

    pref_fmt = group.get("pref_date_fmt")
    mult     = group.get("volume_multiplier", 1)

    df["mail_date"]   = df["mail_date"].apply(lambda x: _parse_any_date(x, pref_fmt))
    df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce").fillna(0) * mult

    if "mail_type" not in df.columns:
        df["mail_type"] = "unknown"

    before = len(df)
    df = df[(df["mail_date"].notna()) & (df["mail_volume"] > 0)]
    bad = before - len(df)
    if bad:
        bad_examples = df[df["mail_date"].isna() | (df["mail_volume"] <= 0)].head(3)
        log.warning(f"    ⚠ dropped {bad} rows – first bad examples:\n{bad_examples}")
    else:
        log.info(f"    ✓ {len(df):,} rows kept")

    df["source_file"] = fp.name
    return df[["mail_date", "mail_volume", "mail_type", "source_file"]]


# ─────────────────────────────── Main ─────────────────────────────── #

def main() -> None:
    log.info("\n──────────── MERGING MAIL FILES ────────────")

    frames: List[pd.DataFrame] = []

    for grp in MAIL_GROUPS:
        log.info(f"\n▶ {grp['name']} (pattern: {grp['pattern']})")
        files = sorted(Path().glob(grp["pattern"]))
        if not files:
            log.warning("  ⚠ no files found")
            continue

        rows = 0
        for fp in files:
            df_norm = _read_file(fp, grp)
            if not df_norm.empty:
                rows += len(df_norm)
                frames.append(df_norm)
        log.info(f"  → total valid rows: {rows:,}")

    if not frames:
        log.error("No valid mail rows – nothing written")
        sys.exit(1)

    all_mail = (pd.concat(frames, ignore_index=True)
                  .sort_values("mail_date")
                  .reset_index(drop=True))
    all_mail.to_csv(OUT_PATH, index=False)

    log.info(f"\n✅ Written {len(all_mail):,} rows → {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()