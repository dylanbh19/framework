#!/usr/bin/env python
"""
merge_mail_files.py
────────────────────
Normalises every monthly mail file (RADAR / Product / Meridian)
into a single CSV →  data/all_mail_data.csv  with columns

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
        logging.FileHandler("merge_mail_files.log", "w", encoding="utf-8"),
    ],
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
            "Work Order Type": "mail_type",
        },
        "pref_date_fmt": "%d/%m/%Y",
        "volume_multiplier": 1,
    },
    # ───── Product ──────────────────────────────────────────────────
    {
        "name": "Product",
        "pattern": "Product/*.xlsx",
        "mappings": {
            "Production Complete Date": "mail_date",
            "Product Count": "mail_volume",
            "Product Type": "mail_type",
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
            "Units": "mail_volume",
            "Description": "mail_type",
        },
        # Meridian dates look like 20250430 (YYYYMMDD) or Excel serials
        "pref_date_fmt": "%Y%m%d",
        "volume_multiplier": 1,
    },
]

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "all_mail_data.csv"

# ──────────────────────────── Helpers ─────────────────────────────── #
_EXCEL_START = pd.Timestamp("1899-12-30")  # Excel’s day-0 (Windows version)


def _excel_serial_to_ts(n: int | float) -> pd.Timestamp | pd.NaT:
    """Convert Excel serial numbers (roughly 25 000-60 000) to Timestamp."""
    if 25_000 <= n <= 60_000:
        try:
            return _EXCEL_START + timedelta(days=float(n))
        except Exception:
            return pd.NaT
    return pd.NaT


def _parse_any_date(
    val,
    pref_fmt: str | None = None,
    debug_name: str = "",
) -> pd.Timestamp | pd.NaT:
    """
    Very tolerant date parser.

    • Tries `pref_fmt` first (important for Meridian YYYYMMDD)
    • Then Excel serials 25000-60000
    • Then raw YYYYMMDD strings / ints
    • Then pandas with US-first, then EU-first
    """
    if pd.isna(val):
        return pd.NaT

    original_val = val

    # ── numeric input ──────────────────────────────────────────────
    if isinstance(val, (int, float)):
        # 8-digit numeric that might be YYYYMMDD
        if 20_000_000 <= val <= 20_401_231:
            ts = pd.to_datetime(str(int(val)), format="%Y%m%d", errors="coerce")
            if not pd.isna(ts):
                return ts

        # Excel serial?
        serial_ts = _excel_serial_to_ts(val)
        if not pd.isna(serial_ts):
            return serial_ts

    # make it string for the rest
    s = str(val).strip()

    # ── preferred format (if supplied) ─────────────────────────────
    if pref_fmt:
        try:
            if pref_fmt == "%Y%m%d":
                digits = re.sub(r"\D", "", s)
                if len(digits) == 8:
                    ts = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
                    if not pd.isna(ts):
                        return ts
            else:
                ts = pd.to_datetime(s, format=pref_fmt, errors="coerce")
                if not pd.isna(ts):
                    return ts
        except Exception:
            pass  # fall through

    # ── raw YYYYMMDD in string ─────────────────────────────────────
    digits_only = re.sub(r"\D", "", s)
    if len(digits_only) == 8 and digits_only.isdigit():
        ts = pd.to_datetime(digits_only, format="%Y%m%d", errors="coerce")
        if not pd.isna(ts):
            return ts

    # ── pandas generic parsing ─────────────────────────────────────
    ts = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if not pd.isna(ts):
        return ts
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if not pd.isna(ts):
        return ts

    # last resort: log a warning once per group
    if debug_name:
        log.debug(f"    {debug_name}: failed to parse '{original_val}'")
    return pd.NaT


def _read_file(fp: Path, group: Dict) -> pd.DataFrame:
    """Load one file and normalise its columns."""
    log.info(f"  • {fp.name}")
    try:
        if fp.suffix.lower() in {".xlsx", ".xls"}:
            if group["name"] == "Meridian":
                df = pd.read_excel(fp, engine="openpyxl", dtype={"BillingDate": str})
            else:
                df = pd.read_excel(fp, engine="openpyxl")
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
    vol_mult = group.get("volume_multiplier", 1)
    dbg_tag  = f"{group['name']}-{fp.name}"

    df["mail_date"] = df["mail_date"].apply(lambda x: _parse_any_date(x, pref_fmt, dbg_tag))
    df["mail_volume"] = (
        pd.to_numeric(df["mail_volume"], errors="coerce").fillna(0) * vol_mult
    )

    if "mail_type" not in df.columns:
        df["mail_type"] = "unknown"

    # drop invalid rows
    before = len(df)
    df = df[(df["mail_date"].notna()) & (df["mail_volume"] > 0)]
    dropped = before - len(df)
    if dropped:
        log.warning(f"    ⚠ dropped {dropped} rows (invalid date / volume)")

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

        total = 0
        for fp in files:
            df_norm = _read_file(fp, grp)
            if not df_norm.empty:
                total += len(df_norm)
                frames.append(df_norm)
        log.info(f"  → total valid rows: {total:,}")

    if not frames:
        log.error("No valid mail rows – nothing written")
        sys.exit(1)

    all_mail = (
        pd.concat(frames, ignore_index=True)
        .sort_values("mail_date")
        .reset_index(drop=True)
    )

    # Summary
    log.info(
        f"\n📅 Date range: {all_mail['mail_date'].min().date()} "
        f"→ {all_mail['mail_date'].max().date()}"
    )
    for src, cnt in all_mail["source_file"].value_counts().items():
        log.info(f"  {src:<30} {cnt:,} rows")

    all_mail.to_csv(OUT_PATH, index=False)
    log.info(f"\n✅ Written {len(all_mail):,} rows → {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()