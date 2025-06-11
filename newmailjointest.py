#!/usr/bin/env python3
"""
newmailcallrunner.py
────────────────────
1. Scans the three “mail families” (RADAR / Product / Meridian).
2. Loads every monthly xlsx / csv found.
3. Normalises to a common schema  ➜  mail_date | mail_volume | mail_type | source_file
4. Concatenates to  all_mail_data.csv

Run:
    python newmailcallrunner.py
"""

from __future__ import annotations
import sys, glob, json, logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─────────────────────────────── Config ────────────────────────────────

MAIL_GROUPS = [
    # ── RADAR ───────────────────────────────────────────────────────────
    {
        "name": "RADAR",
        "pattern": "RADAR/RADAR_*.csv",        # or *.xlsx  – adjust if needed
        "mappings": {                          # raw column ↦ target column
            "Required Mailing Date": "mail_date",
            "Estimated Mail Volume": "mail_volume",
            "Work Order Type": "mail_type",
        },
        "date_format": "%d/%m/%Y",             # example EU format
    },

    # ── Product ────────────────────────────────────────────────────────
    {
        "name": "Product",
        "pattern": "Product/Product Type Report *.csv",
        "mappings": {
            "Production Complete Date": "mail_date",
            "Product Count": "mail_volume",
            "Product Type": "mail_type",
        },
        "date_format": "%m/%d/%Y",             # example US format
    },

    # ── Meridian ────────────────────────────────────────────────────────
    {
        "name": "Meridian",
        "pattern": "Meridian/*.csv",           # adjust if xlsx
        "mappings": {
            "BillingDate": "mail_date",
            "Units": "mail_volume",
            "Description": "mail_type",
        },
        # Meridian BillingDate comes as 20250430 → handled specially below
        "date_format": None,
    },
]

OUT_PATH = Path("all_mail_data.csv")

# ─────────────────────────────── Logging ───────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ────────────────────────── Helper functions ───────────────────────────


def load_and_standardise(path: Path, group: dict) -> pd.DataFrame:
    """Load one file and return a DF with unified columns."""
    # Choose engine
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    mappings = group["mappings"]
    missing = [col for col in mappings if col not in df.columns]
    if missing:
        log.warning(f"    ↳ Skipped – missing columns {missing}")
        return pd.DataFrame()        # empty -> will be ignored

    # Rename + subset
    df = df[list(mappings)].rename(columns=mappings)

    # ── Date handling ────────────────────────────────────────────────
    if group["name"] == "Meridian":
        # Meridian dates are 8-digit yyyymmdd (sometimes int). Convert first
        df["mail_date"] = (
            df["mail_date"]
            .astype(str)
            .str.zfill(8)            # ensure 8 characters
            .replace("00000000", pd.NA)
        )
        df["mail_date"] = pd.to_datetime(df["mail_date"], format="%Y%m%d", errors="coerce")
    else:
        df["mail_date"] = pd.to_datetime(
            df["mail_date"],
            format=group["date_format"],
            errors="coerce"
        )

    # Drop rows with no valid date or volume
    df = df.dropna(subset=["mail_date"])
    df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce").fillna(0)

    # Keep only positive volume rows
    df = df[df["mail_volume"] > 0]

    # Add back-reference to file
    df["source_file"] = path.name
    return df


# ───────────────────────────────  Main  ────────────────────────────────

def main() -> None:
    log.info("\n────────── READING MAIL FILES ──────────")

    all_frames: list[pd.DataFrame] = []

    for group in MAIL_GROUPS:
        log.info(f"\n⮕ Group: {group['name']}")
        files = sorted(Path().glob(group["pattern"]))

        if not files:
            log.warning(f"  ⚠  No files matched pattern {group['pattern']}")
            continue

        for fp in files:
            log.info(f"  • {fp.name}")
            try:
                frame = load_and_standardise(fp, group)
                if len(frame):
                    log.info(f"    ✓ {len(frame):,} rows kept")
                    all_frames.append(frame)
            except Exception as exc:
                log.error(f"    ✗ Error: {exc}")

    if not all_frames:
        log.error("No valid mail rows found – nothing written")
        sys.exit(1)

    all_mail = pd.concat(all_frames, ignore_index=True).sort_values("mail_date")
    all_mail.to_csv(OUT_PATH, index=False)

    log.info(f"\n✅ Written {len(all_mail):,} rows ➜ {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()