#!/usr/bin/env python
# merge_mail_files.py
"""
Merge & standardise monthly mail-campaign workbooks from three families
(RADAR, Product, Meridian) → tidy CSVs.

✔ Walks sub-directories
✔ Works with .xlsx / .xls (needs openpyxl)
✔ Column renaming per-family
✔ Logs warnings instead of crashing
✔ Saves:
    data/combined_RADAR.csv
    data/combined_Product.csv
    data/combined_Meridian.csv
    data/all_mail_data.csv
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
import sys, traceback, logging

##############################################################################
# ---------------------------  USER CONFIG --------------------------------- #
##############################################################################

# Where your three folders live - adjust if different
FAMILY_DIRS = {
    "RADAR":     Path("RADAR"),
    "Product":   Path("Product"),
    "Meridian":  Path("Meridian")
}

# Per-family column maps  →  {raw_name: standard_name}
# *** put exact Excel header text (including spaces / case) on the left ***
COLUMN_MAPS: dict[str, dict[str, str]] = {
    "RADAR": {
        "Processing Date":    "mail_date",
        "Estimated Volume":   "mail_volume",
        "Work Order Type":    "mail_type"
    },
    "Product": {
        "Production Complete Date": "mail_date",
        "Product Count":            "mail_volume",
        "Product Type":             "mail_type"
    },
    "Meridian": {
        "Required Mailing Date":    "mail_date",
        "Estimated Mail Volume":    "mail_volume",
        "Work Order Type":          "mail_type"
    }
}

# Date format(s)  – if day/month order varies you can pass dayfirst=True later
DATE_FORMAT = "%m/%d/%Y"    # change if you know the exact pattern

# Output dir
OUTDIR = Path("data")
##############################################################################

OUTDIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUTDIR / "merge_mail_files.log", encoding="utf-8")]
)
log = logging.getLogger(__name__)


def load_one_workbook(path: Path, family: str) -> pd.DataFrame | None:
    """Read first sheet, rename columns, return tidy DF or None if hopeless."""
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        log.warning(f"❌ {path.name:40s} – cannot read ({exc})")
        return None

    cmap = COLUMN_MAPS[family]
    missing = [col for col in cmap if col not in df.columns]
    if missing:
        log.warning(f"⚠  {path.name:40s} – missing columns {missing}")
        # still attempt if at least mail_date & mail_volume present
        needed = {"mail_date", "mail_volume"}
        have   = {v for k, v in cmap.items() if k in df.columns}
        if not needed.issubset(have):
            return None

    # keep just mapped columns
    df = df[[c for c in cmap if c in df.columns]].rename(columns=cmap)

    # minimal cleaning -------------------------------------------------
    df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce").fillna(0).astype(int)
    df["mail_date"]   = pd.to_datetime(df["mail_date"], format=DATE_FORMAT, errors="coerce")
    df = df.dropna(subset=["mail_date"])

    df.insert(0, "source_family", family)
    df.insert(1, "source_file", path.name)
    return df


def gather_family(family: str, dir_path: Path) -> pd.DataFrame:
    """Load every workbook for that family."""
    if not dir_path.exists():
        log.error(f"Folder not found: {dir_path}")
        return pd.DataFrame()

    all_rows: list[pd.DataFrame] = []
    for xlsx in dir_path.glob("*.xls*"):
        tidy = load_one_workbook(xlsx, family)
        if tidy is not None and len(tidy):
            all_rows.append(tidy)

    if all_rows:
        fam_df = pd.concat(all_rows, ignore_index=True)
        out = OUTDIR / f"combined_{family}.csv"
        fam_df.to_csv(out, index=False)
        log.info(f"✓ {family:<8s} → {len(fam_df):7,} rows   saved {out}")
        return fam_df
    else:
        log.warning(f"No usable rows for {family}")
        return pd.DataFrame()


def main() -> None:
    all_families: list[pd.DataFrame] = []

    for fam, folder in FAMILY_DIRS.items():
        fam_df = gather_family(fam, folder)
        if not fam_df.empty:
            all_families.append(fam_df)

    if all_families:
        big = pd.concat(all_families, ignore_index=True)
        big.to_csv(OUTDIR / "all_mail_data.csv", index=False)
        log.info(f"\n🎉 Combined all families → {len(big):,} rows   "
                 f"({OUTDIR/'all_mail_data.csv'})")
    else:
        log.error("No data produced – check logs and column mappings.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("Fatal error:\n" + traceback.format_exc())