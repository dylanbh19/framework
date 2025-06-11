#!/usr/bin/env python3
"""
grab_all_mail.py
────────────────
Collect *every* .xlsx / .xls / .csv file that sits inside the three
mail-families (Product, RADAR, Meridian), show which files are read,
standardise a few key columns and write one big CSV called
all_mail_data.csv

You can now feed that CSV into comprehensive_mail_call_analysis.py
"""

from pathlib import Path
import sys, re, logging
import pandas as pd

##############################################################################
# CONFIG – edit the column names only if your sheets are different
##############################################################################
FOLDERS         = ["Product", "RADAR", "Meridian"]          # top-level folders
DATE_COLUMNS    = ["BillingDate", "Required Mailing Date",
                   "Production Complete Date"]              # any that may appear
VOLUME_COLUMNS  = ["Units", "Product Count",
                   "Estimated Mail Volume"]                 # possible volume cols
TYPE_COLUMNS    = ["Work Order Type", "Product Type",
                   "Description"]                           # possible type cols
DATE_FORMATS    = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y"]
##############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

def find_files() -> list[Path]:
    """Return every xls/xlsx/csv file inside the three folders"""
    files = []
    here  = Path.cwd()
    for folder in FOLDERS:
        files.extend((here / folder).rglob("*.[cC][sS][vV]"))
        files.extend((here / folder).rglob("*.[xX][lL][sS]"))
        files.extend((here / folder).rglob("*.[xX][lL][sS][xX]"))
    return sorted(set(files))

def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unrecognised file type: {path}")

def _standardise(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    """Rename whichever columns exist → mail_date / mail_volume / mail_type"""
    df = df.copy()

    # --- mail_date ----------------------------------------------------------
    for col in DATE_COLUMNS:
        if col in df.columns:
            df["mail_date"] = pd.to_datetime(df[col], errors="coerce",
                                             dayfirst=True, infer_datetime_format=True)
            break
    if "mail_date" not in df.columns:
        log.warning(f"⚠  {src.name}: no date column, rows will be dropped")
        df["mail_date"] = pd.NaT

    # --- mail_volume --------------------------------------------------------
    for col in VOLUME_COLUMNS:
        if col in df.columns:
            df["mail_volume"] = pd.to_numeric(df[col], errors="coerce")
            break
    if "mail_volume" not in df.columns:
        df["mail_volume"] = 1       # assume 1 per-row if nothing better

    # --- mail_type ----------------------------------------------------------
    for col in TYPE_COLUMNS:
        if col in df.columns:
            df["mail_type"] = df[col].astype(str)
            break
    if "mail_type" not in df.columns:
        df["mail_type"] = pd.NA

    df["source_file"] = src.name
    return df.loc[:, ["mail_date", "mail_volume", "mail_type", "source_file"]]

def main() -> None:
    files = find_files()
    if not files:
        log.error("No files found under Product/ RADAR/ Meridian/")
        return

    log.info("╭───────────────────────────────────────────────╮")
    log.info("│  READING MAIL FILES                           │")
    log.info("╰───────────────────────────────────────────────╯")

    frames = []
    for f in files:
        try:
            log.info(f"→ {f.relative_to(Path.cwd())}")
            raw = _read_any(f)
            std = _standardise(raw, f)
            std = std.dropna(subset=["mail_date"])
            if not std.empty:
                frames.append(std)
            else:
                log.warning(f"   {f.name} produced 0 valid rows")
        except Exception as e:
            log.error(f"   Failed to read {f.name}: {e}")

    if not frames:
        log.error("No valid rows found in ANY file – aborting")
        return

    all_mail = pd.concat(frames, ignore_index=True)
    all_mail.sort_values("mail_date", inplace=True)
    out_path = Path("all_mail_data.csv")
    all_mail.to_csv(out_path, index=False)
    log.info(f"\n✓  Written {len(all_mail):,} rows → {out_path}")

if __name__ == "__main__":
    main()