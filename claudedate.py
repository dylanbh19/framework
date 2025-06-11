#!/usr/bin/env python
“””
merge_mail_files.py
────────────────────
Normalises every monthly mail file (RADAR, Product, Meridian) into one CSV:

```
mail_date | mail_volume | mail_type | source_file
```

Run:
python merge_mail_files.py
“””
from **future** import annotations

import sys, glob, logging, re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd

# ───────────────────────────── Logging ──────────────────────────────

logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s %(levelname)-8s | %(message)s”,
handlers=[
logging.StreamHandler(sys.stdout),
logging.FileHandler(“merge_mail_files.log”, “w”, encoding=“utf-8”)
]
)
log = logging.getLogger(**name**)

# ─────────────────────────── Configuration ──────────────────────────

MAIL_GROUPS: List[Dict] = [

```
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
    # Meridian dates like 20250430 (YYYYMMDD format)
    "pref_date_fmt": "%Y%m%d",
    "volume_multiplier": 1,
},
```

]

OUT_DIR  = Path(“data”)
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / “all_mail_data.csv”

# ──────────────────────────── Helpers ───────────────────────────────

_EXCEL_START = pd.Timestamp(“1899-12-30”)  # Excel’s day-0 (windows)

def _excel_serial_to_ts(n: int | float) -> pd.Timestamp | pd.NaT:
“”“Convert Excel serial numbers (typically 40000-50000 range) to Timestamp.”””
# More restrictive range for actual Excel serials - avoid YYYYMMDD confusion
if 25000 <= n <= 60000:  # Roughly 1968-2164
try:
return _EXCEL_START + timedelta(days=float(n))
except Exception:
return pd.NaT
return pd.NaT

def _parse_any_date(val, pref_fmt: str | None = None, debug_name: str = “”) -> pd.Timestamp | pd.NaT:
“””
Very tolerant date parser with improved YYYYMMDD handling:
• Tries preferred format first (important for Meridian YYYYMMDD)
• Excel serials 25000-60000 (more restrictive to avoid YYYYMMDD collision)
• YYYYMMDD strings/ints (20220101-20301231)
• Generic pandas parsing with US then EU day-first
“””
if pd.isna(val):
return pd.NaT

```
original_val = val

# Convert to string for consistent processing
if isinstance(val, (int, float)):
    # Check if it looks like YYYYMMDD (20220101-20301231 range)
    if 20000000 <= val <= 20301231:
        val_str = str(int(val))
        if len(val_str) == 8:
            try:
                return pd.to_datetime(val_str, format="%Y%m%d", errors="coerce")
            except:
                pass
    
    # Try Excel serial if not YYYYMMDD
    serial_try = _excel_serial_to_ts(val)
    if not pd.isna(serial_try):
        return serial_try

s = str(val).strip()

# Remove any non-digit characters for YYYYMMDD check
digits_only = re.sub(r'\D', '', s)

# Try preferred format first (most important for Meridian)
if pref_fmt:
    try:
        # For YYYYMMDD format, ensure we have exactly 8 digits
        if pref_fmt == "%Y%m%d":
            if len(digits_only) == 8 and digits_only.isdigit():
                ts = pd.to_datetime(digits_only, format="%Y%m%d", errors="coerce")
                if not pd.isna(ts):
                    if debug_name:
                        log.debug(f"    {debug_name}: '{original_val}' -> '{digits_only}' -> {ts.strftime('%Y-%m-%d')}")
                    return ts
        else:
            ts = pd.to_datetime(s, format=pref_fmt, errors="coerce")
            if not pd.isna(ts):
                return ts
    except Exception as e:
        if debug_name:
            log.debug(f"    {debug_name}: preferred format failed for '{s}': {e}")

# YYYYMMDD pattern (backup check)
if len(digits_only) == 8 and digits_only.isdigit():
    year = int(digits_only[:4])
    if 2020 <= year <= 2030:  # Reasonable year range
        try:
            ts = pd.to_datetime(digits_only, format="%Y%m%d", errors="coerce")
            if not pd.isna(ts):
                if debug_name:
                    log.debug(f"    {debug_name}: YYYYMMDD backup: '{original_val}' -> {ts.strftime('%Y-%m-%d')}")
                return ts
        except:
            pass

# Generic pandas tries
ts = pd.to_datetime(s, errors="coerce", dayfirst=False)
if not pd.isna(ts):
    return ts

ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
if not pd.isna(ts):
    return ts

# If all else fails, log the failure for debugging
if debug_name:
    log.warning(f"    {debug_name}: Failed to parse date '{original_val}'")

return pd.NaT
```

def _read_file(fp: Path, group: Dict) -> pd.DataFrame:
“”“Load a file and normalise its columns.”””
log.info(f”  • {fp.name}”)
try:
if fp.suffix.lower() in {”.xlsx”, “.xls”}:
# For Meridian, read BillingDate as string to preserve YYYYMMDD format
if group[“name”] == “Meridian”:
df = pd.read_excel(fp, engine=“openpyxl”, dtype={“BillingDate”: str})
else:
df = pd.read_excel(fp, engine=“openpyxl”)
else:
df = pd.read_csv(fp)
except Exception as exc:
log.error(f”    ✗ read error: {exc}”)
return pd.DataFrame()

```
mapping = group["mappings"]
missing = [raw for raw in mapping if raw not in df.columns]
if missing:
    log.warning(f"    ⚠ missing cols {missing} – skipped")
    return pd.DataFrame()

df = df[list(mapping)].rename(columns=mapping)

pref_fmt = group.get("pref_date_fmt")
mult     = group.get("volume_multiplier", 1)

# Enhanced date parsing with debugging for problematic cases
debug_name = f"{group['name']}-{fp.name}"
df["mail_date"] = df["mail_date"].apply(
    lambda x: _parse_any_date(x, pref_fmt, debug_name)
)

# Show some examples of parsed dates for verification
valid_dates = df[df["mail_date"].notna()]["mail_date"].head(5)
if len(valid_dates) > 0:
    log.info(f"    Sample parsed dates: {', '.join(valid_dates.dt.strftime('%Y-%m-%d').tolist())}")

df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce").fillna(0) * mult

if "mail_type" not in df.columns:
    df["mail_type"] = "unknown"

before = len(df)
invalid_dates = df["mail_date"].isna()
invalid_volumes = df["mail_volume"] <= 0

# Log specific issues
if invalid_dates.sum() > 0:
    bad_date_examples = df[invalid_dates]["mail_date"].head(3).tolist()
    log.warning(f"    ⚠ {invalid_dates.sum()} invalid dates, examples: {bad_date_examples}")

if invalid_volumes.sum() > 0:
    log.warning(f"    ⚠ {invalid_volumes.sum()} invalid volumes (≤0)")

df = df[(~invalid_dates) & (~invalid_volumes)]
dropped = before - len(df)

if dropped > 0:
    log.warning(f"    ⚠ dropped {dropped} rows total")

log.info(f"    ✓ {len(df):,} rows kept")

df["source_file"] = fp.name
return df[["mail_date", "mail_volume", "mail_type", "source_file"]]
```

# ─────────────────────────────── Main ───────────────────────────────

def main() -> None:
log.info(”\n──────────── MERGING MAIL FILES ────────────”)
log.info(“Enhanced date parsing - especially for Meridian YYYYMMDD format”)

```
frames: List[pd.DataFrame] = []

for grp in MAIL_GROUPS:
    log.info(f"\n▶ {grp['name']} (pattern: {grp['pattern']})")
    if grp.get("pref_date_fmt"):
        log.info(f"  Expected date format: {grp['pref_date_fmt']}")
    
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

# Show date range summary
min_date = all_mail["mail_date"].min()
max_date = all_mail["mail_date"].max()
log.info(f"\n📅 Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# Show count by source
source_counts = all_mail["source_file"].value_counts()
log.info(f"\n📊 Rows by source:")
for source, count in source_counts.items():
    log.info(f"  {source}: {count:,} rows")

all_mail.to_csv(OUT_PATH, index=False)

log.info(f"\n✅ Written {len(all_mail):,} rows → {OUT_PATH.resolve()}")
```

if **name** == “**main**”:
main()