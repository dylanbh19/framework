#!/usr/bin/env python
"""
clean_mail_files.py
-------------------
Cleans Meridian and Product files into proper CSV format:

Meridian: Converts YYYYMMDD dates to MM/DD/YYYY format
Product:  Extracts data from Excel tables and normalizes

Creates clean files in:
  - cleaned_meridian/
  - cleaned_product/

Run:
    python clean_mail_files.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import re

# --------------------------- Logging --------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ------------------------- Configuration ------------------------- #

# Input directories
MERIDIAN_DIR = Path("Meridian")
PRODUCT_DIR = Path("Product")

# Output directories
CLEANED_MERIDIAN_DIR = Path("cleaned_meridian")
CLEANED_PRODUCT_DIR = Path("cleaned_product")

# Create output directories
CLEANED_MERIDIAN_DIR.mkdir(exist_ok=True)
CLEANED_PRODUCT_DIR.mkdir(exist_ok=True)

# ------------------------ Date Helpers ------------------------ #

def convert_yyyymmdd_to_mmddyyyy(date_str):
    """
    Convert YYYYMMDD format to MM/DD/YYYY
    Examples: 20250430 -> 04/30/2025
    """
    if not date_str or pd.isna(date_str):
        return None
    
    # Clean the string - remove any non-digits
    clean_str = re.sub(r'\D', '', str(date_str))
    
    if len(clean_str) != 8:
        log.warning(f"Date string '{date_str}' -> '{clean_str}' is not 8 digits")
        return None
    
    try:
        year = clean_str[:4]
        month = clean_str[4:6]
        day = clean_str[6:8]
        
        # Validate the date
        datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
        
        # Return in MM/DD/YYYY format
        return f"{month}/{day}/{year}"
    
    except ValueError as e:
        log.warning(f"Invalid date '{date_str}': {e}")
        return None

def clean_date_column(df, col_name):
    """Clean a date column in a DataFrame"""
    if col_name not in df.columns:
        log.warning(f"Column '{col_name}' not found in DataFrame")
        return df
    
    log.info(f"  Converting {col_name} from YYYYMMDD to MM/DD/YYYY format")
    
    # Show some before examples
    before_samples = df[col_name].dropna().head(3).tolist()
    log.info(f"  Before: {before_samples}")
    
    df[col_name] = df[col_name].apply(convert_yyyymmdd_to_mmddyyyy)
    
    # Show some after examples
    after_samples = df[col_name].dropna().head(3).tolist()
    log.info(f"  After:  {after_samples}")
    
    # Count nulls
    null_count = df[col_name].isna().sum()
    if null_count > 0:
        log.warning(f"  {null_count} dates could not be converted (now null)")
    
    return df

# ------------------------ Meridian Cleaning ------------------------ #

def clean_meridian_file(file_path):
    """
    Clean a single Meridian file:
    - Convert YYYYMMDD dates to MM/DD/YYYY
    - Save as clean CSV
    """
    log.info(f"\nProcessing Meridian file: {file_path.name}")
    
    try:
        # Read the file
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            # Read as string to preserve date format
            df = pd.read_excel(file_path, dtype={'BillingDate': str})
        else:
            df = pd.read_csv(file_path, dtype={'BillingDate': str})
        
        log.info(f"  Read {len(df)} rows, {len(df.columns)} columns")
        log.info(f"  Columns: {list(df.columns)}")
        
        # Clean the BillingDate column
        df = clean_date_column(df, 'BillingDate')
        
        # Clean any other potential date columns
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col != 'BillingDate':  # Already cleaned
                log.info(f"  Found additional date column: {col}")
                df = clean_date_column(df, col)
        
        # Remove rows with null BillingDate
        before_filter = len(df)
        df = df[df['BillingDate'].notna()]
        after_filter = len(df)
        
        if before_filter > after_filter:
            log.warning(f"  Removed {before_filter - after_filter} rows with invalid dates")
        
        # Save clean file
        clean_filename = f"clean_{file_path.stem}.csv"
        output_path = CLEANED_MERIDIAN_DIR / clean_filename
        
        df.to_csv(output_path, index=False)
        log.info(f"  Saved {len(df)} clean rows to {output_path}")
        
        return True
        
    except Exception as e:
        log.error(f"  Error processing {file_path.name}: {e}")
        return False

# ------------------------ Product Cleaning ------------------------ #

def extract_excel_table_data(file_path):
    """
    Extract data from Excel tables/structured ranges
    Tries multiple methods to get the actual data
    """
    log.info(f"  Attempting to extract table data...")
    
    try:
        # Method 1: Read all sheets and find the one with data
        excel_file = pd.ExcelFile(file_path)
        log.info(f"  Found sheets: {excel_file.sheet_names}")
        
        best_df = None
        best_rows = 0
        
        for sheet_name in excel_file.sheet_names:
            try:
                # Try reading the sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip if mostly empty
                if len(df) < 2:
                    continue
                
                # Check if it has reasonable data
                non_empty_cols = df.count().sum()
                if non_empty_cols > best_rows:
                    best_df = df
                    best_rows = non_empty_cols
                    log.info(f"    Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} cols")
                
            except Exception as e:
                log.warning(f"    Could not read sheet '{sheet_name}': {e}")
        
        if best_df is not None:
            log.info(f"  Selected best sheet with {len(best_df)} rows")
            return best_df
        
        # Method 2: Try reading with different parameters
        log.info("  Trying alternative reading methods...")
        
        # Skip rows that might be headers/formatting
        for skip_rows in [0, 1, 2, 3]:
            try:
                df = pd.read_excel(file_path, skiprows=skip_rows)
                if len(df) > 1 and len(df.columns) > 1:
                    log.info(f"  Success with skiprows={skip_rows}: {len(df)} rows")
                    return df
            except:
                continue
        
        return None
        
    except Exception as e:
        log.error(f"  Failed to extract table data: {e}")
        return None

def clean_product_file(file_path):
    """
    Clean a single Product file:
    - Extract data from Excel tables
    - Normalize column names
    - Save as clean CSV
    """
    log.info(f"\nProcessing Product file: {file_path.name}")
    
    try:
        # Extract the data
        df = extract_excel_table_data(file_path)
        
        if df is None:
            log.error(f"  Could not extract data from {file_path.name}")
            return False
        
        log.info(f"  Extracted {len(df)} rows, {len(df.columns)} columns")
        log.info(f"  Columns: {list(df.columns)}")
        
        # Clean column names - remove extra spaces, special characters
        original_columns = list(df.columns)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Show column mapping if changed
        if list(df.columns) != original_columns:
            log.info("  Cleaned column names:")
            for old, new in zip(original_columns, df.columns):
                if old != new:
                    log.info(f"    '{old}' -> '{new}'")
        
        # Remove completely empty rows
        before_clean = len(df)
        df = df.dropna(how='all')
        after_clean = len(df)
        
        if before_clean > after_clean:
            log.info(f"  Removed {before_clean - after_clean} empty rows")
        
        # Clean any date columns (look for common date column names)
        date_columns = [col for col in df.columns 
                       if any(date_word in col.lower() 
                             for date_word in ['date', 'time', 'created', 'modified', 'updated'])]
        
        for col in date_columns:
            log.info(f"  Cleaning date column: {col}")
            # For product files, we'll keep dates as-is or do basic cleaning
            # (you can modify this based on your specific needs)
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Save clean file
        clean_filename = f"clean_{file_path.stem}.csv"
        output_path = CLEANED_PRODUCT_DIR / clean_filename
        
        df.to_csv(output_path, index=False)
        log.info(f"  Saved {len(df)} clean rows to {output_path}")
        
        # Show a sample of the data
        if len(df) > 0:
            log.info("  Sample data:")
            for i, row in df.head(2).iterrows():
                log.info(f"    Row {i}: {dict(row)}")
        
        return True
        
    except Exception as e:
        log.error(f"  Error processing {file_path.name}: {e}")
        return False

# --------------------------- Main --------------------------- #

def main():
    log.info("MAIL FILE CLEANER")
    log.info("=" * 50)
    
    total_processed = 0
    total_success = 0
    
    # Process Meridian files
    if MERIDIAN_DIR.exists():
        log.info(f"\nPROCESSING MERIDIAN FILES from {MERIDIAN_DIR}")
        log.info("-" * 40)
        
        meridian_files = list(MERIDIAN_DIR.glob("*.xlsx")) + list(MERIDIAN_DIR.glob("*.xls")) + list(MERIDIAN_DIR.glob("*.csv"))
        
        if not meridian_files:
            log.warning("No Meridian files found")
        else:
            for file_path in sorted(meridian_files):
                total_processed += 1
                if clean_meridian_file(file_path):
                    total_success += 1
    else:
        log.warning(f"Meridian directory {MERIDIAN_DIR} not found")
    
    # Process Product files
    if PRODUCT_DIR.exists():
        log.info(f"\nPROCESSING PRODUCT FILES from {PRODUCT_DIR}")
        log.info("-" * 40)
        
        product_files = list(PRODUCT_DIR.glob("*.xlsx")) + list(PRODUCT_DIR.glob("*.xls"))
        
        if not product_files:
            log.warning("No Product files found")
        else:
            for file_path in sorted(product_files):
                total_processed += 1
                if clean_product_file(file_path):
                    total_success += 1
    else:
        log.warning(f"Product directory {PRODUCT_DIR} not found")
    
    # Summary
    log.info("\n" + "=" * 50)
    log.info("SUMMARY")
    log.info(f"Files processed: {total_processed}")
    log.info(f"Successfully cleaned: {total_success}")
    log.info(f"Failed: {total_processed - total_success}")
    
    if total_success > 0:
        log.info(f"\nClean files saved to:")
        log.info(f"  {CLEANED_MERIDIAN_DIR.resolve()}")
        log.info(f"  {CLEANED_PRODUCT_DIR.resolve()}")
    
    if total_processed == 0:
        log.warning("\nNo files found to process!")
        log.info("Make sure you have:")
        log.info("  - Meridian/ directory with .xlsx/.xls/.csv files")
        log.info("  - Product/ directory with .xlsx/.xls files")

if __name__ == "__main__":
    main()
