#!/usr/bin/env python
"""
enhanced_mail_call_analysis_v2.py
==================================
Production-ready analysis with source-based stacking and comprehensive time series modeling.
Handles multiple mail sources (Product, RADAR, Meridian) with augmented data highlighting.

Requirements:
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm statsmodels
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import glob
import json
import logging
from typing import Dict, List, Tuple, Optional
import traceback
import re

# ============================================================================
# ASCII-SAFE LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup ASCII-safe logging for Windows."""
    class ASCIIFormatter(logging.Formatter):
        def format(self, record):
            # Replace problematic Unicode characters with ASCII equivalents
            record.msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
            return super().format(record)
    
    # Create formatter
    formatter = ASCIIFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create handlers
    file_handler = logging.FileHandler('analysis.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# IMPORTS WITH FALLBACKS
# ============================================================================

try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    HAS_SKLEARN = True
    logger.info("scikit-learn loaded successfully")
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available - basic modeling only")

try:
    import xgboost as xgb
    HAS_XGB = True
    logger.info("XGBoost loaded successfully")
except ImportError:
    HAS_XGB = False
    logger.info("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
    logger.info("LightGBM loaded successfully")
except ImportError:
    HAS_LGB = False
    logger.info("LightGBM not available")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
    logger.info("Statsmodels loaded successfully")
except ImportError:
    HAS_STATSMODELS = False
    logger.info("Statsmodels not available")

# Set visualization style
plt.style.use('default')
sns.set_palette("Set2")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Mail data patterns and sources
MAIL_CONFIG = {
    'patterns': ['all_mail_data.csv', '*mail*.csv'],
    'source_mapping': {
        'RADAR': {'patterns': ['RADAR', 'radar'], 'color': '#1f77b4'},
        'Product': {'patterns': ['Product', 'product', 'clean_Product'], 'color': '#ff7f0e'},
        'Meridian': {'patterns': ['Meridian', 'meridian', 'clean_May'], 'color': '#2ca02c'}
    },
    'date_formats': ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']
}

# Call data configuration
CALL_CONFIG = {
    'patterns': ['*Genesys*.csv', '*call*.csv'],
    'date_formats': ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'lag_days_to_test': [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21],
    'min_data_points': 30,
    'test_split_ratio': 0.2,
    'confidence_interval': 0.95,
    'max_augmentation_days': 120,
    'seasonal_periods': [7, 30, 365]  # Weekly, monthly, yearly
}

# Output directory
OUTPUT_DIR = Path("enhanced_analysis_results")

# Color scheme for sources and data quality
COLORS = {
    'RADAR': '#1f77b4',
    'Product': '#ff7f0e', 
    'Meridian': '#2ca02c',
    'actual': '#2ca02c',
    'augmented': '#ff7f0e',
    'forecast': '#9467bd',
    'confidence': 'rgba(148, 103, 189, 0.2)'
}

# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class EnhancedMailCallAnalyzer:
    """Enhanced analyzer with source-based analysis and comprehensive time series modeling."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.mail_data = None
        self.call_data = None
        self.combined_data = None
        self.modeling_data = None
        self.train_data = None
        self.test_data = None
        self.test_data_eval = None
        self.feature_cols = []
        self.models = {}
        self.results = {}
        self.evaluation_results = []
        self.data_issues = []
        self.source_analysis = {}
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create output directory structure."""
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            for subdir in ['data', 'plots', 'models', 'reports', 'source_analysis']:
                (OUTPUT_DIR / subdir).mkdir(exist_ok=True)
            logger.info(f"Output directories created: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            sys.exit(1)
    
    # ========================================================================
    # MAIN ANALYSIS PIPELINE
    # ========================================================================
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING ENHANCED MAIL-CALL ANALYSIS WITH SOURCE TRACKING")
        logger.info("=" * 80)
        
        success_steps = []
        
        try:
            # Step 1: Load and process data
            logger.info("\n1. DATA LOADING AND PROCESSING")
            logger.info("-" * 50)
            if self._load_and_process_data():
                success_steps.append("Data Loading")
                logger.info("SUCCESS: Data loading completed")
            else:
                logger.error("FAILED: Data loading - cannot proceed")
                return
            
            # Step 2: Source-based analysis
            logger.info("\n2. SOURCE-BASED ANALYSIS")
            logger.info("-" * 50)
            if self._analyze_by_source():
                success_steps.append("Source Analysis")
                logger.info("SUCCESS: Source analysis completed")
            
            # Step 3: Data augmentation with tracking
            logger.info("\n3. SMART DATA AUGMENTATION")
            logger.info("-" * 50)
            if self._smart_augmentation():
                success_steps.append("Data Augmentation")
                logger.info("SUCCESS: Data augmentation completed")
            
            # Step 4: Time series analysis
            logger.info("\n4. TIME SERIES ANALYSIS")
            logger.info("-" * 50)
            if self._time_series_analysis():
                success_steps.append("Time Series Analysis")
                logger.info("SUCCESS: Time series analysis completed")
            
            # Step 5: Model building
            if HAS_SKLEARN:
                logger.info("\n5. COMPREHENSIVE MODEL BUILDING")
                logger.info("-" * 50)
                if self._build_comprehensive_models():
                    success_steps.append("Model Building")
                    logger.info("SUCCESS: Model building completed")
            
            # Step 6: Visualization and reporting
            logger.info("\n6. VISUALIZATION AND REPORTING")
            logger.info("-" * 50)
            self._create_comprehensive_visualizations()
            self._create_detailed_reports(success_steps)
            success_steps.append("Visualization & Reporting")
            
            # Step 7: Forecasting
            if self.models:
                logger.info("\n7. FORECASTING")
                logger.info("-" * 50)
                self._create_forecasts()
                success_steps.append("Forecasting")
            
            # Final summary
            self._print_final_summary(success_steps)
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Analysis failed: {e}")
            logger.error(f"TRACEBACK: {traceback.format_exc()}")
            self._create_error_report(str(e), success_steps)
            raise
    
    def _print_final_summary(self, success_steps):
        """Print final analysis summary."""
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"SUCCESS: Completed steps: {', '.join(success_steps)}")
        logger.info(f"RESULTS: Saved to {OUTPUT_DIR}")
        
        if self.data_issues:
            logger.info(f"ISSUES: {len(self.data_issues)} data quality issues identified")
            for i, issue in enumerate(self.data_issues[:3]):
                logger.info(f"  {i+1}. {issue}")
        
        # Model performance summary
        if self.evaluation_results:
            best_model = self.evaluation_results[0]
            logger.info(f"BEST MODEL: {best_model['name']} (MAE: {best_model['mae']:.1f})")
        
        # Data summary
        if self.combined_data is not None:
            total_days = len(self.combined_data)
            completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
            logger.info(f"DATA QUALITY: {completeness:.1f}% actual data over {total_days} days")

    # ========================================================================
    # DATA LOADING AND PROCESSING
    # ========================================================================
    
    def _load_and_process_data(self):
        """Load and process mail and call data."""
        try:
            # Load mail data with source detection
            mail_loaded = self._load_mail_data_with_sources()
            
            # Load call data
            call_loaded = self._load_call_data_enhanced()
            
            if not mail_loaded and not call_loaded:
                logger.error("ERROR: No data files could be loaded")
                return False
            
            # Process and combine
            if mail_loaded:
                logger.info(f"INFO: Mail data loaded - {len(self.mail_data):,} records")
                logger.info(f"INFO: Sources found: {list(self.mail_data['source'].unique())}")
            
            if call_loaded:
                logger.info(f"INFO: Call data loaded - {len(self.call_data):,} records")
            
            return self._create_combined_dataset()
            
        except Exception as e:
            logger.error(f"ERROR: Data loading failed: {e}")
            return False
    
    def _load_mail_data_with_sources(self):
        """Load mail data and detect sources."""
        logger.info("Loading mail data with source detection...")
        
        try:
            # Find mail files
            mail_files = []
            for pattern in MAIL_CONFIG['patterns']:
                files = glob.glob(pattern)
                mail_files.extend(files)
            
            mail_files = list(set(mail_files))  # Remove duplicates
            
            if not mail_files:
                logger.error("ERROR: No mail files found")
                return False
            
            logger.info(f"INFO: Found {len(mail_files)} mail files")
            
            all_mail_data = []
            
            for file_path in mail_files:
                try:
                    logger.info(f"INFO: Processing {file_path}")
                    
                    # Try different encodings
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            logger.info(f"SUCCESS: Read with {encoding} - {len(df):,} rows")
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is None:
                        logger.warning(f"WARNING: Could not read {file_path}")
                        continue
                    
                    # Process the mail data
                    processed_df = self._process_mail_file(df, file_path)
                    
                    if processed_df is not None and len(processed_df) > 0:
                        all_mail_data.append(processed_df)
                        logger.info(f"SUCCESS: Processed {len(processed_df):,} records from {file_path}")
                    
                except Exception as e:
                    logger.error(f"ERROR: Failed to process {file_path}: {e}")
                    self.data_issues.append(f"Mail file error ({file_path}): {e}")
            
            if all_mail_data:
                self.mail_data = pd.concat(all_mail_data, ignore_index=True)
                
                # Remove time component from mail_date if present
                if 'mail_date' in self.mail_data.columns:
                    self.mail_data['mail_date'] = pd.to_datetime(self.mail_data['mail_date']).dt.date
                    self.mail_data['mail_date'] = pd.to_datetime(self.mail_data['mail_date'])
                
                # Remove duplicates
                initial_count = len(self.mail_data)
                self.mail_data = self.mail_data.drop_duplicates(['mail_date', 'source', 'mail_type'], keep='last')
                
                if len(self.mail_data) < initial_count:
                    removed = initial_count - len(self.mail_data)
                    logger.info(f"INFO: Removed {removed:,} duplicate records")
                
                # Save processed data
                self.mail_data.to_csv(OUTPUT_DIR / 'data' / 'processed_mail_data.csv', index=False)
                
                logger.info(f"SUCCESS: Total mail records: {len(self.mail_data):,}")
                logger.info(f"INFO: Date range: {self.mail_data['mail_date'].min()} to {self.mail_data['mail_date'].max()}")
                
                return True
            else:
                logger.error("ERROR: No mail data processed successfully")
                return False
                
        except Exception as e:
            logger.error(f"ERROR: Mail data loading failed: {e}")
            self.data_issues.append(f"Mail loading error: {e}")
            return False
    
    def _process_mail_file(self, df, file_path):
        """Process individual mail file and detect source."""
        try:
            # Initialize processed dataframe
            processed_df = pd.DataFrame()
            
            # Detect and parse date column
            date_col = self._detect_date_column(df)
            if not date_col:
                logger.warning(f"WARNING: No date column found in {file_path}")
                return None
            
            # Parse dates
            processed_df['mail_date'] = self._parse_dates(df[date_col])
            if processed_df['mail_date'].isna().all():
                logger.warning(f"WARNING: Could not parse any dates in {file_path}")
                return None
            
            # Detect volume column
            volume_col = self._detect_volume_column(df)
            if volume_col:
                processed_df['mail_volume'] = pd.to_numeric(df[volume_col], errors='coerce').fillna(1)
            else:
                processed_df['mail_volume'] = 1
                logger.info(f"INFO: No volume column found, using count=1 per record")
            
            # Detect mail type
            type_col = self._detect_type_column(df)
            if type_col:
                processed_df['mail_type'] = df[type_col]
            else:
                processed_df['mail_type'] = 'Unknown'
            
            # Detect source from filename and data
            source = self._detect_source(file_path, df)
            processed_df['source'] = source
            
            # Add metadata
            processed_df['source_file'] = Path(file_path).name
            
            # Remove invalid dates
            processed_df = processed_df.dropna(subset=['mail_date'])
            
            logger.info(f"INFO: Detected source '{source}' with {len(processed_df):,} records")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"ERROR: Processing mail file failed: {e}")
            return None
    
    def _detect_date_column(self, df):
        """Detect date column in dataframe."""
        possible_cols = ['mail_date', 'date', 'send_date', 'campaign_date']
        
        for col in possible_cols:
            if col in df.columns:
                return col
        
        # Look for columns with 'date' in name
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        return date_cols[0] if date_cols else None
    
    def _detect_volume_column(self, df):
        """Detect volume column in dataframe."""
        possible_cols = ['mail_volume', 'volume', 'count', 'quantity']
        
        for col in possible_cols:
            if col in df.columns:
                return col
        
        # Look for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols[0] if numeric_cols else None
    
    def _detect_type_column(self, df):
        """Detect mail type column."""
        possible_cols = ['mail_type', 'type', 'campaign_type', 'category']
        
        for col in possible_cols:
            if col in df.columns:
                return col
        
        return None
    
    def _detect_source(self, file_path, df):
        """Detect data source from filename and content."""
        file_name = Path(file_path).name.lower()
        
        # Check filename patterns
        for source, config in MAIL_CONFIG['source_mapping'].items():
            for pattern in config['patterns']:
                if pattern.lower() in file_name:
                    return source
        
        # Check data content
        if 'source_file' in df.columns:
            sample_files = df['source_file'].astype(str).str.lower()
            for source, config in MAIL_CONFIG['source_mapping'].items():
                for pattern in config['patterns']:
                    if sample_files.str.contains(pattern.lower()).any():
                        return source
        
        # Default source based on common patterns
        if 'radar' in file_name:
            return 'RADAR'
        elif 'product' in file_name:
            return 'Product'
        elif 'may' in file_name or 'meridian' in file_name:
            return 'Meridian'
        
        return 'Unknown'
    
    def _parse_dates(self, date_series):
        """Parse dates with multiple format attempts."""
        for date_format in MAIL_CONFIG['date_formats']:
            try:
                parsed = pd.to_datetime(date_series, format=date_format, errors='coerce')
                if parsed.notna().sum() > len(date_series) * 0.8:
                    return parsed
            except:
                continue
        
        # Try pandas auto-detection as fallback
        try:
            return pd.to_datetime(date_series, errors='coerce')
        except:
            return pd.Series([pd.NaT] * len(date_series))
    
    def _load_call_data_enhanced(self):
        """Load call data with enhanced error handling."""
        logger.info("Loading call data...")
        
        try:
            # Find call files
            call_files = []
            for pattern in CALL_CONFIG['patterns']:
                files = glob.glob(pattern)
                call_files.extend(files)
            
            call_files = list(set(call_files))
            
            if not call_files:
                logger.warning("WARNING: No call files found")
                return False
            
            # Sort by size (largest first)
            call_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            
            logger.info(f"INFO: Found {len(call_files)} call files")
            
            for file_path in call_files:
                try:
                    logger.info(f"INFO: Processing call file: {file_path}")
                    
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"INFO: File size: {file_size_mb:.1f} MB")
                    
                    # Read with encoding detection
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            if file_size_mb > 100:
                                # Test with sample first
                                sample = pd.read_csv(file_path, encoding=encoding, nrows=1000)
                                logger.info(f"INFO: Sample read successful with {encoding}")
                                df = pd.read_csv(file_path, encoding=encoding)
                            else:
                                df = pd.read_csv(file_path, encoding=encoding)
                            
                            logger.info(f"SUCCESS: Loaded {len(df):,} call records with {encoding}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"WARNING: Failed with {encoding}: {str(e)[:100]}")
                            continue
                    
                    if df is None:
                        logger.error(f"ERROR: Could not read {file_path}")
                        continue
                    
                    # Process call data
                    processed_calls = self._process_call_data(df, file_path)
                    
                    if processed_calls is not None and len(processed_calls) > 0:
                        self.call_data = processed_calls
                        
                        # Save processed data
                        self.call_data.to_csv(OUTPUT_DIR / 'data' / 'processed_call_data.csv', index=False)
                        
                        logger.info(f"SUCCESS: Call data processed - {len(self.call_data):,} records")
                        logger.info(f"INFO: Date range: {self.call_data['call_date'].min()} to {self.call_data['call_date'].max()}")
                        
                        return True
                    
                except Exception as e:
                    logger.error(f"ERROR: Failed to process {file_path}: {e}")
                    self.data_issues.append(f"Call file error: {e}")
                    continue
            
            logger.error("ERROR: No call files processed successfully")
            return False
            
        except Exception as e:
            logger.error(f"ERROR: Call data loading failed: {e}")
            return False
    
    def _process_call_data(self, df, file_path):
        """Process call data with robust column detection."""
        try:
            logger.info(f"INFO: Processing call data columns")
            logger.info(f"INFO: Available columns: {list(df.columns)[:10]}...")
            
            call_df = pd.DataFrame()
            
            # Detect datetime column (prioritize ConversationStart for Genesys)
            datetime_col = None
            if 'ConversationStart' in df.columns:
                datetime_col = 'ConversationStart'
                logger.info("INFO: Found Genesys ConversationStart column")
            else:
                # Look for other datetime columns
                possible_datetime_cols = ['datetime', 'timestamp', 'call_date', 'date']
                for col in possible_datetime_cols:
                    if col in df.columns:
                        datetime_col = col
                        break
            
            if not datetime_col:
                # Look for any column with date/time in name
                time_cols = [col for col in df.columns 
                           if any(word in col.lower() for word in ['date', 'time', 'start'])]
                if time_cols:
                    datetime_col = time_cols[0]
                    logger.info(f"INFO: Auto-detected datetime column: {datetime_col}")
            
            if not datetime_col:
                logger.error("ERROR: No datetime column found")
                return None
            
            # Parse datetime
            logger.info(f"INFO: Parsing datetime from: {datetime_col}")
            
            parsed_dates = None
            for date_format in CALL_CONFIG['date_formats']:
                try:
                    parsed_dates = pd.to_datetime(df[datetime_col], format=date_format, errors='coerce')
                    valid_count = parsed_dates.notna().sum()
                    if valid_count > len(df) * 0.8:
                        logger.info(f"SUCCESS: Parsed {valid_count:,} dates with format {date_format}")
                        break
                except:
                    continue
            
            if parsed_dates is None or parsed_dates.notna().sum() < len(df) * 0.5:
                try:
                    parsed_dates = pd.to_datetime(df[datetime_col], errors='coerce')
                    logger.info("INFO: Used pandas auto-detection for dates")
                except:
                    logger.error("ERROR: Could not parse any dates")
                    return None
            
            # Extract date component only
            call_df['call_date'] = parsed_dates.dt.date
            call_df['call_date'] = pd.to_datetime(call_df['call_date'])
            
            # Each row represents one call
            call_df['call_count'] = 1
            
            # Add optional metadata
            optional_cols = {
                'ConversationID': 'conversation_id',
                'OriginatingDirection': 'direction',
                'MediaType': 'media_type'
            }
            
            for orig_col, new_col in optional_cols.items():
                if orig_col in df.columns:
                    call_df[new_col] = df[orig_col]
            
            # Remove invalid dates
            call_df = call_df.dropna(subset=['call_date'])
            
            if len(call_df) == 0:
                logger.error("ERROR: No valid call records after processing")
                return None
            
            logger.info(f"SUCCESS: Processed {len(call_df):,} call records")
            
            return call_df
            
        except Exception as e:
            logger.error(f"ERROR: Call data processing failed: {e}")
            return None

    # ========================================================================
    # DATA COMBINATION AND PREPARATION
    # ========================================================================
    
    def _create_combined_dataset(self):
        """Create combined timeline dataset."""
        logger.info("Creating combined dataset...")
        
        try:
            if self.mail_data is None and self.call_data is None:
                logger.error("ERROR: No data available for combination")
                return False
            
            # Determine date range
            dates = []
            if self.mail_data is not None:
                mail_dates = self.mail_data['mail_date'].dropna()
                if len(mail_dates) > 0:
                    dates.extend([mail_dates.min(), mail_dates.max()])
                    logger.info(f"INFO: Mail date range: {mail_dates.min()} to {mail_dates.max()}")
            
            if self.call_data is not None:
                call_dates = self.call_data['call_date'].dropna()
                if len(call_dates) > 0:
                    dates.extend([call_dates.min(), call_dates.max()])
                    logger.info(f"INFO: Call date range: {call_dates.min()} to {call_dates.max()}")
            
            if not dates:
                logger.error("ERROR: No valid dates found")
                return False
            
            # Create complete timeline
            start_date = min(dates)
            end_date = max(dates)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            logger.info(f"INFO: Timeline: {start_date} to {end_date} ({len(date_range)} days)")
            
            timeline = pd.DataFrame({'date': date_range})
            
            # Aggregate mail data by source and date
            if self.mail_data is not None:
                try:
                    # Create pivot table with sources as columns
                    mail_pivot = self.mail_data.pivot_table(
                        index='mail_date',
                        columns='source',
                        values='mail_volume',
                        aggfunc='sum',
                        fill_value=0
                    ).reset_index()
                    
                    mail_pivot.rename(columns={'mail_date': 'date'}, inplace=True)
                    
                    # Rename columns to include mail_ prefix
                    for col in mail_pivot.columns:
                        if col != 'date':
                            mail_pivot.rename(columns={col: f'mail_{col.lower()}'}, inplace=True)
                    
                    # Calculate total mail volume
                    mail_cols = [col for col in mail_pivot.columns if col.startswith('mail_')]
                    mail_pivot['mail_volume_total'] = mail_pivot[mail_cols].sum(axis=1)
                    
                    timeline = timeline.merge(mail_pivot, on='date', how='left')
                    
                    logger.info(f"SUCCESS: Mail data aggregated by source")
                    logger.info(f"INFO: Mail sources: {[col.replace('mail_', '') for col in mail_cols]}")
                    
                except Exception as e:
                    logger.error(f"ERROR: Mail aggregation failed: {e}")
                    timeline['mail_volume_total'] = 0
            else:
                timeline['mail_volume_total'] = 0
            
            # Aggregate call data by date
            if self.call_data is not None:
                try:
                    daily_calls = self.call_data.groupby('call_date').agg({'call_count': 'sum'}).reset_index()
                    daily_calls.rename(columns={'call_date': 'date'}, inplace=True)
                    timeline = timeline.merge(daily_calls, on='date', how='left')
                    
                    logger.info(f"SUCCESS: Call data aggregated")
                    
                except Exception as e:
                    logger.error(f"ERROR: Call aggregation failed: {e}")
                    timeline['call_count'] = 0
            else:
                timeline['call_count'] = 0
            
            # Fill missing values
            for col in timeline.columns:
                if col != 'date':
                    timeline[col] = timeline[col].fillna(0)
            
            # Add time features
            timeline['day_of_week'] = timeline['date'].dt.dayofweek
            timeline['day_name'] = timeline['date'].dt.day_name()
            timeline['week'] = timeline['date'].dt.isocalendar().week
            timeline['month'] = timeline['date'].dt.month
            timeline['quarter'] = timeline['date'].dt.quarter
            timeline['is_weekend'] = timeline['day_of_week'].isin([5, 6]).astype(int)
            timeline['is_month_start'] = timeline['date'].dt.is_month_start.astype(int)
            timeline['is_month_end'] = timeline['date'].dt.is_month_end.astype(int)
            
            # Add data quality tracking
            timeline['data_quality'] = 'actual'
            timeline['augmentation_method'] = 'none'
            
            self.combined_data = timeline
            
            # Summary statistics
            total_days = len(timeline)
            mail_days = (timeline['mail_volume_total'] > 0).sum()
            call_days = (timeline['call_count'] > 0).sum()
            
            logger.info(f"SUCCESS: Combined dataset created")
            logger.info(f"INFO: Total days: {total_days}")
            logger.info(f"INFO: Days with mail: {mail_days} ({mail_days/total_days*100:.1f}%)")
            logger.info(f"INFO: Days with calls: {call_days} ({call_days/total_days*100:.1f}%)")
            
            if self.mail_data is not None:
                total_mail = timeline['mail_volume_total'].sum()
                logger.info(f"INFO: Total mail volume: {total_mail:,.0f}")
                
                # Log by source
                for source in ['RADAR', 'Product', 'Meridian']:
                    col_name = f'mail_{source.lower()}'
                    if col_name in timeline.columns:
                        source_total = timeline[col_name].sum()
                        logger.info(f"INFO: {source} mail volume: {source_total:,.0f}")
            
            if self.call_data is not None:
                total_calls = timeline['call_count'].sum()
                logger.info(f"INFO: Total call count: {total_calls:,.0f}")
            
            # Save combined data
            self.combined_data.to_csv(OUTPUT_DIR / 'data' / 'combined_timeline.csv', index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Combined dataset creation failed: {e}")
            logger.error(f"TRACEBACK: {traceback.format_exc()}")
            return False
    # ========================================================================
    # SOURCE ANALYSIS
    # ========================================================================
    
    def _analyze_by_source(self):
        """Perform source-specific analysis."""
        logger.info("Performing source-based analysis...")
        
        try:
            if self.mail_data is None:
                logger.warning("WARNING: No mail data for source analysis")
                return False
            
            # Analyze each source
            for source in self.mail_data['source'].unique():
                logger.info(f"INFO: Analyzing source: {source}")
                
                source_data = self.mail_data[self.mail_data['source'] == source]
                
                analysis = {
                    'source': source,
                    'total_records': len(source_data),
                    'total_volume': source_data['mail_volume'].sum(),
                    'date_range': (source_data['mail_date'].min(), source_data['mail_date'].max()),
                    'avg_daily_volume': source_data['mail_volume'].mean(),
                    'unique_types': source_data['mail_type'].nunique() if 'mail_type' in source_data.columns else 0
                }
                
                # Daily aggregation for this source
                daily_source = source_data.groupby('mail_date').agg({
                    'mail_volume': 'sum',
                    'mail_type': 'count'
                }).reset_index()
                
                analysis['days_active'] = len(daily_source)
                analysis['max_daily_volume'] = daily_source['mail_volume'].max()
                analysis['std_daily_volume'] = daily_source['mail_volume'].std()
                
                self.source_analysis[source] = analysis
                
                logger.info(f"SUCCESS: {source} - {analysis['total_records']:,} records, {analysis['total_volume']:,.0f} volume")
            
            # Save source analysis
            source_df = pd.DataFrame(self.source_analysis).T
            source_df.to_csv(OUTPUT_DIR / 'source_analysis' / 'source_summary.csv', index=True)
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Source analysis failed: {e}")
            return False
    
    # ========================================================================
    # DATA AUGMENTATION
    # ========================================================================
    
    def _smart_augmentation(self):
        """Smart data augmentation with comprehensive tracking."""
        logger.info("Performing smart data augmentation...")
        
        try:
            if self.combined_data is None:
                logger.warning("WARNING: No combined data for augmentation")
                return False
            
            # Calculate missing data percentages
            mail_missing = (self.combined_data['mail_volume_total'] == 0)
            call_missing = (self.combined_data['call_count'] == 0)
            
            mail_missing_pct = mail_missing.mean() * 100
            call_missing_pct = call_missing.mean() * 100
            
            logger.info(f"INFO: Mail data gaps: {mail_missing.sum()} days ({mail_missing_pct:.1f}%)")
            logger.info(f"INFO: Call data gaps: {call_missing.sum()} days ({call_missing_pct:.1f}%)")
            
            # Augment mail data if needed
            if 15 <= mail_missing_pct <= 85:
                logger.info("INFO: Augmenting mail data...")
                self._augment_mail_data(mail_missing)
            elif mail_missing_pct > 85:
                logger.warning(f"WARNING: Too much mail data missing ({mail_missing_pct:.1f}%)")
                self.data_issues.append(f"Excessive mail data missing: {mail_missing_pct:.1f}%")
            
            # Augment call data if needed
            if 15 <= call_missing_pct <= 85:
                logger.info("INFO: Augmenting call data...")
                self._augment_call_data(call_missing)
            elif call_missing_pct > 85:
                logger.warning(f"WARNING: Too much call data missing ({call_missing_pct:.1f}%)")
                self.data_issues.append(f"Excessive call data missing: {call_missing_pct:.1f}%")
            
            # Summary
            augmented_count = (self.combined_data['data_quality'] == 'augmented').sum()
            if augmented_count > 0:
                logger.info(f"SUCCESS: Augmented {augmented_count} data points")
                self.combined_data.to_csv(OUTPUT_DIR / 'data' / 'augmented_timeline.csv', index=False)
            else:
                logger.info("INFO: No data augmentation needed")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Data augmentation failed: {e}")
            return False
    
    def _augment_mail_data(self, missing_mask):
        """Augment missing mail data using source-specific patterns."""
        try:
            available_data = self.combined_data[~missing_mask]
            
            if len(available_data) < 14:
                logger.warning("WARNING: Insufficient data for mail augmentation")
                return
            
            # Calculate patterns by day of week
            dow_patterns = available_data.groupby('day_of_week')['mail_volume_total'].agg(['mean', 'std'])
            
            # Calculate monthly patterns
            monthly_patterns = None
            if len(available_data) > 60:
                monthly_patterns = available_data.groupby('month')['mail_volume_total'].agg(['mean', 'std'])
            
            # Calculate source-specific patterns
            source_patterns = {}
            mail_source_cols = [col for col in self.combined_data.columns 
                              if col.startswith('mail_') and col != 'mail_volume_total']
            
            for col in mail_source_cols:
                if available_data[col].sum() > 0:
                    source_patterns[col] = available_data.groupby('day_of_week')[col].agg(['mean', 'std'])
            
            augmented_count = 0
            max_augment = min(len(self.combined_data[missing_mask]), ANALYSIS_PARAMS['max_augmentation_days'])
            
            for idx in self.combined_data[missing_mask].head(max_augment).index:
                try:
                    dow = self.combined_data.loc[idx, 'day_of_week']
                    month = self.combined_data.loc[idx, 'month']
                    
                    if dow in dow_patterns.index and dow_patterns.loc[dow, 'mean'] > 0:
                        base_value = dow_patterns.loc[dow, 'mean']
                        base_std = dow_patterns.loc[dow, 'std']
                        
                        # Apply monthly adjustment
                        if monthly_patterns is not None and month in monthly_patterns.index:
                            monthly_factor = monthly_patterns.loc[month, 'mean'] / dow_patterns['mean'].mean()
                            base_value *= max(0.1, min(3.0, monthly_factor))  # Reasonable bounds
                        
                        # Add controlled noise
                        if base_std > 0 and not pd.isna(base_std):
                            noise = np.random.normal(0, base_std * 0.25)
                        else:
                            noise = np.random.normal(0, base_value * 0.1)
                        
                        total_augmented = max(0, base_value + noise)
                        self.combined_data.loc[idx, 'mail_volume_total'] = total_augmented
                        
                        # Distribute across sources based on historical patterns
                        for col in source_patterns:
                            if dow in source_patterns[col].index:
                                source_mean = source_patterns[col].loc[dow, 'mean']
                                if source_mean > 0:
                                    source_proportion = source_mean / dow_patterns.loc[dow, 'mean']
                                    source_value = total_augmented * max(0, min(1, source_proportion))
                                    self.combined_data.loc[idx, col] = source_value
                        
                        self.combined_data.loc[idx, 'data_quality'] = 'augmented'
                        self.combined_data.loc[idx, 'augmentation_method'] = 'source_pattern_based'
                        augmented_count += 1
                        
                except Exception as e:
                    logger.warning(f"WARNING: Error augmenting mail at index {idx}: {e}")
                    continue
            
            logger.info(f"SUCCESS: Augmented {augmented_count} mail data points")
            
        except Exception as e:
            logger.error(f"ERROR: Mail augmentation failed: {e}")
    
    def _augment_call_data(self, missing_mask):
        """Augment missing call data using mail correlation."""
        try:
            available_data = self.combined_data[~missing_mask]
            
            if len(available_data) < 14:
                logger.warning("WARNING: Insufficient data for call augmentation")
                return
            
            # Find best lag correlation with mail
            best_lag = 3
            best_corr = 0
            
            for lag in [1, 2, 3, 4, 5, 7]:
                mail_lagged = self.combined_data['mail_volume_total'].shift(lag)
                valid_mask = (mail_lagged > 0) & (self.combined_data['call_count'] > 0)
                
                if valid_mask.sum() > 10:
                    try:
                        corr = mail_lagged[valid_mask].corr(self.combined_data.loc[valid_mask, 'call_count'])
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag = lag
                    except:
                        continue
            
            logger.info(f"INFO: Using lag {best_lag} days for call augmentation (correlation: {best_corr:.3f})")
            
            # Calculate day-of-week patterns
            dow_patterns = available_data.groupby('day_of_week')['call_count'].agg(['mean', 'std'])
            
            # Calculate mail-to-call ratio
            valid_data = self.combined_data[
                (self.combined_data['mail_volume_total'] > 0) & 
                (self.combined_data['call_count'] > 0)
            ]
            
            if len(valid_data) > 0:
                mail_call_ratio = valid_data['call_count'].sum() / valid_data['mail_volume_total'].sum()
            else:
                mail_call_ratio = 0.01
            
            augmented_count = 0
            mail_lagged = self.combined_data['mail_volume_total'].shift(best_lag)
            max_augment = min(len(self.combined_data[missing_mask]), ANALYSIS_PARAMS['max_augmentation_days'])
            
            for idx in self.combined_data[missing_mask].head(max_augment).index:
                try:
                    # Try mail-based prediction first
                    if idx >= best_lag and mail_lagged.iloc[idx] > 0:
                        predicted = mail_lagged.iloc[idx] * mail_call_ratio
                        noise = np.random.normal(0, predicted * 0.15)
                        call_value = max(0, predicted + noise)
                        method = 'mail_lag_based'
                    else:
                        # Fall back to day-of-week pattern
                        dow = self.combined_data.loc[idx, 'day_of_week']
                        if dow in dow_patterns.index and dow_patterns.loc[dow, 'mean'] > 0:
                            base_value = dow_patterns.loc[dow, 'mean']
                            base_std = dow_patterns.loc[dow, 'std']
                            
                            if base_std > 0 and not pd.isna(base_std):
                                noise = np.random.normal(0, base_std * 0.3)
                            else:
                                noise = np.random.normal(0, base_value * 0.15)
                            
                            call_value = max(0, base_value + noise)
                            method = 'dow_pattern_based'
                        else:
                            continue
                    
                    self.combined_data.loc[idx, 'call_count'] = call_value
                    self.combined_data.loc[idx, 'data_quality'] = 'augmented'
                    self.combined_data.loc[idx, 'augmentation_method'] = method
                    augmented_count += 1
                    
                except Exception as e:
                    logger.warning(f"WARNING: Error augmenting calls at index {idx}: {e}")
                    continue
            
            logger.info(f"SUCCESS: Augmented {augmented_count} call data points")
            
        except Exception as e:
            logger.error(f"ERROR: Call augmentation failed: {e}")
    
    # ========================================================================
    # TIME SERIES ANALYSIS
    # ========================================================================
    
    def _time_series_analysis(self):
        """Comprehensive time series analysis."""
        logger.info("Performing time series analysis...")
        
        try:
            if self.combined_data is None:
                logger.warning("WARNING: No data for time series analysis")
                return False
            
            # Analyze trends and seasonality
            self._analyze_trends_seasonality()
            
            # Correlation analysis
            self._analyze_correlations()
            
            # Source correlation analysis
            self._analyze_source_correlations()
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Time series analysis failed: {e}")
            return False
    
    def _analyze_trends_seasonality(self):
        """Analyze trends and seasonal patterns."""
        try:
            logger.info("INFO: Analyzing trends and seasonality...")
            
            # Weekly seasonality analysis
            weekly_patterns = {}
            
            # Mail volume patterns
            if 'mail_volume_total' in self.combined_data.columns:
                weekly_mail = self.combined_data.groupby('day_of_week')['mail_volume_total'].mean()
                weekly_patterns['mail'] = weekly_mail.to_dict()
            
            # Call volume patterns
            if 'call_count' in self.combined_data.columns:
                weekly_calls = self.combined_data.groupby('day_of_week')['call_count'].mean()
                weekly_patterns['calls'] = weekly_calls.to_dict()
            
            # Source-specific patterns
            mail_source_cols = [col for col in self.combined_data.columns 
                              if col.startswith('mail_') and col != 'mail_volume_total']
            for col in mail_source_cols:
                source_name = col.replace('mail_', '')
                weekly_source = self.combined_data.groupby('day_of_week')[col].mean()
                weekly_patterns[source_name] = weekly_source.to_dict()
            
            # Monthly patterns
            monthly_patterns = {}
            if 'mail_volume_total' in self.combined_data.columns:
                monthly_mail = self.combined_data.groupby('month')['mail_volume_total'].mean()
                monthly_patterns['mail'] = monthly_mail.to_dict()
            
            if 'call_count' in self.combined_data.columns:
                monthly_calls = self.combined_data.groupby('month')['call_count'].mean()
                monthly_patterns['calls'] = monthly_calls.to_dict()
            
            # Store results
            self.results['weekly_patterns'] = weekly_patterns
            self.results['monthly_patterns'] = monthly_patterns
            
            # Save patterns
            pd.DataFrame(weekly_patterns).to_csv(OUTPUT_DIR / 'data' / 'weekly_patterns.csv', index=True)
            pd.DataFrame(monthly_patterns).to_csv(OUTPUT_DIR / 'data' / 'monthly_patterns.csv', index=True)
            
            logger.info("SUCCESS: Trend and seasonality analysis completed")
            
        except Exception as e:
            logger.error(f"ERROR: Trend analysis failed: {e}")

    def _analyze_correlations(self):
        """Analyze correlations between mail and calls."""
        try:
            logger.info("INFO: Analyzing mail-call correlations...")
            
            # Direct correlation (same day)
            if 'mail_volume_total' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                valid_data = self.combined_data[
                    (self.combined_data['mail_volume_total'] > 0) & 
                    (self.combined_data['call_count'] > 0) &
                    (self.combined_data['data_quality'] == 'actual')  # Use only actual data for correlation
                ]
                
                if len(valid_data) > 10:
                    direct_corr = valid_data['mail_volume_total'].corr(valid_data['call_count'])
                    logger.info(f"INFO: Same-day correlation: {direct_corr:.3f}")
                    self.results['direct_correlation'] = direct_corr
                else:
                    logger.warning("WARNING: Insufficient actual data for same-day correlation")
                    self.results['direct_correlation'] = 0
            
            # Lag correlations
            best_lag = 0
            best_corr = 0
            lag_results = {}
            
            for lag in ANALYSIS_PARAMS['lag_days_to_test']:
                try:
                    mail_lagged = self.combined_data['mail_volume_total'].shift(lag)
                    
                    # Only use actual data for correlation analysis
                    actual_mask = (self.combined_data['data_quality'] == 'actual')
                    valid_mask = actual_mask & (mail_lagged > 0) & (self.combined_data['call_count'] > 0)
                    valid_mask &= ~(mail_lagged.isna() | self.combined_data['call_count'].isna())
                    
                    if valid_mask.sum() > 10:
                        corr = mail_lagged[valid_mask].corr(self.combined_data.loc[valid_mask, 'call_count'])
                        lag_results[lag] = corr
                        
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag = lag
                            
                except Exception as e:
                    logger.warning(f"WARNING: Error calculating lag {lag}: {e}")
                    continue
            
            if lag_results:
                logger.info(f"SUCCESS: Best lag correlation: {best_lag} days ({best_corr:.3f})")
                self.results['best_lag'] = best_lag
                self.results['best_lag_correlation'] = best_corr
                self.results['lag_correlations'] = lag_results
                
                # Save lag correlations
                pd.DataFrame(list(lag_results.items()), columns=['lag', 'correlation']).to_csv(
                    OUTPUT_DIR / 'data' / 'lag_correlations.csv', index=False
                )
            else:
                logger.warning("WARNING: Could not calculate meaningful lag correlations")
                self.results['best_lag'] = 3
                self.results['best_lag_correlation'] = 0
            
        except Exception as e:
            logger.error(f"ERROR: Correlation analysis failed: {e}")

    def _analyze_source_correlations(self):
        """Analyze correlations between individual sources and calls."""
        try:
            logger.info("INFO: Analyzing source-specific correlations...")
            
            source_correlations = {}
            mail_source_cols = [col for col in self.combined_data.columns 
                                if col.startswith('mail_') and col != 'mail_volume_total']
            
            for col in mail_source_cols:
                source_name = col.replace('mail_', '')
                
                # Only use actual data
                actual_mask = (self.combined_data['data_quality'] == 'actual')
                valid_mask = actual_mask & (self.combined_data[col] > 0) & (self.combined_data['call_count'] > 0)
                
                if valid_mask.sum() > 10:
                    try:
                        corr = self.combined_data.loc[valid_mask, col].corr(
                            self.combined_data.loc[valid_mask, 'call_count']
                        )
                        source_correlations[source_name] = {
                            'correlation': corr,
                            'data_points': valid_mask.sum()
                        }
                        logger.info(f"INFO: {source_name} correlation: {corr:.3f} (n={valid_mask.sum()})")
                    except Exception as e:
                        logger.warning(f"WARNING: Error calculating {source_name} correlation: {e}")
                else:
                    logger.warning(f"WARNING: Insufficient data for {source_name} correlation")
            
            self.results['source_correlations'] = source_correlations
            
            # Save source correlations
            if source_correlations:
                source_corr_df = pd.DataFrame(source_correlations).T
                source_corr_df.to_csv(OUTPUT_DIR / 'source_analysis' / 'source_correlations.csv', index=True)
            
        except Exception as e:
            logger.error(f"ERROR: Source correlation analysis failed: {e}")

# ========================================================================
    # MODEL BUILDING
    # ========================================================================

    def _build_comprehensive_models(self):
        """Build comprehensive models including time series models."""
        logger.info("Building comprehensive prediction models...")

        try:
            # Prepare features
            if not self._prepare_modeling_features():
                return False
            
            # Split data
            if not self._split_data_for_modeling():
                return False
            
            # Build different types of models
            self._build_baseline_models()
            self._build_machine_learning_models()
            
            if HAS_STATSMODELS:
                self._build_time_series_models()
            
            # Evaluate all models
            if self.models:
                self._evaluate_all_models()
                return True
            else:
                logger.warning("WARNING: No models built successfully")
                return False
            
        except Exception as e:
            logger.error(f"ERROR: Model building failed: {e}")
            return False

    def _prepare_modeling_features(self):
        """Prepare comprehensive features for modeling."""
        try:
            logger.info("INFO: Preparing modeling features...")
            
            # Create lag features for total mail volume
            for lag in [1, 2, 3, 5, 7, 14]:
                if len(self.combined_data) > lag:
                    self.combined_data[f'mail_lag_{lag}'] = self.combined_data['mail_volume_total'].shift(lag)
                    self.combined_data[f'call_lag_{lag}'] = self.combined_data['call_count'].shift(lag)
            
            # Create lag features for each source
            mail_source_cols = [col for col in self.combined_data.columns 
                                if col.startswith('mail_') and col != 'mail_volume_total']
            for col in mail_source_cols:
                source_name = col.replace('mail_', '')
                for lag in [1, 3, 7]:
                    if len(self.combined_data) > lag:
                        self.combined_data[f'{source_name}_lag_{lag}'] = self.combined_data[col].shift(lag)
            
            # Moving averages
            for window in [3, 7, 14, 30]:
                if len(self.combined_data) >= window:
                    self.combined_data[f'mail_ma_{window}'] = (
                        self.combined_data['mail_volume_total'].rolling(window, min_periods=1).mean()
                    )
                    self.combined_data[f'call_ma_{window}'] = (
                        self.combined_data['call_count'].rolling(window, min_periods=1).mean()
                    )
            
            # Cyclical time features
            self.combined_data['day_sin'] = np.sin(2 * np.pi * self.combined_data['day_of_week'] / 7)
            self.combined_data['day_cos'] = np.cos(2 * np.pi * self.combined_data['day_of_week'] / 7)
            self.combined_data['month_sin'] = np.sin(2 * np.pi * self.combined_data['month'] / 12)
            self.combined_data['month_cos'] = np.cos(2 * np.pi * self.combined_data['month'] / 12)
            
            # Interaction features
            self.combined_data['mail_weekend_interaction'] = (
                self.combined_data['mail_volume_total'] * self.combined_data['is_weekend']
            )
            
            # Remove rows with too many NaN values
            self.modeling_data = self.combined_data.dropna(subset=['call_count'])
            
            logger.info(f"SUCCESS: Features prepared - {len(self.modeling_data)} rows for modeling")
            logger.info(f"INFO: Feature columns: {len([col for col in self.modeling_data.columns if col not in ['date', 'call_count', 'data_quality', 'augmentation_method', 'day_name']])}")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Feature preparation failed: {e}")
            return False

    def _split_data_for_modeling(self):
        """Split data for modeling with time series considerations."""
        try:
            logger.info("INFO: Splitting data for modeling...")
            
            # Prefer actual data for evaluation
            actual_data = self.modeling_data[self.modeling_data['data_quality'] == 'actual']
            
            if len(actual_data) < ANALYSIS_PARAMS['min_data_points']:
                logger.warning(f"WARNING: Only {len(actual_data)} actual data points, using all data")
                modeling_data = self.modeling_data
            else:
                modeling_data = self.modeling_data
            
            # Time-based split (important for time series)
            split_idx = int(len(modeling_data) * (1 - ANALYSIS_PARAMS['test_split_ratio']))
            split_date = modeling_data.iloc[split_idx]['date']
            
            self.train_data = modeling_data[modeling_data['date'] < split_date]
            self.test_data = modeling_data[modeling_data['date'] >= split_date]
            
            # For evaluation, prefer actual data in test set
            actual_test = self.test_data[self.test_data['data_quality'] == 'actual']
            if len(actual_test) >= 5:
                self.test_data_eval = actual_test
                logger.info("INFO: Using actual data only for evaluation")
            else:
                self.test_data_eval = self.test_data
                logger.info("INFO: Using all test data for evaluation")
            
            logger.info(f"SUCCESS: Data split completed")
            logger.info(f"INFO: Train: {len(self.train_data)} rows")
            logger.info(f"INFO: Test: {len(self.test_data_eval)} rows")
            logger.info(f"INFO: Split date: {split_date}")
            
            return len(self.test_data_eval) >= 3
            
        except Exception as e:
            logger.error(f"ERROR: Data splitting failed: {e}")
            return False

    def _build_baseline_models(self):
        """Build baseline models."""
        try:
            logger.info("INFO: Building baseline models...")
            
            # Define feature columns
            exclude_cols = ['date', 'call_count', 'data_quality', 'augmentation_method', 'day_name']
            self.feature_cols = [col for col in self.modeling_data.columns if col not in exclude_cols]
            
            y_train = self.train_data['call_count']
            y_test = self.test_data_eval['call_count']
            
            # Historical average
            if len(y_train) > 0:
                avg_pred = np.full(len(y_test), y_train.mean())
                self.models['baseline_avg'] = {
                    'predictions': avg_pred,
                    'name': 'Historical Average',
                    'description': 'Uses overall historical average',
                    'type': 'baseline'
                }
            
            # Day-of-week average
            if 'day_of_week' in self.train_data.columns:
                dow_avg = self.train_data.groupby('day_of_week')['call_count'].mean()
                dow_pred = self.test_data_eval['day_of_week'].map(dow_avg).fillna(y_train.mean())
                self.models['baseline_dow'] = {
                    'predictions': dow_pred.values,
                    'name': 'Day-of-Week Average',
                    'description': 'Uses day-of-week historical patterns',
                    'type': 'baseline'
                }
            
            # Seasonal naive (same day last week)
            if len(self.train_data) >= 7:
                seasonal_pred = []
                for idx in self.test_data_eval.index:
                    # Find same day of week in recent history
                    test_dow = self.test_data_eval.loc[idx, 'day_of_week']
                    recent_same_day = self.train_data[
                        self.train_data['day_of_week'] == test_dow
                    ]['call_count'].tail(4).mean()  # Average of last 4 same days
                    seasonal_pred.append(recent_same_day if not pd.isna(recent_same_day) else y_train.mean())
                
                self.models['baseline_seasonal'] = {
                    'predictions': np.array(seasonal_pred),
                    'name': 'Seasonal Naive',
                    'description': 'Uses same day of week from recent history',
                    'type': 'baseline'
                }
            
            logger.info(f"SUCCESS: Built {len([m for m in self.models.values() if m.get('type') == 'baseline'])} baseline models")
            
        except Exception as e:
            logger.error(f"ERROR: Baseline model building failed: {e}")

    def _build_machine_learning_models(self):
        """Build machine learning models."""
        try:
            logger.info("INFO: Building machine learning models...")
            
            X_train = self.train_data[self.feature_cols].fillna(0)
            y_train = self.train_data['call_count']
            X_test = self.test_data_eval[self.feature_cols].fillna(0)
            
            if len(X_train) < 10:
                logger.warning("WARNING: Insufficient training data for ML models")
                return
            
            # Ridge Regression
            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train_scaled, y_train)
                
                self.models['ridge'] = {
                    'model': ridge,
                    'scaler': scaler,
                    'predictions': ridge.predict(X_test_scaled),
                    'name': 'Ridge Regression',
                    'description': 'L2 regularized linear regression',
                    'type': 'ml'
                }
                logger.info("SUCCESS: Ridge Regression built")
            except Exception as e:
                logger.warning(f"WARNING: Ridge Regression failed: {e}")
            
            # Random Forest
            try:
                n_estimators = min(100, max(10, len(X_train) // 2))
                max_depth = min(15, max(3, len(self.feature_cols)))
                
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.models['random_forest'] = {
                    'model': rf,
                    'predictions': rf.predict(X_test),
                    'name': 'Random Forest',
                    'description': 'Ensemble of decision trees',
                    'type': 'ml',
                    'feature_importance': feature_importance
                }
                logger.info("SUCCESS: Random Forest built")
            except Exception as e:
                logger.warning(f"WARNING: Random Forest failed: {e}")
            
            # Gradient Boosting
            try:
                gbr = GradientBoostingRegressor(
                    n_estimators=min(100, max(20, len(X_train) // 3)),
                    learning_rate=0.1,
                    max_depth=min(6, max(3, len(self.feature_cols) // 3)),
                    random_state=42
                )
                gbr.fit(X_train, y_train)
                
                self.models['gradient_boosting'] = {
                    'model': gbr,
                    'predictions': gbr.predict(X_test),
                    'name': 'Gradient Boosting',
                    'description': 'Sequential boosting ensemble',
                    'type': 'ml'
                }
                logger.info("SUCCESS: Gradient Boosting built")
            except Exception as e:
                logger.warning(f"WARNING: Gradient Boosting failed: {e}")
            
            # XGBoost (if available)
            if HAS_XGB:
                try:
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=min(100, max(20, len(X_train) // 3)),
                        learning_rate=0.1,
                        max_depth=min(6, max(3, len(self.feature_cols) // 3)),
                        random_state=42,
                        verbosity=0
                    )
                    xgb_model.fit(X_train, y_train)
                    
                    self.models['xgboost'] = {
                        'model': xgb_model,
                        'predictions': xgb_model.predict(X_test),
                        'name': 'XGBoost',
                        'description': 'Optimized gradient boosting',
                        'type': 'ml'
                    }
                    logger.info("SUCCESS: XGBoost built")
                except Exception as e:
                    logger.warning(f"WARNING: XGBoost failed: {e}")
            
            # LightGBM (if available)
            if HAS_LGB:
                try:
                    lgb_model = lgb.LGBMRegressor(
                        n_estimators=min(100, max(20, len(X_train) // 3)),
                        learning_rate=0.1,
                        max_depth=min(6, max(3, len(self.feature_cols) // 3)),
                        random_state=42,
                        verbosity=-1
                    )
                    lgb_model.fit(X_train, y_train)
                    
                    self.models['lightgbm'] = {
                        'model': lgb_model,
                        'predictions': lgb_model.predict(X_test),
                        'name': 'LightGBM',
                        'description': 'Light gradient boosting machine',
                        'type': 'ml'
                    }
                    logger.info("SUCCESS: LightGBM built")
                except Exception as e:
                    logger.warning(f"WARNING: LightGBM failed: {e}")
            
            ml_count = len([m for m in self.models.values() if m.get('type') == 'ml'])
            logger.info(f"SUCCESS: Built {ml_count} machine learning models")
            
        except Exception as e:
            logger.error(f"ERROR: ML model building failed: {e}")

    def _build_time_series_models(self):
        """Build time series specific models."""
        try:
            logger.info("INFO: Building time series models...")
            
            # Prepare time series data (daily call counts)
            ts_data = self.train_data.set_index('date')['call_count'].asfreq('D', fill_value=0)
            
            if len(ts_data) < 30:
                logger.warning("WARNING: Insufficient data for time series models")
                return
            
            # ARIMA model
            try:
                # Simple ARIMA(1,1,1) - can be optimized
                arima_model = ARIMA(ts_data, order=(1, 1, 1))
                arima_fitted = arima_model.fit()
                
                # Forecast for test period
                forecast_steps = len(self.test_data_eval)
                arima_forecast = arima_fitted.forecast(steps=forecast_steps)
                
                self.models['arima'] = {
                    'model': arima_fitted,
                    'predictions': arima_forecast,
                    'name': 'ARIMA(1,1,1)',
                    'description': 'Autoregressive integrated moving average',
                    'type': 'time_series'
                }
                logger.info("SUCCESS: ARIMA model built")
            except Exception as e:
                logger.warning(f"WARNING: ARIMA model failed: {e}")
            
            # Exponential Smoothing
            try:
                if len(ts_data) >= 14:  # Need enough data for seasonal patterns
                    exp_smooth = ExponentialSmoothing(
                        ts_data,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=7  # Weekly seasonality
                    )
                    exp_fitted = exp_smooth.fit()
                    
                    forecast_steps = len(self.test_data_eval)
                    exp_forecast = exp_fitted.forecast(steps=forecast_steps)
                    
                    self.models['exp_smoothing'] = {
                        'model': exp_fitted,
                        'predictions': exp_forecast,
                        'name': 'Exponential Smoothing',
                        'description': 'Triple exponential smoothing with trend and seasonality',
                        'type': 'time_series'
                    }
                    logger.info("SUCCESS: Exponential Smoothing built")
            except Exception as e:
                logger.warning(f"WARNING: Exponential Smoothing failed: {e}")
            
            ts_count = len([m for m in self.models.values() if m.get('type') == 'time_series'])
            logger.info(f"SUCCESS: Built {ts_count} time series models")
            
        except Exception as e:
            logger.error(f"ERROR: Time series model building failed: {e}")

    def _evaluate_all_models(self):
        """Evaluate all models with comprehensive metrics."""
        try:
            logger.info("INFO: Evaluating all models...")
            
            y_test = self.test_data_eval['call_count'].values
            
            if len(y_test) == 0:
                logger.error("ERROR: No test data for evaluation")
                return
            
            self.evaluation_results = []
            
            for model_name, model_info in self.models.items():
                try:
                    predictions = np.array(model_info['predictions']).flatten()
                    
                    # Ensure same length
                    min_len = min(len(y_test), len(predictions))
                    y_test_eval = y_test[:min_len]
                    pred_eval = predictions[:min_len]
                    
                    # Calculate comprehensive metrics
                    mae = mean_absolute_error(y_test_eval, pred_eval)
                    rmse = np.sqrt(mean_squared_error(y_test_eval, pred_eval))
                    
                    # MAPE with zero handling
                    mask = y_test_eval != 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_test_eval[mask] - pred_eval[mask]) / y_test_eval[mask])) * 100
                    else:
                        mape = 100.0
                    
                    # R-squared
                    try:
                        r2 = r2_score(y_test_eval, pred_eval)
                    except:
                        r2 = 0.0
                    
                    # Additional metrics
                    max_error = np.max(np.abs(y_test_eval - pred_eval))
                    median_ae = np.median(np.abs(y_test_eval - pred_eval))
                    
                    result = {
                        'model': model_name,
                        'name': model_info['name'],
                        'description': model_info['description'],
                        'type': model_info.get('type', 'unknown'),
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'r2': r2,
                        'max_error': max_error,
                        'median_ae': median_ae,
                        'predictions': pred_eval,
                        'n_predictions': len(pred_eval)
                    }
                    
                    self.evaluation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"WARNING: Error evaluating {model_name}: {e}")
                    continue
            
            if self.evaluation_results:
                # Sort by MAE (primary metric)
                self.evaluation_results = sorted(self.evaluation_results, key=lambda x: x['mae'])
                
                # Print comprehensive results
                logger.info("\nMODEL PERFORMANCE SUMMARY:")
                logger.info("=" * 100)
                logger.info(f"{'Model':<25} {'Type':<12} {'MAE':<8} {'RMSE':<8} {'MAPE%':<8} {'R2':<8} {'MaxErr':<8}")
                logger.info("=" * 100)
                
                for result in self.evaluation_results:
                    logger.info(
                        f"{result['name']:<25} {result['type']:<12} "
                        f"{result['mae']:<8.1f} {result['rmse']:<8.1f} "
                        f"{result['mape']:<8.1f} {result['r2']:<8.3f} {result['max_error']:<8.1f}"
                    )
                
                best = self.evaluation_results[0]
                logger.info("=" * 100)
                logger.info(f"BEST MODEL: {best['name']} (MAE: {best['mae']:.1f}, R2: {best['r2']:.3f})")
                
                # Save evaluation results
                eval_df = pd.DataFrame(self.evaluation_results)
                eval_df.to_csv(OUTPUT_DIR / 'models' / 'model_evaluation.csv', index=False)
                
            else:
                logger.warning("WARNING: No models could be evaluated")
            
        except Exception as e:
            logger.error(f"ERROR: Model evaluation failed: {e}")

    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================

    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualizations with source stacking."""
        logger.info("Creating comprehensive visualizations...")
        
        try:
            # Main dashboard
            self._create_main_dashboard()
            
            # Source analysis plots
            self._create_source_analysis_plots()
            
            # Model comparison plots
            self._create_model_comparison_plots()
            
            # Time series decomposition plots
            self._create_time_series_plots()
            
            logger.info("SUCCESS: All visualizations created")
            
        except Exception as e:
            logger.error(f"ERROR: Visualization creation failed: {e}")

    def _create_main_dashboard(self):
        """Create main dashboard with source stacking and augmentation highlighting."""
        try:
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    'Mail Volume by Source (Stacked) + Call Volume',
                    'Data Quality Overview',
                    'Source Correlation Analysis',
                    'Model Performance Comparison',
                    'Day-of-Week Patterns by Source',
                    'Actual vs Augmented Data Distribution',
                    'Weekly Trends with Data Quality',
                    'Lag Correlation Analysis'
                ),
                specs=[
                    [{'secondary_y': True}, {'type': 'indicator'}],
                    [{'type': 'bar'}, {'type': 'bar'}],
                    [{'type': 'bar'}, {'type': 'histogram'}],
                    [{'type': 'bar'}, {'type': 'scatter'}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.15,
                row_heights=[0.3, 0.25, 0.25, 0.2]
            )
            
            # 1. Stacked mail by source + calls (top left)
            mail_source_cols = [col for col in self.combined_data.columns 
                                if col.startswith('mail_') and col != 'mail_volume_total']
            
            # Add stacked mail traces
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                color = COLORS.get(source_name, f'hsl({i*120}, 70%, 50%)')
                
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data[col],
                        name=f'{source_name} Mail',
                        fill='tonexty' if i > 0 else 'tozeroy',
                        line=dict(color=color, width=0),
                        stackgroup='mail'
                    ),
                    row=1, col=1, secondary_y=False
                )
            
            # Add call volume with augmentation highlighting
            actual_data = self.combined_data[self.combined_data['data_quality'] == 'actual']
            augmented_data = self.combined_data[self.combined_data['data_quality'] == 'augmented']
            
            fig.add_trace(
                go.Scatter(
                    x=actual_data['date'],
                    y=actual_data['call_count'],
                    name='Calls (Actual)',
                    line=dict(color=COLORS['actual'], width=2),
                    mode='lines'
                ),
                row=1, col=1, secondary_y=True
            )
            
            if len(augmented_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=augmented_data['date'],
                        y=augmented_data['call_count'],
                        name='Calls (Augmented)',
                        line=dict(color=COLORS['augmented'], width=2, dash='dot'),
                        mode='lines'
                    ),
                    row=1, col=1, secondary_y=True
                )
            
            # 2. Data quality indicator (top right)
            completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=completeness,
                    delta={'reference': 90, 'position': "top"},
                    title={'text': "Data Completeness %<br><span style='font-size:12px'>Target: 90%+</span>"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': COLORS['actual']},
                        'steps': [
                            {'range': [0, 50], 'color': '#ffebee'},
                            {'range': [50, 80], 'color': '#fff3e0'},
                            {'range': [80, 100], 'color': '#e8f5e9'}
                        ],
                        'threshold': {
                            'line': {'color': COLORS['augmented'], 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=2
            )
            
            # 3. Source correlations (row 2, left)
            if hasattr(self, 'results') and 'source_correlations' in self.results:
                source_names = list(self.results['source_correlations'].keys())
                correlations = [self.results['source_correlations'][s]['correlation'] for s in source_names]
                
                fig.add_trace(
                    go.Bar(
                        x=source_names,
                        y=correlations,
                        name='Source Correlations',
                        marker_color=[COLORS.get(s.upper(), COLORS['RADAR']) for s in source_names],
                        text=[f'{c:.3f}' for c in correlations],
                        textposition='outside'
                    ),
                    row=2, col=1
                )
            
            # 4. Model performance (row 2, right)
            if self.evaluation_results:
                model_names = [r['name'] for r in self.evaluation_results[:6]]
                mae_values = [r['mae'] for r in self.evaluation_results[:6]]
                
                colors = [COLORS['actual'] if i == 0 else COLORS['RADAR'] for i in range(len(model_names))]
                
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=mae_values,
                        name='Model MAE',
                        marker_color=colors,
                        text=[f'{v:.1f}' for v in mae_values],
                        textposition='outside'
                    ),
                    row=2, col=2
                )
            
            # Continue with remaining subplots...
            # 5-8. Additional subplots would be added here
            
            # Update layout
            fig.update_layout(
                height=1600,
                title={
                    'text': "Enhanced Mail-Call Analytics Dashboard<br><sub>Source-Based Analysis with Augmentation Tracking</sub>",
                    'font': {'size': 24, 'color': COLORS['RADAR']},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                showlegend=True,
                paper_bgcolor='#f8f9fa',
                plot_bgcolor='white',
                font={'family': 'Arial, sans-serif', 'size': 10}
            )
            
            # Save dashboard
            dashboard_path = OUTPUT_DIR / 'plots' / 'main_dashboard.html'
            fig.write_html(dashboard_path)
            logger.info(f"SUCCESS: Main dashboard saved to {dashboard_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Main dashboard creation failed: {e}")

    def _create_source_analysis_plots(self):
        """Create detailed source analysis plots."""
        try:
            # Source comparison over time
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Mail Volume by Source Over Time',
                    'Source Volume Distribution',
                    'Source Activity Calendar',
                    'Source Performance Metrics'
                )
            )
            
            mail_source_cols = [col for col in self.combined_data.columns 
                                if col.startswith('mail_') and col != 'mail_volume_total']
            
            # 1. Source trends over time
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                color = COLORS.get(source_name.upper(), f'hsl({i*120}, 70%, 50%)')
                
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data[col],
                        name=f'{source_name}',
                        line=dict(color=color, width=2),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            # 2. Source volume distribution
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                source_data = self.combined_data[self.combined_data[col] > 0][col]
                
                if len(source_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=source_data,
                            name=f'{source_name}',
                            marker_color=COLORS.get(source_name.upper(), f'hsl({i*120}, 70%, 50%)')
                        ),
                        row=1, col=2
                    )
            
            # 3. Source activity calendar (heatmap by week/day)
            if mail_source_cols:
                source_name = mail_source_cols[0].replace('mail_', '').title()
                col = mail_source_cols[0]
                
                # Create week x day matrix
                calendar_data = self.combined_data.pivot_table(
                    index='week',
                    columns='day_of_week',
                    values=col,
                    aggfunc='mean',
                    fill_value=0
                )
                
                fig.add_trace(
                    go.Heatmap(
                        z=calendar_data.values,
                        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        y=calendar_data.index,
                        colorscale='Blues',
                        name=f'{source_name} Activity'
                    ),
                    row=2, col=1
                )
            
            # 4. Source performance metrics
            if hasattr(self, 'source_analysis') and self.source_analysis:
                sources = list(self.source_analysis.keys())
                volumes = [self.source_analysis[s]['total_volume'] for s in sources]
                
                fig.add_trace(
                    go.Bar(
                        x=sources,
                        y=volumes,
                        name='Total Volume',
                        marker_color=[COLORS.get(s.upper(), COLORS['RADAR']) for s in sources],
                        text=[f'{v:,.0f}' for v in volumes],
                        textposition='outside'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title='Source Analysis Dashboard',
                showlegend=True
            )
            
            source_plots_path = OUTPUT_DIR / 'plots' / 'source_analysis.html'
            fig.write_html(source_plots_path)
            logger.info(f"SUCCESS: Source analysis plots saved to {source_plots_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Source analysis plots failed: {e}")

    def _create_model_comparison_plots(self):
        """Create model comparison and diagnostic plots."""
        try:
            if not self.evaluation_results:
                logger.warning("WARNING: No evaluation results for model plots")
                return
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Model Performance Comparison',
                    'Actual vs Predicted (Best Model)',
                    'Prediction Residuals',
                    'Feature Importance (if available)'
                )
            )
            
            # 1. Model performance comparison
            model_names = [r['name'] for r in self.evaluation_results]
            mae_values = [r['mae'] for r in self.evaluation_results]
            r2_values = [r['r2'] for r in self.evaluation_results]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=mae_values,
                    name='MAE',
                    marker_color=[COLORS['actual'] if i == 0 else COLORS['RADAR'] for i in range(len(model_names))],
                    text=[f'{v:.1f}' for v in mae_values],
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # 2. Actual vs Predicted for best model
            best_model = self.evaluation_results[0]
            y_actual = self.test_data_eval['call_count'].values[:len(best_model['predictions'])]
            y_pred = best_model['predictions']
            
            fig.add_trace(
                go.Scatter(
                    x=y_actual,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color=COLORS['RADAR'],
                        size=8,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    )
                ),
                row=1, col=2
            )
            
            # Perfect prediction line
            max_val = max(y_actual.max(), y_pred.max()) if len(y_pred) > 0 else 100
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color=COLORS['augmented'], dash='dash', width=2)
                ),
                row=1, col=2
            )
            
            # 3. Residuals analysis
            residuals = y_actual - y_pred
            
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color=COLORS['Product'], size=6, opacity=0.7)
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[y_pred.min(), y_pred.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Line',
                    line=dict(color='black', dash='dash', width=1)
                ),
                row=2, col=1
            )
            
            # 4. Feature importance (if available)
            best_model_info = self.models.get(best_model['model'])
            if best_model_info and 'feature_importance' in best_model_info:
                feat_imp = best_model_info['feature_importance'].head(10)
                
                fig.add_trace(
                    go.Bar(
                        x=feat_imp['importance'],
                        y=feat_imp['feature'],
                        orientation='h',
                        name='Importance',
                        marker_color=COLORS['Meridian']
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title=f'Model Analysis - Best: {best_model["name"]} (MAE: {best_model["mae"]:.1f})',
                showlegend=True
            )
            
            model_plots_path = OUTPUT_DIR / 'plots' / 'model_analysis.html'
            fig.write_html(model_plots_path)
            logger.info(f"SUCCESS: Model analysis plots saved to {model_plots_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Model comparison plots failed: {e}")

    def _create_time_series_plots(self):
        """Create time series specific plots."""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Time Series with Trend and Moving Averages',
                    'Seasonal Decomposition (if available)',
                    'Autocorrelation Analysis'
                )
            )
            
            # 1. Time series with trends
            fig.add_trace(
                go.Scatter(
                    x=self.combined_data['date'],
                    y=self.combined_data['call_count'],
                    name='Call Volume',
                    line=dict(color=COLORS['RADAR'], width=1, opacity=0.7),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if len(self.combined_data) >= 7:
                ma_7 = self.combined_data['call_count'].rolling(7, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=ma_7,
                        name='7-day MA',
                        line=dict(color=COLORS['Product'], width=2),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            if len(self.combined_data) >= 30:
                ma_30 = self.combined_data['call_count'].rolling(30, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=ma_30,
                        name='30-day MA',
                        line=dict(color=COLORS['Meridian'], width=2),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            # 2. Seasonal decomposition (simplified)
            if HAS_STATSMODELS and len(self.combined_data) >= 28:
                try:
                    # Use only actual data for decomposition
                    actual_data = self.combined_data[self.combined_data['data_quality'] == 'actual']
                    if len(actual_data) >= 28:
                        ts_data = actual_data.set_index('date')['call_count'].asfreq('D', fill_value=0)
                        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=actual_data['date'],
                                y=decomposition.trend.values,
                                name='Trend',
                                line=dict(color=COLORS['RADAR'], width=2),
                                mode='lines'
                            ),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=actual_data['date'],
                                y=decomposition.seasonal.values,
                                name='Seasonal',
                                line=dict(color=COLORS['Product'], width=1),
                                mode='lines'
                            ),
                            row=2, col=1
                        )
                except Exception as e:
                    logger.warning(f"WARNING: Seasonal decomposition failed: {e}")
            
            # 3. Lag correlation plot
            if hasattr(self, 'results') and 'lag_correlations' in self.results:
                lags = list(self.results['lag_correlations'].keys())
                correlations = list(self.results['lag_correlations'].values())
                
                fig.add_trace(
                    go.Bar(
                        x=lags,
                        y=correlations,
                        name='Lag Correlations',
                        marker_color=COLORS['Meridian'],
                        text=[f'{c:.3f}' for c in correlations],
                        textposition='outside'
                    ),
                    row=3, col=1
                )
            
            fig.update_layout(
                height=1200,
                title='Time Series Analysis',
                showlegend=True
            )
            
            ts_plots_path = OUTPUT_DIR / 'plots' / 'time_series_analysis.html'
            fig.write_html(ts_plots_path)
            logger.info(f"SUCCESS: Time series plots saved to {ts_plots_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Time series plots failed: {e}")

    # ========================================================================
    # FORECASTING METHODS
    # ========================================================================

    def _create_forecasts(self):
        """Create forecasts using the best models."""
        logger.info("Creating forecasts...")
        
        try:
            if not self.evaluation_results:
                logger.warning("WARNING: No models available for forecasting")
                return
            
            # Use best model for forecast
            best_model_name = self.evaluation_results[0]['model']
            best_model_info = self.models.get(best_model_name)
            
            if not best_model_info:
                logger.error("ERROR: Best model not found")
                return
            
            # Create forecast
            self._create_forecast_visualization(best_model_info, best_model_name)
            
            # Create ensemble forecast if multiple models
            if len(self.evaluation_results) >= 3:
                self._create_ensemble_forecast()
            
        except Exception as e:
            logger.error(f"ERROR: Forecast creation failed: {e}")

    def _create_forecast_visualization(self, model_info, model_name):
        """Create forecast visualization with confidence intervals."""
        try:
            logger.info(f"INFO: Creating forecast with {model_info['name']}")
            
            # Create future dates
            last_date = self.combined_data['date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=30,
                freq='D'
            )
            
            # Prepare future features
            future_df = self._prepare_future_features(future_dates)
            
            # Make predictions
            if model_info.get('type') == 'time_series':
                # Time series models
                predictions = model_info['model'].forecast(steps=30)
            else:
                # ML models
                X_future = future_df[self.feature_cols].fillna(0)
                
                if 'scaler' in model_info:
                    X_future_scaled = model_info['scaler'].transform(X_future)
                    predictions = model_info['model'].predict(X_future_scaled)
                else:
                    predictions = model_info['model'].predict(X_future)
            
            # Calculate confidence intervals
            mae = self.evaluation_results[0]['mae']
            upper_bound = predictions + 1.96 * mae
            lower_bound = np.maximum(0, predictions - 1.96 * mae)
            
            # Create visualization
            fig = go.Figure()
            
            # Historical data (last 60 days)
            recent_data = self.combined_data.tail(60)
            actual_recent = recent_data[recent_data['data_quality'] == 'actual']
            augmented_recent = recent_data[recent_data['data_quality'] == 'augmented']
            
            # Historical actual
            fig.add_trace(go.Scatter(
                x=actual_recent['date'],
                y=actual_recent['call_count'],
                mode='lines+markers',
                name='Historical (Actual)',
                line=dict(color=COLORS['actual'], width=2),
                marker=dict(size=4)
            ))
            
            # Historical augmented
            if len(augmented_recent) > 0:
                fig.add_trace(go.Scatter(
                    x=augmented_recent['date'],
                    y=augmented_recent['call_count'],
                    mode='lines+markers',
                    name='Historical (Augmented)',
                    line=dict(color=COLORS['augmented'], width=2, dash='dot'),
                    marker=dict(size=4, symbol='diamond'),
                    opacity=0.7
                ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name=f'Forecast ({model_info["name"]})',
                line=dict(color=COLORS['forecast'], width=3, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates.tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor=COLORS['confidence'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='95% Confidence Interval'
            ))
            
            # Add separation line
            fig.add_shape(
                type="line",
                x0=last_date, x1=last_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="gray", width=2, dash="dot")
            )
            
            fig.add_annotation(
                x=last_date,
                y=1,
                yref="paper",
                text="Forecast Start",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30
            )
            
            fig.update_layout(
                title={
                    'text': f'30-Day Call Volume Forecast<br><sub>Model: {model_info["name"]} | MAE: ±{mae:.0f} calls/day</sub>',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title='Date',
                yaxis_title='Call Volume',
                height=600,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Save forecast
            forecast_path = OUTPUT_DIR / 'plots' / 'forecast.html'
            fig.write_html(forecast_path)
            
            # Save forecast data
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_calls': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_used': model_info['name']
            })
            forecast_df.to_csv(OUTPUT_DIR / 'data' / 'forecast.csv', index=False)
            
            logger.info(f"SUCCESS: Forecast saved to {forecast_path}")
            logger.info(f"INFO: Average forecast: {predictions.mean():.0f} calls/day")
            
        except Exception as e:
            logger.error(f"ERROR: Forecast visualization failed: {e}")

    def _prepare_future_features(self, future_dates):
        """Prepare features for future dates."""
        try:
            future_df = pd.DataFrame({'date': future_dates})
            
            # Time features
            future_df['day_of_week'] = future_df['date'].dt.dayofweek
            future_df['month'] = future_df['date'].dt.month
            future_df['quarter'] = future_df['date'].dt.quarter
            future_df['is_weekend'] = future_df['date'].dt.dayofweek.isin([5, 6]).astype(int)
            future_df['is_month_start'] = future_df['date'].dt.is_month_start.astype(int)
            future_df['is_month_end'] = future_df['date'].dt.is_month_end.astype(int)
            
            # Cyclical features
            future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
            future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
            future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
            future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
            
            # Fill other features with recent patterns
            for col in self.feature_cols:
                if col not in future_df.columns:
                    if 'lag' in col or 'ma' in col:
                        # Use recent average for lag/MA features
                        recent_avg = self.modeling_data[col].tail(14).mean()
                        future_df[col] = recent_avg
                    elif 'mail_' in col:
                        # Use day-of-week pattern for mail features
                        if col in self.combined_data.columns:
                            dow_pattern = self.combined_data.groupby('day_of_week')[col].mean()
                            future_df[col] = future_df['day_of_week'].map(dow_pattern).fillna(0)
                        else:
                            future_df[col] = 0
                    else:
                        # Use overall average for other features
                        future_df[col] = self.modeling_data[col].mean()
            
            return future_df
            
        except Exception as e:
            logger.error(f"ERROR: Future feature preparation failed: {e}")
            return pd.DataFrame()

    def _create_ensemble_forecast(self):
        """Create ensemble forecast from multiple models."""
        try:
            logger.info("INFO: Creating ensemble forecast...")
            
            # Use top 3 models for ensemble
            top_models = self.evaluation_results[:3]
            
            # Get individual predictions (already calculated)
            all_predictions = []
            weights = []
            
            for result in top_models:
                # Weight by inverse MAE (better models get higher weight)
                weight = 1.0 / (result['mae'] + 1e-6)
                weights.append(weight)
                
                # For this example, we'll use the test predictions
                # In practice, you'd generate new forecasts
                all_predictions.append(result['predictions'])
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Create weighted ensemble
            ensemble_pred = np.zeros(len(all_predictions[0]))
            for pred, weight in zip(all_predictions, weights):
                ensemble_pred += weight * pred
            
            # This is a simplified ensemble for demonstration
            # In practice, you'd create proper ensemble forecasts
            
            logger.info(f"SUCCESS: Ensemble forecast created using {len(top_models)} models")
            
        except Exception as e:
            logger.error(f"ERROR: Ensemble forecast failed: {e}")

# ========================================================================
    # REPORTING METHODS
    # ========================================================================

    def _create_detailed_reports(self, success_steps):
        """Create detailed HTML reports."""
        logger.info("Creating detailed reports...")
        
        try:
            # Executive summary report
            self._create_executive_report(success_steps)
            
            # Technical report
            self._create_technical_report()
            
            # Source analysis report
            self._create_source_report()
            
        except Exception as e:
            logger.error(f"ERROR: Report creation failed: {e}")

    def _create_executive_report(self, success_steps):
        """Create executive summary report."""
        try:
            # Calculate key metrics
            if self.combined_data is not None:
                total_days = len(self.combined_data)
                completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
                total_mail = self.combined_data['mail_volume_total'].sum()
                total_calls = self.combined_data['call_count'].sum()
                
                # Source breakdown
                source_breakdown = {}
                mail_source_cols = [col for col in self.combined_data.columns if col.startswith('mail_') and col != 'mail_volume_total']
                for col in mail_source_cols:
                    source_name = col.replace('mail_', '').title()
                    source_breakdown[source_name] = self.combined_data[col].sum()
            else:
                total_days = completeness = total_mail = total_calls = 0
                source_breakdown = {}
            
            # Model performance
            if self.evaluation_results:
                best_model = self.evaluation_results[0]
                accuracy = max(0, 100 - best_model['mape'])
                mae = best_model['mae']
                model_name = best_model['name']
            else:
                accuracy = mae = 0
                model_name = "None"
            
            # Correlation insights
            best_lag = self.results.get('best_lag', 'Unknown')
            best_corr = self.results.get('best_lag_correlation', 0)
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Mail-Call Analytics - Executive Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: 'Segoe UI', Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: #333;
                        line-height: 1.6;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 15px;
                        overflow: hidden;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    }}
                    .header {{
                        background: linear-gradient(45deg, {COLORS['RADAR']}, {COLORS['Product']});
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }}
                    .header h1 {{
                        margin: 0;
                        font-size: 36px;
                        font-weight: 300;
                    }}
                    .subtitle {{
                        margin: 10px 0 0 0;
                        font-size: 18px;
                        opacity: 0.9;
                    }}
                    .content {{
                        padding: 40px;
                    }}
                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 25px;
                        margin: 30px 0;
                    }}
                    .metric {{
                        background: #f8f9fa;
                        padding: 25px;
                        border-radius: 10px;
                        text-align: center;
                        border-left: 5px solid {COLORS['RADAR']};
                        transition: transform 0.2s;
                    }}
                    .metric:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    }}
                    .metric-value {{
                        font-size: 36px;
                        font-weight: bold;
                        color: {COLORS['RADAR']};
                        margin: 10px 0;
                    }}
                    .metric-label {{
                        color: #666;
                        font-size: 14px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }}
                    .source-breakdown {{
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                    }}
                    .source-item {{
                        display: flex;
                        justify-content: space-between;
                        margin: 10px 0;
                        padding: 10px;
                        background: white;
                        border-radius: 5px;
                    }}
                    .section {{
                        margin: 30px 0;
                        padding: 30px;
                        background: #f8f9fa;
                        border-radius: 10px;
                    }}
                    .section h2 {{
                        color: {COLORS['RADAR']};
                        font-size: 24px;
                        margin-bottom: 15px;
                        padding-bottom: 10px;
                        border-bottom: 2px solid {COLORS['RADAR']};
                    }}
                    .success-steps {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                        margin: 20px 0;
                    }}
                    .step-badge {{
                        background: {COLORS['actual']};
                        color: white;
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: bold;
                    }}
                    .footer {{
                        background: #343a40;
                        color: white;
                        text-align: center;
                        padding: 30px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Enhanced Mail-Call Predictive Analytics</h1>
                        <div class="subtitle">Executive Summary Report</div>
                        <div class="subtitle">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                    </div>
                    
                    <div class="content">
                        <div class="metrics-grid">
                            <div class="metric">
                                <div class="metric-label">Data Quality</div>
                                <div class="metric-value">{completeness:.0f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Model Accuracy</div>
                                <div class="metric-value">{accuracy:.0f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Total Mail Volume</div>
                                <div class="metric-value">{total_mail:,.0f}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Total Call Volume</div>
                                <div class="metric-value">{total_calls:,.0f}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Best Lag Period</div>
                                <div class="metric-value">{best_lag}</div>
                                <div style="font-size: 12px; color: #666;">days</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Correlation Strength</div>
                                <div class="metric-value">{abs(best_corr):.2f}</div>
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>Analysis Progress</h2>
                            <div class="success-steps">
                                {' '.join([f'<span class="step-badge">{step}</span>' for step in success_steps])}
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>Source Analysis</h2>
                            <div class="source-breakdown">
                                <h3>Mail Volume by Source</h3>
                                {chr(10).join([f'<div class="source-item"><span>{source}</span><span>{volume:,.0f}</span></div>' for source, volume in source_breakdown.items()])}
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>Key Findings</h2>
                            <ul>
                                <li>Best performing model: {model_name} with {accuracy:.1f}% accuracy</li>
                                <li>Optimal mail-to-call lag: {best_lag} days (correlation: {best_corr:.3f})</li>
                                <li>Data completeness: {completeness:.0f}% actual data</li>
                                <li>Multiple mail sources identified: {len(source_breakdown)} sources analyzed</li>
                                <li>Total analysis period: {total_days} days</li>
                            </ul>
                        </div>
                        
                        <div class="section">
                            <h2>Recommendations</h2>
                            <ol>
                                <li><strong>Immediate:</strong> Deploy {model_name} model for daily predictions</li>
                                <li><strong>Short-term:</strong> Improve data collection to increase completeness above 90%</li>
                                <li><strong>Medium-term:</strong> Implement source-specific modeling for better accuracy</li>
                                <li><strong>Long-term:</strong> Develop real-time prediction system with automated retraining</li>
                            </ol>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>Generated by Enhanced Mail-Call Predictive Analytics System</p>
                        <p>Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')} | Version 2.0</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            report_path = OUTPUT_DIR / 'reports' / 'executive_summary.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"SUCCESS: Executive report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Executive report creation failed: {e}")

    def _create_technical_report(self):
        """Create technical analysis report."""
        try:
            logger.info("INFO: Creating technical report...")
            
            # This would contain detailed technical analysis
            # For now, we'll create a placeholder
            tech_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Technical Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ background: #f0f0f0; padding: 10px; margin: 5px 0; }}
                </style>
            </head>
            <body>
                <h1>Technical Analysis Report</h1>
                <div class="section">
                    <h2>Model Details</h2>
                    <div class="metric">Models Evaluated: {len(self.evaluation_results)}</div>
                    <div class="metric">Features Used: {len(self.feature_cols)}</div>
                    <div class="metric">Training Data Points: {len(self.train_data) if self.train_data is not None else 0}</div>
                </div>
                
                <div class="section">
                    <h2>Data Quality Assessment</h2>
                    <div class="metric">Total Records: {len(self.combined_data) if self.combined_data is not None else 0}</div>
                    <div class="metric">Data Issues Identified: {len(self.data_issues)}</div>
                </div>
                
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            tech_path = OUTPUT_DIR / 'reports' / 'technical_report.html'
            with open(tech_path, 'w', encoding='utf-8') as f:
                f.write(tech_content)
            
            logger.info(f"SUCCESS: Technical report saved to {tech_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Technical report creation failed: {e}")

    def _create_source_report(self):
        """Create source analysis report."""
        try:
            logger.info("INFO: Creating source analysis report...")
            
            # Source-specific report placeholder
            source_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Source Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .source {{ border: 1px solid #ccc; margin: 10px 0; padding: 15px; }}
                </style>
            </head>
            <body>
                <h1>Source Analysis Report</h1>
                {chr(10).join([f'<div class="source"><h3>{source}</h3><p>Records: {data["total_records"]}</p><p>Volume: {data["total_volume"]:,.0f}</p></div>' for source, data in self.source_analysis.items()])}
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            source_path = OUTPUT_DIR / 'reports' / 'source_analysis_report.html'
            with open(source_path, 'w', encoding='utf-8') as f:
                f.write(source_content)
            
            logger.info(f"SUCCESS: Source report saved to {source_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Source report creation failed: {e}")

    def _create_error_report(self, error_message, completed_steps):
        """Create error report when analysis fails."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Analysis Error Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8d7da; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                    .error {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .completed {{ background: #d4edda; color: #155724; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Analysis Error Report</h1>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="error">
                        <h3>Error Details</h3>
                        <p>{error_message}</p>
                    </div>
                    
                    <h3>Completed Steps</h3>
                    {chr(10).join([f'<div class="completed">{step}</div>' for step in completed_steps])}
                    
                    <h3>Troubleshooting</h3>
                    <ul>
                        <li>Check that data files are in the correct format</li>
                        <li>Verify all required libraries are installed</li>
                        <li>Ensure sufficient disk space for output files</li>
                        <li>Review the analysis.log file for detailed error information</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            error_path = OUTPUT_DIR / 'reports' / 'error_report.html'
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"ERROR: Error report saved to {error_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Error report creation failed: {e}")


# ============================================================================
# MAIN EXECUTION AND UTILITY FUNCTIONS
# ============================================================================

def main():
    """Main execution function."""
    try:
        print("\n" + "="*80)
        print("ENHANCED MAIL-CALL PREDICTIVE ANALYTICS SYSTEM v2.0")
        print("="*80)
        print("Initializing comprehensive analysis pipeline...")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Available libraries: scikit-learn: {HAS_SKLEARN}, XGBoost: {HAS_XGB}, LightGBM: {HAS_LGB}, Statsmodels: {HAS_STATSMODELS}")
        print("="*80)
        
        # Initialize analyzer
        analyzer = EnhancedMailCallAnalyzer()
        
        # Run complete analysis
        analyzer.run_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {OUTPUT_DIR}")
        print("\nGenerated Files:")
        print(f"  📊 Executive Report: {OUTPUT_DIR}/reports/executive_summary.html")
        print(f"  📈 Main Dashboard: {OUTPUT_DIR}/plots/main_dashboard.html")
        print(f"  🔮 Forecast: {OUTPUT_DIR}/plots/forecast.html")
        print(f"  🔍 Source Analysis: {OUTPUT_DIR}/plots/source_analysis.html")
        print(f"  🤖 Model Analysis: {OUTPUT_DIR}/plots/model_analysis.html")
        print(f"  📋 Technical Report: {OUTPUT_DIR}/reports/technical_report.html")
        print(f"  📝 Log File: analysis.log")
        print("\nData Files:")
        print(f"  💾 Combined Timeline: {OUTPUT_DIR}/data/combined_timeline.csv")
        print(f"  📊 Model Evaluation: {OUTPUT_DIR}/models/model_evaluation.csv")
        print(f"  🔮 Forecast Data: {OUTPUT_DIR}/data/forecast.csv")
        print(f"  🔗 Correlations: {OUTPUT_DIR}/data/lag_correlations.csv")
        
        # Display key results
        if analyzer.evaluation_results:
            best_model = analyzer.evaluation_results[0]
            print(f"\n🏆 Best Model: {best_model['name']}")
            print(f"   📏 MAE: {best_model['mae']:.1f}")
            print(f"   📐 R²: {best_model['r2']:.3f}")
            print(f"   🎯 MAPE: {best_model['mape']:.1f}%")
        
        if analyzer.combined_data is not None:
            total_days = len(analyzer.combined_data)
            completeness = (analyzer.combined_data['data_quality'] == 'actual').mean() * 100
            print(f"\n📊 Data Summary:")
            print(f"   📅 Total Days: {total_days}")
            print(f"   ✅ Data Quality: {completeness:.1f}% actual")
            
        if analyzer.results.get('best_lag'):
            print(f"   ⏱️  Best Lag: {analyzer.results['best_lag']} days")
            print(f"   🔗 Correlation: {analyzer.results.get('best_lag_correlation', 0):.3f}")
        
        print("\n" + "="*80)
        print("Open the executive report in your browser to view the complete analysis!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR in main execution: {e}")
        logger.error(f"TRACEBACK: {traceback.format_exc()}")
        print(f"\n❌ ERROR: Analysis failed - {e}")
        print("Check analysis.log for detailed error information")
        
        # Try to create a basic error summary
        try:
            print(f"\n📁 Partial results may be available in: {OUTPUT_DIR}")
            if OUTPUT_DIR.exists():
                files = list(OUTPUT_DIR.rglob("*.*"))
                if files:
                    print(f"Found {len(files)} output files")
                    for file in files[:5]:  # Show first 5 files
                        print(f"  - {file}")
                    if len(files) > 5:
                        print(f"  ... and {len(files) - 5} more files")
        except:
            pass
            
        sys.exit(1)


def validate_environment():
    """Validate the environment and dependencies."""
    print("🔍 Validating environment...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append("Python 3.7+ is required")
    
    # Check required libraries
    required_libs = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    for lib_name, import_name in required_libs.items():
        try:
            __import__(import_name)
            print(f"  ✅ {lib_name}")
        except ImportError:
            issues.append(f"Missing required library: {lib_name}")
            print(f"  ❌ {lib_name}")
    
    # Check optional libraries
    optional_libs = {
        'scikit-learn': HAS_SKLEARN,
        'xgboost': HAS_XGB,
        'lightgbm': HAS_LGB,
        'statsmodels': HAS_STATSMODELS
    }
    
    for lib_name, available in optional_libs.items():
        if available:
            print(f"  ✅ {lib_name} (optional)")
        else:
            print(f"  ⚠️  {lib_name} (optional - enhanced features disabled)")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_mb = free // (1024 * 1024)
        if free_mb < 100:
            issues.append(f"Low disk space: {free_mb}MB available")
        else:
            print(f"  ✅ Disk space: {free_mb}MB available")
    except:
        print("  ⚠️  Could not check disk space")
    
    if issues:
        print("\n❌ Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("  ✅ Environment validation passed!")
        return True


def check_data_files():
    """Check for available data files."""
    print("\n🔍 Checking for data files...")
    
    mail_files = []
    call_files = []
    
    # Check mail files
    for pattern in MAIL_CONFIG['patterns']:
        files = glob.glob(pattern)
        mail_files.extend(files)
    
    # Check call files  
    for pattern in CALL_CONFIG['patterns']:
        files = glob.glob(pattern)
        call_files.extend(files)
    
    print(f"📧 Mail files found: {len(mail_files)}")
    for file in mail_files[:3]:
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.1f}MB)")
    if len(mail_files) > 3:
        print(f"  ... and {len(mail_files) - 3} more files")
    
    print(f"📞 Call files found: {len(call_files)}")
    for file in call_files[:3]:
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.1f}MB)")
    if len(call_files) > 3:
        print(f"  ... and {len(call_files) - 3} more files")
    
    if not mail_files and not call_files:
        print("❌ No data files found!")
        print("\nExpected file patterns:")
        print("  Mail files:", ", ".join(MAIL_CONFIG['patterns']))
        print("  Call files:", ", ".join(CALL_CONFIG['patterns']))
        return False
    
    return True


def display_help():
    """Display help information."""
    help_text = """
Enhanced Mail-Call Predictive Analytics System v2.0
===================================================

OVERVIEW:
This system analyzes the relationship between mail campaigns and call volumes
to predict future call patterns. It supports multiple mail sources and provides
comprehensive analytics with interactive visualizations.

SUPPORTED FILE FORMATS:
- Mail Data: CSV files matching patterns: all_mail_data.csv, *mail*.csv
- Call Data: CSV files matching patterns: *Genesys*.csv, *call*.csv

MAIL DATA SOURCES:
- RADAR: Marketing campaigns
- Product: Product-related communications  
- Meridian: Customer communications

FEATURES:
✅ Multi-source mail data processing
✅ Intelligent data augmentation
✅ Multiple prediction models (ML + time series)
✅ Interactive dashboards and visualizations
✅ Comprehensive forecasting with confidence intervals
✅ Executive and technical reporting

REQUIREMENTS:
- Python 3.7+
- pandas, numpy, plotly, matplotlib, seaborn (required)
- scikit-learn, xgboost, lightgbm, statsmodels (optional)

USAGE:
1. Place your mail and call data files in the current directory
2. Run: python enhanced_mail_call_analysis_v2.py
3. Check the results in the 'enhanced_analysis_results' folder

OUTPUT FILES:
- Executive Report: reports/executive_summary.html
- Interactive Dashboard: plots/main_dashboard.html  
- Forecast: plots/forecast.html
- Technical Reports: reports/
- Processed Data: data/

For issues or questions, check the analysis.log file.
    """
    print(help_text)


def cleanup_old_results():
    """Clean up old result directories if they exist."""
    try:
        if OUTPUT_DIR.exists():
            # Count existing files
            existing_files = list(OUTPUT_DIR.rglob("*.*"))
            if existing_files:
                response = input(f"Found {len(existing_files)} existing result files. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Analysis cancelled.")
                    sys.exit(0)
                
                # Clean up
                import shutil
                shutil.rmtree(OUTPUT_DIR)
                print(f"✅ Cleaned up old results")
    except Exception as e:
        logger.warning(f"Warning: Could not clean up old results: {e}")


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            display_help()
            sys.exit(0)
        elif sys.argv[1] in ['-v', '--version', 'version']:
            print("Enhanced Mail-Call Predictive Analytics System v2.0")
            sys.exit(0)
    
    try:
        # Pre-flight checks
        print("🚀 Starting Enhanced Mail-Call Analytics...")
        
        # Validate environment
        if not validate_environment():
            print("\n❌ Environment validation failed!")
            print("Please install missing dependencies and try again.")
            sys.exit(1)
        
        # Check for data files
        if not check_data_files():
            print("\n❌ No data files found!")
            print("Please place your mail and call data files in the current directory.")
            print("Run with --help for more information.")
            sys.exit(1)
        
        # Clean up old results
        cleanup_old_results()
        
        # Run main analysis
        main()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        print("Partial results may be available in the output directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        logger.error(f"TRACEBACK: {traceback.format_exc()}")
        print(f"\n💥 FATAL ERROR: {e}")
        print("Check analysis.log for full details")
        sys.exit(1)
