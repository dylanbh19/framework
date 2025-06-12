# #!/usr/bin/env python
“””
enhanced_mail_call_analysis.py

Production-ready analysis with comprehensive error handling and data validation.
Automatically detects and handles various data formats and structures.

Requirements:
pip install pandas numpy matplotlib seaborn plotly scikit-learn
“””

import warnings
warnings.filterwarnings(‘ignore’)

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

# Set up comprehensive logging with UTF-8 encoding fix

logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(levelname)s - %(message)s’,
handlers=[
logging.FileHandler(‘analysis.log’, encoding=‘utf-8’),
logging.StreamHandler(sys.stdout)
]
)

# Fix for Windows console encoding

import sys
if sys.platform == “win32”:
import codecs
sys.stdout = codecs.getwriter(‘utf-8’)(sys.stdout.buffer, ‘strict’)
sys.stderr = codecs.getwriter(‘utf-8’)(sys.stderr.buffer, ‘strict’)
logger = logging.getLogger(**name**)

# Import ML libraries with fallbacks

try:
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
HAS_SKLEARN = True
logger.info(“✓ scikit-learn available”)
except ImportError:
logger.warning(“⚠ scikit-learn not available - basic modeling only”)
HAS_SKLEARN = False

try:
import xgboost as xgb
HAS_XGB = True
logger.info(“✓ XGBoost available”)
except ImportError:
HAS_XGB = False

# Set visualization style

plt.style.use(‘default’)
sns.set_palette(“husl”)

# ============================================================================

# ENHANCED CONFIGURATION WITH AUTO-DETECTION

# ============================================================================

# Mail files configuration - will auto-detect if patterns don’t match

MAIL_FILES = {
‘patterns’: [’*mail*.csv’, ‘all_mail_data.csv’, ‘merged_mail.csv’],  # Multiple patterns to try
‘possible_date_columns’: [‘mail_date’, ‘date’, ‘send_date’, ‘campaign_date’],
‘possible_volume_columns’: [‘mail_volume’, ‘volume’, ‘count’, ‘quantity’],
‘possible_campaign_columns’: [‘campaign_type’, ‘mail_type’, ‘campaign’, ‘type’],
‘date_formats’: [’%Y-%m-%d’, ‘%d/%m/%Y’, ‘%m/%d/%Y’, ‘%Y-%m-%d %H:%M:%S’],
}

# Call data configuration - enhanced for your Genesys data

CALL_FILE = {
‘patterns’: [’*Genesys*.csv’, ‘*call*.csv’, ‘call_data.csv’],  # Multiple patterns
‘possible_date_columns’: [‘ConversationStart’, ‘call_date’, ‘date’, ‘timestamp’],
‘possible_datetime_columns’: [‘ConversationStart’, ‘datetime’, ‘timestamp’],
‘possible_id_columns’: [‘ConversationID’, ‘call_id’, ‘id’],
‘date_formats’: [’%Y-%m-%d %H:%M:%S.%f’, ‘%Y-%m-%d %H:%M:%S’, ‘%Y-%m-%d’, ‘%d/%m/%Y’],
}

# Analysis parameters

ANALYSIS_PARAMS = {
‘lag_days_to_test’: [0, 1, 2, 3, 4, 5, 6, 7, 10, 14],
‘min_data_points’: 20,  # Reduced threshold
‘test_split_ratio’: 0.2,
‘confidence_interval’: 0.95,
‘max_augmentation_days’: 90,  # Limit augmentation
}

# Output directory

OUTPUT_DIR = Path(“mail_call_analysis_results”)

# Professional color scheme

COLORS = {
‘primary’: ‘#1f77b4’,
‘secondary’: ‘#ff7f0e’,
‘success’: ‘#2ca02c’,
‘warning’: ‘#d62728’,
‘info’: ‘#9467bd’,
‘muted’: ‘#7f7f7f’
}

class EnhancedMailCallAnalyzer:
“”“Enhanced analyzer with comprehensive error handling and auto-detection.”””

```
def __init__(self):
    """Initialize the analyzer with enhanced error handling."""
    self.mail_data =     def _create_enhanced_forecast(self):
    """Create enhanced forecast with confidence intervals."""
    logger.info("Creating enhanced forecast...")
    
    try:
        if not self.evaluation_results:
            logger.warning("No models available for forecasting")
            return
        
        # Use best model
        best_model_name = self.evaluation_results[0]['model']
        model_info = self.models.get(best_model_name)
        
        if not model_info or 'model' not in model_info:
            logger.warning("Best model not suitable for forecasting")
            return
        
        # Create future dates
        last_date = self.combined_data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=30,
            freq='D'
        )
        
        # Prepare features for future dates
        future_df = pd.DataFrame({'date': future_dates})
        
        # Add time features
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        
        # Fill other features with recent averages or patterns
        for col in self.feature_cols:
            if col not in future_df.columns:
                if 'lag' in col or 'ma' in col:
                    # Use recent average for lag/moving average features
                    recent_avg = self.modeling_data[col].tail(14).mean()
                    future_df[col] = recent_avg
                else:
                    # Use overall average for other features
                    overall_avg = self.modeling_data[col].mean()
                    future_df[col] = overall_avg
        
        # Make predictions
        X_future = future_df[self.feature_cols].fillna(0)
        
        if 'scaler' in model_info:
            X_future_scaled = model_info['scaler'].transform(X_future)
            predictions = model_info['model'].predict(X_future_scaled)
        else:
            predictions = model_info['model'].predict(X_future)
        
        # Calculate confidence intervals (simplified approach)
        if self.evaluation_results:
            mae = self.evaluation_results[0]['mae']
            upper_bound = predictions + 1.96 * mae  # Approximate 95% CI
            lower_bound = np.maximum(0, predictions - 1.96 * mae)  # Ensure non-negative
        else:
            upper_bound = predictions * 1.2
            lower_bound = predictions * 0.8
        
        # Create forecast visualization
        fig = go.Figure()
        
        # Historical data (last 60 days) - separate actual vs augmented
        recent_data = self.combined_data.tail(60)
        actual_recent = recent_data[recent_data['data_quality'] == 'actual']
        augmented_recent = recent_data[recent_data['data_quality'] == 'augmented']
        
        # Historical actual data
        fig.add_trace(go.Scatter(
            x=actual_recent['date'],
            y=actual_recent['call_count'],
            mode='lines+markers',
            name='Historical Calls (Actual)',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=4)
        ))
        
        # Historical augmented data
        if len(augmented_recent) > 0:
            fig.add_trace(go.Scatter(
                x=augmented_recent['date'],
                y=augmented_recent['call_count'],
                mode='lines+markers',
                name='Historical Calls (Augmented)',
                line=dict(color=COLORS['warning'], width=2, dash='dot'),
                marker=dict(size=4, symbol='diamond'),
                opacity=0.7
            ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name=f'Forecast ({model_info["name"]})',
            line=dict(color=COLORS['success'], width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(46, 160, 46, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
        # Add vertical line to separate historical from forecast using shapes
        fig.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dot")
        )
        
        # Add annotation for the separation line
        fig.add_annotation(
            x=last_date,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="gray",
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
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save forecast plot
        forecast_plot_path = OUTPUT_DIR / 'plots' / 'forecast.html'
        fig.write_html(forecast_plot_path)
        
        # Create model visualization
        self._create_model_visualization()
        
        # Save forecast data
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_calls': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_used': model_info['name']
        })
        forecast_data_path = OUTPUT_DIR / 'data' / 'forecast.csv'
        forecast_df.to_csv(forecast_data_path, index=False)
        
        # Summary statistics
        avg_forecast = predictions.mean()
        total_forecast = predictions.sum()
        
        logger.info(f"30-day forecast created:")
        logger.info(f"  Average daily calls: {avg_forecast:.0f}")
        logger.info(f"  Total forecasted calls: {total_forecast:.0f}")
        logger.info(f"  Model used: {model_info['name']}")
        logger.info(f"  Forecast saved to: {forecast_plot_path}")
        
        # Store forecast results
        self
    self.call_data = None
    self.combined_data = None
    self.models = {}
    self.results = {}
    self.evaluation_results = []
    self.data_issues = []  # Track data quality issues
    
    # Create output directories
    self._setup_directories()
    
def _setup_directories(self):
    """Create output directory structure with error handling."""
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        for subdir in ['data', 'plots', 'models', 'reports']:
            (OUTPUT_DIR / subdir).mkdir(exist_ok=True)
        logger.info(f"✓ Output directories created at: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"✗ Failed to create directories: {e}")
        sys.exit(1)

def run_analysis(self):
    """Run the complete analysis pipeline with comprehensive error handling."""
    logger.info("="*80)
    logger.info("STARTING ENHANCED MAIL-CALL PREDICTIVE ANALYSIS")
    logger.info("="*80)
    
    success_steps = []
    
    try:
        # Step 1: Load data with auto-detection
        logger.info("\n1. DATA LOADING WITH AUTO-DETECTION")
        logger.info("-"*50)
        if self._load_all_data_enhanced():
            success_steps.append("Data Loading")
            logger.info("✓ Data loading completed successfully")
        else:
            logger.error("✗ Data loading failed - cannot proceed")
            return
        
        # Step 2: Data validation and cleaning
        logger.info("\n2. DATA VALIDATION AND CLEANING")
        logger.info("-"*50)
        if self._validate_and_clean_data():
            success_steps.append("Data Validation")
            logger.info("✓ Data validation completed")
        
        # Step 3: Data augmentation (if needed)
        if self.combined_data is not None:
            logger.info("\n3. DATA AUGMENTATION")
            logger.info("-"*50)
            self._smart_data_augmentation()
            success_steps.append("Data Augmentation")
        
        # Step 4: Exploratory analysis
        logger.info("\n4. EXPLORATORY ANALYSIS")
        logger.info("-"*50)
        if self._perform_enhanced_eda():
            success_steps.append("Exploratory Analysis")
            logger.info("✓ EDA completed")
        
        # Step 5: Model building (if possible)
        if HAS_SKLEARN and self.combined_data is not None and len(self.combined_data) >= ANALYSIS_PARAMS['min_data_points']:
            logger.info("\n5. MODEL BUILDING")
            logger.info("-"*50)
            if self._build_robust_models():
                success_steps.append("Model Building")
                logger.info("✓ Model building completed")
        else:
            logger.warning("⚠ Skipping model building - insufficient data or missing dependencies")
        
        # Step 6: Visualization and reporting
        logger.info("\n6. VISUALIZATION AND REPORTING")
        logger.info("-"*50)
        self._create_comprehensive_dashboard()
        self._create_detailed_report(success_steps)
        success_steps.append("Reporting")
        
        # Step 7: Forecast (if models available)
        if self.models:
            logger.info("\n7. FORECASTING")
            logger.info("-"*50)
            self._create_enhanced_forecast()
            success_steps.append("Forecasting")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETION SUMMARY")
        logger.info("="*80)
        logger.info(f"✓ Successful steps: {', '.join(success_steps)}")
        logger.info(f"✓ Results saved to: {OUTPUT_DIR}")
        
        if self.data_issues:
            logger.info(f"⚠ Data issues identified: {len(self.data_issues)}")
            for issue in self.data_issues[:5]:  # Show first 5 issues
                logger.info(f"  - {issue}")
        
        logger.info("\nKey outputs:")
        logger.info(f"  - Executive Report: {OUTPUT_DIR}/reports/executive_summary.html")
        logger.info(f"  - Dashboard: {OUTPUT_DIR}/plots/comprehensive_dashboard.html")
        logger.info(f"  - Data Quality Report: {OUTPUT_DIR}/reports/data_quality.html")
        
    except Exception as e:
        logger.error(f"✗ Analysis failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to create a failure report
        try:
            self._create_failure_report(str(e), success_steps)
        except:
            pass
        
        raise

def _load_all_data_enhanced(self):
    """Enhanced data loading with auto-detection and error handling."""
    mail_loaded = self._load_mail_files_enhanced()
    call_loaded = self._load_call_data_enhanced() 
    
    if not mail_loaded and not call_loaded:
        logger.error("✗ No data files could be loaded")
        return False
    
    if mail_loaded:
        logger.info(f"✓ Mail data loaded: {len(self.mail_data)} records")
    else:
        logger.warning("⚠ No mail data loaded")
        
    if call_loaded:
        logger.info(f"✓ Call data loaded: {len(self.call_data)} records")
    else:
        logger.warning("⚠ No call data loaded")
    
    # Combine datasets
    return self._prepare_combined_dataset_enhanced()

def _load_mail_files_enhanced(self):
    """Load mail files with auto-detection."""
    logger.info("Loading mail files with auto-detection...")
    
    try:
        # Try multiple patterns
        all_files = []
        for pattern in MAIL_FILES['patterns']:
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Remove duplicates
        all_files = list(set(all_files))
        
        if not all_files:
            logger.warning("⚠ No mail files found with standard patterns")
            # Try to find any CSV files
            csv_files = glob.glob("*.csv")
            logger.info(f"Found {len(csv_files)} CSV files in directory")
            for f in csv_files[:5]:  # Show first 5
                logger.info(f"  - {f}")
            return False
        
        logger.info(f"Found {len(all_files)} potential mail files")
        
        all_mail_data = []
        
        for file_path in sorted(all_files):
            try:
                logger.info(f"Attempting to load: {file_path}")
                
                # Read with multiple encodings
                df = None
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"  ✓ Loaded with {encoding} encoding ({len(df)} rows, {len(df.columns)} columns)")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    logger.warning(f"  ✗ Could not read {file_path} with any encoding")
                    continue
                
                # Auto-detect columns
                mail_df = self._detect_mail_columns(df, file_path)
                
                if mail_df is not None and len(mail_df) > 0:
                    all_mail_data.append(mail_df)
                    logger.info(f"  ✓ Processed {len(mail_df)} valid mail records")
                else:
                    logger.warning(f"  ⚠ No valid mail data found in {file_path}")
                    
            except Exception as e:
                logger.error(f"  ✗ Error processing {file_path}: {e}")
                self.data_issues.append(f"Mail file error ({file_path}): {e}")
        
        # Combine all mail data
        if all_mail_data:
            self.mail_data = pd.concat(all_mail_data, ignore_index=True)
            
            # Remove duplicates
            initial_count = len(self.mail_data)
            self.mail_data = self.mail_data.drop_duplicates()
            if len(self.mail_data) < initial_count:
                logger.info(f"  Removed {initial_count - len(self.mail_data)} duplicate records")
            
            # Save combined data
            self.mail_data.to_csv(OUTPUT_DIR / 'data' / 'combined_mail_data.csv', index=False)
            logger.info(f"✓ Total mail records: {len(self.mail_data):,}")
            
            if 'mail_date' in self.mail_data.columns:
                date_range = f"{self.mail_data['mail_date'].min()} to {self.mail_data['mail_date'].max()}"
                logger.info(f"  Date range: {date_range}")
            
            return True
        else:
            logger.error("✗ No mail data loaded successfully")
            return False
            
    except Exception as e:
        logger.error(f"✗ Mail file loading failed: {e}")
        self.data_issues.append(f"Mail loading error: {e}")
        return False

def _detect_mail_columns(self, df, file_path):
    """Auto-detect mail data columns."""
    try:
        mail_df = pd.DataFrame()
        
        # Detect date column
        date_col = None
        for col_name in MAIL_FILES['possible_date_columns']:
            if col_name in df.columns:
                date_col = col_name
                break
        
        if not date_col:
            # Look for any column with "date" in name
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                logger.info(f"  Auto-detected date column: {date_col}")
        
        if not date_col:
            logger.warning(f"  No date column found in {file_path}")
            return None
        
        # Parse dates with multiple formats
        mail_df['mail_date'] = None
        for date_format in MAIL_FILES['date_formats']:
            try:
                mail_df['mail_date'] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                valid_dates = mail_df['mail_date'].notna().sum()
                if valid_dates > len(df) * 0.8:  # If 80%+ dates are valid
                    logger.info(f"  ✓ Date format detected: {date_format} ({valid_dates} valid dates)")
                    break
            except:
                continue
        
        # If no format worked, try pandas auto-detection
        if mail_df['mail_date'].isna().all():
            try:
                mail_df['mail_date'] = pd.to_datetime(df[date_col], errors='coerce')
                logger.info(f"  ✓ Used pandas auto-detection for dates")
            except:
                logger.warning(f"  ✗ Could not parse dates in {file_path}")
                return None
        
        # Detect volume column
        volume_col = None
        for col_name in MAIL_FILES['possible_volume_columns']:
            if col_name in df.columns:
                volume_col = col_name
                break
        
        if not volume_col:
            # Look for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                volume_col = numeric_cols[0]
                logger.info(f"  Auto-detected volume column: {volume_col}")
            else:
                # Default to count of 1 per row
                mail_df['mail_volume'] = 1
                logger.info(f"  No volume column found, using count of 1 per record")
        
        if volume_col:
            mail_df['mail_volume'] = pd.to_numeric(df[volume_col], errors='coerce').fillna(1)
        
        # Optional campaign column
        campaign_col = None
        for col_name in MAIL_FILES['possible_campaign_columns']:
            if col_name in df.columns:
                campaign_col = col_name
                mail_df['campaign_type'] = df[col_name]
                break
        
        # Add metadata
        mail_df['source_file'] = Path(file_path).name
        
        # Remove invalid dates
        valid_mask = mail_df['mail_date'].notna()
        mail_df = mail_df[valid_mask]
        
        if len(mail_df) == 0:
            logger.warning(f"  No valid records after cleaning in {file_path}")
            return None
        
        return mail_df
        
    except Exception as e:
        logger.error(f"  Error detecting columns in {file_path}: {e}")
        return None

def _load_call_data_enhanced(self):
    """Load call data with enhanced detection for Genesys format."""
    logger.info("Loading call data with enhanced detection...")
    
    try:
        # Try multiple patterns to find call files
        all_files = []
        for pattern in CALL_FILE['patterns']:
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Remove duplicates and sort by size (larger files first)
        all_files = list(set(all_files))
        if all_files:
            all_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        
        if not all_files:
            logger.warning("⚠ No call files found with standard patterns")
            # List available CSV files
            csv_files = glob.glob("*.csv")
            logger.info(f"Available CSV files: {[f for f in csv_files[:10]]}")
            return False
        
        logger.info(f"Found {len(all_files)} potential call files")
        
        # Try to load the largest file first (likely the main data file)
        for file_path in all_files:
            try:
                logger.info(f"Attempting to load call file: {file_path}")
                
                # Check file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logger.info(f"  File size: {file_size:.1f} MB")
                
                # Read with multiple encodings and handle large files
                df = None
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        # For large files, read in chunks to test
                        if file_size > 100:  # If larger than 100MB
                            logger.info(f"  Large file detected, reading sample first...")
                            df_sample = pd.read_csv(file_path, encoding=encoding, nrows=1000)
                            logger.info(f"  Sample read successfully with {encoding}")
                            # If sample works, read full file
                            df = pd.read_csv(file_path, encoding=encoding)
                        else:
                            df = pd.read_csv(file_path, encoding=encoding)
                        
                        logger.info(f"  ✓ Loaded with {encoding} encoding ({len(df):,} rows, {len(df.columns)} columns)")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"  Error with {encoding}: {e}")
                        continue
                
                if df is None:
                    logger.warning(f"  ✗ Could not read {file_path}")
                    continue
                
                # Auto-detect call data structure
                call_df = self._detect_call_columns(df, file_path)
                
                if call_df is not None and len(call_df) > 0:
                    self.call_data = call_df
                    
                    # Save processed data
                    self.call_data.to_csv(OUTPUT_DIR / 'data' / 'processed_call_data.csv', index=False)
                    
                    logger.info(f"✓ Call data loaded: {len(self.call_data):,} records")
                    if 'call_date' in self.call_data.columns:
                        date_range = f"{self.call_data['call_date'].min()} to {self.call_data['call_date'].max()}"
                        logger.info(f"  Date range: {date_range}")
                    
                    return True
                else:
                    logger.warning(f"  ⚠ No valid call data found in {file_path}")
                    
            except Exception as e:
                logger.error(f"  ✗ Error processing {file_path}: {e}")
                self.data_issues.append(f"Call file error ({file_path}): {e}")
                continue
        
        logger.error("✗ No call files could be processed")
        return False
        
    except Exception as e:
        logger.error(f"✗ Call data loading failed: {e}")
        self.data_issues.append(f"Call loading error: {e}")
        return False

def _detect_call_columns(self, df, file_path):
    """Auto-detect call data columns, especially for Genesys format."""
    try:
        logger.info(f"  Detecting call data structure in {file_path}")
        logger.info(f"  Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        call_df = pd.DataFrame()
        
        # Detect datetime/date column - prioritize ConversationStart for Genesys
        date_col = None
        datetime_col = None
        
        # Check for Genesys-specific columns first
        if 'ConversationStart' in df.columns:
            datetime_col = 'ConversationStart'
            logger.info(f"  ✓ Found Genesys ConversationStart column")
        else:
            # Try other datetime columns
            for col_name in CALL_FILE['possible_datetime_columns']:
                if col_name in df.columns:
                    datetime_col = col_name
                    break
            
            # Try date columns
            if not datetime_col:
                for col_name in CALL_FILE['possible_date_columns']:
                    if col_name in df.columns:
                        date_col = col_name
                        break
        
        # Use datetime column if available, otherwise date column
        source_col = datetime_col or date_col
        
        if not source_col:
            # Look for any column with "date" or "time" in name
            time_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'start', 'end'])]
            if time_cols:
                source_col = time_cols[0]
                logger.info(f"  Auto-detected time column: {source_col}")
        
        if not source_col:
            logger.warning(f"  No date/time column found in {file_path}")
            return None
        
        # Parse the datetime/date column
        logger.info(f"  Parsing time data from column: {source_col}")
        
        # For Genesys ConversationStart format: "2025-03-27 04:05:30.397"
        parsed_dates = None
        for date_format in CALL_FILE['date_formats']:
            try:
                parsed_dates = pd.to_datetime(df[source_col], format=date_format, errors='coerce')
                valid_count = parsed_dates.notna().sum()
                if valid_count > len(df) * 0.8:  # If 80%+ are valid
                    logger.info(f"  ✓ Date format detected: {date_format} ({valid_count} valid)")
                    break
            except:
                continue
        
        # If no format worked, try pandas auto-detection
        if parsed_dates is None or parsed_dates.notna().sum() < len(df) * 0.5:
            try:
                parsed_dates = pd.to_datetime(df[source_col], errors='coerce')
                logger.info(f"  ✓ Used pandas auto-detection for dates")
            except:
                logger.warning(f"  ✗ Could not parse dates in {file_path}")
                return None
        
        # Extract date component
        call_df['call_date'] = parsed_dates.dt.date
        call_df['call_date'] = pd.to_datetime(call_df['call_date'])
        
        # For individual call records (not pre-aggregated), each row = 1 call
        call_df['call_count'] = 1
        
        # Add optional fields if available
        if 'ConversationID' in df.columns:
            call_df['conversation_id'] = df['ConversationID']
        
        # Add call direction if available
        if 'OriginatingDirection' in df.columns:
            call_df['direction'] = df['OriginatingDirection']
        
        # Add media type if available
        if 'MediaType' in df.columns:
            call_df['media_type'] = df['MediaType']
        
        # Remove invalid dates
        valid_mask = call_df['call_date'].notna()
        call_df = call_df[valid_mask]
        
        if len(call_df) == 0:
            logger.warning(f"  No valid call records after date parsing")
            return None
        
        logger.info(f"  ✓ Processed {len(call_df)} call records")
        logger.info(f"  Date range: {call_df['call_date'].min()} to {call_df['call_date'].max()}")
        
        return call_df
        
    except Exception as e:
        logger.error(f"  Error detecting call columns: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None

def _prepare_combined_dataset_enhanced(self):
    """Prepare combined dataset with enhanced error handling."""
    logger.info("Preparing combined dataset...")
    
    try:
        if self.mail_data is None and self.call_data is None:
            logger.error("✗ No data available for combination")
            return False
        
        # Determine date range
        dates = []
        if self.mail_data is not None and 'mail_date' in self.mail_data.columns:
            mail_dates = self.mail_data['mail_date'].dropna()
            if len(mail_dates) > 0:
                dates.extend([mail_dates.min(), mail_dates.max()])
                logger.info(f"  Mail date range: {mail_dates.min()} to {mail_dates.max()}")
        
        if self.call_data is not None and 'call_date' in self.call_data.columns:
            call_dates = self.call_data['call_date'].dropna()
            if len(call_dates) > 0:
                dates.extend([call_dates.min(), call_dates.max()])
                logger.info(f"  Call date range: {call_dates.min()} to {call_dates.max()}")
        
        if not dates:
            logger.error("✗ No valid dates found in any dataset")
            return False
        
        # Create timeline
        start_date = min(dates)
        end_date = max(dates)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        logger.info(f"  Creating timeline: {start_date} to {end_date} ({len(date_range)} days)")
        
        timeline = pd.DataFrame({'date': date_range})
        
        # Aggregate mail data by date
        if self.mail_data is not None:
            try:
                daily_mail = self.mail_data.groupby('mail_date').agg({
                    'mail_volume': 'sum'
                }).reset_index()
                daily_mail.rename(columns={'mail_date': 'date'}, inplace=True)
                timeline = timeline.merge(daily_mail, on='date', how='left')
                logger.info(f"  ✓ Mail data aggregated: {len(daily_mail)} unique dates")
            except Exception as e:
                logger.error(f"  ✗ Error aggregating mail data: {e}")
                timeline['mail_volume'] = 0
        else:
            timeline['mail_volume'] = 0
        
        # Aggregate call data by date
        if self.call_data is not None:
            try:
                daily_calls = self.call_data.groupby('call_date').agg({
                    'call_count': 'sum'
                }).reset_index()
                daily_calls.rename(columns={'call_date': 'date'}, inplace=True)
                timeline = timeline.merge(daily_calls, on='date', how='left')
                logger.info(f"  ✓ Call data aggregated: {len(daily_calls)} unique dates")
            except Exception as e:
                logger.error(f"  ✗ Error aggregating call data: {e}")
                timeline['call_count'] = 0
        else:
            timeline['call_count'] = 0
        
        # Fill missing values
        timeline['mail_volume'] = timeline['mail_volume'].fillna(0)
        timeline['call_count'] = timeline['call_count'].fillna(0)
        
        # Add time features
        timeline['day_of_week'] = timeline['date'].dt.dayofweek
        timeline['day_name'] = timeline['date'].dt.day_name()
        timeline['week'] = timeline['date'].dt.isocalendar().week
        timeline['month'] = timeline['date'].dt.month
        timeline['is_weekend'] = timeline['day_of_week'].isin([5, 6]).astype(int)
        
        # Add data quality tracking
        timeline['data_quality'] = 'actual'
        timeline['augmentation_method'] = 'none'
        
        self.combined_data = timeline
        
        # Data quality summary
        total_days = len(timeline)
        mail_days = (timeline['mail_volume'] > 0).sum()
        call_days = (timeline['call_count'] > 0).sum()
        
        logger.info(f"✓ Combined dataset created:")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Days with mail: {mail_days} ({mail_days/total_days*100:.1f}%)")
        logger.info(f"  Days with calls: {call_days} ({call_days/total_days*100:.1f}%)")
        logger.info(f"  Total mail volume: {timeline['mail_volume'].sum():,.0f}")
        logger.info(f"  Total call count: {timeline['call_count'].sum():,.0f}")
        
        # Save combined data
        self.combined_data.to_csv(OUTPUT_DIR / 'data' / 'combined_timeline.csv', index=False)
        
        # Check if we have enough data for analysis
        if mail_days < 5 and call_days < 5:
            logger.warning("⚠ Very limited data available - results may not be reliable")
            self.data_issues.append("Limited data: fewer than 5 days of mail and call data")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to prepare combined dataset: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def _validate_and_clean_data(self):
    """Validate and clean the combined dataset."""
    if self.combined_data is None:
        return False
    
    try:
        logger.info("Validating and cleaning data...")
        
        initial_rows = len(self.combined_data)
        
        # Remove extreme outliers (more than 5 standard deviations)
        for col in ['mail_volume', 'call_count']:
            if col in self.combined_data.columns:
                mean_val = self.combined_data[col].mean()
                std_val = self.combined_data[col].std()
                
                if std_val > 0:
                    outlier_threshold = mean_val + 5 * std_val
                    outliers = self.combined_data[col] > outlier_threshold
                    
                    if outliers.sum() > 0:
                        logger.info(f"  Found {outliers.sum()} extreme outliers in {col}")
                        self.combined_data.loc[outliers, col] = outlier_threshold
                        self.data_issues.append(f"Extreme outliers capped in {col}: {outliers.sum()} values")
        
        # Check for data consistency
        mail_negative = (self.combined_data['mail_volume'] < 0).sum()
        call_negative = (self.combined_data['call_count'] < 0).sum()
        
        if mail_negative > 0:
            logger.warning(f"  Found {mail_negative} negative mail volumes - setting to 0")
            self.combined_data.loc[self.combined_data['mail_volume'] < 0, 'mail_volume'] = 0
            self.data_issues.append(f"Negative mail volumes corrected: {mail_negative}")
        
        if call_negative > 0:
            logger.warning(f"  Found {call_negative} negative call counts - setting to 0")
            self.combined_data.loc[self.combined_data['call_count'] < 0, 'call_count'] = 0
            self.data_issues.append(f"Negative call counts corrected: {call_negative}")
        
        logger.info(f"✓ Data validation completed - {len(self.combined_data)} rows retained")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data validation failed: {e}")
        return False

def _smart_data_augmentation(self):
    """Smart data augmentation with conservative approach."""
    if self.combined_data is None:
        return
    
    try:
        logger.info("Performing smart data augmentation...")
        
        # Calculate missing data percentages
        mail_missing = (self.combined_data['mail_volume'] == 0)
        call_missing = (self.combined_data['call_count'] == 0)
        
        mail_missing_pct = mail_missing.mean() * 100
        call_missing_pct = call_missing.mean() * 100
        
        logger.info(f"  Mail data gaps: {mail_missing.sum()} days ({mail_missing_pct:.1f}%)")
        logger.info(f"  Call data gaps: {call_missing.sum()} days ({call_missing_pct:.1f}%)")
        
        # Only augment if gaps are significant but not overwhelming
        if 20 <= mail_missing_pct <= 80:
            logger.info("  Augmenting mail data...")
            self._augment_data_conservative('mail_volume', mail_missing)
        elif mail_missing_pct > 80:
            logger.warning("  Too much mail data missing for reliable augmentation")
            self.data_issues.append(f"Mail data mostly missing: {mail_missing_pct:.1f}%")
        
        if 20 <= call_missing_pct <= 80:
            logger.info("  Augmenting call data...")
            self._augment_data_conservative('call_count', call_missing)
        elif call_missing_pct > 80:
            logger.warning("  Too much call data missing for reliable augmentation")
            self.data_issues.append(f"Call data mostly missing: {call_missing_pct:.1f}%")
        
        # Summary
        augmented_count = (self.combined_data['data_quality'] == 'augmented').sum()
        if augmented_count > 0:
            logger.info(f"✓ Augmented {augmented_count} data points")
            self.combined_data.to_csv(OUTPUT_DIR / 'data' / 'augmented_data.csv', index=False)
        
    except Exception as e:
        logger.error(f"✗ Data augmentation failed: {e}")

def _augment_data_conservative(self, column, missing_mask):
    """Conservative data augmentation using patterns."""
    try:
        available_data = self.combined_data[~missing_mask]
        
        if len(available_data) < 14:  # Need at least 2 weeks of data
            logger.warning(f"    Insufficient data for {column} augmentation")
            return
        
        # Calculate day-of-week patterns
        dow_patterns = available_data.groupby('day_of_week')[column].agg(['mean', 'std'])
        
        # Calculate monthly patterns if we have enough data
        monthly_patterns = None
        if len(available_data) > 60:  # More than 2 months
            monthly_patterns = available_data.groupby('month')[column].agg(['mean', 'std'])
        
        augmented_count = 0
        max_augment = min(len(self.combined_data[missing_mask]), ANALYSIS_PARAMS['max_augmentation_days'])
        
        for idx in self.combined_data[missing_mask].head(max_augment).index:
            try:
                dow = self.combined_data.loc[idx, 'day_of_week']
                month = self.combined_data.loc[idx, 'month']
                
                # Start with day-of-week pattern
                if dow in dow_patterns.index and dow_patterns.loc[dow, 'mean'] > 0:
                    base_value = dow_patterns.loc[dow, 'mean']
                    base_std = dow_patterns.loc[dow, 'std']
                    
                    # Adjust with monthly pattern if available
                    if monthly_patterns is not None and month in monthly_patterns.index:
                        monthly_factor = monthly_patterns.loc[month, 'mean'] / dow_patterns['mean'].mean()
                        base_value *= monthly_factor
                    
                    # Add controlled randomness
                    if base_std > 0 and not pd.isna(base_std):
                        noise = np.random.normal(0, base_std * 0.3)  # Conservative noise
                    else:
                        noise = np.random.normal(0, base_value * 0.1)
                    
                    augmented_value = max(0, base_value + noise)
                    
                    self.combined_data.loc[idx, column] = augmented_value
                    self.combined_data.loc[idx, 'data_quality'] = 'augmented'
                    self.combined_data.loc[idx, 'augmentation_method'] = 'pattern_based'
                    augmented_count += 1
                    
            except Exception as e:
                logger.warning(f"    Error augmenting index {idx}: {e}")
                continue
        
        logger.info(f"    Augmented {augmented_count} {column} values")
        
    except Exception as e:
        logger.error(f"    Conservative augmentation failed for {column}: {e}")

def _perform_enhanced_eda(self):
    """Enhanced exploratory data analysis."""
    if self.combined_data is None:
        return False
    
    try:
        logger.info("Performing enhanced EDA...")
        
        # Basic statistics
        total_mail = self.combined_data['mail_volume'].sum()
        total_calls = self.combined_data['call_count'].sum()
        avg_daily_mail = self.combined_data['mail_volume'].mean()
        avg_daily_calls = self.combined_data['call_count'].mean()
        
        logger.info(f"  Total mail volume: {total_mail:,.0f}")
        logger.info(f"  Total call count: {total_calls:,.0f}")
        logger.info(f"  Average daily mail: {avg_daily_mail:.1f}")
        logger.info(f"  Average daily calls: {avg_daily_calls:.1f}")
        
        # Correlation analysis
        self._analyze_correlations()
        
        # Create visualizations
        self._create_eda_visualizations()
        
        # Store results
        self.results.update({
            'total_mail': total_mail,
            'total_calls': total_calls,
            'avg_daily_mail': avg_daily_mail,
            'avg_daily_calls': avg_daily_calls,
            'data_quality_score': (self.combined_data['data_quality'] == 'actual').mean()
        })
        
        return True
        
    except Exception as e:
        logger.error(f"✗ EDA failed: {e}")
        return False

def _analyze_correlations(self):
    """Analyze correlations between mail and calls."""
    try:
        logger.info("  Analyzing mail-call correlations...")
        
        # Direct correlation (same day)
        valid_data = self.combined_data[
            (self.combined_data['mail_volume'] > 0) & 
            (self.combined_data['call_count'] > 0)
        ]
        
        if len(valid_data) > 10:
            direct_corr = valid_data['mail_volume'].corr(valid_data['call_count'])
            logger.info(f"    Same-day correlation: {direct_corr:.3f}")
            self.results['direct_correlation'] = direct_corr
        else:
            logger.warning("    Insufficient data for same-day correlation")
            self.results['direct_correlation'] = 0
        
        # Lag correlations
        best_lag = 0
        best_corr = 0
        lag_results = {}
        
        for lag in ANALYSIS_PARAMS['lag_days_to_test']:
            try:
                mail_lagged = self.combined_data['mail_volume'].shift(lag)
                
                # Only use data where both values are positive
                valid_mask = (mail_lagged > 0) & (self.combined_data['call_count'] > 0)
                valid_mask &= ~(mail_lagged.isna() | self.combined_data['call_count'].isna())
                
                if valid_mask.sum() > 10:
                    corr = mail_lagged[valid_mask].corr(self.combined_data.loc[valid_mask, 'call_count'])
                    lag_results[lag] = corr
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                        
            except Exception as e:
                logger.warning(f"    Error calculating lag {lag}: {e}")
                continue
        
        if lag_results:
            logger.info(f"    Best lag correlation: {best_lag} days ({best_corr:.3f})")
            self.results['best_lag'] = best_lag
            self.results['best_lag_correlation'] = best_corr
            self.results['lag_correlations'] = lag_results
        else:
            logger.warning("    Could not calculate lag correlations")
            self.results['best_lag'] = 3  # Default
            self.results['best_lag_correlation'] = 0
        
    except Exception as e:
        logger.error(f"    Correlation analysis failed: {e}")

def _create_eda_visualizations(self):
    """Create EDA visualizations."""
    try:
        logger.info("  Creating EDA visualizations...")
        
        # Time series plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Mail volume
        axes[0].plot(self.combined_data['date'], self.combined_data['mail_volume'], 
                    color=COLORS['primary'], linewidth=1, alpha=0.7, label='Daily Volume')
        
        # Add 7-day moving average
        if len(self.combined_data) >= 7:
            ma7 = self.combined_data['mail_volume'].rolling(7, center=True).mean()
            axes[0].plot(self.combined_data['date'], ma7, color=COLORS['warning'], 
                        linewidth=2, label='7-day MA')
        
        axes[0].set_ylabel('Mail Volume')
        axes[0].set_title('Daily Mail Volume Over Time', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Call volume
        axes[1].plot(self.combined_data['date'], self.combined_data['call_count'], 
                    color=COLORS['success'], linewidth=1, alpha=0.7, label='Daily Calls')
        
        # Add 7-day moving average
        if len(self.combined_data) >= 7:
            call_ma7 = self.combined_data['call_count'].rolling(7, center=True).mean()
            axes[1].plot(self.combined_data['date'], call_ma7, color=COLORS['warning'], 
                        linewidth=2, label='7-day MA')
        
        axes[1].set_ylabel('Call Count')
        axes[1].set_xlabel('Date')
        axes[1].set_title('Daily Call Volume Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plots' / 'time_series_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Day of week patterns
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Mail by day of week
        dow_mail = self.combined_data.groupby('day_name')['mail_volume'].mean()
        dow_mail = dow_mail.reindex(days_order)
        
        bars1 = axes[0].bar(range(7), dow_mail.values, color=COLORS['primary'], alpha=0.7)
        axes[0].set_xticks(range(7))
        axes[0].set_xticklabels(days_order, rotation=45)
        axes[0].set_ylabel('Average Mail Volume')
        axes[0].set_title('Mail Volume by Day of Week', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars1, dow_mail.values):
            if not pd.isna(val):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dow_mail.values)*0.01,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Calls by day of week
        dow_calls = self.combined_data.groupby('day_name')['call_count'].mean()
        dow_calls = dow_calls.reindex(days_order)
        
        bars2 = axes[1].bar(range(7), dow_calls.values, color=COLORS['success'], alpha=0.7)
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(days_order, rotation=45)
        axes[1].set_ylabel('Average Call Count')
        axes[1].set_title('Call Volume by Day of Week', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars2, dow_calls.values):
            if not pd.isna(val):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dow_calls.values)*0.01,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plots' / 'day_of_week_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ EDA visualizations created")
        
    except Exception as e:
        logger.error(f"  ✗ EDA visualization creation failed: {e}")

def _build_robust_models(self):
    """Build models with robust error handling."""
    if self.combined_data is None:
        return False
    
    try:
        logger.info("Building robust prediction models...")
        
        # Prepare features
        if not self._prepare_modeling_features():
            return False
        
        # Split data
        if not self._split_data_robust():
            return False
        
        # Build models
        self._build_baseline_models_robust()
        self._build_ml_models_robust()
        
        # Evaluate models
        if self.models:
            self._evaluate_models_robust()
            return True
        else:
            logger.warning("⚠ No models were successfully built")
            return False
        
    except Exception as e:
        logger.error(f"✗ Model building failed: {e}")
        return False

def _prepare_modeling_features(self):
    """Prepare features for modeling with error handling."""
    try:
        logger.info("  Preparing modeling features...")
        
        # Create lag features
        for lag in [1, 3, 7]:
            if len(self.combined_data) > lag:
                self.combined_data[f'mail_lag_{lag}'] = self.combined_data['mail_volume'].shift(lag)
                self.combined_data[f'call_lag_{lag}'] = self.combined_data['call_count'].shift(lag)
        
        # Moving averages
        for window in [3, 7]:
            if len(self.combined_data) >= window:
                self.combined_data[f'mail_ma_{window}'] = (
                    self.combined_data['mail_volume'].rolling(window, min_periods=1).mean()
                )
                self.combined_data[f'call_ma_{window}'] = (
                    self.combined_data['call_count'].rolling(window, min_periods=1).mean()
                )
        
        # Cyclical time features
        self.combined_data['day_sin'] = np.sin(2 * np.pi * self.combined_data['day_of_week'] / 7)
        self.combined_data['day_cos'] = np.cos(2 * np.pi * self.combined_data['day_of_week'] / 7)
        self.combined_data['month_sin'] = np.sin(2 * np.pi * self.combined_data['month'] / 12)
        self.combined_data['month_cos'] = np.cos(2 * np.pi * self.combined_data['month'] / 12)
        
        # Remove rows with too many NaN values
        self.modeling_data = self.combined_data.dropna(subset=['call_count'])
        
        logger.info(f"  ✓ Features prepared: {len(self.modeling_data)} rows for modeling")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Feature preparation failed: {e}")
        return False

def _split_data_robust(self):
    """Split data for modeling with robust validation."""
    try:
        logger.info("  Splitting data for validation...")
        
        # Only use actual data for evaluation
        actual_data = self.modeling_data[self.modeling_data['data_quality'] == 'actual']
        
        if len(actual_data) < ANALYSIS_PARAMS['min_data_points']:
            logger.warning(f"  Insufficient actual data: {len(actual_data)} rows")
            # Use all data if actual data is insufficient
            actual_data = self.modeling_data
        
        # Time-based split to respect temporal structure
        split_idx = int(len(self.modeling_data) * (1 - ANALYSIS_PARAMS['test_split_ratio']))
        split_date = self.modeling_data.iloc[split_idx]['date']
        
        self.train_data = self.modeling_data[self.modeling_data['date'] < split_date]
        self.test_data = self.modeling_data[self.modeling_data['date'] >= split_date]
        
        # Use actual data only for testing when possible
        actual_test = self.test_data[self.test_data['data_quality'] == 'actual']
        if len(actual_test) >= 5:
            self.test_data_eval = actual_test
        else:
            self.test_data_eval = self.test_data
        
        logger.info(f"  ✓ Train: {len(self.train_data)} rows, Test: {len(self.test_data_eval)} rows")
        
        return len(self.test_data_eval) >= 3  # Need at least 3 test points
        
    except Exception as e:
        logger.error(f"  ✗ Data splitting failed: {e}")
        return False

def _build_baseline_models_robust(self):
    """Build baseline models with error handling."""
    try:
        logger.info("  Building baseline models...")
        
        # Define feature columns
        self.feature_cols = [col for col in self.modeling_data.columns 
                           if col not in ['date', 'call_count', 'mail_volume', 'data_quality', 
                                        'augmentation_method', 'day_name']]
        
        y_train = self.train_data['call_count']
        y_test = self.test_data_eval['call_count']
        
        # Simple average baseline
        if len(y_train) > 0:
            avg_pred = np.full(len(y_test), y_train.mean())
            self.models['baseline_avg'] = {
                'predictions': avg_pred,
                'name': 'Simple Average',
                'description': 'Uses overall average call volume'
            }
        
        # Day of week baseline
        if 'day_of_week' in self.train_data.columns:
            dow_avg = self.train_data.groupby('day_of_week')['call_count'].mean()
            dow_pred = self.test_data_eval['day_of_week'].map(dow_avg).fillna(y_train.mean())
            self.models['baseline_dow'] = {
                'predictions': dow_pred.values,
                'name': 'Day-of-Week Average',
                'description': 'Uses day-of-week patterns'
            }
        
        logger.info(f"  ✓ Built {len(self.models)} baseline models")
        
    except Exception as e:
        logger.error(f"  ✗ Baseline model building failed: {e}")

def _build_ml_models_robust(self):
    """Build ML models with comprehensive error handling."""
    try:
        logger.info("  Building ML models...")
        
        # Prepare training data
        X_train = self.train_data[self.feature_cols].fillna(0)
        y_train = self.train_data['call_count']
        X_test = self.test_data_eval[self.feature_cols].fillna(0)
        
        if len(X_train) < 10:
            logger.warning("  Insufficient training data for ML models")
            return
        
        # Ridge Regression (robust to multicollinearity)
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            
            self.models['ridge'] = {
                'model': ridge,
                'scaler': scaler,
                'predictions': ridge.predict(X_test_scaled),
                'name': 'Ridge Regression',
                'description': 'L2 regularized linear regression'
            }
            logger.info("    ✓ Ridge Regression built")
        except Exception as e:
            logger.warning(f"    ✗ Ridge Regression failed: {e}")
        
        # Random Forest (robust to outliers)
        try:
            # Adjust parameters based on data size
            n_estimators = min(100, max(10, len(X_train) // 2))
            max_depth = min(10, max(3, len(self.feature_cols)))
            
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            self.models['random_forest'] = {
                'model': rf,
                'predictions': rf.predict(X_test),
                'name': 'Random Forest',
                'description': 'Ensemble of decision trees',
                'feature_importance': pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
            }
            logger.info("    ✓ Random Forest built")
        except Exception as e:
            logger.warning(f"    ✗ Random Forest failed: {e}")
        
        # Gradient Boosting
        try:
            gbr = GradientBoostingRegressor(
                n_estimators=min(100, max(10, len(X_train) // 3)),
                learning_rate=0.1,
                max_depth=min(5, max(2, len(self.feature_cols) // 2)),
                random_state=42
            )
            gbr.fit(X_train, y_train)
            
            self.models['gradient_boost'] = {
                'model': gbr,
                'predictions': gbr.predict(X_test),
                'name': 'Gradient Boosting',
                'description': 'Sequential boosting ensemble'
            }
            logger.info("    ✓ Gradient Boosting built")
        except Exception as e:
            logger.warning(f"    ✗ Gradient Boosting failed: {e}")
        
        logger.info(f"  ✓ Built {len([m for m in self.models.values() if 'model' in m])} ML models")
        
    except Exception as e:
        logger.error(f"  ✗ ML model building failed: {e}")

def _evaluate_models_robust(self):
    """Evaluate models with robust metrics."""
    try:
        logger.info("  Evaluating models...")
        
        y_test = self.test_data_eval['call_count'].values
        
        if len(y_test) == 0:
            logger.error("  No test data available for evaluation")
            return
        
        self.evaluation_results = []
        
        for model_name, model_info in self.models.items():
            try:
                predictions = np.array(model_info['predictions']).flatten()
                
                # Ensure same length
                min_len = min(len(y_test), len(predictions))
                y_test_eval = y_test[:min_len]
                pred_eval = predictions[:min_len]
                
                # Calculate metrics with error handling
                mae = mean_absolute_error(y_test_eval, pred_eval)
                rmse = np.sqrt(mean_squared_error(y_test_eval, pred_eval))
                
                # MAPE with zero handling
                mask = y_test_eval != 0
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_test_eval[mask] - pred_eval[mask]) / y_test_eval[mask])) * 100
                else:
                    mape = 100.0
                
                # R² with fallback
                try:
                    r2 = r2_score(y_test_eval, pred_eval)
                except:
                    r2 = 0.0
                
                result = {
                    'model': model_name,
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'predictions': pred_eval,
                    'n_predictions': len(pred_eval)
                }
                
                self.evaluation_results.append(result)
                
            except Exception as e:
                logger.warning(f"    Error evaluating {model_name}: {e}")
                continue
        
        if self.evaluation_results:
            # Sort by MAE
            self.evaluation_results = sorted(self.evaluation_results, key=lambda x: x['mae'])
            
            # Print results
            logger.info("\n  Model Performance Summary:")
            logger.info("  " + "-"*70)
            logger.info(f"  {'Model':<25} {'MAE':<10} {'RMSE':<10} {'MAPE %':<10} {'R²':<10}")
            logger.info("  " + "-"*70)
            
            for result in self.evaluation_results:
                logger.info(
                    f"  {result['name']:<25} {result['mae']:<10.2f} "
                    f"{result['rmse']:<10.2f} {result['mape']:<10.2f} {result['r2']:<10.3f}"
                )
            
            best = self.evaluation_results[0]
            logger.info(f"\n  ✓ Best Model: {best['name']} (MAE: {best['mae']:.2f})")
        else:
            logger.warning("  No models could be evaluated")
        
    except Exception as e:
        logger.error(f"  ✗ Model evaluation failed: {e}")

def _create_comprehensive_dashboard(self):
    """Create comprehensive dashboard with error handling."""
    logger.info("Creating comprehensive dashboard...")
    
    try:
        # Create main dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Mail & Call Volume Trends',
                'Data Quality Overview', 
                'Day-of-Week Patterns',
                'Model Performance',
                'Correlation Analysis',
                'Weekly Trends'
            ),
            specs=[
                [{'secondary_y': True}, {'type': 'indicator'}],
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Time series (top left)
        fig.add_trace(
            go.Scatter(
                x=self.combined_data['date'],
                y=self.combined_data['mail_volume'],
                name='Mail Volume',
                line=dict(color=COLORS['primary'], width=2),
                opacity=0.8
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.combined_data['date'],
                y=self.combined_data['call_count'],
                name='Call Volume',
                line=dict(color=COLORS['success'], width=2),
                opacity=0.8
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Data quality indicator (top right)
        if self.combined_data is not None:
            completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
        else:
            completeness = 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=completeness,
                title={'text': "Data Completeness %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': COLORS['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgray'},
                        {'range': [50, 80], 'color': 'yellow'},
                        {'range': [80, 100], 'color': 'lightgreen'}
                    ],
                    'threshold': {
                        'line': {'color': COLORS['warning'], 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=2
        )
        
        # 3. Day-of-week patterns (middle left)
        if self.combined_data is not None:
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_mail = self.combined_data.groupby('day_name')['mail_volume'].mean().reindex(days_order)
            
            fig.add_trace(
                go.Bar(
                    x=days_order,
                    y=dow_mail.values,
                    name='Avg Mail',
                    marker_color=COLORS['primary'],
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 4. Model performance (middle right)
        if self.evaluation_results:
            model_names = [r['name'] for r in self.evaluation_results[:4]]
            mae_values = [r['mae'] for r in self.evaluation_results[:4]]
            
            colors = [COLORS['success'] if i == 0 else COLORS['primary'] for i in range(len(model_names))]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=mae_values,
                    marker_color=colors,
                    text=[f"{v:.1f}" for v in mae_values],
                    textposition='outside',
                    name='MAE'
                ),
                row=2, col=2
            )
        
        # 5. Correlation scatter (bottom left)
        if self.combined_data is not None:
            # Use best lag if available
            lag = self.results.get('best_lag', 3)
            mail_lagged = self.combined_data['mail_volume'].shift(lag)
            
            valid_mask = (mail_lagged > 0) & (self.combined_data['call_count'] > 0)
            if valid_mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=mail_lagged[valid_mask],
                        y=self.combined_data.loc[valid_mask, 'call_count'],
                        mode='markers',
                        marker=dict(
                            color=COLORS['info'],
                            size=6,
                            opacity=0.6
                        ),
                        name=f'Mail vs Calls (+{lag}d)',
                        showlegend=False
                    ),
                    row=3, col=1
                )
        
        # 6. Weekly trends (bottom right)
        if self.combined_data is not None and 'week' in self.combined_data.columns:
            weekly = self.combined_data.groupby('week').agg({
                'call_count': 'sum'
            }).tail(12)  # Last 12 weeks
            
            fig.add_trace(
                go.Bar(
                    x=[f"W{w}" for w in weekly.index],
                    y=weekly['call_count'],
                    name='Weekly Calls',
                    marker_color=COLORS['success'],
                    opacity=0.7,
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title={
                'text': "Mail-Call Predictive Analytics Dashboard",
                'font': {'size': 24, 'color': COLORS['primary']},
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            paper_bgcolor='#f8f9fa',
            plot_bgcolor='white',
            font={'family': 'Arial, sans-serif', 'size': 11}
        )
        
        # Update axes labels
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5')
        
        # Set y-axis labels for secondary_y
        fig.update_yaxes(title_text="Mail Volume", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Call Volume", secondary_y=True, row=1, col=1)
        
        # Save dashboard
        dashboard_path = OUTPUT_DIR / 'plots' / 'comprehensive_dashboard.html'
        fig.write_html(dashboard_path)
        logger.info(f"✓ Dashboard saved to: {dashboard_path}")
        
    except Exception as e:
        logger.error(f"✗ Dashboard creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def _create_detailed_report(self, success_steps):
    """Create detailed HTML report."""
    logger.info("Creating detailed report...")
    
    try:
        # Calculate key metrics
        if self.combined_data is not None:
            total_days = len(self.combined_data)
            mail_days = (self.combined_data['mail_volume'] > 0).sum()
            call_days = (self.combined_data['call_count'] > 0).sum()
            completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
            total_mail = self.combined_data['mail_volume'].sum()
            total_calls = self.combined_data['call_count'].sum()
            
            # Data quality assessment
            if completeness >= 90:
                quality_status = "Excellent"
                quality_color = "#4CAF50"
            elif completeness >= 70:
                quality_status = "Good"
                quality_color = "#FF9800"
            else:
                quality_status = "Needs Improvement"
                quality_color = "#F44336"
        else:
            total_days = mail_days = call_days = 0
            completeness = 0
            total_mail = total_calls = 0
            quality_status = "No Data"
            quality_color = "#F44336"
        
        # Model performance
        if self.evaluation_results:
            best_model = self.evaluation_results[0]
            accuracy = max(0, 100 - best_model['mape'])
            mae = best_model['mae']
            model_name = best_model['name']
            model_status = "Ready for Testing" if accuracy > 70 else "Needs Improvement"
        else:
            accuracy = mae = 0
            model_name = "None Built"
            model_status = "Not Available"
        
        # Correlation insights
        best_lag = self.results.get('best_lag', 'Unknown')
        best_corr = self.results.get('best_lag_correlation', 0)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(completeness, accuracy, best_corr)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mail-Call Predictive Analytics - Detailed Report</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
                    background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']});
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
                    border-left: 5px solid {COLORS['primary']};
                    transition: transform 0.2s;
                }}
                .metric:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: {COLORS['primary']};
                    margin: 10px 0;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 8px 16px;
                    border-radius: 20px;
                    color: white;
                    font-weight: bold;
                    margin: 5px;
                }}
                .section {{
                    margin: 40px 0;
                    padding: 30px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                .section h2 {{
                    color: {COLORS['primary']};
                    font-size: 24px;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid {COLORS['primary']};
                }}
                .two-column {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin: 20px 0;
                }}
                .data-insights {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid {COLORS['success']};
                }}
                .warning-box {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .warning-box h3 {{
                    color: #856404;
                    margin-top: 0;
                }}
                .recommendations {{
                    background: white;
                    padding: 25px;
                    border-radius: 8px;
                }}
                .rec-item {{
                    padding: 15px;
                    margin: 10px 0;
                    background: #f8f9fa;
                    border-left: 4px solid {COLORS['info']};
                    border-radius: 5px;
                }}
                .rec-priority {{
                    font-weight: bold;
                    color: {COLORS['primary']};
                }}
                .footer {{
                    background: #343a40;
                    color: white;
                    text-align: center;
                    padding: 30px;
                }}
                .success-steps {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .step-badge {{
                    background: {COLORS['success']};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Mail-Call Predictive Analytics</h1>
                    <div class="subtitle">Comprehensive Analysis Report</div>
                    <div class="subtitle">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                </div>
                
                <div class="content">
                    <!-- Executive Summary -->
                    <div class="section">
                        <h2>📊 Executive Summary</h2>
                        
                        <div class="metrics-grid">
                            <div class="metric">
                                <div class="metric-label">Data Quality</div>
                                <div class="metric-value">{completeness:.0f}%</div>
                                <div class="status-badge" style="background-color: {quality_color};">{quality_status}</div>
                            </div>
                            
                            <div class="metric">
                                <div class="metric-label">Model Accuracy</div>
                                <div class="metric-value">{accuracy:.0f}%</div>
                                <div class="status-badge" style="background-color: {'#4CAF50' if accuracy > 70 else '#FF9800'};">{model_status}</div>
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
                                <div style="font-size: 12px; color: #666;">{'Strong' if abs(best_corr) > 0.5 else 'Moderate' if abs(best_corr) > 0.3 else 'Weak'}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Analysis Progress -->
                    <div class="section">
                        <h2>✅ Analysis Progress</h2>
                        <p>Successfully completed analysis steps:</p>
                        <div class="success-steps">
                            {' '.join([f'<span class="step-badge">{step}</span>' for step in success_steps])}
                        </div>
                    </div>
                    
                    <!-- Data Insights -->
                    <div class="two-column">
                        <div class="data-insights">
                            <h3>📈 Data Insights</h3>
                            <ul>
                                <li><strong>Timeline:</strong> {total_days} days analyzed</li>
                                <li><strong>Mail Activity:</strong> {mail_days} days with mail campaigns</li>
                                <li><strong>Call Activity:</strong> {call_days} days with call volume</li>
                                <li><strong>Average Daily Mail:</strong> {total_mail/max(total_days,1):,.0f}</li>
                                <li><strong>Average Daily Calls:</strong> {total_calls/max(total_days,1):,.0f}</li>
                            </ul>
                        </div>
                        
                        <div class="data-insights">
                            <h3>🎯 Model Performance</h3>
                            <ul>
                                <li><strong>Best Model:</strong> {model_name}</li>
                                <li><strong>Prediction Error:</strong> ±{mae:.0f} calls/day</li>
                                <li><strong>Accuracy Rate:</strong> {accuracy:.1f}%</li>
                                {'<li><strong>Status:</strong> Ready for pilot testing</li>' if accuracy > 70 else '<li><strong>Status:</strong> Needs more data</li>'}
                            </ul>
                        </div>
                    </div>
                    
                    <!-- Data Quality Issues -->
                    {f'''
                    <div class="warning-box">
                        <h3>⚠️ Data Quality Issues Identified</h3>
                        <ul>
                            {chr(10).join([f"<li>{issue}</li>" for issue in self.data_issues[:5]])}
                        </ul>
                        {f"<p><em>And {len(self.data_issues)-5} more issues...</em></p>" if len(self.data_issues) > 5 else ""}
                    </div>
                    ''' if self.data_issues else ''}
                    
                    <!-- Recommendations -->
                    <div class="section">
                        <h2>💡 Strategic Recommendations</h2>
                        <div class="recommendations">
                            {chr(10).join([f'''
                            <div class="rec-item">
                                <div class="rec-priority">{rec["priority"]}</div>
                                <div><strong>{rec["title"]}</strong></div>
                                <div>{rec["description"]}</div>
                            </div>
                            ''' for rec in recommendations])}
                        </div>
                    </div>
                    
                    <!-- Technical Details -->
                    <div class="section">
                        <h2>🔧 Technical Details</h2>
                        <div class="two-column">
                            <div>
                                <h4>Analysis Configuration</h4>
                                <ul>
                                    <li>Lag periods tested: {', '.join(map(str, ANALYSIS_PARAMS['lag_days_to_test']))}</li>
                                    <li>Minimum data points: {ANALYSIS_PARAMS['min_data_points']}</li>
                                    <li>Test split ratio: {ANALYSIS_PARAMS['test_split_ratio']}</li>
                                </ul>
                            </div>
                            <div>
                                <h4>Models Built</h4>
                                <ul>
                                    {chr(10).join([f"<li>{model['name']}: {model['description']}</li>" for model in self.models.values() if 'name' in model])}
                                </ul>
                            </div>
                        </div>
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
        
        # Save report
        report_path = OUTPUT_DIR / 'reports' / 'executive_summary.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✓ Detailed report saved to: {report_path}")
        
        # Also create data quality report
        self._create_data_quality_report()
        
    except Exception as e:
        logger.error(f"✗ Report creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def _generate_recommendations(self, completeness, accuracy, correlation):
    """Generate strategic recommendations based on analysis results."""
    recommendations = []
    
    # Data quality recommendations
    if completeness < 70:
        recommendations.append({
            'priority': 'HIGH PRIORITY',
            'title': 'Improve Data Collection',
            'description': f'Current data completeness is {completeness:.0f}%. Focus on capturing missing historical data and implementing robust data collection processes.'
        })
    
    # Model performance recommendations
    if accuracy > 80:
        recommendations.append({
            'priority': 'IMMEDIATE ACTION',
            'title': 'Deploy Predictive Model',
            'description': f'Model accuracy of {accuracy:.0f}% is excellent. Ready for production deployment with monitoring.'
        })
    elif accuracy > 60:
        recommendations.append({
            'priority': 'SHORT TERM',
            'title': 'Pilot Model Testing',
            'description': f'Model accuracy of {accuracy:.0f}% is good for pilot testing. Monitor performance and refine.'
        })
    else:
        recommendations.append({
            'priority': 'MEDIUM TERM',
            'title': 'Enhance Model Training',
            'description': f'Model accuracy of {accuracy:.0f}% needs improvement. Collect more data and try advanced algorithms.'
        })
    
    # Correlation insights
    if abs(correlation) > 0.5:
        recommendations.append({
            'priority': 'INSIGHT',
            'title': 'Strong Mail-Call Relationship',
            'description': f'Strong correlation ({correlation:.2f}) detected. This validates the predictive approach and suggests reliable forecasting capability.'
        })
    elif abs(correlation) < 0.2:
        recommendations.append({
            'priority': 'INVESTIGATE',
            'title': 'Weak Mail-Call Relationship',
            'description': f'Low correlation ({correlation:.2f}) suggests other factors may drive call volume. Consider additional data sources.'
        })
    
    # Operational recommendations
    recommendations.append({
        'priority': 'OPERATIONAL',
        'title': 'Implement Monitoring Dashboard',
        'description': 'Set up real-time monitoring of predictions vs actuals to ensure model performance remains stable.'
    })
    
    recommendations.append({
        'priority': 'STRATEGIC',
        'title': 'Expand Predictive Capabilities',
        'description': 'Consider incorporating additional factors like seasonality, campaign types, and external events for enhanced accuracy.'
    })
    
    return recommendations

def _create_data_quality_report(self):
    """Create detailed data quality report."""
    try:
        logger.info("  Creating data quality report...")
        
        # Data quality analysis
        if self.combined_data is not None:
            quality_metrics = {
                'total_records': len(self.combined_data),
                'actual_data_pct': (self.combined_data['data_quality'] == 'actual').mean() * 100,
                'augmented_data_pct': (self.combined_data['data_quality'] == 'augmented').mean() * 100,
                'mail_coverage': (self.combined_data['mail_volume'] > 0).mean() * 100,
                'call_coverage': (self.combined_data['call_count'] > 0).mean() * 100,
            }
        else:
            quality_metrics = {k: 0 for k in ['total_records', 'actual_data_pct', 'augmented_data_pct', 'mail_coverage', 'call_coverage']}
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: {COLORS['primary']}; text-align: center; }}
                .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid {COLORS['primary']}; }}
                .issue {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                .good {{ background: #d4edda; padding: 10px; margin: 5px 0; border-left: 4px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Data Quality Assessment</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Quality Metrics</h2>
                <div class="metric"><strong>Total Records:</strong> {quality_metrics['total_records']:,}</div>
                <div class="metric"><strong>Actual Data:</strong> {quality_metrics['actual_data_pct']:.1f}%</div>
                <div class="metric"><strong>Augmented Data:</strong> {quality_metrics['augmented_data_pct']:.1f}%</div>
                <div class="metric"><strong>Mail Coverage:</strong> {quality_metrics['mail_coverage']:.1f}%</div>
                <div class="metric"><strong>Call Coverage:</strong> {quality_metrics['call_coverage']:.1f}%</div>
                
                <h2>Issues Identified</h2>
                {chr(10).join([f'<div class="issue">{issue}</div>' for issue in self.data_issues]) if self.data_issues else '<div class="good">No major data quality issues identified.</div>'}
                
                <h2>Recommendations</h2>
                <ul>
                    <li>Target 90%+ data completeness for production use</li>
                    <li>Implement automated data validation checks</li>
                    <li>Set up monitoring for data quality degradation</li>
                    <li>Regular audits of data collection processes</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        quality_report_path = OUTPUT_DIR / 'reports' / 'data_quality.html'
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"  ✓ Data quality report saved to: {quality_report_path}")
        
    except Exception as e:
        logger.error(f"  ✗ Data quality report creation failed: {e}")

def _create_enhanced_forecast(self):
    """Create enhanced forecast with confidence intervals."""
    logger.info("Creating enhanced forecast...")
    
    try:
        if not self.evaluation_results:
            logger.warning("⚠ No models available for forecasting")
            return
        
        # Use best model
        best_model_name = self.evaluation_results[0]['model']
        model_info = self.models.get(best_model_name)
        
        if not model_info or 'model' not in model_info:
            logger.warning("Best model not suitable for forecasting")
            return
        
        # Create future dates
        last_date = self.combined_data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=30,
            freq='D'
        )
        
        # Prepare features for future dates
        future_df = pd.DataFrame({'date': future_dates})
        
        # Add time features
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        
        # Fill other features with recent averages or patterns
        for col in self.feature_cols:
            if col not in future_df.columns:
                if 'lag' in col or 'ma' in col:
                    # Use recent average for lag/moving average features
                    recent_avg = self.modeling_data[col].tail(14).mean()
                    future_df[col] = recent_avg
                else:
                    # Use overall average for other features
                    overall_avg = self.modeling_data[col].mean()
                    future_df[col] = overall_avg
        
        # Make predictions
        X_future = future_df[self.feature_cols].fillna(0)
        
        if 'scaler' in model_info:
            X_future_scaled = model_info['scaler'].transform(X_future)
            predictions = model_info['model'].predict(X_future_scaled)
        else:
            predictions = model_info['model'].predict(X_future)
        
        # Calculate confidence intervals (simplified approach)
        if self.evaluation_results:
            mae = self.evaluation_results[0]['mae']
            upper_bound = predictions + 1.96 * mae  # Approximate 95% CI
            lower_bound = np.maximum(0, predictions - 1.96 * mae)  # Ensure non-negative
        else:
            upper_bound = predictions * 1.2
            lower_bound = predictions * 0.8
        
        # Create forecast visualization
        fig = go.Figure()
        
        # Historical data (last 60 days)
        recent_data = self.combined_data.tail(60)
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['call_count'],
            mode='lines+markers',
            name='Historical Calls',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=4)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color=COLORS['success'], width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(46, 160, 46, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
        # Add vertical line to separate historical from forecast
        fig.add_vline(
            x=last_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top"
        )
        
        fig.update_layout(
            title={
                'text': f'30-Day Call Volume Forecast ({model_info["name"]})',
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
        
        # Save forecast plot
        forecast_plot_path = OUTPUT_DIR / 'plots' / 'forecast.html'
        fig.write_html(forecast_plot_path)
        
        # Save forecast data
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_calls': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_used': model_info['name']
        })
        forecast_data_path = OUTPUT_DIR / 'data' / 'forecast.csv'
        forecast_df.to_csv(forecast_data_path, index=False)
        
        # Summary statistics
        avg_forecast = predictions.mean()
        total_forecast = predictions.sum()
        
        logger.info(f"✓ 30-day forecast created:")
        logger.info(f"  Average daily calls: {avg_forecast:.0f}")
        logger.info(f"  Total forecasted calls: {total_forecast:.0f}")
        logger.info(f"  Model used: {model_info['name']}")
        logger.info(f"  Forecast saved to: {forecast_plot_path}")
        
        # Store forecast results
        self.results['forecast'] = {
            'avg_daily': avg_forecast,
            'total_30day': total_forecast,
            'model_used': model_info['name'],
            'confidence_interval': 95
        }
        
    except Exception as e:
        logger.error(f"✗ Forecast creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def _create_failure_report(self, error_message, completed_steps):
    """Create a failure report when analysis cannot complete."""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Failure Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8d7da; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                .error {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .completed {{ background: #d4edda; color: #155724; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚠️ Analysis Failure Report</h1>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="error">
                    <h3>Error Details</h3>
                    <p>{error_message}</p>
                </div>
                
                <h3>Completed Steps</h3>
                {chr(10).join([f'<div class="completed">✓ {step}</div>' for step in completed_steps]) if completed_steps else '<p>No steps completed successfully.</p>'}
                
                <h3>Troubleshooting Steps</h3>
                <ol>
                    <li>Check that data files exist and are readable</li>
                    <li>Verify column names match the configuration</li>
                    <li>Ensure data files have proper date formats</li>
                    <li>Check file permissions and encoding</li>
                    <li>Review the analysis.log file for detailed errors</li>
                </ol>
            </div>
        </body>
        </html>
        """
        
        failure_path = OUTPUT_DIR / 'reports' / 'failure_report.html'
        with open(failure_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Failure report saved to: {failure_path}")
        
    except Exception as e:
        logger.error(f"Could not create failure report: {e}")
```

def main():
“”“Enhanced main execution function with comprehensive error handling.”””
print(”””
╔═══════════════════════════════════════════════════════════════════╗
║          Enhanced Mail-Call Predictive Analytics System           ║
║                        Production Version 2.0                     ║
║                     With Robust Error Handling                    ║
╚═══════════════════════════════════════════════════════════════════╝
“””)

```
try:
    # Create and run analyzer
    analyzer = EnhancedMailCallAnalyzer()
    analyzer.run_analysis()
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"📊 Open the dashboard: {OUTPUT_DIR}/plots/comprehensive_dashboard.html")
    print(f"📋 View the report: {OUTPUT_DIR}/reports/executive_summary.html")
    
except KeyboardInterrupt:
    print("\n⚠️ Analysis interrupted by user")
    logger.info("Analysis interrupted by user")
except Exception as e:
    print(f"\n❌ Analysis failed: {e}")
    logger.error(f"Main execution failed: {e}")
    print(f"📝 Check the log file: analysis.log for detailed error information")
    print(f"📁 Partial results (if any) saved to: {OUTPUT_DIR}")

print("\n" + "="*80)
```

if **name** == “**main**”:
main()