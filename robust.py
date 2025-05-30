"""
Call Center Volume Prediction - Robust Production-Ready Solution
================================================================

This comprehensive solution prepares call center data for volume prediction modeling
based on customer mailing campaigns. It handles various data formats and column names.

Requirements:
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- holidays >= 0.13
- openpyxl >= 3.0.0 (for Excel file handling)

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import re
from pathlib import Path

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RobustCallCenterPipeline:
    """
    Robust pipeline for call center data analysis with comprehensive error handling
    """
    
    def __init__(self, genesys_path=None, contact_path=None, country='US', output_dir='output'):
        """
        Initialize the pipeline
        """
        self.genesys_path = genesys_path
        self.contact_path = contact_path
        self.country = country
        self.holidays = holidays.US() if country == 'US' else holidays.UK()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.genesys_df = None
        self.contact_df = None
        self.merged_df = None
        self.daily_summary = None
        self.hourly_summary = None
        self.modeling_data = None
        
        # Column mappings for flexibility
        self.column_mappings = {
            'connection_id': ['ConnectionID', 'CNID', 'connectionid', 'connection_id', 'CallID'],
            'reference_no': ['ReferenceNo', 'Reference_No', 'referenceno', 'RefNum', 'Reference'],
            'timestamp': ['ConversationStart', 'StartTime', 'CallStartTime', 'timestamp', 'CreatedDate'],
            'wait_time': ['WaitTime', 'Wait_Time', 'waittime', 'QueueTime'],
            'talk_time': ['TalkTime', 'Talk_Time', 'talktime', 'ConversationTime'],
            'hold_time': ['HoldTime', 'Hold_Time', 'holdtime'],
            'call_centre': ['CallCentre', 'Call_Centre', 'call_centre', 'Location', 'Site']
        }
        
        # Documentation
        self.transformation_log = []
        self.feature_definitions = {}
        
    def find_column(self, df, column_aliases):
        """
        Find a column in dataframe using multiple possible names
        """
        for alias in column_aliases:
            if alias in df.columns:
                return alias
        return None
    
    def load_data(self, genesys_df=None, contact_df=None):
        """
        Load data from files or dataframes with robust error handling
        """
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        try:
            if genesys_df is not None and contact_df is not None:
                self.genesys_df = genesys_df.copy()
                self.contact_df = contact_df.copy()
            else:
                # Load from files
                if self.genesys_path:
                    if self.genesys_path.endswith('.xlsx'):
                        self.genesys_df = pd.read_excel(self.genesys_path)
                    else:
                        self.genesys_df = pd.read_csv(self.genesys_path, encoding='utf-8', low_memory=False)
                        
                if self.contact_path:
                    if self.contact_path.endswith('.xlsx'):
                        self.contact_df = pd.read_excel(self.contact_path)
                    else:
                        self.contact_df = pd.read_csv(self.contact_path, encoding='utf-8', low_memory=False)
            
            print(f"✓ Loaded Genesys data: {len(self.genesys_df)} records, {len(self.genesys_df.columns)} columns")
            print(f"✓ Loaded Contact data: {len(self.contact_df)} records, {len(self.contact_df.columns)} columns")
            
            # Identify key columns
            print("\nIdentifying key columns...")
            for key, aliases in self.column_mappings.items():
                genesys_col = self.find_column(self.genesys_df, aliases)
                contact_col = self.find_column(self.contact_df, aliases)
                print(f"  {key}: Genesys='{genesys_col}', Contact='{contact_col}'")
                
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
            
    def clean_connection_ids(self, df, col_name):
        """
        Clean and standardize connection IDs with multiple strategies
        """
        if col_name not in df.columns:
            return df
            
        print(f"  Cleaning {col_name}...")
        
        # Convert to string
        df[col_name] = df[col_name].astype(str)
        
        # Strategy 1: Remove common prefixes/suffixes
        df[f'{col_name}_clean1'] = df[col_name].str.replace(r'^00E303907400', '', regex=True)
        df[f'{col_name}_clean1'] = df[f'{col_name}_clean1'].str.replace(r'_\d+$', '', regex=True)
        
        # Strategy 2: Extract alphanumeric core
        df[f'{col_name}_clean2'] = df[col_name].str.extract(r'([A-Z0-9]{4,})')
        
        # Strategy 3: Extract last significant part
        df[f'{col_name}_clean3'] = df[col_name].str.split('_').str[0]
        
        # Strategy 4: Remove all non-alphanumeric
        df[f'{col_name}_clean4'] = df[col_name].str.replace(r'[^A-Z0-9]', '', regex=True)
        
        return df
        
    def clean_and_standardize_data(self):
        """
        Clean and standardize both datasets with robust error handling
        """
        print("\n" + "=" * 80)
        print("DATA CLEANING AND STANDARDIZATION")
        print("=" * 80)
        
        # Find connection ID columns
        genesys_conn_col = self.find_column(self.genesys_df, self.column_mappings['connection_id'])
        contact_ref_col = self.find_column(self.contact_df, self.column_mappings['reference_no'])
        
        if not genesys_conn_col or not contact_ref_col:
            raise ValueError("Could not find connection ID columns in data")
            
        print(f"\nUsing '{genesys_conn_col}' from Genesys and '{contact_ref_col}' from Contact data")
        
        # Clean Genesys data
        print("\nCleaning Genesys data...")
        
        # Remove rows with missing connection IDs
        self.genesys_df = self.genesys_df.dropna(subset=[genesys_conn_col])
        
        # Clean connection IDs with multiple strategies
        self.genesys_df = self.clean_connection_ids(self.genesys_df, genesys_conn_col)
        
        # Handle timestamps
        timestamp_col = self.find_column(self.genesys_df, self.column_mappings['timestamp'])
        if timestamp_col:
            self.genesys_df[timestamp_col] = pd.to_datetime(self.genesys_df[timestamp_col], errors='coerce')
            print(f"  ✓ Converted {timestamp_col} to datetime")
            
        # Clean Contact data
        print("\nCleaning Contact data...")
        
        # Remove rows with missing reference numbers
        self.contact_df = self.contact_df.dropna(subset=[contact_ref_col])
        
        # Clean reference numbers with multiple strategies
        self.contact_df = self.clean_connection_ids(self.contact_df, contact_ref_col)
        
        # Standardize activity names
        if 'ActivityName' in self.contact_df.columns:
            self.contact_df['ActivityName'] = self.contact_df['ActivityName'].str.strip().str.title()
            
        # Handle duration data
        if 'ActivityDuration' in self.contact_df.columns:
            self.contact_df['ActivityDuration'] = pd.to_numeric(self.contact_df['ActivityDuration'], errors='coerce').fillna(0)
            
        print("✓ Data cleaning completed")
        
    def create_contact_aggregations(self):
        """
        Create comprehensive aggregations from contact data
        """
        print("\n" + "=" * 80)
        print("CREATING CONTACT DATA AGGREGATIONS")
        print("=" * 80)
        
        # Find reference column
        ref_col = self.find_column(self.contact_df, self.column_mappings['reference_no'])
        
        # Build aggregation dict dynamically based on available columns
        agg_dict = {}
        
        if 'ActivityName' in self.contact_df.columns:
            agg_dict['ActivityName'] = [
                'count',
                'nunique',
                lambda x: '|'.join(x.astype(str)) if len(x) > 0 else '',
                lambda x: x.iloc[0] if len(x) > 0 else 'Unknown',
                lambda x: x.iloc[-1] if len(x) > 0 else 'Unknown'
            ]
            
        if 'ActivityDuration' in self.contact_df.columns:
            agg_dict['ActivityDuration'] = ['sum', 'mean', 'max', 'min', 'std', 'median']
            
        # Find call centre column
        call_centre_col = self.find_column(self.contact_df, self.column_mappings['call_centre'])
        if call_centre_col:
            agg_dict[call_centre_col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            
        if 'CallerType' in self.contact_df.columns:
            agg_dict['CallerType'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            
        if 'CompanyCode' in self.contact_df.columns:
            agg_dict['CompanyCode'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            
        print("Creating aggregations...")
        
        # Create multiple aggregations for different ID cleaning strategies
        contact_aggs = []
        
        for clean_col in [ref_col] + [f'{ref_col}_clean{i}' for i in range(1, 5)]:
            if clean_col in self.contact_df.columns:
                try:
                    agg = self.contact_df.groupby(clean_col).agg(agg_dict).reset_index()
                    agg.columns = self._flatten_column_names(agg.columns, clean_col)
                    contact_aggs.append(agg)
                except:
                    continue
                    
        # Use the first successful aggregation
        if contact_aggs:
            self.contact_agg = contact_aggs[0]
        else:
            raise ValueError("Could not create contact aggregations")
            
        # Create derived features
        self._create_derived_features()
        
        print(f"✓ Created {len(self.contact_agg)} aggregated contact records")
        
    def _flatten_column_names(self, columns, ref_col):
        """
        Flatten multi-level column names from aggregation
        """
        flattened = [ref_col]  # Keep reference column name
        
        for col in columns[1:]:
            if isinstance(col, tuple):
                if col[1] == 'count':
                    flattened.append('activity_count')
                elif col[1] == 'nunique':
                    flattened.append('unique_activities')
                elif col[1] == '<lambda_0>':
                    flattened.append('activity_sequence')
                elif col[1] == '<lambda_1>':
                    flattened.append('first_activity')
                elif col[1] == '<lambda_2>':
                    flattened.append('last_activity')
                elif col[1] == 'sum':
                    flattened.append('total_duration')
                elif col[1] == 'mean':
                    flattened.append('avg_duration')
                elif col[1] == 'max':
                    flattened.append('max_duration')
                elif col[1] == 'min':
                    flattened.append('min_duration')
                elif col[1] == 'std':
                    flattened.append('std_duration')
                elif col[1] == 'median':
                    flattened.append('median_duration')
                elif col[1] == '<lambda>':
                    if 'CallCentre' in col[0] or 'call_centre' in col[0].lower():
                        flattened.append('call_centre')
                    elif 'CallerType' in col[0]:
                        flattened.append('caller_type')
                    elif 'CompanyCode' in col[0]:
                        flattened.append('company_code')
                    else:
                        flattened.append(col[0].lower())
                else:
                    flattened.append(f"{col[0].lower()}_{col[1]}")
            else:
                flattened.append(col)
                
        return flattened
        
    def _create_derived_features(self):
        """
        Create derived features from aggregations
        """
        df = self.contact_agg
        
        # Transfer indicators
        if 'activity_count' in df.columns:
            df['was_transferred_count'] = (df['activity_count'] > 2).astype(int)
            
        if 'unique_activities' in df.columns:
            df['was_transferred_unique'] = (df['unique_activities'] > 1).astype(int)
            
        if 'activity_sequence' in df.columns:
            df['was_transferred_keyword'] = df['activity_sequence'].str.contains(
                'Transfer|Escalat|Specialist', case=False, na=False
            ).astype(int)
            
        # Complexity scores
        if 'activity_count' in df.columns:
            df['complexity_simple'] = df['activity_count']
            
        if 'total_duration' in df.columns:
            df['complexity_duration'] = df['total_duration'] / 60
            
        if 'activity_count' in df.columns and 'total_duration' in df.columns:
            df['complexity_combined'] = (
                (df['activity_count'] - 1) * 2 + 
                (df['total_duration'] / 180)
            )
            
        # Call patterns
        if 'activity_count' in df.columns and 'total_duration' in df.columns:
            df['is_simple_call'] = (
                (df['activity_count'] <= 2) & 
                (df['total_duration'] < 300)
            ).astype(int)
            
            df['is_complex_call'] = (
                (df['activity_count'] > 5) | 
                (df['total_duration'] > 900)
            ).astype(int)
            
    def merge_datasets_robust(self):
        """
        Merge datasets with multiple fallback strategies
        """
        print("\n" + "=" * 80)
        print("MERGING DATASETS - ROBUST METHOD")
        print("=" * 80)
        
        # Find columns
        genesys_conn_col = self.find_column(self.genesys_df, self.column_mappings['connection_id'])
        
        # Try multiple merge strategies
        merge_strategies = []
        
        # Strategy 1: Try each cleaning method
        for i in [''] + [f'_clean{j}' for j in range(1, 5)]:
            genesys_col = f'{genesys_conn_col}{i}'
            
            for j in [''] + [f'_clean{k}' for k in range(1, 5)]:
                contact_col = self.contact_agg.columns[0] + j if j else self.contact_agg.columns[0]
                
                if genesys_col in self.genesys_df.columns and contact_col in self.contact_agg.columns:
                    merge_strategies.append((genesys_col, contact_col))
                    
        # Try each merge strategy
        best_merge = None
        best_match_rate = 0
        
        for genesys_col, contact_col in merge_strategies:
            try:
                # Sample merge to check match rate
                sample_size = min(10000, len(self.genesys_df))
                sample_merge = pd.merge(
                    self.genesys_df.sample(n=sample_size, random_state=42),
                    self.contact_agg,
                    left_on=genesys_col,
                    right_on=contact_col,
                    how='left'
                )
                
                match_rate = sample_merge[contact_col].notna().sum() / len(sample_merge)
                
                print(f"  Strategy {genesys_col} <-> {contact_col}: {match_rate:.2%} match rate")
                
                if match_rate > best_match_rate:
                    best_match_rate = match_rate
                    best_merge = (genesys_col, contact_col)
                    
            except Exception as e:
                continue
                
        # Use the best merge strategy
        if best_merge and best_match_rate > 0:
            print(f"\n✓ Using best strategy: {best_merge[0]} <-> {best_merge[1]} ({best_match_rate:.2%} match rate)")
            
            self.merged_df = pd.merge(
                self.genesys_df,
                self.contact_agg,
                left_on=best_merge[0],
                right_on=best_merge[1],
                how='left'
            )
        else:
            print("\n⚠️  No good merge strategy found, using left join on original columns")
            self.merged_df = pd.merge(
                self.genesys_df,
                self.contact_agg,
                left_on=genesys_conn_col,
                right_on=self.contact_agg.columns[0],
                how='left'
            )
            
        # Calculate final statistics
        total_records = len(self.merged_df)
        matched_records = self.merged_df[self.contact_agg.columns[0]].notna().sum()
        match_rate = (matched_records / total_records) * 100
        
        print(f"\n✓ Merged {total_records} records")
        print(f"✓ Final match rate: {match_rate:.2f}%")
        print(f"✓ Unmatched records: {total_records - matched_records}")
        
    def calculate_time_metrics(self):
        """
        Calculate comprehensive time metrics from millisecond fields
        """
        print("\n" + "=" * 80)
        print("CALCULATING TIME METRICS")
        print("=" * 80)
        
        df = self.merged_df
        
        # Find time columns
        wait_col = self.find_column(df, self.column_mappings['wait_time'])
        talk_col = self.find_column(df, self.column_mappings['talk_time'])
        hold_col = self.find_column(df, self.column_mappings['hold_time'])
        
        # List of possible time columns
        time_columns = [
            ('wait_time', wait_col),
            ('talk_time', talk_col),
            ('hold_time', hold_col),
            ('acw_time', 'ACWTime'),
            ('alert_time', 'AlertTime'),
            ('aban_time', 'AbanTime')
        ]
        
        # Convert milliseconds to seconds
        total_time_components = []
        
        for name, col in time_columns:
            if col and col in df.columns:
                # Convert to numeric, handling errors
                df[f'{name}_seconds'] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 1000
                total_time_components.append(f'{name}_seconds')
                print(f"  ✓ Converted {col} to seconds")
                
        # Calculate total call time
        if total_time_components:
            df['total_call_time_seconds'] = df[total_time_components].sum(axis=1)
        else:
            df['total_call_time_seconds'] = 0
            
        # Create time categories
        if 'wait_time_seconds' in df.columns:
            df['wait_time_category'] = pd.cut(
                df['wait_time_seconds'],
                bins=[0, 30, 60, 120, 300, float('inf')],
                labels=['No Wait', 'Short Wait', 'Medium Wait', 'Long Wait', 'Very Long Wait']
            )
            
        if 'talk_time_seconds' in df.columns:
            df['talk_time_category'] = pd.cut(
                df['talk_time_seconds'],
                bins=[0, 60, 180, 300, 600, 1200, float('inf')],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extremely Long']
            )
            
        # Compare with contact activity duration if available
        if 'total_duration' in df.columns and 'talk_time_seconds' in df.columns:
            df['duration_difference'] = abs(df['talk_time_seconds'] - df['total_duration'])
            df['duration_match_quality'] = pd.cut(
                df['duration_difference'],
                bins=[0, 10, 30, 60, 120, float('inf')],
                labels=['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']
            )
            
        # Summary statistics
        if 'wait_time_seconds' in df.columns:
            print(f"\n  Average wait time: {df['wait_time_seconds'].mean():.1f} seconds")
        if 'talk_time_seconds' in df.columns:
            print(f"  Average talk time: {df['talk_time_seconds'].mean():.1f} seconds")
        if 'total_call_time_seconds' in df.columns:
            print(f"  Average total call time: {df['total_call_time_seconds'].mean():.1f} seconds")
            
        self.merged_df = df
        
    def create_temporal_features_robust(self):
        """
        Create temporal features with robust error handling
        """
        print("\n" + "=" * 80)
        print("CREATING TEMPORAL FEATURES")
        print("=" * 80)
        
        df = self.merged_df
        
        # Find timestamp column
        timestamp_col = self.find_column(df, self.column_mappings['timestamp'])
        
        if not timestamp_col:
            print("⚠️  Warning: No timestamp column found. Skipping temporal features.")
            return
            
        # Create timestamp column
        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Handle null timestamps
        null_timestamps = df['timestamp'].isna().sum()
        if null_timestamps > 0:
            print(f"  Warning: {null_timestamps} timestamps could not be parsed")
            
            if null_timestamps == len(df):
                print("  ❌ All timestamps are invalid. Skipping temporal features.")
                return
                
            # Keep only valid timestamps for temporal features
            valid_timestamp_mask = df['timestamp'].notna()
            
        # Create date column for valid timestamps
        df.loc[valid_timestamp_mask, 'date'] = df.loc[valid_timestamp_mask, 'timestamp'].dt.date
        
        # Basic temporal features for valid timestamps
        print("  Creating basic temporal features...")
        
        temporal_features = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'hour': lambda x: x.dt.hour,
            'minute': lambda x: x.dt.minute,
            'day_of_week': lambda x: x.dt.dayofweek,
            'day_name': lambda x: x.dt.day_name(),
            'quarter': lambda x: x.dt.quarter,
            'day_of_year': lambda x: x.dt.dayofyear
        }
        
        for feature_name, feature_func in temporal_features.items():
            try:
                df.loc[valid_timestamp_mask, feature_name] = feature_func(df.loc[valid_timestamp_mask, 'timestamp'])
            except:
                print(f"    Could not create {feature_name}")
                
        # Week of year - handle different pandas versions
        try:
            df.loc[valid_timestamp_mask, 'week_of_year'] = df.loc[valid_timestamp_mask, 'timestamp'].dt.isocalendar().week
        except:
            try:
                df.loc[valid_timestamp_mask, 'week_of_year'] = df.loc[valid_timestamp_mask, 'timestamp'].dt.week
            except:
                print("    Could not create week_of_year")
                
        # Advanced temporal features
        print("  Creating advanced temporal features...")
        
        # Weekend
        if 'day_of_week' in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
        # Holiday features
        if 'date' in df.columns:
            df['is_holiday'] = df['date'].apply(lambda x: x in self.holidays if pd.notna(x) else False).astype(int)
            
        # Business hours
        if 'hour' in df.columns:
            df['is_business_hours'] = df['hour'].between(8, 17).astype(int)
            
        # Time of day
        if 'hour' in df.columns:
            df['time_of_day'] = pd.cut(
                df['hour'],
                bins=[-1, 6, 9, 12, 15, 18, 21, 24],
                labels=['Night', 'Early Morning', 'Morning', 'Lunch', 'Afternoon', 'Evening', 'Late Evening'],
                include_lowest=True
            )
            
        print(f"  ✓ Created temporal features for {valid_timestamp_mask.sum()} records")
        
        self.merged_df = df
        
    # ------------------------------------------------------------------
    # Intent augmentation  (new version)  # <<< CHANGED
    # ------------------------------------------------------------------
    def augment_intent_data(self):
        """
        Derive a final intent label for every IVR row.

        Priority
        --------
        1. Use an existing non-blank IVR/Genesys intent if it already maps.
        2. Keyword match against the concatenated Contact activity_sequence.
        3. Keyword match against the single longest-duration activity.
        4. Otherwise keep "Unknown".

        Extra columns added
        -------------------
        • intent_augmented        – final label
        • intent_confidence       – crude 0-1 score
        • intent_source           – 'ivr' / 'contact_keyword' / 'contact_duration' / 'unknown'
        • intent_augmented_flag   – 1 if we replaced an Unknown, else 0
        """
        print("\n" + "=" * 80)
        print("INTENT AUGMENTATION")
        print("=" * 80)

        df = self.merged_df

        # ------------------------------------------------------------------
        # 1. Dictionaries
        # ------------------------------------------------------------------
        STANDARD_INTENTS = {
            "Fraud Assistance": ["fraud", "unauthorized", "suspicious"],
            "Escheatment": ["escheat", "unclaimed"],
            "Balance/Value": ["balance", "value", "worth"],
            "Sell": ["sell", "liquidate", "redeem"],
            "Repeat Caller": ["repeat", "again"],
            "Name Change": ["name change", "marriage", "divorce"],
            "Buy Stock": ["buy", "purchase", "acquire"],
            "Statement": ["statement", "document"],
            "Recent Activity": ["recent activity", "history"],
            "Corporate Action": ["corporate action", "merger", "split"],
            "Data Protection": ["gdpr", "privacy"],
            "Press and Media": ["press", "media"],
            "Privacy Breach": ["breach", "leak"],
            "Consolidation": ["consolidate", "combine"],
            "Proxy Inquiry": ["proxy", "vote"],
            "Complaint Call": ["complaint", "unhappy"],
            "General Inquiry": ["question", "information"],
            "Tax Information": ["tax", "irs", "1099"],
            "Banking Details": ["bank", "wire", "ach"],
            "Dividend Payment": ["dividend", "payout"],
            "Address Change": ["address"],
            "Check Replacement": ["check", "cheque", "reissue"],
            "Stock Quote": ["quote", "price"],
            "Beneficiary Information": ["beneficiary", "estate"],
            "Dividend Reinvestment": ["drip", "reinvest"],
            "Certificate Issuance": ["certificate", "paper"],
            "Transfer": ["transfer"],
            "Existing IC User Login Problem": ["login", "password"],
            "New IC User Login Problem": ["register", "signup"],
            "Fulfillment": ["fulfill", "mail"],
            "Enrolment": ["enrol", "subscribe"],
            "Associate": ["associate", "link"],
            "Lost Certificate": ["lost certificate"],
        }

        LETTER_CODE = {
            "a": "Balance/Value",
            "b": "Sell",
            "c": "Repeat Caller",
            "d": "Name Change",
            "e": "Buy Stock",
            "f": "Statement",
            "g": "Recent Activity",
            "h": "Tax Information",
            "i": "Banking Details",
            "j": "Dividend Payment",
            "k": "Address Change",
            "l": "Check Replacement",
            "m": "Stock Quote",
            "n": "Beneficiary Information",
            "o": "Dividend Reinvestment",
            "p": "Certificate Issuance",
            "q": "Transfer",
            "r": "Existing IC User Login Problem",
            "s": "New IC User Login Problem",
            "t": "Fulfillment",
            "u": "Enrolment",
            "w": "Associate",
            "x": "Lost Certificate",
            "y": "Blank",
            "z": "Unknown",
        }

        # ------------------------------------------------------------------
        # 2. Locate base columns
        # ------------------------------------------------------------------
        ivr_intent_col = next(
            (c for c in df.columns if c.lower() in ["intent", "callintent", "uui_intent"]), None
        )
        longest_activity_col = (
            "last_activity" if "last_activity" in df.columns else None
        )

        # ------------------------------------------------------------------
        # 3. Inference helper
        # ------------------------------------------------------------------
        def infer_intent(row):
            # 1️⃣  use IVR if valid
            if ivr_intent_col and pd.notna(row[ivr_intent_col]):
                raw = str(row[ivr_intent_col]).strip()
                if raw in STANDARD_INTENTS:
                    return raw, 1.0, "ivr"
                if raw.lower() in LETTER_CODE:
                    return LETTER_CODE[raw.lower()], 1.0, "ivr"

            # 2️⃣  keyword search on activity_sequence
            if pd.notna(row.get("activity_sequence")):
                seq = str(row["activity_sequence"]).lower()
                for intent, kws in STANDARD_INTENTS.items():
                    hits = sum(kw in seq for kw in kws)
                    if hits:
                        return intent, min(hits / 3, 1.0), "contact_keyword"

            # 3️⃣  look at single longest activity (last_activity)
            if longest_activity_col and pd.notna(row.get(longest_activity_col)):
                act = str(row[longest_activity_col]).lower()
                for intent, kws in STANDARD_INTENTS.items():
                    if any(kw in act for kw in kws):
                        return intent, 0.4, "contact_duration"

            # 4️⃣  give up
            return "Unknown", 0.0, "unknown"

        # ------------------------------------------------------------------
        # 4. Apply and store
        # ------------------------------------------------------------------
        res = df.apply(lambda r: pd.Series(infer_intent(r)), axis=1)
        res.columns = ["intent_augmented", "intent_confidence", "intent_source"]
        df[["intent_augmented", "intent_confidence", "intent_source"]] = res
        df["intent_augmented_flag"] = (df["intent_source"] != "ivr").astype(int)

        # ------------------------------ DEBUG ------------------------------
        unknown_before = (
            df[ivr_intent_col].str.lower().eq("unknown").sum()
            if ivr_intent_col else len(df)
        )
        unknown_after = df["intent_augmented"].eq("Unknown").sum()
        print(
            f"✓ Unknown intents reduced  {unknown_before:,} ➜ {unknown_after:,} "
            f"({unknown_after/len(df):.1%})"
        )

        print("\nTop 10 intents:")
        print(df["intent_augmented"].value_counts().head(10).to_string())
        print("\nIntent sources:")
        print(df["intent_source"].value_counts(dropna=False).to_string())
        print("=" * 80)
        # ------------------------------------------------------------------

        self.merged_df = df
    def create_call_metrics(self):
        """
        Create comprehensive call-level metrics
        """
        print("\n" + "=" * 80)
        print("CREATING CALL METRICS")
        print("=" * 80)
        
        df = self.merged_df
        
        # Duration categories
        if 'total_duration' in df.columns:
            df['duration_category'] = pd.cut(
                df['total_duration'].fillna(0),
                bins=[0, 60, 180, 300, 600, 1200, float('inf')],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extremely Long']
            )
            
        # Call efficiency metrics
        if 'total_duration' in df.columns and 'activity_count' in df.columns:
            df['activities_per_minute'] = np.where(
                df['total_duration'] > 0,
                df['activity_count'] / (df['total_duration'] / 60),
                0
            )
            
        # Outcome indicators
        df['is_abandoned'] = 0
        if 'Abandoned' in df.columns:
            df['is_abandoned'] = pd.to_numeric(df['Abandoned'], errors='coerce').fillna(0).astype(int)
            
        df['has_callback'] = 0
        if 'CallbackDNIS' in df.columns:
            df['has_callback'] = df['CallbackDNIS'].notna().astype(int)
            
        # Multi-touch indicator
        if 'unique_activities' in df.columns:
            df['is_multi_touch'] = (df['unique_activities'] > 3).astype(int)
            
        print("  ✓ Created call metrics and categorizations")
        
        self.merged_df = df
        
    def create_aggregated_views(self):
        """
        Create hourly and daily aggregated views with error handling
        """
        print("\n" + "=" * 80)
        print("CREATING AGGREGATED VIEWS")
        print("=" * 80)
        
        df = self.merged_df
        
        # Find call centre column
        call_centre_col = self.find_column(df, self.column_mappings['call_centre'])
        if not call_centre_col:
            call_centre_col = 'call_centre'
            df[call_centre_col] = 'Unknown'
            
        # Check if we have date column
        if 'date' not in df.columns:
            print("  ⚠️  No date column available. Creating limited aggregations...")
            
            # Create a simple daily summary without dates
            self.daily_summary = pd.DataFrame({
                'total_calls': [len(df)],
                'call_centre': ['All Centers']
            })
            
            self.hourly_summary = pd.DataFrame({
                'total_calls': [len(df)],
                'call_centre': ['All Centers']
            })
            
            return
            
        # Daily aggregation
        print("  Creating daily aggregation...")
        
        # Build aggregation dict based on available columns
        daily_agg_dict = {
            'ConnectionID': 'count' if 'ConnectionID' in df.columns else 'size'
        }
        
        # Add optional columns if they exist
        optional_aggs = {
            'was_transferred_count': 'sum',
            'is_abandoned': 'sum',
            'total_duration': ['sum', 'mean'],
            'complexity_simple': 'mean',
            'activity_count': 'mean',
            'is_multi_touch': 'sum',
            'intent_confidence': 'mean',
            'is_weekend': 'first',
            'is_holiday': 'first'
        }
        
        for col, agg_func in optional_aggs.items():
            if col in df.columns:
                daily_agg_dict[col] = agg_func
                
        # Perform aggregation
        daily_agg = df.groupby(['date', call_centre_col]).agg(daily_agg_dict).reset_index()
        
        # Rename columns
        new_columns = ['date', 'call_centre']
        for col in daily_agg.columns[2:]:
            if isinstance(col, tuple):
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
                
        daily_agg.columns = new_columns[:len(daily_agg.columns)]
        
        # Rename the count column
        if 'ConnectionID' in daily_agg.columns:
            daily_agg.rename(columns={'ConnectionID': 'total_calls'}, inplace=True)
        elif 'size' in daily_agg.columns:
            daily_agg.rename(columns={'size': 'total_calls'}, inplace=True)
            
        # Add derived metrics safely
        if 'total_calls' in daily_agg.columns:
            if 'was_transferred_count' in daily_agg.columns:
                daily_agg['transfer_rate'] = daily_agg['was_transferred_count'] / daily_agg['total_calls']
            if 'is_abandoned' in daily_agg.columns:
                daily_agg['abandon_rate'] = daily_agg['is_abandoned'] / daily_agg['total_calls']
                
        self.daily_summary = daily_agg
        
        # Hourly aggregation (if hour column exists)
        if 'hour' in df.columns:
            print("  Creating hourly aggregation...")
            hourly_agg = df.groupby(['date', 'hour', call_centre_col]).size().reset_index(name='call_volume')
            self.hourly_summary = hourly_agg
        else:
            self.hourly_summary = pd.DataFrame()
            
        print(f"  ✓ Created daily summary: {len(self.daily_summary)} records")
        if len(self.hourly_summary) > 0:
            print(f"  ✓ Created hourly summary: {len(self.hourly_summary)} records")
            
    def create_lag_features(self):
        """
        Create lag features for time series modeling
        """
        print("\n" + "=" * 80)
        print("CREATING LAG FEATURES")
        print("=" * 80)
        
        if self.daily_summary is None or len(self.daily_summary) == 0:
            print("  ⚠️  No daily summary available. Skipping lag features.")
            self.modeling_data = pd.DataFrame()
            return
            
        df = self.daily_summary.copy()
        
        # Only create lag features if we have date and call centre columns
        if 'date' in df.columns and 'call_centre' in df.columns:
            df = df.sort_values(['call_centre', 'date'])
            
            # Lag features
            lag_periods = [1, 2, 3, 7, 14, 21, 28]
            
            for lag in lag_periods:
                if 'total_calls' in df.columns:
                    df[f'calls_lag_{lag}d'] = df.groupby('call_centre')['total_calls'].shift(lag)
                    
            # Rolling statistics
            windows = [7, 14, 28]
            
            for window in windows:
                if 'total_calls' in df.columns:
                    df[f'calls_ma_{window}d'] = df.groupby('call_centre')['total_calls'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    
            print(f"  ✓ Created lag and rolling features")
        else:
            print("  ⚠️  Cannot create lag features without date information")
            
        self.modeling_data = df
        
    def create_visualizations_safe(self):
        """
        Create visualizations with comprehensive error handling
        """
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create output directory for plots
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Check what data is available
        has_daily = self.daily_summary is not None and len(self.daily_summary) > 0
        has_hourly = self.hourly_summary is not None and len(self.hourly_summary) > 0
        has_temporal = 'date' in self.merged_df.columns
        
        if not has_temporal:
            print("  ⚠️  Limited visualizations available without temporal data")
            
        # Create a figure for basic statistics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Call Center Analysis Overview', fontsize=16)
        
        # 1. Call distribution by center
        ax = axes[0, 0]
        call_centre_col = self.find_column(self.merged_df, self.column_mappings['call_centre'])
        if call_centre_col and call_centre_col in self.merged_df.columns:
            center_counts = self.merged_df[call_centre_col].value_counts()
            center_counts.plot(kind='bar', ax=ax)
            ax.set_title('Calls by Center')
            ax.set_xlabel('Call Center')
            ax.set_ylabel('Number of Calls')
        else:
            ax.text(0.5, 0.5, 'No call center data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Calls by Center')
            
        # 2. Intent distribution
        ax = axes[0, 1]
        if 'intent_augmented' in self.merged_df.columns:
            intent_counts = self.merged_df['intent_augmented'].value_counts().head(10)
            intent_counts.plot(kind='barh', ax=ax)
            ax.set_title('Top 10 Call Intents')
            ax.set_xlabel('Count')
        else:
            ax.text(0.5, 0.5, 'No intent data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Call Intents')
            
        # 3. Duration distribution
        ax = axes[1, 0]
        if 'total_duration' in self.merged_df.columns:
            duration_data = self.merged_df['total_duration'].dropna()
            if len(duration_data) > 0:
                duration_data_filtered = duration_data[duration_data < duration_data.quantile(0.95)]
                ax.hist(duration_data_filtered, bins=50, edgecolor='black')
                ax.set_title('Call Duration Distribution')
                ax.set_xlabel('Duration (seconds)')
                ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Call Duration Distribution')
            
        # 4. Summary statistics
        ax = axes[1, 1]
        summary_text = f"Total Calls: {len(self.merged_df):,}\n"
        
        if 'call_centre' in self.merged_df.columns:
            summary_text += f"Call Centers: {self.merged_df['call_centre'].nunique()}\n"
            
        if 'intent_augmented' in self.merged_df.columns:
            unknown_rate = (self.merged_df['intent_augmented'] == 'Unknown').sum() / len(self.merged_df) * 100
            summary_text += f"Unknown Intent Rate: {unknown_rate:.1f}%\n"
            
        if 'total_duration' in self.merged_df.columns:
            avg_duration = self.merged_df['total_duration'].mean() / 60
            summary_text += f"Avg Duration: {avg_duration:.1f} min\n"
            
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12, verticalalignment='center')
        ax.set_title('Summary Statistics')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'overview_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create time series plot if we have daily data
        if has_daily and 'date' in self.daily_summary.columns:
            fig, ax = plt.subplots(figsize=(15, 6))
            
            for centre in self.daily_summary['call_centre'].unique():
                centre_data = self.daily_summary[self.daily_summary['call_centre'] == centre]
                if 'total_calls' in centre_data.columns:
                    ax.plot(centre_data['date'], centre_data['total_calls'], 
                           marker='o', markersize=4, label=centre, alpha=0.7)
                    
            ax.set_title('Daily Call Volume Trend', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Calls')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / 'daily_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"  ✓ Created visualizations in {plots_dir}")
        
    def save_outputs(self):
        """
        Save all processed data and reports
        """
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)
        
        # Save processed datasets
        print("  Saving processed datasets...")
        
        # Main datasets
        self.merged_df.to_csv(self.output_dir / 'merged_call_data.csv', index=False)
        print(f"    ✓ Saved merged_call_data.csv ({len(self.merged_df)} records)")
        
        if self.daily_summary is not None and len(self.daily_summary) > 0:
            self.daily_summary.to_csv(self.output_dir / 'daily_summary.csv', index=False)
            print(f"    ✓ Saved daily_summary.csv ({len(self.daily_summary)} records)")
            
        if self.hourly_summary is not None and len(self.hourly_summary) > 0:
            self.hourly_summary.to_csv(self.output_dir / 'hourly_summary.csv', index=False)
            print(f"    ✓ Saved hourly_summary.csv ({len(self.hourly_summary)} records)")
            
        if self.modeling_data is not None and len(self.modeling_data) > 0:
            self.modeling_data.to_csv(self.output_dir / 'modeling_ready_data.csv', index=False)
            print(f"    ✓ Saved modeling_ready_data.csv ({len(self.modeling_data)} records)")
            
        print(f"\n  ✓ All outputs saved to {self.output_dir}")
        
    def generate_business_report(self):
        """
        Generate a business-friendly summary report
        """
        print("\n" + "=" * 80)
        print("GENERATING BUSINESS REPORT")
        print("=" * 80)
        
        # Gather statistics
        total_calls = len(self.merged_df)
        
        # Call center info
        call_centre_col = self.find_column(self.merged_df, self.column_mappings['call_centre'])
        num_centers = self.merged_df[call_centre_col].nunique() if call_centre_col else 'Unknown'
        
        # Date range
        if 'date' in self.merged_df.columns:
            date_range = f"{self.merged_df['date'].min()} to {self.merged_df['date'].max()}"
        else:
            date_range = "Date information not available"
            
        # Match rate
        contact_cols = [col for col in self.merged_df.columns if col.startswith('activity_')]
        if contact_cols:
            match_rate = self.merged_df[contact_cols[0]].notna().sum() / len(self.merged_df) * 100
        else:
            match_rate = 0
            
        report = f"""
CALL CENTER DATA ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

EXECUTIVE SUMMARY
----------------
This report summarizes the data preparation and analysis completed for the call 
center volume prediction project.

DATA OVERVIEW
-------------
• Total call records analyzed: {total_calls:,}
• Date range: {date_range}
• Number of call centers: {num_centers}
• Data match rate: {match_rate:.1f}%

KEY COLUMNS IDENTIFIED
---------------------
"""
        
        # Add identified columns
        for key, aliases in self.column_mappings.items():
            genesys_col = self.find_column(self.genesys_df, aliases) if self.genesys_df is not None else None
            report += f"• {key}: {genesys_col or 'Not found'}\n"
            
        report += """

DATA QUALITY NOTES
-----------------
• Connection ID matching used multiple strategies to maximize match rate
• Temporal features created where timestamp data was available
• Intent augmentation applied to categorize calls into 26 standard categories

OUTPUTS CREATED
--------------
• merged_call_data.csv - Complete integrated dataset
• daily_summary.csv - Daily aggregated metrics (if dates available)
• modeling_ready_data.csv - Prepared for predictive modeling
• plots/ - Visualization directory

NEXT STEPS
----------
1. Review the match rate and consider manual validation of unmatched records
2. Integrate mailing campaign data when available
3. Build predictive models using the prepared features

================================================================================
        """
        
        # Save report
        with open(self.output_dir / 'business_report.txt', 'w') as f:
            f.write(report)
            
        print(f"  ✓ Business report saved to {self.output_dir / 'business_report.txt'}")
        
    def run_complete_pipeline(self):
        """
        Run the entire pipeline with comprehensive error handling
        """
        print("\n" + "="*80)
        print("RUNNING ROBUST CALL CENTER ANALYSIS PIPELINE")
        print("="*80)
        
        steps = [
            ("Loading data", self.load_data),
            ("Cleaning and standardizing data", self.clean_and_standardize_data),
            ("Creating contact aggregations", self.create_contact_aggregations),
            ("Merging datasets", self.merge_datasets_robust),
            ("Calculating time metrics", self.calculate_time_metrics),
            ("Creating temporal features", self.create_temporal_features_robust),
            ("Augmenting intent data", self.augment_intent_data),
            ("Creating call metrics", self.create_call_metrics),
            ("Creating aggregated views", self.create_aggregated_views),
            ("Creating lag features", self.create_lag_features),
            ("Creating visualizations", self.create_visualizations_safe),
            ("Saving outputs", self.save_outputs),
            ("Generating business report", self.generate_business_report)
        ]
        
        completed_steps = []
        
        for step_name, step_func in steps:
            try:
                print(f"\n→ {step_name}...")
                step_func()
                completed_steps.append(step_name)
            except Exception as e:
                print(f"\n❌ ERROR in {step_name}: {str(e)}")
                print(f"   Continuing with pipeline...")
                
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"\nCompleted {len(completed_steps)} out of {len(steps)} steps:")
        for step in completed_steps:
            print(f"  ✓ {step}")
            
        print(f"\nAll outputs saved to: {self.output_dir}")
        
        return len(completed_steps) == len(steps)


def main():
    """
    Main execution function
    """
    import sys
    
    print("\n" + "="*80)
    print("ROBUST CALL CENTER VOLUME PREDICTION - DATA PREPARATION")
    print("="*80)
    
    # Check command line arguments
    if len(sys.argv) >= 3:
        genesys_path = sys.argv[1]
        contact_path = sys.argv[2]
        country = sys.argv[3] if len(sys.argv) > 3 else 'US'
    else:
        # Interactive mode
        print("\nPlease provide the file paths:")
        genesys_path = input("Genesys/IVR data file path: ").strip()
        contact_path = input("Contact data file path: ").strip()
        country = input("Country for holidays (US/UK) [default: US]: ").strip() or 'US'
    
    # Validate file paths
    if not os.path.exists(genesys_path):
        print(f"❌ Error: Genesys file not found: {genesys_path}")
        return
    
    if not os.path.exists(contact_path):
        print(f"❌ Error: Contact file not found: {contact_path}")
        return
    
    # Create output directory
    output_dir = 'call_center_analysis_output_robust'
    
    # Initialize pipeline
    pipeline = RobustCallCenterPipeline(
        genesys_path=genesys_path,
        contact_path=contact_path,
        country=country,
        output_dir=output_dir
    )
    
    # Run pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ Analysis complete! Check the output directory for results.")
    else:
        print("\n⚠️  Analysis completed with some errors. Check the output directory for available results.")


if __name__ == "__main__":
    main()