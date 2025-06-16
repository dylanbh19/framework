## **1. Replace the entire `load_all_data()` method:**

**Find this method (around line 430):**

```python
def load_all_data(self):
    """Load all available data."""
    # ... existing code ...
```

**Replace the entire method with:**

```python
def load_all_data(self):
    """Load all available data."""
    print("📂 Loading analysis data...")
    
    success_count = 0
    
    # Combined timeline data
    timeline_path = self.data_dir / 'data' / 'combined_timeline.csv'
    if timeline_path.exists():
        try:
            self.combined_data = pd.read_csv(timeline_path)
            self.combined_data['date'] = pd.to_datetime(self.combined_data['date'])
            print(f"✅ Timeline data: {len(self.combined_data)} rows")
            success_count += 1
        except Exception as e:
            print(f"⚠️ Timeline data error: {e}")
    
    # Model evaluation - SAFE LOADING
    eval_path = self.data_dir / 'models' / 'model_evaluation.csv'
    if eval_path.exists():
        try:
            eval_df = pd.read_csv(eval_path)
            # SAFE MODEL LOADING - Only keep valid rows
            eval_df = eval_df.dropna(subset=['mae', 'r2', 'mape']).head(10)
            for col in ['mae', 'r2', 'mape']:
                eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce')
            eval_df = eval_df.dropna(subset=['mae', 'r2', 'mape'])
            
            if len(eval_df) > 0:
                self.evaluation_results = eval_df.to_dict('records')
                print(f"✅ Evaluation data: {len(self.evaluation_results)} models")
                success_count += 1
            else:
                print("⚠️ No valid model evaluation data")
        except Exception as e:
            print(f"⚠️ Evaluation data error: {e}")
    
    # Source analysis
    source_path = self.data_dir / 'source_analysis' / 'source_summary.csv'
    if source_path.exists():
        try:
            source_df = pd.read_csv(source_path, index_col=0)
            self.source_analysis = source_df.to_dict('index')
            print(f"✅ Source data: {len(self.source_analysis)} sources")
            success_count += 1
        except Exception as e:
            print(f"⚠️ Source data error: {e}")
    
    # Correlation data
    corr_path = self.data_dir / 'data' / 'lag_correlations.csv'
    if corr_path.exists():
        try:
            self.correlation_data = pd.read_csv(corr_path)
            print(f"✅ Correlation data: {len(self.correlation_data)} points")
            success_count += 1
        except Exception as e:
            print(f"⚠️ Correlation data error: {e}")
    
    # Forecast data
    forecast_path = self.data_dir / 'data' / 'forecast.csv'
    if forecast_path.exists():
        try:
            self.forecast_data = pd.read_csv(forecast_path)
            self.forecast_data['date'] = pd.to_datetime(self.forecast_data['date'])
            print(f"✅ Forecast data: {len(self.forecast_data)} points")
            success_count += 1
        except Exception as e:
            print(f"⚠️ Forecast data error: {e}")
    
    print(f"📊 Loaded {success_count}/5 data sources")
    
    # QUICK DATA CLEANING
    if success_count > 0 and self.combined_data is not None:
        print("🔧 Quick data cleaning...")
        
        # Filter out 2023 dates
        original_len = len(self.combined_data)
        self.combined_data = self.combined_data[self.combined_data['date'].dt.year != 2023].copy()
        print(f"📅 Filtered out {original_len - len(self.combined_data)} records from 2023")
        
        # Clean and normalize key columns
        for col in ['mail_volume_total', 'call_count']:
            if col in self.combined_data.columns:
                # Replace zeros with forward fill, then normalize
                self.combined_data[col] = self.combined_data[col].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill').fillna(1)
                # Cap extreme outliers
                q99 = self.combined_data[col].quantile(0.99)
                self.combined_data[col] = np.minimum(self.combined_data[col], q99)
                # Create normalized version (0-100 scale)
                max_val = self.combined_data[col].max()
                if max_val > 0:
                    self.combined_data[f'{col}_norm'] = (self.combined_data[col] / max_val) * 100
        
        print(f"✅ Data cleaned and normalized: {len(self.combined_data)} rows")
    
    return success_count > 0
```

## **2. Update the `_create_executive_dashboard()` method to use normalized data:**

**Find this section around line 600 in `_create_executive_dashboard()`:**

```python
# 1. Mail volume time series (top left)
ax1 = fig.add_subplot(gs[0, :2])
if 'mail_volume_total' in self.combined_data.columns:
```

**Replace that entire section with:**

```python
# 1. Mail volume time series (top left) - USE NORMALIZED DATA
ax1 = fig.add_subplot(gs[0, :2])
ax1_twin = ax1.twinx()

# Use normalized data for better comparison
if 'mail_volume_total_norm' in self.combined_data.columns:
    ax1.plot(self.combined_data['date'], 
            self.combined_data['mail_volume_total_norm'],
            color=COLORS['primary'], linewidth=2.5, label='Mail Volume (Normalized)')
    ax1.set_ylabel('Mail Volume (0-100)', color=COLORS['primary'])
elif 'mail_volume_total' in self.combined_data.columns:
    ax1.plot(self.combined_data['date'], 
            self.combined_data['mail_volume_total'],
            color=COLORS['primary'], linewidth=2.5, label='Mail Volume')
    ax1.set_ylabel('Mail Volume', color=COLORS['primary'])

# Calls on secondary axis
if 'call_count_norm' in self.combined_data.columns:
    ax1_twin.plot(self.combined_data['date'], 
                 self.combined_data['call_count_norm'],
                 color=COLORS['success'], linewidth=2.5, label='Calls (Normalized)')
    ax1_twin.set_ylabel('Call Count (0-100)', color=COLORS['success'])
elif 'call_count' in self.combined_data.columns:
    ax1_twin.plot(self.combined_data['date'], 
                 self.combined_data['call_count'],
                 color=COLORS['success'], linewidth=2.5, label='Calls')
    ax1_twin.set_ylabel('Call Count', color=COLORS['success'])

ax1.set_title('📧📞 Mail & Call Volumes (Cleaned & Normalized)', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
```

That’s it! Just replace those 2 sections and your data will be:

- ✅ **2023 dates filtered out**
- 📊 **Normalized to 0-100 scale**
- 🔧 **Zeros replaced intelligently**
- 🛡️ **Model errors handled safely**
- 📈 **Better plotting with comparable scales**

Much cleaner and easier to implement! 🚀​​​​​​​​​​​​​​​​