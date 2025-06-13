#!/usr/bin/env python
"""
advanced_matplotlib_plotter.py
===============================
Professional analytics visualization using Matplotlib + Seaborn
Creates publication-quality plots that always work

Usage: python advanced_matplotlib_plotter.py

Features:
- Beautiful modern styling with seaborn
- Interactive elements where possible
- High-quality PNG/PDF exports
- Professional color schemes
- Advanced statistical visualizations
- No browser dependencies

Requirements:
    pip install matplotlib seaborn pandas numpy pillow
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import webbrowser

# Essential imports
try:
    import pandas as pd
    import numpy as np
    print("✅ Core libraries loaded")
except ImportError as e:
    print(f"❌ Missing core libraries: {e}")
    print("💡 Run: pip install pandas numpy")
    sys.exit(1)

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    
    # Set matplotlib backend for better compatibility
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    print("✅ Matplotlib and Seaborn loaded")
    PLOTTING_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Plotting libraries missing: {e}")
    print("💡 Run: pip install matplotlib seaborn")
    PLOTTING_AVAILABLE = False

# ============================================================================
# MODERN STYLING AND CONFIGURATION
# ============================================================================

# Set modern style
if PLOTTING_AVAILABLE:
    # Use seaborn modern style
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8')
    
    # Modern color palette
    COLORS = {
        'primary': '#2E86C1',      # Modern blue
        'secondary': '#F39C12',    # Vibrant orange
        'success': '#27AE60',      # Green
        'danger': '#E74C3C',       # Red
        'warning': '#F1C40F',      # Yellow
        'info': '#8E44AD',         # Purple
        'dark': '#2C3E50',         # Dark blue-gray
        'light': '#ECF0F1',        # Light gray
        'RADAR': '#3498DB',        # Light blue
        'Product': '#E67E22',      # Orange
        'Meridian': '#16A085',     # Teal
        'actual': '#27AE60',       # Green
        'augmented': '#F39C12',    # Orange
        'forecast': '#9B59B6'      # Purple
    }
    
    # Modern font settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

# Default paths
DEFAULT_DATA_DIR = Path("enhanced_analysis_results")
DEFAULT_OUTPUT_DIR = Path("enhanced_analysis_results/plots_matplotlib")

# ============================================================================
# ADVANCED PLOTTER CLASS
# ============================================================================

class AdvancedMatplotlibPlotter:
    """Advanced analytics plotter using Matplotlib + Seaborn."""
    
    def __init__(self, data_dir=None, output_dir=None):
        """Initialize the plotter."""
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.combined_data = None
        self.evaluation_results = None
        self.source_analysis = None
        self.correlation_data = None
        self.forecast_data = None
        
        # Tracking
        self.created_plots = []
        self.failed_plots = []
        
        print(f"🎨 Advanced Matplotlib Plotter initialized")
        print(f"📁 Data: {self.data_dir}")
        print(f"📊 Output: {self.output_dir}")
    
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
        
        # Model evaluation
        eval_path = self.data_dir / 'models' / 'model_evaluation.csv'
        if eval_path.exists():
            try:
                eval_df = pd.read_csv(eval_path)
                self.evaluation_results = eval_df.to_dict('records')
                print(f"✅ Evaluation data: {len(self.evaluation_results)} models")
                success_count += 1
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
        return success_count > 0
    
    def create_all_visualizations(self):
        """Create all visualizations."""
        if not PLOTTING_AVAILABLE:
            print("❌ Plotting libraries not available")
            return False
        
        print("🎨 Creating comprehensive visualizations...")
        
        # Load data
        if not self.load_all_data():
            print("⚠️ No data loaded - creating sample visualizations")
            return self._create_sample_plots()
        
        # Create all plots
        plot_functions = [
            ("Executive Dashboard", self._create_executive_dashboard),
            ("Time Series Analysis", self._create_time_series_plots),
            ("Source Analysis", self._create_source_plots),
            ("Model Performance", self._create_model_plots),
            ("Correlation Analysis", self._create_correlation_plots),
            ("Data Quality Dashboard", self._create_quality_plots),
            ("Forecast Visualization", self._create_forecast_plots),
            ("Statistical Summary", self._create_statistical_plots)
        ]
        
        for name, func in plot_functions:
            try:
                print(f"📊 Creating {name}...")
                if func():
                    self.created_plots.append(name)
                    print(f"✅ {name} completed")
                else:
                    self.failed_plots.append(name)
                    print(f"⚠️ {name} failed")
            except Exception as e:
                self.failed_plots.append(name)
                print(f"❌ {name} error: {e}")
        
        # Create summary
        self._create_html_summary()
        
        # Print results
        self._print_results()
        
        # Try to open results
        self._open_results()
        
        return len(self.created_plots) > 0
    
    def _save_plot(self, fig, filename, title="Plot", dpi=300):
        """Save plot in multiple formats."""
        base_path = self.output_dir / filename
        
        try:
            # Save as PNG (high quality)
            png_path = base_path.with_suffix('.png')
            fig.savefig(png_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 PNG saved: {png_path.name}")
            
            # Save as PDF (vector)
            pdf_path = base_path.with_suffix('.pdf')
            fig.savefig(pdf_path, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📄 PDF saved: {pdf_path.name}")
            
            # Close figure to free memory
            plt.close(fig)
            
            return png_path
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            plt.close(fig)
            return None
    
    def _create_executive_dashboard(self):
        """Create executive dashboard."""
        try:
            if self.combined_data is None:
                return False
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # Main title
            fig.suptitle('📊 Executive Analytics Dashboard', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            # 1. Mail volume time series (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            if 'mail_volume_total' in self.combined_data.columns:
                ax1.plot(self.combined_data['date'], 
                        self.combined_data['mail_volume_total'],
                        color=COLORS['primary'], linewidth=2.5, label='Mail Volume')
                ax1.set_title('📧 Mail Volume Over Time', fontweight='bold')
                ax1.set_ylabel('Mail Volume')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add trend line
                x_numeric = range(len(self.combined_data))
                z = np.polyfit(x_numeric, self.combined_data['mail_volume_total'], 1)
                p = np.poly1d(z)
                ax1.plot(self.combined_data['date'], p(x_numeric), 
                        color=COLORS['danger'], linestyle='--', alpha=0.7, label='Trend')
                ax1.legend()
            
            # 2. Key metrics (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis('off')
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            y_pos = 0.9
            for key, value in metrics.items():
                ax2.text(0.1, y_pos, f'{key}:', fontweight='bold', fontsize=12)
                ax2.text(0.1, y_pos-0.1, f'{value}', fontsize=11, color=COLORS['primary'])
                y_pos -= 0.25
            
            ax2.set_title('📈 Key Metrics', fontweight='bold')
            
            # 3. Call volume with quality indicators (middle left)
            ax3 = fig.add_subplot(gs[1, :2])
            if 'call_count' in self.combined_data.columns:
                # Actual data
                actual_data = self.combined_data[
                    self.combined_data.get('data_quality', 'actual') == 'actual'
                ]
                if len(actual_data) > 0:
                    ax3.plot(actual_data['date'], actual_data['call_count'],
                            color=COLORS['success'], linewidth=2.5, 
                            label='Calls (Actual)', marker='o', markersize=3)
                
                # Augmented data
                augmented_data = self.combined_data[
                    self.combined_data.get('data_quality', 'none') == 'augmented'
                ]
                if len(augmented_data) > 0:
                    ax3.plot(augmented_data['date'], augmented_data['call_count'],
                            color=COLORS['warning'], linewidth=2, linestyle='--',
                            label='Calls (Augmented)', alpha=0.8)
                
                ax3.set_title('📞 Call Volume Over Time', fontweight='bold')
                ax3.set_ylabel('Call Count')
                ax3.tick_params(axis='x', rotation=45)
                ax3.legend()
            
            # 4. Weekly pattern (middle right)
            ax4 = fig.add_subplot(gs[1, 2])
            if 'day_of_week' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                weekly_data = self.combined_data.groupby('day_of_week')['call_count'].mean()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                bars = ax4.bar(day_names, weekly_data.values, 
                              color=COLORS['info'], alpha=0.8, edgecolor='white')
                ax4.set_title('📅 Weekly Pattern', fontweight='bold')
                ax4.set_ylabel('Avg Calls')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, weekly_data.values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=9)
            
            # 5. Source performance (bottom left)
            ax5 = fig.add_subplot(gs[2, 0])
            if self.source_analysis:
                sources = list(self.source_analysis.keys())
                volumes = [self.source_analysis[s]['total_volume'] for s in sources]
                colors = [COLORS.get(s.upper(), COLORS['primary']) for s in sources]
                
                bars = ax5.bar(sources, volumes, color=colors, alpha=0.8, edgecolor='white')
                ax5.set_title('🎯 Source Performance', fontweight='bold')
                ax5.set_ylabel('Total Volume')
                ax5.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, volumes):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes)*0.01,
                            f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
            
            # 6. Data quality pie chart (bottom middle)
            ax6 = fig.add_subplot(gs[2, 1])
            if 'data_quality' in self.combined_data.columns:
                quality_counts = self.combined_data['data_quality'].value_counts()
                colors_pie = [COLORS['success'], COLORS['warning']][:len(quality_counts)]
                
                wedges, texts, autotexts = ax6.pie(quality_counts.values, 
                                                  labels=quality_counts.index,
                                                  colors=colors_pie, autopct='%1.1f%%',
                                                  startangle=90)
                ax6.set_title('✅ Data Quality', fontweight='bold')
            
            # 7. Correlation heatmap (bottom right)
            ax7 = fig.add_subplot(gs[2, 2])
            if self.correlation_data is not None:
                # Create a simple correlation visualization
                lags = self.correlation_data['lag'].values
                corrs = self.correlation_data['correlation'].values
                
                bars = ax7.bar(lags, corrs, color=COLORS['secondary'], alpha=0.8)
                ax7.set_title('🔗 Lag Correlations', fontweight='bold')
                ax7.set_xlabel('Lag (days)')
                ax7.set_ylabel('Correlation')
                ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Highlight best correlation
                best_idx = np.argmax(np.abs(corrs))
                bars[best_idx].set_color(COLORS['danger'])
            
            # Add timestamp
            plt.figtext(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                       ha='right', va='bottom', fontsize=8, style='italic')
            
            return self._save_plot(fig, 'executive_dashboard', 'Executive Dashboard')
            
        except Exception as e:
            print(f"❌ Executive dashboard error: {e}")
            return False
    
    def _create_time_series_plots(self):
        """Create detailed time series analysis."""
        try:
            if self.combined_data is None:
                return False
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            fig.suptitle('📈 Time Series Analysis', fontsize=18, fontweight='bold')
            
            # 1. Main time series with dual y-axis
            ax1 = axes[0]
            ax1_twin = ax1.twinx()
            
            if 'mail_volume_total' in self.combined_data.columns:
                line1 = ax1.plot(self.combined_data['date'], 
                               self.combined_data['mail_volume_total'],
                               color=COLORS['primary'], linewidth=2.5, 
                               label='Mail Volume')
                ax1.set_ylabel('Mail Volume', color=COLORS['primary'])
                ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
            
            if 'call_count' in self.combined_data.columns:
                line2 = ax1_twin.plot(self.combined_data['date'], 
                                    self.combined_data['call_count'],
                                    color=COLORS['success'], linewidth=2.5,
                                    label='Call Volume')
                ax1_twin.set_ylabel('Call Volume', color=COLORS['success'])
                ax1_twin.tick_params(axis='y', labelcolor=COLORS['success'])
            
            ax1.set_title('📊 Mail and Call Volume Trends', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 2. Moving averages
            ax2 = axes[1]
            if 'call_count' in self.combined_data.columns and len(self.combined_data) > 7:
                # Original data
                ax2.plot(self.combined_data['date'], self.combined_data['call_count'],
                        color=COLORS['light'], alpha=0.5, linewidth=1, label='Daily Values')
                
                # 7-day moving average
                ma7 = self.combined_data['call_count'].rolling(7, center=True).mean()
                ax2.plot(self.combined_data['date'], ma7,
                        color=COLORS['secondary'], linewidth=2.5, label='7-day MA')
                
                # 30-day moving average (if enough data)
                if len(self.combined_data) > 30:
                    ma30 = self.combined_data['call_count'].rolling(30, center=True).mean()
                    ax2.plot(self.combined_data['date'], ma30,
                            color=COLORS['danger'], linewidth=2.5, label='30-day MA')
                
                ax2.set_title('📈 Moving Averages Analysis', fontweight='bold')
                ax2.set_ylabel('Call Volume')
                ax2.legend()
                ax2.tick_params(axis='x', rotation=45)
            
            # 3. Seasonal decomposition (simplified)
            ax3 = axes[2]
            if 'call_count' in self.combined_data.columns and 'day_of_week' in self.combined_data.columns:
                # Box plot by day of week
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                daily_data = []
                
                for day in range(7):
                    day_data = self.combined_data[
                        self.combined_data['day_of_week'] == day
                    ]['call_count'].values
                    daily_data.append(day_data)
                
                bp = ax3.boxplot(daily_data, labels=day_names, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], 
                                      [COLORS['primary'], COLORS['secondary'], COLORS['success'],
                                       COLORS['warning'], COLORS['info'], COLORS['danger'], COLORS['dark']]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax3.set_title('📅 Daily Pattern Distribution', fontweight='bold')
                ax3.set_ylabel('Call Volume')
                ax3.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return self._save_plot(fig, 'time_series_analysis', 'Time Series Analysis')
            
        except Exception as e:
            print(f"❌ Time series plots error: {e}")
            return False
    
    def _create_source_plots(self):
        """Create source analysis plots."""
        try:
            if self.combined_data is None:
                return False
            
            # Get mail source columns
            mail_source_cols = [col for col in self.combined_data.columns 
                               if col.startswith('mail_') and col != 'mail_volume_total']
            
            if not mail_source_cols:
                print("⚠️ No mail source columns found")
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('🎯 Source Analysis Dashboard', fontsize=18, fontweight='bold')
            
            # 1. Time series by source (top left)
            ax1 = axes[0, 0]
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                color = COLORS.get(source_name.upper(), plt.cm.Set3(i))
                
                ax1.plot(self.combined_data['date'], self.combined_data[col],
                        label=source_name, linewidth=2.5, color=color)
            
            ax1.set_title('📧 Mail Volume by Source Over Time', fontweight='bold')
            ax1.set_ylabel('Mail Volume')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Source comparison bar chart (top right)
            ax2 = axes[0, 1]
            if self.source_analysis:
                sources = list(self.source_analysis.keys())
                volumes = [self.source_analysis[s]['total_volume'] for s in sources]
                colors = [COLORS.get(s.upper(), COLORS['primary']) for s in sources]
                
                bars = ax2.bar(sources, volumes, color=colors, alpha=0.8, edgecolor='white')
                ax2.set_title('📊 Total Volume by Source', fontweight='bold')
                ax2.set_ylabel('Total Volume')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, volumes):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volumes)*0.01,
                            f'{value:,.0f}', ha='center', va='bottom', fontsize=10)
            
            # 3. Source distribution violin plot (bottom left)
            ax3 = axes[1, 0]
            source_data_list = []
            source_labels = []
            
            for col in mail_source_cols:
                source_name = col.replace('mail_', '').title()
                source_values = self.combined_data[self.combined_data[col] > 0][col]
                if len(source_values) > 0:
                    source_data_list.append(source_values)
                    source_labels.append(source_name)
            
            if source_data_list:
                parts = ax3.violinplot(source_data_list, positions=range(len(source_labels)))
                
                # Color the violin plots
                for i, pc in enumerate(parts['bodies']):
                    source_name = source_labels[i]
                    color = COLORS.get(source_name.upper(), COLORS['primary'])
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax3.set_xticks(range(len(source_labels)))
                ax3.set_xticklabels(source_labels, rotation=45)
                ax3.set_title('📈 Source Volume Distribution', fontweight='bold')
                ax3.set_ylabel('Mail Volume')
            
            # 4. Source correlation with calls (bottom right)
            ax4 = axes[1, 1]
            if 'call_count' in self.combined_data.columns:
                correlations = []
                source_names = []
                
                for col in mail_source_cols:
                    source_name = col.replace('mail_', '').title()
                    valid_mask = (self.combined_data[col] > 0) & (self.combined_data['call_count'] > 0)
                    
                    if valid_mask.sum() > 10:
                        try:
                            corr = self.combined_data.loc[valid_mask, col].corr(
                                self.combined_data.loc[valid_mask, 'call_count']
                            )
                            correlations.append(corr)
                            source_names.append(source_name)
                        except:
                            continue
                
                if correlations:
                    colors = [COLORS.get(name.upper(), COLORS['info']) for name in source_names]
                    bars = ax4.bar(source_names, correlations, color=colors, alpha=0.8)
                    
                    ax4.set_title('🔗 Source-Call Correlations', fontweight='bold')
                    ax4.set_ylabel('Correlation Coefficient')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Add value labels
                    for bar, value in zip(bars, correlations):
                        y_pos = value + (0.02 if value >= 0 else -0.05)
                        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                                f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top',
                                fontsize=10)
            
            plt.tight_layout()
            return self._save_plot(fig, 'source_analysis', 'Source Analysis')
            
        except Exception as e:
            print(f"❌ Source plots error: {e}")
            return False
    
    def _create_model_plots(self):
        """Create model performance analysis."""
        try:
            if not self.evaluation_results:
                print("⚠️ No evaluation results available")
                return False
            
            models = self.evaluation_results[:8]  # Top 8 models
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('🤖 Model Performance Analysis', fontsize=18, fontweight='bold')
            
            # Extract data
            names = [m['name'] for m in models]
            mae_vals = [m['mae'] for m in models]
            r2_vals = [m['r2'] for m in models]
            mape_vals = [m['mape'] for m in models]
            types = [m.get('type', 'unknown') for m in models]
            
            # 1. MAE comparison (top left)
            ax1 = axes[0, 0]
            bars1 = ax1.barh(names, mae_vals, color=COLORS['danger'], alpha=0.8)
            ax1.set_title('📏 Mean Absolute Error (Lower = Better)', fontweight='bold')
            ax1.set_xlabel('MAE')
            
            # Add value labels
            for bar, value in zip(bars1, mae_vals):
                ax1.text(bar.get_width() + max(mae_vals)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}', ha='left', va='center', fontsize=9)
            
            # 2. R² comparison (top right)
            ax2 = axes[0, 1]
            bars2 = ax2.barh(names, r2_vals, color=COLORS['success'], alpha=0.8)
            ax2.set_title('📐 R-Squared Score (Higher = Better)', fontweight='bold')
            ax2.set_xlabel('R² Score')
            
            # Add value labels
            for bar, value in zip(bars2, r2_vals):
                ax2.text(bar.get_width() + max(r2_vals)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontsize=9)
            
            # 3. MAPE comparison (bottom left)
            ax3 = axes[1, 0]
            bars3 = ax3.barh(names, mape_vals, color=COLORS['warning'], alpha=0.8)
            ax3.set_title('🎯 MAPE Percentage (Lower = Better)', fontweight='bold')
            ax3.set_xlabel('MAPE %')
            
            # Add value labels
            for bar, value in zip(bars3, mape_vals):
                ax3.text(bar.get_width() + max(mape_vals)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left', va='center', fontsize=9)
            
            # 4. Model type distribution (bottom right)
            ax4 = axes[1, 1]
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            colors_pie = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info']]
            wedges, texts, autotexts = ax4.pie(type_counts.values(), 
                                              labels=type_counts.keys(),
                                              colors=colors_pie[:len(type_counts)],
                                              autopct='%1.1f%%', startangle=90)
            ax4.set_title('🏷️ Model Type Distribution', fontweight='bold')
            
            plt.tight_layout()
            return self._save_plot(fig, 'model_performance', 'Model Performance')
            
        except Exception as e:
            print(f"❌ Model plots error: {e}")
            return False
    
    def _create_correlation_plots(self):
        """Create correlation analysis plots."""
        try:
            if self.combined_data is None:
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('🔗 Correlation Analysis Dashboard', fontsize=18, fontweight='bold')
            
            # 1. Mail vs Calls scatter plot (top left)
            ax1 = axes[0, 0]
            if 'mail_volume_total' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                valid_data = self.combined_data[
                    (self.combined_data['mail_volume_total'] > 0) & 
                    (self.combined_data['call_count'] > 0)
                ]
                
                if len(valid_data) > 0:
                    # Scatter plot
                    scatter = ax1.scatter(valid_data['mail_volume_total'], 
                                        valid_data['call_count'],
                                        c=COLORS['primary'], alpha=0.6, s=50, edgecolors='white')
                    
                    # Add trend line
                    try:
                        z = np.polyfit(valid_data['mail_volume_total'], valid_data['call_count'], 1)
                        p = np.poly1d(z)
                        ax1.plot(valid_data['mail_volume_total'], 
                                p(valid_data['mail_volume_total']),
                                color=COLORS['danger'], linestyle='--', linewidth=2)
                        
                        # Calculate and display correlation
                        corr = valid_data['mail_volume_total'].corr(valid_data['call_count'])
                        ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=12, fontweight='bold')
                    except:
                        pass
                    
                    ax1.set_title('📧📞 Mail vs Calls Relationship', fontweight='bold')
                    ax1.set_xlabel('Mail Volume')
                    ax1.set_ylabel('Call Count')
            
            # 2. Lag correlation analysis (top right)
            ax2 = axes[0, 1]
            if self.correlation_data is not None:
                bars = ax2.bar(self.correlation_data['lag'], 
                              self.correlation_data['correlation'],
                              color=COLORS['success'], alpha=0.8, edgecolor='white')
                
                ax2.set_title('⏰ Lag Correlation Analysis', fontweight='bold')
                ax2.set_xlabel('Lag (days)')
                ax2.set_ylabel('Correlation Coefficient')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Highlight best correlation
                best_idx = np.argmax(np.abs(self.correlation_data['correlation']))
                bars[best_idx].set_color(COLORS['danger'])
                
                # Add value labels for significant correlations
                for bar, value in zip(bars, self.correlation_data['correlation']):
                    if abs(value) > 0.1:  # Only label significant correlations
                        y_pos = value + (0.02 if value >= 0 else -0.05)
                        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                                f'{value:.3f}', ha='center', 
                                va='bottom' if value >= 0 else 'top', fontsize=9)
            
            # 3. Source correlations (bottom left)
            ax3 = axes[1, 0]
            mail_source_cols = [col for col in self.combined_data.columns 
                               if col.startswith('mail_') and col != 'mail_volume_total']
            
            if mail_source_cols and 'call_count' in self.combined_data.columns:
                source_corrs = []
                source_names = []
                
                for col in mail_source_cols:
                    source_name = col.replace('mail_', '').title()
                    valid_mask = (self.combined_data[col] > 0) & (self.combined_data['call_count'] > 0)
                    
                    if valid_mask.sum() > 10:
                        try:
                            corr = self.combined_data.loc[valid_mask, col].corr(
                                self.combined_data.loc[valid_mask, 'call_count']
                            )
                            source_corrs.append(corr)
                            source_names.append(source_name)
                        except:
                            continue
                
                if source_corrs:
                    colors = [COLORS.get(name.upper(), COLORS['info']) for name in source_names]
                    bars = ax3.bar(source_names, source_corrs, color=colors, alpha=0.8, edgecolor='white')
                    
                    ax3.set_title('🎯 Source-Call Correlations', fontweight='bold')
                    ax3.set_ylabel('Correlation Coefficient')
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Add value labels
                    for bar, value in zip(bars, source_corrs):
                        y_pos = value + (0.02 if value >= 0 else -0.05)
                        ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                                f'{value:.3f}', ha='center', 
                                va='bottom' if value >= 0 else 'top', fontsize=10)
            
            # 4. Correlation heatmap (bottom right)
            ax4 = axes[1, 1]
            
            # Create correlation matrix for numerical columns
            numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns
            corr_cols = [col for col in numeric_cols if col not in ['day_of_week', 'week', 'month', 'quarter']]
            
            if len(corr_cols) > 1:
                corr_matrix = self.combined_data[corr_cols].corr()
                
                # Create heatmap
                im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Set ticks and labels
                ax4.set_xticks(range(len(corr_cols)))
                ax4.set_yticks(range(len(corr_cols)))
                ax4.set_xticklabels([col.replace('_', '\n').replace('mail ', '') for col in corr_cols], 
                                   rotation=45, ha='right')
                ax4.set_yticklabels([col.replace('_', '\n').replace('mail ', '') for col in corr_cols])
                
                # Add correlation values as text
                for i in range(len(corr_cols)):
                    for j in range(len(corr_cols)):
                        value = corr_matrix.iloc[i, j]
                        color = 'white' if abs(value) > 0.5 else 'black'
                        ax4.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                color=color, fontsize=8, fontweight='bold')
                
                ax4.set_title('🌡️ Correlation Heatmap', fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
                cbar.set_label('Correlation Coefficient')
            
            plt.tight_layout()
            return self._save_plot(fig, 'correlation_analysis', 'Correlation Analysis')
            
        except Exception as e:
            print(f"❌ Correlation plots error: {e}")
            return False
    
    def _create_quality_plots(self):
        """Create data quality dashboard."""
        try:
            if self.combined_data is None:
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('🔍 Data Quality Assessment', fontsize=18, fontweight='bold')
            
            # 1. Data quality over time (top left)
            ax1 = axes[0, 0]
            if 'data_quality' in self.combined_data.columns:
                # Group by date and quality
                quality_pivot = self.combined_data.groupby(['date', 'data_quality']).size().unstack(fill_value=0)
                
                if 'actual' in quality_pivot.columns:
                    ax1.fill_between(quality_pivot.index, 0, quality_pivot['actual'],
                                   color=COLORS['success'], alpha=0.8, label='Actual Data')
                
                if 'augmented' in quality_pivot.columns:
                    bottom = quality_pivot.get('actual', 0)
                    ax1.fill_between(quality_pivot.index, bottom, 
                                   bottom + quality_pivot['augmented'],
                                   color=COLORS['warning'], alpha=0.8, label='Augmented Data')
                
                ax1.set_title('✅ Data Quality Over Time', fontweight='bold')
                ax1.set_ylabel('Number of Records')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)
            
            # 2. Missing data analysis (top right)
            ax2 = axes[0, 1]
            missing_data = {}
            for col in ['mail_volume_total', 'call_count']:
                if col in self.combined_data.columns:
                    missing_pct = (self.combined_data[col] == 0).mean() * 100
                    missing_data[col.replace('_', ' ').title()] = missing_pct
            
            if missing_data:
                bars = ax2.bar(list(missing_data.keys()), list(missing_data.values()),
                              color=[COLORS['danger'], COLORS['warning']], alpha=0.8, edgecolor='white')
                ax2.set_title('📊 Missing Data Analysis', fontweight='bold')
                ax2.set_ylabel('Missing Data %')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, missing_data.values()):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # 3. Data completeness by source (bottom left)
            ax3 = axes[1, 0]
            mail_source_cols = [col for col in self.combined_data.columns 
                               if col.startswith('mail_') and col != 'mail_volume_total']
            
            if mail_source_cols:
                source_completeness = {}
                for col in mail_source_cols:
                    source_name = col.replace('mail_', '').title()
                    completeness_pct = (self.combined_data[col] > 0).mean() * 100
                    source_completeness[source_name] = completeness_pct
                
                sources = list(source_completeness.keys())
                completeness = list(source_completeness.values())
                colors = [COLORS.get(s.upper(), COLORS['info']) for s in sources]
                
                bars = ax3.bar(sources, completeness, color=colors, alpha=0.8, edgecolor='white')
                ax3.set_title('🎯 Completeness by Source', fontweight='bold')
                ax3.set_ylabel('Completeness %')
                ax3.tick_params(axis='x', rotation=45)
                ax3.set_ylim(0, 100)
                
                # Add value labels
                for bar, value in zip(bars, completeness):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
                
                # Add quality threshold line
                ax3.axhline(y=90, color=COLORS['danger'], linestyle='--', alpha=0.7, 
                           label='Quality Threshold (90%)')
                ax3.legend()
            
            # 4. Data quality gauge (bottom right)
            ax4 = axes[1, 1]
            ax4.set_xlim(-1.5, 1.5)
            ax4.set_ylim(-1.5, 1.5)
            ax4.set_aspect('equal')
            
            # Calculate overall quality score
            overall_quality = 100
            if 'data_quality' in self.combined_data.columns:
                overall_quality = (self.combined_data['data_quality'] == 'actual').mean() * 100
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            
            # Background arc
            ax4.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=20, alpha=0.3)
            
            # Quality arc
            quality_theta = np.linspace(0, np.pi * (overall_quality / 100), 100)
            
            if overall_quality >= 90:
                color = COLORS['success']
            elif overall_quality >= 70:
                color = COLORS['warning']
            else:
                color = COLORS['danger']
            
            ax4.plot(np.cos(quality_theta), np.sin(quality_theta), 
                    color=color, linewidth=20, alpha=0.8)
            
            # Add pointer
            pointer_angle = np.pi * (overall_quality / 100)
            ax4.arrow(0, 0, 0.8*np.cos(pointer_angle), 0.8*np.sin(pointer_angle),
                     head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            # Add labels
            ax4.text(0, -0.3, f'{overall_quality:.1f}%', ha='center', va='center',
                    fontsize=24, fontweight='bold', color=color)
            ax4.text(0, -0.6, 'Data Quality Score', ha='center', va='center',
                    fontsize=12, fontweight='bold')
            
            # Add scale labels
            for i, label in enumerate(['0%', '50%', '100%']):
                angle = np.pi * (i / 2)
                x, y = 1.2 * np.cos(angle), 1.2 * np.sin(angle)
                ax4.text(x, y, label, ha='center', va='center', fontsize=10)
            
            ax4.set_title('📈 Overall Quality Score', fontweight='bold')
            ax4.axis('off')
            
            plt.tight_layout()
            return self._save_plot(fig, 'data_quality', 'Data Quality')
            
        except Exception as e:
            print(f"❌ Quality plots error: {e}")
            return False
    
    def _create_forecast_plots(self):
        """Create forecast visualization."""
        try:
            if self.forecast_data is None or self.combined_data is None:
                print("⚠️ No forecast data available")
                return False
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle('🔮 Call Volume Forecast Analysis', fontsize=18, fontweight='bold')
            
            # 1. Main forecast plot (top)
            ax1 = axes[0]
            
            # Historical data (last 60 days)
            recent_data = self.combined_data.tail(60)
            
            # Actual historical data
            actual_recent = recent_data[recent_data.get('data_quality', 'actual') == 'actual']
            if len(actual_recent) > 0:
                ax1.plot(actual_recent['date'], actual_recent['call_count'],
                        color=COLORS['success'], linewidth=3, marker='o', markersize=4,
                        label='Historical (Actual)', alpha=0.8)
            
            # Augmented historical data
            augmented_recent = recent_data[recent_data.get('data_quality', 'none') == 'augmented']
            if len(augmented_recent) > 0:
                ax1.plot(augmented_recent['date'], augmented_recent['call_count'],
                        color=COLORS['warning'], linewidth=2, linestyle='--',
                        marker='s', markersize=3, label='Historical (Augmented)', alpha=0.7)
            
            # Forecast line
            ax1.plot(self.forecast_data['date'], self.forecast_data['predicted_calls'],
                    color=COLORS['forecast'], linewidth=4, linestyle='-.', 
                    marker='D', markersize=5, label='Forecast')
            
            # Confidence interval (if available)
            if 'upper_bound' in self.forecast_data.columns and 'lower_bound' in self.forecast_data.columns:
                ax1.fill_between(self.forecast_data['date'],
                               self.forecast_data['lower_bound'],
                               self.forecast_data['upper_bound'],
                               color=COLORS['forecast'], alpha=0.2, label='Confidence Interval')
            
            # Add vertical line at forecast start
            last_date = self.combined_data['date'].max()
            ax1.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax1.text(last_date, ax1.get_ylim()[1] * 0.9, 'Forecast Start',
                    rotation=90, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax1.set_title('📈 Historical Data and Forecast', fontweight='bold')
            ax1.set_ylabel('Call Volume')
            ax1.legend(loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 2. Forecast statistics (bottom)
            ax2 = axes[1]
            
            # Calculate forecast statistics
            forecast_stats = {
                'Average': self.forecast_data['predicted_calls'].mean(),
                'Minimum': self.forecast_data['predicted_calls'].min(),
                'Maximum': self.forecast_data['predicted_calls'].max(),
                'Std Dev': self.forecast_data['predicted_calls'].std(),
                'Total': self.forecast_data['predicted_calls'].sum()
            }
            
            # Create bar chart of forecast statistics
            stats_names = list(forecast_stats.keys())
            stats_values = list(forecast_stats.values())
            
            bars = ax2.bar(stats_names, stats_values, 
                          color=[COLORS['primary'], COLORS['danger'], COLORS['success'], 
                                COLORS['warning'], COLORS['info']], 
                          alpha=0.8, edgecolor='white')
            
            ax2.set_title('📊 Forecast Statistics', fontweight='bold')
            ax2.set_ylabel('Call Volume')
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_values)*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Add forecast period info
            forecast_period = len(self.forecast_data)
            avg_forecast = self.forecast_data['predicted_calls'].mean()
            
            ax2.text(0.02, 0.98, f'Forecast Period: {forecast_period} days\nAverage Daily Forecast: {avg_forecast:.0f} calls',
                    transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], alpha=0.8),
                    fontsize=11)
            
            plt.tight_layout()
            return self._save_plot(fig, 'forecast_visualization', 'Forecast')
            
        except Exception as e:
            print(f"❌ Forecast plots error: {e}")
            return False
    
    def _create_statistical_plots(self):
        """Create advanced statistical analysis plots."""
        try:
            if self.combined_data is None:
                return False
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📊 Statistical Analysis Dashboard', fontsize=18, fontweight='bold')
            
            # 1. Distribution analysis (top left)
            ax1 = axes[0, 0]
            if 'call_count' in self.combined_data.columns:
                call_data = self.combined_data[self.combined_data['call_count'] > 0]['call_count']
                
                # Histogram
                n, bins, patches = ax1.hist(call_data, bins=30, alpha=0.7, 
                                          color=COLORS['primary'], edgecolor='white')
                
                # Overlay normal distribution curve
                mu, sigma = call_data.mean(), call_data.std()
                x = np.linspace(call_data.min(), call_data.max(), 100)
                y = ((1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((x-mu)/sigma)**2)) * len(call_data) * (bins[1]-bins[0])
                ax1.plot(x, y, color=COLORS['danger'], linewidth=2, label=f'Normal (μ={mu:.1f}, σ={sigma:.1f})')
                
                ax1.set_title('📈 Call Volume Distribution', fontweight='bold')
                ax1.set_xlabel('Call Count')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Monthly trends (top right)
            ax2 = axes[0, 1]
            if 'month' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                monthly_stats = self.combined_data.groupby('month')['call_count'].agg(['mean', 'std'])
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Line plot with error bars
                ax2.errorbar(monthly_stats.index, monthly_stats['mean'], 
                           yerr=monthly_stats['std'], marker='o', linewidth=2,
                           color=COLORS['secondary'], capsize=5, capthick=2,
                           markersize=8, alpha=0.8)
                
                ax2.set_title('📅 Monthly Trends with Variability', fontweight='bold')
                ax2.set_xlabel('Month')
                ax2.set_ylabel('Average Call Count')
                ax2.set_xticks(range(1, 13))
                ax2.set_xticklabels([month_names[i-1] for i in monthly_stats.index])
                ax2.grid(True, alpha=0.3)
            
            # 3. Outlier analysis (bottom left)
            ax3 = axes[1, 0]
            if 'call_count' in self.combined_data.columns:
                call_data = self.combined_data['call_count']
                
                # Box plot
                bp = ax3.boxplot(call_data, patch_artist=True, widths=0.6)
                bp['boxes'][0].set_facecolor(COLORS['info'])
                bp['boxes'][0].set_alpha(0.7)
                
                # Calculate and highlight outliers
                Q1 = call_data.quantile(0.25)
                Q3 = call_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = call_data[(call_data < lower_bound) | (call_data > upper_bound)]
                
                ax3.set_title(f'📦 Outlier Analysis ({len(outliers)} outliers)', fontweight='bold')
                ax3.set_ylabel('Call Count')
                ax3.set_xticklabels(['Call Volume'])
                
                # Add statistics text
                stats_text = f'Q1: {Q1:.0f}\nMedian: {call_data.median():.0f}\nQ3: {Q3:.0f}\nIQR: {IQR:.0f}'
                ax3.text(1.2, Q3, stats_text, va='center', ha='left',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # 4. Trend decomposition (bottom right)
            ax4 = axes[1, 1]
            if 'call_count' in self.combined_data.columns and len(self.combined_data) > 14:
                # Simple trend analysis
                call_data = self.combined_data['call_count'].values
                
                # Calculate rolling statistics
                window = min(14, len(call_data) // 4)
                rolling_mean = pd.Series(call_data).rolling(window, center=True).mean()
                rolling_std = pd.Series(call_data).rolling(window, center=True).std()
                
                # Plot original data
                ax4.plot(range(len(call_data)), call_data, 
                        color=COLORS['light'], alpha=0.5, linewidth=1, label='Original')
                
                # Plot trend
                ax4.plot(range(len(call_data)), rolling_mean, 
                        color=COLORS['danger'], linewidth=3, label=f'Trend ({window}-day MA)')
                
                # Plot variability envelope
                ax4.fill_between(range(len(call_data)), 
                               rolling_mean - rolling_std, rolling_mean + rolling_std,
                               color=COLORS['danger'], alpha=0.2, label='±1 Std Dev')
                
                ax4.set_title('📈 Trend Analysis', fontweight='bold')
                ax4.set_xlabel('Days')
                ax4.set_ylabel('Call Count')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._save_plot(fig, 'statistical_analysis', 'Statistical Analysis')
            
        except Exception as e:
            print(f"❌ Statistical plots error: {e}")
            return False
    
    def _create_sample_plots(self):
        """Create sample visualizations when no real data is available."""
        print("📊 Creating sample visualizations...")
        
        try:
            # Generate sample data
            dates = pd.date_range('2024-01-01', periods=90, freq='D')
            np.random.seed(42)
            
            # Create realistic sample data
            base_mail = 200 + 50 * np.sin(np.arange(90) * 2 * np.pi / 7)  # Weekly pattern
            mail_data = base_mail + np.random.normal(0, 30, 90)
            mail_data = np.maximum(mail_data, 0)
            
            call_data = mail_data * 0.25 + np.random.normal(0, 15, 90)
            call_data = np.maximum(call_data, 0)
            
            # Create sample dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📊 Sample Analytics Dashboard (No Real Data Available)', 
                        fontsize=18, fontweight='bold')
            
            # 1. Sample time series (top left)
            ax1 = axes[0, 0]
            ax1.plot(dates, mail_data, color=COLORS['primary'], linewidth=2.5, label='Sample Mail Volume')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(dates, call_data, color=COLORS['success'], linewidth=2.5, 
                         label='Sample Call Volume')
            
            ax1.set_title('📧📞 Sample Mail & Call Volumes', fontweight='bold')
            ax1.set_ylabel('Mail Volume', color=COLORS['primary'])
            ax1_twin.set_ylabel('Call Volume', color=COLORS['success'])
            ax1.tick_params(axis='x', rotation=45)
            
            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 2. Sample correlation (top right)
            ax2 = axes[0, 1]
            scatter = ax2.scatter(mail_data, call_data, c=COLORS['primary'], 
                                 alpha=0.6, s=50, edgecolors='white')
            
            # Add trend line
            z = np.polyfit(mail_data, call_data, 1)
            p = np.poly1d(z)
            ax2.plot(mail_data, p(mail_data), color=COLORS['danger'], 
                    linestyle='--', linewidth=2)
            
            # Calculate correlation
            corr = np.corrcoef(mail_data, call_data)[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=12, fontweight='bold')
            
            ax2.set_title('🔗 Sample Correlation Analysis', fontweight='bold')
            ax2.set_xlabel('Mail Volume')
            ax2.set_ylabel('Call Volume')
            
            # 3. Sample weekly pattern (bottom left)
            ax3 = axes[1, 0]
            
            # Create weekly pattern
            sample_df = pd.DataFrame({'date': dates, 'call_count': call_data})
            sample_df['day_of_week'] = sample_df['date'].dt.dayofweek
            weekly_pattern = sample_df.groupby('day_of_week')['call_count'].mean()
            
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            bars = ax3.bar(day_names, weekly_pattern.values, 
                          color=COLORS['secondary'], alpha=0.8, edgecolor='white')
            
            ax3.set_title('📅 Sample Weekly Pattern', fontweight='bold')
            ax3.set_ylabel('Average Call Count')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, weekly_pattern.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=10)
            
            # 4. Sample statistics table (bottom right)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calculate sample statistics
            stats = {
                'Total Days': len(dates),
                'Avg Daily Mail': f'{mail_data.mean():.0f}',
                'Avg Daily Calls': f'{call_data.mean():.0f}',
                'Mail Std Dev': f'{mail_data.std():.0f}',
                'Call Std Dev': f'{call_data.std():.0f}',
                'Correlation': f'{corr:.3f}'
            }
            
            # Create table
            table_data = []
            for key, value in stats.items():
                table_data.append([key, value])
            
            table = ax4.table(cellText=table_data,
                             colLabels=['Metric', 'Value'],
                             cellLoc='left',
                             loc='center',
                             colWidths=[0.6, 0.4])
            
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(stats) + 1):
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor(COLORS['primary'])
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor(COLORS['light'] if i % 2 == 0 else 'white')
                    cell.set_edgecolor('white')
            
            ax4.set_title('📊 Sample Statistics', fontweight='bold', pad=20)
            
            plt.tight_layout()
            return self._save_plot(fig, 'sample_dashboard', 'Sample Dashboard')
            
        except Exception as e:
            print(f"❌ Sample plots error: {e}")
            return False
    
    def _calculate_metrics(self):
        """Calculate key metrics for display."""
        metrics = {}
        
        try:
            if self.combined_data is not None:
                metrics['Total Days'] = len(self.combined_data)
                
                if 'mail_volume_total' in self.combined_data.columns:
                    metrics['Total Mail'] = f"{self.combined_data['mail_volume_total'].sum():,.0f}"
                    metrics['Avg Daily Mail'] = f"{self.combined_data['mail_volume_total'].mean():.0f}"
                
                if 'call_count' in self.combined_data.columns:
                    metrics['Total Calls'] = f"{self.combined_data['call_count'].sum():,.0f}"
                    metrics['Avg Daily Calls'] = f"{self.combined_data['call_count'].mean():.0f}"
                
                if 'data_quality' in self.combined_data.columns:
                    completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
                    metrics['Data Quality'] = f"{completeness:.1f}%"
            
            if self.evaluation_results:
                best_model = self.evaluation_results[0]
                metrics['Best Model'] = best_model['name']
                metrics['Model MAE'] = f"{best_model['mae']:.1f}"
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {'Error': 'Could not calculate metrics'}
    
    def _create_html_summary(self):
        """Create HTML summary page with links to all plots."""
        try:
            # Get list of created plot files
            png_files = list(self.output_dir.glob("*.png"))
            pdf_files = list(self.output_dir.glob("*.pdf"))
            
            # Calculate summary stats
            stats = self._calculate_metrics()
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>📊 Advanced Analytics Dashboard</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                        color: #333;
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    .header {{
                        background: linear-gradient(45deg, #2E86C1, #F39C12);
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }}
                    .header h1 {{ font-size: 2.8em; margin-bottom: 10px; font-weight: 300; }}
                    .header p {{ font-size: 1.2em; opacity: 0.9; }}
                    .content {{ padding: 40px; }}
                    
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 30px 0;
                    }}
                    .stat-card {{
                        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                        padding: 25px;
                        border-radius: 15px;
                        text-align: center;
                        border-left: 5px solid #2E86C1;
                        transition: transform 0.3s ease;
                    }}
                    .stat-card:hover {{ transform: translateY(-8px); box-shadow: 0 15px 30px rgba(0,0,0,0.1); }}
                    .stat-value {{ font-size: 2.5em; font-weight: bold; color: #2E86C1; margin: 10px 0; }}
                    .stat-label {{ color: #666; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
                    
                    .plots-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                        gap: 25px;
                        margin: 40px 0;
                    }}
                    .plot-card {{
                        background: white;
                        border-radius: 15px;
                        padding: 25px;
                        text-align: center;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
                        transition: all 0.3s ease;
                        border: 1px solid #e9ecef;
                    }}
                    .plot-card:hover {{
                        transform: translateY(-10px);
                        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                        border-color: #2E86C1;
                    }}
                    .plot-icon {{ font-size: 3.5em; margin-bottom: 15px; }}
                    .plot-title {{ font-size: 1.4em; font-weight: 600; margin-bottom: 10px; color: #2E86C1; }}
                    .plot-description {{ color: #666; margin-bottom: 20px; line-height: 1.5; }}
                    
                    .plot-links {{
                        display: flex;
                        gap: 10px;
                        justify-content: center;
                        flex-wrap: wrap;
                    }}
                    .plot-link {{
                        display: inline-block;
                        padding: 10px 20px;
                        text-decoration: none;
                        border-radius: 25px;
                        font-weight: 500;
                        transition: all 0.3s ease;
                        font-size: 14px;
                    }}
                    .png-link {{
                        background: #2E86C1;
                        color: white;
                    }}
                    .png-link:hover {{
                        background: #1B4F72;
                        transform: translateY(-2px);
                    }}
                    .pdf-link {{
                        background: #E74C3C;
                        color: white;
                    }}
                    .pdf-link:hover {{
                        background: #C0392B;
                        transform: translateY(-2px);
                    }}
                    
                    .section {{
                        margin: 40px 0;
                        padding: 30px;
                        background: #f8f9fa;
                        border-radius: 15px;
                    }}
                    .section h2 {{
                        color: #2E86C1;
                        font-size: 1.8em;
                        margin-bottom: 20px;
                        border-bottom: 3px solid #2E86C1;
                        padding-bottom: 10px;
                    }}
                    
                    .footer {{
                        background: #2C3E50;
                        color: white;
                        text-align: center;
                        padding: 30px;
                    }}
                    
                    .badge {{
                        display: inline-block;
                        background: #27AE60;
                        color: white;
                        padding: 5px 15px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: bold;
                        margin: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Advanced Analytics Dashboard</h1>
                        <p>Professional Mail-Call Analytics • Matplotlib + Seaborn</p>
                        <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    </div>
                    
                    <div class="content">
                        <div class="section">
                            <h2>📈 Key Performance Metrics</h2>
                            <div class="stats-grid">
                                {self._generate_stats_html(stats)}
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>🎯 Professional Visualizations</h2>
                            <p>High-quality publication-ready plots in PNG and PDF formats:</p>
                            <div class="plots-grid">
                                {self._generate_plot_cards_html(png_files, pdf_files)}
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>✨ Features & Benefits</h2>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                                <div>
                                    <h3>🖼️ High-Quality Output</h3>
                                    <ul>
                                        <li>Publication-ready PNG images (300 DPI)</li>
                                        <li>Vector PDF files for scalability</li>
                                        <li>Professional color schemes</li>
                                        <li>Clean, modern styling</li>
                                    </ul>
                                </div>
                                <div>
                                    <h3>📊 Advanced Analytics</h3>
                                    <ul>
                                        <li>Statistical analysis and distributions</li>
                                        <li>Correlation and trend analysis</li>
                                        <li>Data quality assessment</li>
                                        <li>Forecasting visualizations</li>
                                    </ul>
                                </div>
                                <div>
                                    <h3>🔧 Technical Excellence</h3>
                                    <ul>
                                        <li>No browser dependencies</li>
                                        <li>Cross-platform compatibility</li>
                                        <li>Robust error handling</li>
                                        <li>Memory-efficient processing</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>ℹ️ Usage Guide</h2>
                            <ul>
                                <li><strong>PNG Files:</strong> Perfect for presentations, reports, and web use</li>
                                <li><strong>PDF Files:</strong> Ideal for print materials and publications</li>
                                <li><strong>High Resolution:</strong> All images are 300 DPI for crisp quality</li>
                                <li><strong>Color Schemes:</strong> Carefully chosen for accessibility and professionalism</li>
                            </ul>
                        </div>
                        
                        <div class="section">
                            <h2>🎯 Analysis Summary</h2>
                            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
                                <span class="badge">✅ {len(self.created_plots)} Plots Created</span>
                                <span class="badge">📄 {len(pdf_files)} PDF Files</span>
                                <span class="badge">🖼️ {len(png_files)} PNG Files</span>
                                <span class="badge">🎨 Matplotlib + Seaborn</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>🚀 Advanced Analytics System | Built with Matplotlib & Seaborn</p>
                        <p>Professional • Reliable • Publication-Ready</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML summary
            summary_path = self.output_dir / 'analytics_summary.html'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📄 HTML summary created: {summary_path.name}")
            return summary_path
            
        except Exception as e:
            print(f"❌ HTML summary creation failed: {e}")
            return None
    
    def _generate_stats_html(self, stats):
        """Generate HTML for statistics cards."""
        stats_html = ""
        for key, value in stats.items():
            stats_html += f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{key}</div>
            </div>
            """
        return stats_html
    
    def _generate_plot_cards_html(self, png_files, pdf_files):
        """Generate HTML for plot cards."""
        plot_configs = {
            'executive_dashboard': ('📊', 'Executive Dashboard', 'Comprehensive overview with key metrics and trends'),
            'time_series_analysis': ('📈', 'Time Series Analysis', 'Temporal patterns, moving averages, and seasonality'),
            'source_analysis': ('🎯', 'Source Analysis', 'Performance breakdown by mail source with correlations'),
            'model_performance': ('🤖', 'Model Performance', 'Detailed comparison of prediction model accuracy'),
            'correlation_analysis': ('🔗', 'Correlation Analysis', 'Statistical relationships and lag analysis'),
            'data_quality': ('🔍', 'Data Quality', 'Completeness assessment and quality metrics'),
            'forecast_visualization': ('🔮', 'Forecast Analysis', 'Future predictions with confidence intervals'),
            'statistical_analysis': ('📊', 'Statistical Analysis', 'Advanced statistical insights and distributions'),
            'sample_dashboard': ('📊', 'Sample Dashboard', 'Demonstration with synthetic data')
        }
        
        # Get base names of files
        png_bases = {f.stem for f in png_files}
        pdf_bases = {f.stem for f in pdf_files}
        all_bases = png_bases.union(pdf_bases)
        
        cards_html = ""
        for base_name in all_bases:
            if base_name in plot_configs:
                icon, title, description = plot_configs[base_name]
            else:
                icon, title, description = ('📊', base_name.replace('_', ' ').title(), 'Professional visualization')
            
            # Build links
            links_html = ""
            if base_name in png_bases:
                links_html += f'<a href="{base_name}.png" class="plot-link png-link" target="_blank">📸 View PNG</a>'
            if base_name in pdf_bases:
                links_html += f'<a href="{base_name}.pdf" class="plot-link pdf-link" target="_blank">📄 View PDF</a>'
            
            cards_html += f"""
            <div class="plot-card">
                <div class="plot-icon">{icon}</div>
                <div class="plot-title">{title}</div>
                <div class="plot-description">{description}</div>
                <div class="plot-links">
                    {links_html}
                </div>
            </div>
            """
        
        return cards_html
    
    def _open_results(self):
        """Try to open the results."""
        try:
            # Find HTML summary file
            summary_path = self.output_dir / 'analytics_summary.html'
            
            if summary_path.exists():
                print(f"🌐 Opening summary: {summary_path.name}")
                self._try_open_file(summary_path)
            else:
                print("📁 Summary file not found, opening output directory")
                self._try_open_directory()
                
        except Exception as e:
            print(f"⚠️ Could not auto-open results: {e}")
            print(f"📁 Manually open: {self.output_dir}")
    
    def _try_open_file(self, file_path):
        """Try to open a file with system default application."""
        try:
            abs_path = Path(file_path).resolve()
            
            if sys.platform.startswith('win'):
                os.startfile(str(abs_path))
                print("✅ Opened with Windows default application")
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', str(abs_path)])
                print("✅ Opened with macOS default application")
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(abs_path)])
                print("✅ Opened with Linux default application")
            else:
                print(f"📋 Please manually open: {abs_path}")
                
        except Exception as e:
            print(f"⚠️ Could not open file: {e}")
            print(f"📋 Please manually open: {file_path}")
    
    def _try_open_directory(self):
        """Try to open the output directory."""
        try:
            abs_path = self.output_dir.resolve()
            
            if sys.platform.startswith('win'):
                subprocess.run(['explorer', str(abs_path)])
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', str(abs_path)])
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', str(abs_path)])
                
            print(f"✅ Opened directory: {abs_path}")
            
        except Exception as e:
            print(f"⚠️ Could not open directory: {e}")
            print(f"📁 Please manually navigate to: {self.output_dir}")
    
    def _print_results(self):
        """Print comprehensive results summary."""
        print("\n" + "=" * 80)
        print("🎉 ADVANCED ANALYTICS RESULTS")
        print("=" * 80)
        
        if self.created_plots:
            print(f"✅ Successfully created {len(self.created_plots)} visualizations:")
            for plot in self.created_plots:
                print(f"   📊 {plot}")
        
        if self.failed_plots:
            print(f"\n⚠️ Failed to create {len(self.failed_plots)} visualizations:")
            for plot in self.failed_plots:
                print(f"   ❌ {plot}")
        
        print(f"\n📁 Results location: {self.output_dir}")
        
        # Count files
        png_files = list(self.output_dir.glob("*.png"))
        pdf_files = list(self.output_dir.glob("*.pdf"))
        html_files = list(self.output_dir.glob("*.html"))
        
        print(f"\n📊 Generated Files:")
        print(f"   🖼️ PNG files: {len(png_files)}")
        print(f"   📄 PDF files: {len(pdf_files)}")
        print(f"   🌐 HTML files: {len(html_files)}")
        
        print(f"\n🎯 Key Features:")
        print(f"   📐 High resolution: 300 DPI")
        print(f"   🎨 Professional styling")
        print(f"   📱 No browser dependencies")
        print(f"   🔧 Cross-platform compatible")
        
        if html_files:
            main_file = self.output_dir / 'analytics_summary.html'
            if main_file.exists():
                print(f"\n🚀 Start here: {main_file}")
            else:
                print(f"\n🚀 View files in: {self.output_dir}")
        
        print("=" * 80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🎨 ADVANCED MATPLOTLIB ANALYTICS SYSTEM")
    print("=" * 60)
    
    if not PLOTTING_AVAILABLE:
        print("❌ Required plotting libraries not available!")
        print("💡 Install with: pip install matplotlib seaborn pandas numpy")
        return False
    
    try:
        # Create plotter
        plotter = AdvancedMatplotlibPlotter()
        
        # Run analysis
        success = plotter.create_all_visualizations()
        
        if success:
            print("\n🎉 SUCCESS! Professional visualizations created!")
            print("🎯 High-quality PNG and PDF files ready for use")
        else:
            print("\n⚠️ Some visualizations failed, check output directory")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['-h', '--help', 'help']:
            print("""
🎨 Advanced Matplotlib Analytics System

Usage:
    python advanced_matplotlib_plotter.py

Features:
- Publication-quality PNG images (300 DPI)
- Vector PDF files for print
- Professional color schemes
- No browser dependencies
- Cross-platform compatibility

Requirements:
    pip install matplotlib seaborn pandas numpy

Output:
- High-resolution PNG files
- Vector PDF files
- Professional HTML summary
            """)
            sys.exit(0)
    
    # Run main analysis
    success = main()
    sys.exit(0 if success else 1)
