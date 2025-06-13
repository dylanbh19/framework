#!/usr/bin/env python
"""
plotting.py
===========
Standalone plotting module for Enhanced Mail-Call Analytics
Creates comprehensive visualizations from analysis results

Usage:
    python plotting.py
    or
    from plotting import AnalyticsPlotter
    plotter = AnalyticsPlotter()
    plotter.create_all_plots()

Requirements:
    pip install plotly pandas numpy seaborn matplotlib kaleido
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure Plotly
pyo.init_notebook_mode(connected=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Color scheme
COLORS = {
    'RADAR': '#1f77b4',
    'Product': '#ff7f0e', 
    'Meridian': '#2ca02c',
    'actual': '#2ca02c',
    'augmented': '#ff7f0e',
    'forecast': '#9467bd',
    'confidence': 'rgba(148, 103, 189, 0.2)',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8'
}

# Default paths
DEFAULT_DATA_DIR = Path("enhanced_analysis_results")
DEFAULT_OUTPUT_DIR = Path("enhanced_analysis_results/plots")

# Plotting configuration
PLOT_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'responsive': True
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_plotting_logger():
    """Setup logger for plotting module."""
    logger = logging.getLogger('plotting')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - PLOTTING - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('plotting.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_plotting_logger()

# ============================================================================
# MAIN PLOTTER CLASS
# ============================================================================

class AnalyticsPlotter:
    """Comprehensive plotting class for mail-call analytics."""
    
    def __init__(self, data_dir: Path = None, output_dir: Path = None):
        """Initialize the plotter.
        
        Args:
            data_dir: Directory containing analysis results
            output_dir: Directory to save plots
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.combined_data = None
        self.evaluation_results = None
        self.source_analysis = None
        self.correlation_data = None
        self.forecast_data = None
        
        logger.info(f"Plotter initialized - Data: {self.data_dir}, Output: {self.output_dir}")
    
    def load_data(self):
        """Load all analysis data files."""
        try:
            logger.info("Loading analysis data...")
            
            # Load combined timeline data
            timeline_path = self.data_dir / 'data' / 'combined_timeline.csv'
            if timeline_path.exists():
                self.combined_data = pd.read_csv(timeline_path)
                self.combined_data['date'] = pd.to_datetime(self.combined_data['date'])
                logger.info(f"Loaded timeline data: {len(self.combined_data)} rows")
            else:
                logger.warning(f"Timeline data not found: {timeline_path}")
            
            # Load model evaluation results
            eval_path = self.data_dir / 'models' / 'model_evaluation.csv'
            if eval_path.exists():
                eval_df = pd.read_csv(eval_path)
                self.evaluation_results = eval_df.to_dict('records')
                logger.info(f"Loaded evaluation results: {len(self.evaluation_results)} models")
            else:
                logger.warning(f"Evaluation results not found: {eval_path}")
            
            # Load source analysis
            source_path = self.data_dir / 'source_analysis' / 'source_summary.csv'
            if source_path.exists():
                source_df = pd.read_csv(source_path, index_col=0)
                self.source_analysis = source_df.to_dict('index')
                logger.info(f"Loaded source analysis: {len(self.source_analysis)} sources")
            else:
                logger.warning(f"Source analysis not found: {source_path}")
            
            # Load correlation data
            corr_path = self.data_dir / 'data' / 'lag_correlations.csv'
            if corr_path.exists():
                self.correlation_data = pd.read_csv(corr_path)
                logger.info(f"Loaded correlation data: {len(self.correlation_data)} lag points")
            else:
                logger.warning(f"Correlation data not found: {corr_path}")
            
            # Load forecast data
            forecast_path = self.data_dir / 'data' / 'forecast.csv'
            if forecast_path.exists():
                self.forecast_data = pd.read_csv(forecast_path)
                self.forecast_data['date'] = pd.to_datetime(self.forecast_data['date'])
                logger.info(f"Loaded forecast data: {len(self.forecast_data)} forecast points")
            else:
                logger.warning(f"Forecast data not found: {forecast_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_all_plots(self):
        """Create all visualization plots."""
        logger.info("=" * 60)
        logger.info("CREATING COMPREHENSIVE VISUALIZATIONS")
        logger.info("=" * 60)
        
        try:
            # Load data first
            if not self.load_data():
                logger.error("Failed to load data - cannot create plots")
                return False
            
            # Create plots
            plots_created = []
            
            # 1. Executive Dashboard
            if self.create_executive_dashboard():
                plots_created.append("Executive Dashboard")
            
            # 2. Time Series Analysis
            if self.create_time_series_analysis():
                plots_created.append("Time Series Analysis")
            
            # 3. Source Analysis
            if self.create_source_analysis():
                plots_created.append("Source Analysis")
            
            # 4. Model Performance
            if self.create_model_performance():
                plots_created.append("Model Performance")
            
            # 5. Forecast Visualization
            if self.create_forecast_visualization():
                plots_created.append("Forecast Visualization")
            
            # 6. Correlation Analysis
            if self.create_correlation_analysis():
                plots_created.append("Correlation Analysis")
            
            # 7. Data Quality Dashboard
            if self.create_data_quality_dashboard():
                plots_created.append("Data Quality Dashboard")
            
            # 8. Summary Report
            if self.create_summary_report(plots_created):
                plots_created.append("Summary Report")
            
            logger.info(f"\n✅ Successfully created {len(plots_created)} visualizations:")
            for plot in plots_created:
                logger.info(f"   📊 {plot}")
            
            logger.info(f"\n📁 All plots saved to: {self.output_dir}")
            logger.info("🌐 Open the HTML files in your browser to view interactive plots")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            return False
    
    def create_executive_dashboard(self):
        """Create executive summary dashboard."""
        try:
            logger.info("Creating executive dashboard...")
            
            if self.combined_data is None:
                logger.warning("No combined data available for dashboard")
                return False
            
            # Create subplot layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Mail Volume vs Call Volume Over Time',
                    'Data Quality Overview',
                    'Weekly Pattern Analysis',
                    'Source Performance Summary',
                    'Call Volume Distribution',
                    'Key Metrics Summary'
                ),
                specs=[
                    [{'secondary_y': True}, {'type': 'indicator'}],
                    [{'type': 'bar'}, {'type': 'bar'}],
                    [{'type': 'histogram'}, {'type': 'table'}]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.12
            )
            
            # 1. Main time series (top left)
            if 'mail_volume_total' in self.combined_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data['mail_volume_total'],
                        name='Mail Volume',
                        line=dict(color=COLORS['primary'], width=2),
                        mode='lines'
                    ),
                    row=1, col=1, secondary_y=False
                )
            
            if 'call_count' in self.combined_data.columns:
                # Separate actual and augmented data
                actual_data = self.combined_data[self.combined_data.get('data_quality', 'actual') == 'actual']
                augmented_data = self.combined_data[self.combined_data.get('data_quality', 'none') == 'augmented']
                
                # Actual calls
                fig.add_trace(
                    go.Scatter(
                        x=actual_data['date'],
                        y=actual_data['call_count'],
                        name='Calls (Actual)',
                        line=dict(color=COLORS['actual'], width=2),
                        mode='lines+markers',
                        marker=dict(size=4)
                    ),
                    row=1, col=1, secondary_y=True
                )
                
                # Augmented calls (if any)
                if len(augmented_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=augmented_data['date'],
                            y=augmented_data['call_count'],
                            name='Calls (Augmented)',
                            line=dict(color=COLORS['augmented'], width=2, dash='dot'),
                            mode='lines',
                            opacity=0.7
                        ),
                        row=1, col=1, secondary_y=True
                    )
            
            # 2. Data quality gauge (top right)
            if 'data_quality' in self.combined_data.columns:
                completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
            else:
                completeness = 100  # Assume all actual if no quality column
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=completeness,
                    delta={'reference': 90},
                    title={'text': "Data Completeness %"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': COLORS['success']},
                        'steps': [
                            {'range': [0, 50], 'color': '#ffebee'},
                            {'range': [50, 80], 'color': '#fff3e0'},
                            {'range': [80, 100], 'color': '#e8f5e9'}
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
            
            # 3. Weekly patterns (middle left)
            if 'day_of_week' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                weekly_data = self.combined_data.groupby('day_of_week')['call_count'].mean()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                fig.add_trace(
                    go.Bar(
                        x=day_names,
                        y=weekly_data.values,
                        name='Avg Daily Calls',
                        marker_color=COLORS['info'],
                        text=[f'{v:.0f}' for v in weekly_data.values],
                        textposition='outside'
                    ),
                    row=2, col=1
                )
            
            # 4. Source performance (middle right)
            if self.source_analysis:
                sources = list(self.source_analysis.keys())
                volumes = [self.source_analysis[s]['total_volume'] for s in sources]
                colors = [COLORS.get(s.upper(), COLORS['primary']) for s in sources]
                
                fig.add_trace(
                    go.Bar(
                        x=sources,
                        y=volumes,
                        name='Total Volume',
                        marker_color=colors,
                        text=[f'{v:,.0f}' for v in volumes],
                        textposition='outside'
                    ),
                    row=2, col=2
                )
            
            # 5. Call volume distribution (bottom left)
            if 'call_count' in self.combined_data.columns:
                call_data = self.combined_data[self.combined_data['call_count'] > 0]['call_count']
                
                fig.add_trace(
                    go.Histogram(
                        x=call_data,
                        name='Call Distribution',
                        marker_color=COLORS['secondary'],
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=3, col=1
                )
            
            # 6. Key metrics table (bottom right)
            if self.combined_data is not None:
                metrics_data = self._calculate_key_metrics()
                
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=['Metric', 'Value'],
                            fill_color=COLORS['primary'],
                            font=dict(color='white', size=12),
                            align='left'
                        ),
                        cells=dict(
                            values=[
                                list(metrics_data.keys()),
                                list(metrics_data.values())
                            ],
                            fill_color='white',
                            align='left',
                            font=dict(size=11)
                        )
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title={
                    'text': "📊 Executive Analytics Dashboard",
                    'x': 0.5,
                    'font': {'size': 24, 'color': COLORS['primary']}
                },
                showlegend=True,
                template='plotly_white',
                font={'family': 'Arial, sans-serif'}
            )
            
            # Set axis labels
            fig.update_yaxes(title_text="Mail Volume", secondary_y=False, row=1, col=1)
            fig.update_yaxes(title_text="Call Volume", secondary_y=True, row=1, col=1)
            fig.update_xaxes(title_text="Day of Week", row=2, col=1)
            fig.update_yaxes(title_text="Average Calls", row=2, col=1)
            fig.update_xaxes(title_text="Source", row=2, col=2)
            fig.update_yaxes(title_text="Total Volume", row=2, col=2)
            fig.update_xaxes(title_text="Call Count", row=3, col=1)
            fig.update_yaxes(title_text="Frequency", row=3, col=1)
            
            # Save dashboard
            dashboard_path = self.output_dir / 'executive_dashboard.html'
            fig.write_html(
                dashboard_path,
                include_plotlyjs=True,
                config=PLOT_CONFIG
            )
            
            logger.info(f"✅ Executive dashboard saved: {dashboard_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Executive dashboard failed: {e}")
            return False
    
    def create_time_series_analysis(self):
        """Create comprehensive time series analysis."""
        try:
            logger.info("Creating time series analysis...")
            
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    'Mail Volume and Call Volume Trends',
                    'Moving Averages and Seasonality',
                    'Correlation Analysis'
                ),
                vertical_spacing=0.1
            )
            
            # 1. Main trends
            if 'mail_volume_total' in self.combined_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data['mail_volume_total'],
                        name='Mail Volume',
                        line=dict(color=COLORS['primary'], width=2),
                        yaxis='y1'
                    ),
                    row=1, col=1
                )
            
            if 'call_count' in self.combined_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data['call_count'],
                        name='Call Volume',
                        line=dict(color=COLORS['actual'], width=2),
                        yaxis='y2'
                    ),
                    row=1, col=1
                )
            
            # 2. Moving averages
            if 'call_count' in self.combined_data.columns and len(self.combined_data) > 7:
                # 7-day moving average
                ma_7 = self.combined_data['call_count'].rolling(7, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=ma_7,
                        name='7-day MA',
                        line=dict(color=COLORS['secondary'], width=3),
                        mode='lines'
                    ),
                    row=2, col=1
                )
                
                # 30-day moving average (if enough data)
                if len(self.combined_data) > 30:
                    ma_30 = self.combined_data['call_count'].rolling(30, center=True).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=self.combined_data['date'],
                            y=ma_30,
                            name='30-day MA',
                            line=dict(color=COLORS['info'], width=3),
                            mode='lines'
                        ),
                        row=2, col=1
                    )
            
            # 3. Correlation analysis
            if self.correlation_data is not None:
                fig.add_trace(
                    go.Bar(
                        x=self.correlation_data['lag'],
                        y=self.correlation_data['correlation'],
                        name='Lag Correlation',
                        marker_color=COLORS['success'],
                        text=[f'{c:.3f}' for c in self.correlation_data['correlation']],
                        textposition='outside'
                    ),
                    row=3, col=1
                )
            
            fig.update_layout(
                height=900,
                title="📈 Time Series Analysis Dashboard",
                template='plotly_white'
            )
            
            # Save plot
            ts_path = self.output_dir / 'time_series_analysis.html'
            fig.write_html(ts_path, include_plotlyjs=True, config=PLOT_CONFIG)
            
            logger.info(f"✅ Time series analysis saved: {ts_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Time series analysis failed: {e}")
            return False
    
    def create_source_analysis(self):
        """Create source-specific analysis plots."""
        try:
            logger.info("Creating source analysis...")
            
            if self.combined_data is None:
                return False
            
            # Get mail source columns
            mail_source_cols = [col for col in self.combined_data.columns 
                               if col.startswith('mail_') and col != 'mail_volume_total']
            
            if not mail_source_cols:
                logger.warning("No mail source columns found")
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Mail Volume by Source Over Time',
                    'Source Volume Distribution',
                    'Source Performance Metrics',
                    'Daily Average by Source'
                )
            )
            
            # 1. Time series by source
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                color = COLORS.get(source_name.upper(), f'hsl({i*60}, 70%, 50%)')
                
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data[col],
                        name=source_name,
                        line=dict(color=color, width=2),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            # 2. Box plots for distribution
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                source_data = self.combined_data[self.combined_data[col] > 0][col]
                
                if len(source_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=source_data,
                            name=source_name,
                            marker_color=COLORS.get(source_name.upper(), f'hsl({i*60}, 70%, 50%)')
                        ),
                        row=1, col=2
                    )
            
            # 3. Performance metrics
            if self.source_analysis:
                sources = list(self.source_analysis.keys())
                volumes = [self.source_analysis[s]['total_volume'] for s in sources]
                colors = [COLORS.get(s.upper(), COLORS['primary']) for s in sources]
                
                fig.add_trace(
                    go.Bar(
                        x=sources,
                        y=volumes,
                        name='Total Volume',
                        marker_color=colors,
                        text=[f'{v:,.0f}' for v in volumes],
                        textposition='outside'
                    ),
                    row=2, col=1
                )
            
            # 4. Daily averages
            source_averages = {}
            for col in mail_source_cols:
                source_name = col.replace('mail_', '').title()
                avg_volume = self.combined_data[col].mean()
                source_averages[source_name] = avg_volume
            
            sources = list(source_averages.keys())
            averages = list(source_averages.values())
            colors = [COLORS.get(s.upper(), COLORS['secondary']) for s in sources]
            
            fig.add_trace(
                go.Bar(
                    x=sources,
                    y=averages,
                    name='Daily Average',
                    marker_color=colors,
                    text=[f'{v:.0f}' for v in averages],
                    textposition='outside'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title="📊 Source Analysis Dashboard",
                template='plotly_white'
            )
            
            # Save plot
            source_path = self.output_dir / 'source_analysis.html'
            fig.write_html(source_path, include_plotlyjs=True, config=PLOT_CONFIG)
            
            logger.info(f"✅ Source analysis saved: {source_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Source analysis failed: {e}")
            return False
    
    def create_model_performance(self):
        """Create model performance comparison."""
        try:
            logger.info("Creating model performance analysis...")
            
            if not self.evaluation_results:
                logger.warning("No evaluation results available")
                return False
            
            # Get top models
            models = self.evaluation_results[:8]  # Top 8 models
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Mean Absolute Error (Lower is Better)',
                    'R-Squared Score (Higher is Better)',
                    'MAPE % (Lower is Better)',
                    'Model Type Distribution'
                )
            )
            
            # Extract data
            names = [m['name'] for m in models]
            mae_vals = [m['mae'] for m in models]
            r2_vals = [m['r2'] for m in models]
            mape_vals = [m['mape'] for m in models]
            types = [m.get('type', 'unknown') for m in models]
            
            # 1. MAE comparison
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=mae_vals,
                    name='MAE',
                    marker_color=COLORS['danger'],
                    text=[f'{v:.1f}' for v in mae_vals],
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # 2. R² comparison
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=r2_vals,
                    name='R²',
                    marker_color=COLORS['success'],
                    text=[f'{v:.3f}' for v in r2_vals],
                    textposition='outside'
                ),
                row=1, col=2
            )
            
            # 3. MAPE comparison
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=mape_vals,
                    name='MAPE %',
                    marker_color=COLORS['warning'],
                    text=[f'{v:.1f}%' for v in mape_vals],
                    textposition='outside'
                ),
                row=2, col=1
            )
            
            # 4. Model type pie chart
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            fig.add_trace(
                go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    name="Model Types",
                    marker_colors=[COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info']]
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title="🤖 Model Performance Dashboard",
                template='plotly_white'
            )
            
            # Rotate x-axis labels for readability
            fig.update_xaxes(tickangle=45)
            
            # Save plot
            model_path = self.output_dir / 'model_performance.html'
            fig.write_html(model_path, include_plotlyjs=True, config=PLOT_CONFIG)
            
            logger.info(f"✅ Model performance saved: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model performance failed: {e}")
            return False
    
    def create_forecast_visualization(self):
        """Create forecast visualization."""
        try:
            logger.info("Creating forecast visualization...")
            
            if self.forecast_data is None or self.combined_data is None:
                logger.warning("No forecast data available")
                return False
            
            fig = go.Figure()
            
            # Historical data (last 60 days)
            recent_data = self.combined_data.tail(60)
            
            # Actual historical data
            actual_recent = recent_data[recent_data.get('data_quality', 'actual') == 'actual']
            if len(actual_recent) > 0:
                fig.add_trace(go.Scatter(
                    x=actual_recent['date'],
                    y=actual_recent['call_count'],
                    mode='lines+markers',
                    name='Historical (Actual)',
                    line=dict(color=COLORS['actual'], width=2),
                    marker=dict(size=5)
                ))
            
            # Augmented historical data
            augmented_recent = recent_data[recent_data.get('data_quality', 'none') == 'augmented']
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
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=self.forecast_data['date'],
                y=self.forecast_data['predicted_calls'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=COLORS['forecast'], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Confidence interval
            if 'upper_bound' in self.forecast_data.columns and 'lower_bound' in self.forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.forecast_data['date'].tolist() + self.forecast_data['date'].tolist()[::-1],
                    y=self.forecast_data['upper_bound'].tolist() + self.forecast_data['lower_bound'].tolist()[::-1],fill='toself',
                    fillcolor=COLORS['confidence'],
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
            
            # Add vertical line at forecast start
            last_historical_date = self.combined_data['date'].max()
            fig.add_vline(
                x=last_historical_date,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top"
            )
            
            # Calculate average forecast for annotation
            avg_forecast = self.forecast_data['predicted_calls'].mean()
            
            fig.update_layout(
                title={
                    'text': f"🔮 Call Volume Forecast<br><sub>Average forecast: {avg_forecast:.0f} calls/day</sub>",
                    'x': 0.5,
                    'font': {'size': 18}
                },
                xaxis_title='Date',
                yaxis_title='Call Volume',
                height=600,
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Save forecast
            forecast_path = self.output_dir / 'forecast_visualization.html'
            fig.write_html(forecast_path, include_plotlyjs=True, config=PLOT_CONFIG)
            
            logger.info(f"✅ Forecast visualization saved: {forecast_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Forecast visualization failed: {e}")
            return False
    
    def create_correlation_analysis(self):
        """Create correlation analysis plots."""
        try:
            logger.info("Creating correlation analysis...")
            
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Mail vs Calls Scatter Plot',
                    'Lag Correlation Analysis',
                    'Source vs Calls Correlation',
                    'Daily Pattern Correlation'
                )
            )
            
            # 1. Scatter plot: Mail vs Calls
            if 'mail_volume_total' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                valid_data = self.combined_data[
                    (self.combined_data['mail_volume_total'] > 0) & 
                    (self.combined_data['call_count'] > 0)
                ]
                
                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data['mail_volume_total'],
                            y=valid_data['call_count'],
                            mode='markers',
                            name='Mail vs Calls',
                            marker=dict(
                                color=COLORS['primary'],
                                size=8,
                                opacity=0.6,
                                line=dict(width=1, color='white')
                            )
                        ),
                        row=1, col=1
                    )
                    
                    # Add trend line
                    z = np.polyfit(valid_data['mail_volume_total'], valid_data['call_count'], 1)
                    p = np.poly1d(z)
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data['mail_volume_total'],
                            y=p(valid_data['mail_volume_total']),
                            mode='lines',
                            name='Trend Line',
                            line=dict(color=COLORS['danger'], width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
            
            # 2. Lag correlation
            if self.correlation_data is not None:
                fig.add_trace(
                    go.Bar(
                        x=self.correlation_data['lag'],
                        y=self.correlation_data['correlation'],
                        name='Lag Correlation',
                        marker_color=COLORS['success'],
                        text=[f'{c:.3f}' for c in self.correlation_data['correlation']],
                        textposition='outside'
                    ),
                    row=1, col=2
                )
            
            # 3. Source correlations (if available)
            mail_source_cols = [col for col in self.combined_data.columns 
                               if col.startswith('mail_') and col != 'mail_volume_total']
            
            if mail_source_cols and 'call_count' in self.combined_data.columns:
                source_corrs = {}
                for col in mail_source_cols:
                    source_name = col.replace('mail_', '').title()
                    valid_mask = (self.combined_data[col] > 0) & (self.combined_data['call_count'] > 0)
                    if valid_mask.sum() > 10:
                        corr = self.combined_data.loc[valid_mask, col].corr(
                            self.combined_data.loc[valid_mask, 'call_count']
                        )
                        source_corrs[source_name] = corr
                
                if source_corrs:
                    sources = list(source_corrs.keys())
                    correlations = list(source_corrs.values())
                    colors = [COLORS.get(s.upper(), COLORS['info']) for s in sources]
                    
                    fig.add_trace(
                        go.Bar(
                            x=sources,
                            y=correlations,
                            name='Source Correlation',
                            marker_color=colors,
                            text=[f'{c:.3f}' for c in correlations],
                            textposition='outside'
                        ),
                        row=2, col=1
                    )
            
            # 4. Day-of-week correlation
            if 'day_of_week' in self.combined_data.columns:
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                # Calculate average calls by day of week
                if 'call_count' in self.combined_data.columns:
                    dow_calls = self.combined_data.groupby('day_of_week')['call_count'].mean()
                    
                    fig.add_trace(
                        go.Bar(
                            x=dow_names,
                            y=dow_calls.values,
                            name='Avg Calls by Day',
                            marker_color=COLORS['warning'],
                            text=[f'{v:.0f}' for v in dow_calls.values],
                            textposition='outside'
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=800,
                title="🔗 Correlation Analysis Dashboard",
                template='plotly_white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Mail Volume", row=1, col=1)
            fig.update_yaxes(title_text="Call Count", row=1, col=1)
            fig.update_xaxes(title_text="Lag (days)", row=1, col=2)
            fig.update_yaxes(title_text="Correlation", row=1, col=2)
            fig.update_xaxes(title_text="Source", row=2, col=1)
            fig.update_yaxes(title_text="Correlation", row=2, col=1)
            fig.update_xaxes(title_text="Day of Week", row=2, col=2)
            fig.update_yaxes(title_text="Average Calls", row=2, col=2)
            
            # Save plot
            corr_path = self.output_dir / 'correlation_analysis.html'
            fig.write_html(corr_path, include_plotlyjs=True, config=PLOT_CONFIG)
            
            logger.info(f"✅ Correlation analysis saved: {corr_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Correlation analysis failed: {e}")
            return False
    
    def create_data_quality_dashboard(self):
        """Create data quality assessment dashboard."""
        try:
            logger.info("Creating data quality dashboard...")
            
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Data Quality Over Time',
                    'Missing Data Analysis',
                    'Data Completeness by Source',
                    'Quality Metrics Summary'
                ),
                specs=[
                    [{'type': 'scatter'}, {'type': 'bar'}],
                    [{'type': 'bar'}, {'type': 'indicator'}]
                ]
            )
            
            # 1. Data quality over time
            if 'data_quality' in self.combined_data.columns:
                quality_over_time = self.combined_data.groupby(['date', 'data_quality']).size().unstack(fill_value=0)
                
                if 'actual' in quality_over_time.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=quality_over_time.index,
                            y=quality_over_time['actual'],
                            name='Actual Data',
                            line=dict(color=COLORS['actual'], width=2),
                            mode='lines',
                            stackgroup='one'
                        ),
                        row=1, col=1
                    )
                
                if 'augmented' in quality_over_time.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=quality_over_time.index,
                            y=quality_over_time['augmented'],
                            name='Augmented Data',
                            line=dict(color=COLORS['augmented'], width=2),
                            mode='lines',
                            stackgroup='one'
                        ),
                        row=1, col=1
                    )
            
            # 2. Missing data analysis
            missing_data = {}
            for col in ['mail_volume_total', 'call_count']:
                if col in self.combined_data.columns:
                    missing_pct = (self.combined_data[col] == 0).mean() * 100
                    missing_data[col.replace('_', ' ').title()] = missing_pct
            
            if missing_data:
                fig.add_trace(
                    go.Bar(
                        x=list(missing_data.keys()),
                        y=list(missing_data.values()),
                        name='Missing %',
                        marker_color=COLORS['danger'],
                        text=[f'{v:.1f}%' for v in missing_data.values()],
                        textposition='outside'
                    ),
                    row=1, col=2
                )
            
            # 3. Completeness by source
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
                
                fig.add_trace(
                    go.Bar(
                        x=sources,
                        y=completeness,
                        name='Completeness %',
                        marker_color=colors,
                        text=[f'{v:.1f}%' for v in completeness],
                        textposition='outside'
                    ),
                    row=2, col=1
                )
            
            # 4. Overall quality score
            overall_quality = 100
            if 'data_quality' in self.combined_data.columns:
                overall_quality = (self.combined_data['data_quality'] == 'actual').mean() * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_quality,
                    delta={'reference': 90},
                    title={'text': "Overall Data Quality Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': COLORS['success']},
                        'steps': [
                            {'range': [0, 50], 'color': '#ffebee'},
                            {'range': [50, 80], 'color': '#fff3e0'},
                            {'range': [80, 100], 'color': '#e8f5e9'}
                        ],
                        'threshold': {
                            'line': {'color': COLORS['warning'], 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title="🔍 Data Quality Dashboard",
                template='plotly_white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Record Count", row=1, col=1)
            fig.update_xaxes(title_text="Data Type", row=1, col=2)
            fig.update_yaxes(title_text="Missing %", row=1, col=2)
            fig.update_xaxes(title_text="Source", row=2, col=1)
            fig.update_yaxes(title_text="Completeness %", row=2, col=1)
            
            # Save plot
            quality_path = self.output_dir / 'data_quality_dashboard.html'
            fig.write_html(quality_path, include_plotlyjs=True, config=PLOT_CONFIG)
            
            logger.info(f"✅ Data quality dashboard saved: {quality_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data quality dashboard failed: {e}")
            return False
    
    def create_summary_report(self, plots_created):
        """Create HTML summary report with links to all plots."""
        try:
            logger.info("Creating summary report...")
            
            # Calculate summary statistics
            stats = self._calculate_summary_statistics()
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>📊 Analytics Visualization Summary</title>
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 15px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }}
                    
                    .header {{
                        background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']});
                        color: white;
                        padding: 40px;
                        text-align: center;
                    }}
                    
                    .header h1 {{
                        font-size: 2.5em;
                        margin-bottom: 10px;
                        font-weight: 300;
                    }}
                    
                    .subtitle {{
                        font-size: 1.2em;
                        opacity: 0.9;
                    }}
                    
                    .content {{
                        padding: 40px;
                    }}
                    
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 30px 0;
                    }}
                    
                    .stat-card {{
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        border-left: 4px solid {COLORS['primary']};
                        transition: transform 0.2s;
                    }}
                    
                    .stat-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                    }}
                    
                    .stat-value {{
                        font-size: 2.5em;
                        font-weight: bold;
                        color: {COLORS['primary']};
                        margin: 10px 0;
                    }}
                    
                    .stat-label {{
                        color: #666;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        font-size: 0.9em;
                    }}
                    
                    .plots-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        margin: 40px 0;
                    }}
                    
                    .plot-card {{
                        background: white;
                        border: 1px solid #e9ecef;
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        transition: all 0.3s;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    }}
                    
                    .plot-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                        border-color: {COLORS['primary']};
                    }}
                    
                    .plot-icon {{
                        font-size: 3em;
                        margin-bottom: 15px;
                    }}
                    
                    .plot-title {{
                        font-size: 1.3em;
                        font-weight: 600;
                        margin-bottom: 10px;
                        color: {COLORS['primary']};
                    }}
                    
                    .plot-description {{
                        color: #666;
                        margin-bottom: 20px;
                        font-size: 0.95em;
                    }}
                    
                    .plot-link {{
                        display: inline-block;
                        background: {COLORS['primary']};
                        color: white;
                        padding: 10px 20px;
                        text-decoration: none;
                        border-radius: 5px;
                        transition: all 0.3s;
                        font-weight: 500;
                    }}
                    
                    .plot-link:hover {{
                        background: {COLORS['secondary']};
                        transform: translateY(-2px);
                    }}
                    
                    .section {{
                        margin: 40px 0;
                        padding: 30px;
                        background: #f8f9fa;
                        border-radius: 10px;
                    }}
                    
                    .section h2 {{
                        color: {COLORS['primary']};
                        margin-bottom: 20px;
                        font-size: 1.8em;
                        border-bottom: 2px solid {COLORS['primary']};
                        padding-bottom: 10px;
                    }}
                    
                    .footer {{
                        background: #343a40;
                        color: white;
                        text-align: center;
                        padding: 20px;
                    }}
                    
                    .timestamp {{
                        opacity: 0.8;
                        font-size: 0.9em;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Analytics Visualization Dashboard</h1>
                        <div class="subtitle">Comprehensive Mail-Call Analytics Results</div>
                        <div class="timestamp">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                    </div>
                    
                    <div class="content">
                        <div class="section">
                            <h2>📈 Key Statistics</h2>
                            <div class="stats-grid">
                                {self._generate_stats_cards(stats)}
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>🎯 Interactive Visualizations</h2>
                            <p>Click on any visualization below to open the interactive plot in your browser:</p>
                            <div class="plots-grid">
                                {self._generate_plot_cards()}
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>ℹ️ How to Use</h2>
                            <ul>
                                <li><strong>Interactive Plots:</strong> All visualizations are interactive - hover, zoom, and click to explore</li>
                                <li><strong>Export Options:</strong> Use the toolbar in each plot to download as PNG or PDF</li>
                                <li><strong>Data Filtering:</strong> Click legend items to show/hide data series</li>
                                <li><strong>Mobile Friendly:</strong> All plots are responsive and work on mobile devices</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>🚀 Enhanced Mail-Call Analytics System | Generated with ❤️ using Plotly</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save report
            report_path = self.output_dir / 'visualization_summary.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ Summary report saved: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Summary report failed: {e}")
            return False
    
    def _calculate_key_metrics(self):
        """Calculate key metrics for dashboard."""
        try:
            metrics = {}
            
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
                metrics['Model Accuracy'] = f"{max(0, 100 - best_model['mape']):.1f}%"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_summary_statistics(self):
        """Calculate comprehensive summary statistics."""
        try:
            stats = {}
            
            if self.combined_data is not None:
                stats['total_days'] = len(self.combined_data)
                stats['date_range'] = f"{self.combined_data['date'].min().strftime('%Y-%m-%d')} to {self.combined_data['date'].max().strftime('%Y-%m-%d')}"
                
                if 'mail_volume_total' in self.combined_data.columns:
                    stats['total_mail'] = int(self.combined_data['mail_volume_total'].sum())
                    stats['avg_daily_mail'] = int(self.combined_data['mail_volume_total'].mean())
                
                if 'call_count' in self.combined_data.columns:
                    stats['total_calls'] = int(self.combined_data['call_count'].sum())
                    stats['avg_daily_calls'] = int(self.combined_data['call_count'].mean())
                
                if 'data_quality' in self.combined_data.columns:
                    stats['data_quality'] = f"{(self.combined_data['data_quality'] == 'actual').mean() * 100:.1f}%"
            
            if self.evaluation_results:
                stats['models_tested'] = len(self.evaluation_results)
                best_model = self.evaluation_results[0]
                stats['best_model'] = best_model['name']
                stats['best_mae'] = f"{best_model['mae']:.1f}"
            
            if self.source_analysis:
                stats['sources_analyzed'] = len(self.source_analysis)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
    
    def _generate_stats_cards(self, stats):
        """Generate HTML for statistics cards."""
        cards_html = ""
        
        stat_configs = {
            'total_days': ('📅', 'Total Days Analyzed'),
            'total_mail': ('📧', 'Total Mail Volume'),
            'total_calls': ('📞', 'Total Call Volume'),
            'avg_daily_calls': ('📊', 'Average Daily Calls'),
            'data_quality': ('✅', 'Data Quality'),
            'models_tested': ('🤖', 'Models Tested'),
            'best_mae': ('🎯', 'Best Model MAE'),
            'sources_analyzed': ('🔍', 'Sources Analyzed')
        }
        
        for key, (icon, label) in stat_configs.items():
            if key in stats:
                value = stats[key]
                cards_html += f"""
                <div class="stat-card">
                    <div style="font-size: 2em;">{icon}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """
        
        return cards_html
    
    def _generate_plot_cards(self):
        """Generate HTML for plot cards."""
        plot_configs = [
            ('executive_dashboard.html', '📊', 'Executive Dashboard', 'High-level overview with key metrics and trends'),
            ('time_series_analysis.html', '📈', 'Time Series Analysis', 'Detailed temporal patterns and moving averages'),
            ('source_analysis.html', '🎯', 'Source Analysis', 'Performance breakdown by mail source'),
            ('model_performance.html', '🤖', 'Model Performance', 'Comparison of prediction model accuracy'),
            ('forecast_visualization.html', '🔮', 'Forecast Visualization', 'Future call volume predictions with confidence intervals'),
            ('correlation_analysis.html', '🔗', 'Correlation Analysis', 'Relationship analysis between mail and calls'),
            ('data_quality_dashboard.html', '🔍', 'Data Quality Dashboard', 'Assessment of data completeness and quality')
        ]
        
        cards_html = ""
        for filename, icon, title, description in plot_configs:
            file_path = self.output_dir / filename
            if file_path.exists():
                cards_html += f"""
                <div class="plot-card">
                    <div class="plot-icon">{icon}</div>
                    <div class="plot-title">{title}</div>
                    <div class="plot-description">{description}</div>
                    <a href="{filename}" class="plot-link" target="_blank">View Interactive Plot</a>
                </div>
                """
        
        return cards_html

# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    """Main function for standalone execution."""
    print("🎨 Enhanced Analytics Plotting System")
    print("=" * 50)
    
    try:
        # Initialize plotter
        plotter = AnalyticsPlotter()
        
        # Create all plots
        success = plotter.create_all_plots()
        
        if success:
            print("\n🎉 SUCCESS! All visualizations created successfully!")
            print(f"\n📁 Results saved to: {plotter.output_dir}")
            print("\n🌐 Open 'visualization_summary.html' to access all plots")
            
            # Try to open the summary in default browser
            try:
                import webbrowser
                summary_path = plotter.output_dir / 'visualization_summary.html'
                if summary_path.exists():
                    webbrowser.open(summary_path.as_uri())
                    print("🚀 Opening summary in your default browser...")
            except Exception as e:
                print(f"Note: Could not auto-open browser: {e}")
        else:
            print("\n❌ Some visualizations failed to create. Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
