#!/usr/bin/env python
"""
robust_plotting.py
==================
Ultra-robust plotting module for Enhanced Mail-Call Analytics
Includes multiple fallbacks for browser opening and PNG backup generation

Usage:
    python robust_plotting.py

Features:
- Multiple browser opening methods with fallbacks
- Automatic PNG generation if HTML fails
- Cross-platform compatibility
- Comprehensive error handling
- Works in any Python environment

Requirements:
    pip install plotly pandas numpy kaleido pillow
"""

import os
import sys
import platform
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
import logging
import traceback
import warnings
warnings.filterwarnings('ignore')

# Essential imports
try:
    import pandas as pd
    import numpy as np
    print("✅ NumPy and Pandas loaded")
except ImportError as e:
    print(f"❌ Missing core libraries: {e}")
    print("💡 Run: pip install pandas numpy")
    sys.exit(1)

# Plotly imports with configuration
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    
    # Force browser/file output mode
    pio.renderers.default = "browser"
    os.environ["PLOTLY_RENDERER"] = "browser"
    
    # Disable notebook mode
    import plotly.offline
    plotly.offline.init_notebook_mode(connected=False)
    
    PLOTLY_AVAILABLE = True
    print("✅ Plotly configured for standalone execution")
    
except ImportError as e:
    print(f"❌ Plotly not available: {e}")
    print("💡 Run: pip install plotly")
    PLOTLY_AVAILABLE = False

# Image export capability
try:
    import kaleido
    PNG_EXPORT_AVAILABLE = True
    print("✅ PNG export available (kaleido)")
except ImportError:
    try:
        from PIL import Image
        PNG_EXPORT_AVAILABLE = True
        print("✅ PNG export available (pillow)")
    except ImportError:
        PNG_EXPORT_AVAILABLE = False
        print("⚠️ PNG export not available (install kaleido or pillow)")

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
    'info': '#17a2b8',
    'dark': '#343a40',
    'light': '#f8f9fa'
}

# Default paths
DEFAULT_DATA_DIR = Path("enhanced_analysis_results")
DEFAULT_OUTPUT_DIR = Path("enhanced_analysis_results/plots")

# Plot configuration
PLOT_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'responsive': True,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'plot',
        'height': 800,
        'width': 1200,
        'scale': 1
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger():
    """Setup comprehensive logging."""
    logger = logging.getLogger('robust_plotting')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Simple formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler('robust_plotting.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Skip file logging if it fails
    
    return logger

logger = setup_logger()

# ============================================================================
# BROWSER OPENING WITH MULTIPLE FALLBACKS
# ============================================================================

class BrowserOpener:
    """Comprehensive browser opening with multiple fallback methods."""
    
    @staticmethod
    def open_file(file_path):
        """Try every possible method to open file in browser."""
        file_path = Path(file_path).resolve()
        
        print(f"\n🌐 Attempting to open: {file_path.name}")
        print(f"📁 Full path: {file_path}")
        
        # Method 1: Default webbrowser
        if BrowserOpener._try_default_webbrowser(file_path):
            return True
        
        # Method 2: OS-specific system commands
        if BrowserOpener._try_system_open(file_path):
            return True
        
        # Method 3: Specific browser executables
        if BrowserOpener._try_browser_executables(file_path):
            return True
        
        # Method 4: Manual URL construction
        if BrowserOpener._try_manual_url(file_path):
            return True
        
        # Method 5: Environment-specific methods
        if BrowserOpener._try_environment_specific(file_path):
            return True
        
        # All methods failed
        print("❌ All browser opening methods failed")
        BrowserOpener._print_manual_instructions(file_path)
        return False
    
    @staticmethod
    def _try_default_webbrowser(file_path):
        """Try default webbrowser module."""
        try:
            import webbrowser
            
            # Try as_uri() method
            try:
                url = file_path.as_uri()
                webbrowser.open(url)
                print(f"✅ Opened with webbrowser.open() using as_uri()")
                return True
            except Exception as e:
                logger.debug(f"as_uri() failed: {e}")
            
            # Try string path
            try:
                webbrowser.open(str(file_path))
                print(f"✅ Opened with webbrowser.open() using string path")
                return True
            except Exception as e:
                logger.debug(f"string path failed: {e}")
            
            # Try file:// URL
            try:
                file_url = f"file://{file_path.as_posix()}"
                webbrowser.open(file_url)
                print(f"✅ Opened with webbrowser.open() using file:// URL")
                return True
            except Exception as e:
                logger.debug(f"file:// URL failed: {e}")
                
        except Exception as e:
            logger.debug(f"Default webbrowser failed: {e}")
        
        return False
    
    @staticmethod
    def _try_system_open(file_path):
        """Try OS-specific system commands."""
        try:
            system = platform.system().lower()
            
            if system == 'windows':
                os.startfile(str(file_path))
                print("✅ Opened with Windows os.startfile()")
                return True
            elif system == 'darwin':  # macOS
                subprocess.run(['open', str(file_path)], check=True)
                print("✅ Opened with macOS 'open' command")
                return True
            elif system == 'linux':
                subprocess.run(['xdg-open', str(file_path)], check=True)
                print("✅ Opened with Linux 'xdg-open' command")
                return True
                
        except Exception as e:
            logger.debug(f"System open failed: {e}")
        
        return False
    
    @staticmethod
    def _try_browser_executables(file_path):
        """Try launching specific browser executables."""
        browsers = [
            # Chrome variants
            'google-chrome',
            'chrome',
            'chromium',
            'google-chrome-stable',
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            'C:/Program Files/Google/Chrome/Application/chrome.exe',
            'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
            
            # Firefox variants
            'firefox',
            'mozilla-firefox',
            '/Applications/Firefox.app/Contents/MacOS/firefox',
            'C:/Program Files/Mozilla Firefox/firefox.exe',
            'C:/Program Files (x86)/Mozilla Firefox/firefox.exe',
            
            # Edge variants
            'microsoft-edge',
            'msedge',
            'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
            
            # Safari
            '/Applications/Safari.app/Contents/MacOS/Safari'
        ]
        
        for browser in browsers:
            try:
                if os.path.exists(browser):
                    subprocess.run([browser, str(file_path)], check=True)
                    print(f"✅ Opened with {browser}")
                    return True
                else:
                    # Try as command name
                    subprocess.run([browser, str(file_path)], check=True, 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"✅ Opened with {browser} command")
                    return True
            except Exception:
                continue
        
        return False
    
    @staticmethod
    def _try_manual_url(file_path):
        """Try manual URL construction methods."""
        try:
            import webbrowser
            
            # Windows-style file URL
            if platform.system().lower() == 'windows':
                url = f"file:///{file_path.as_posix()}"
            else:
                url = f"file://{file_path}"
            
            webbrowser.open(url)
            print(f"✅ Opened with manual URL construction: {url}")
            return True
            
        except Exception as e:
            logger.debug(f"Manual URL failed: {e}")
        
        return False
    
    @staticmethod
    def _try_environment_specific(file_path):
        """Try environment-specific methods."""
        try:
            # Check if running in specific environments
            
            # WSL (Windows Subsystem for Linux)
            if 'microsoft' in platform.uname().release.lower():
                subprocess.run(['cmd.exe', '/c', 'start', str(file_path)], check=True)
                print("✅ Opened with WSL cmd.exe start")
                return True
            
            # Jupyter environment
            if 'JUPYTER_SERVER_ROOT' in os.environ:
                # Try jupyter-specific opening
                try:
                    from IPython.display import HTML, display
                    display(HTML(f'<a href="{file_path.name}" target="_blank">Open Plot</a>'))
                    print("✅ Created Jupyter link")
                    return True
                except:
                    pass
            
            # Google Colab
            if 'COLAB_GPU' in os.environ:
                print("📋 Colab detected - download and open file manually")
                return False
                
        except Exception as e:
            logger.debug(f"Environment-specific methods failed: {e}")
        
        return False
    
    @staticmethod
    def _print_manual_instructions(file_path):
        """Print manual opening instructions."""
        print("\n" + "="*60)
        print("🛠️ MANUAL OPENING REQUIRED")
        print("="*60)
        print("Copy one of these URLs to your browser:")
        print(f"1. file://{file_path}")
        print(f"2. file:///{file_path.as_posix()}")
        print(f"\nOr navigate to this folder and double-click the file:")
        print(f"📁 {file_path.parent}")
        print(f"📄 {file_path.name}")
        print("="*60)

# ============================================================================
# PNG FALLBACK GENERATOR
# ============================================================================

class PNGFallback:
    """Generate PNG images as fallback when HTML fails."""
    
    @staticmethod
    def save_figure_as_png(fig, output_path, width=1200, height=800):
        """Save plotly figure as PNG with multiple methods."""
        output_path = Path(output_path)
        png_path = output_path.with_suffix('.png')
        
        print(f"💾 Creating PNG fallback: {png_path.name}")
        
        # Method 1: Kaleido (recommended)
        if PNGFallback._try_kaleido(fig, png_path, width, height):
            return png_path
        
        # Method 2: Plotly built-in
        if PNGFallback._try_plotly_builtin(fig, png_path, width, height):
            return png_path
        
        # Method 3: HTML to image conversion
        if PNGFallback._try_html_conversion(fig, png_path, width, height):
            return png_path
        
        print("❌ All PNG generation methods failed")
        return None
    
    @staticmethod
    def _try_kaleido(fig, png_path, width, height):
        """Try PNG export using kaleido."""
        try:
            fig.write_image(png_path, width=width, height=height, engine="kaleido")
            print("✅ PNG created with kaleido")
            return True
        except Exception as e:
            logger.debug(f"Kaleido PNG failed: {e}")
            return False
    
    @staticmethod
    def _try_plotly_builtin(fig, png_path, width, height):
        """Try PNG export using plotly built-in methods."""
        try:
            fig.write_image(png_path, width=width, height=height)
            print("✅ PNG created with plotly built-in")
            return True
        except Exception as e:
            logger.debug(f"Plotly built-in PNG failed: {e}")
            return False
    
    @staticmethod
    def _try_html_conversion(fig, png_path, width, height):
        """Try HTML to PNG conversion."""
        try:
            # Save as HTML first
            html_path = png_path.with_suffix('.html')
            fig.write_html(html_path, include_plotlyjs=True)
            
            # Try to convert HTML to PNG using various methods
            # This is a simplified version - could be expanded
            print("⚠️ HTML created, but PNG conversion not implemented")
            return False
            
        except Exception as e:
            logger.debug(f"HTML conversion failed: {e}")
            return False

# ============================================================================
# MAIN PLOTTER CLASS
# ============================================================================

class RobustAnalyticsPlotter:
    """Ultra-robust analytics plotter with comprehensive fallbacks."""
    
    def __init__(self, data_dir=None, output_dir=None):
        """Initialize plotter with comprehensive setup."""
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
        
        logger.info(f"🚀 Robust plotter initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📊 Output directory: {self.output_dir}")
        
        # Validate environment
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate that we have everything needed."""
        issues = []
        
        if not PLOTLY_AVAILABLE:
            issues.append("Plotly not available")
        
        if not self.data_dir.exists():
            issues.append(f"Data directory not found: {self.data_dir}")
        
        if issues:
            print("⚠️ Environment issues detected:")
            for issue in issues:
                print(f"   - {issue}")
            
            if not PLOTLY_AVAILABLE:
                print("💡 Run: pip install plotly kaleido")
            
            if not self.data_dir.exists():
                print(f"💡 Make sure analysis results exist in: {self.data_dir}")
    
    def load_all_data(self):
        """Load all available analysis data with robust error handling."""
        logger.info("📂 Loading analysis data...")
        
        success_count = 0
        
        # Load combined timeline data
        if self._load_timeline_data():
            success_count += 1
        
        # Load evaluation results
        if self._load_evaluation_data():
            success_count += 1
        
        # Load source analysis
        if self._load_source_data():
            success_count += 1
        
        # Load correlation data
        if self._load_correlation_data():
            success_count += 1
        
        # Load forecast data
        if self._load_forecast_data():
            success_count += 1
        
        logger.info(f"📊 Loaded {success_count}/5 data sources")
        return success_count > 0
    
    def _load_timeline_data(self):
        """Load combined timeline data."""
        try:
            timeline_path = self.data_dir / 'data' / 'combined_timeline.csv'
            if timeline_path.exists():
                self.combined_data = pd.read_csv(timeline_path)
                self.combined_data['date'] = pd.to_datetime(self.combined_data['date'])
                logger.info(f"✅ Timeline data: {len(self.combined_data)} rows")
                return True
            else:
                # Try alternative locations
                for alt_path in ['combined_timeline.csv', 'data/timeline.csv']:
                    alt_file = self.data_dir / alt_path
                    if alt_file.exists():
                        self.combined_data = pd.read_csv(alt_file)
                        self.combined_data['date'] = pd.to_datetime(self.combined_data['date'])
                        logger.info(f"✅ Timeline data found at: {alt_path}")
                        return True
                
                logger.warning(f"⚠️ Timeline data not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Timeline data loading failed: {e}")
            return False
    
    def _load_evaluation_data(self):
        """Load model evaluation results."""
        try:
            eval_path = self.data_dir / 'models' / 'model_evaluation.csv'
            if eval_path.exists():
                eval_df = pd.read_csv(eval_path)
                self.evaluation_results = eval_df.to_dict('records')
                logger.info(f"✅ Evaluation data: {len(self.evaluation_results)} models")
                return True
            else:
                logger.warning("⚠️ Evaluation data not found")
                return False
        except Exception as e:
            logger.error(f"❌ Evaluation data loading failed: {e}")
            return False
    
    def _load_source_data(self):
        """Load source analysis data."""
        try:
            source_path = self.data_dir / 'source_analysis' / 'source_summary.csv'
            if source_path.exists():
                source_df = pd.read_csv(source_path, index_col=0)
                self.source_analysis = source_df.to_dict('index')
                logger.info(f"✅ Source data: {len(self.source_analysis)} sources")
                return True
            else:
                logger.warning("⚠️ Source data not found")
                return False
        except Exception as e:
            logger.error(f"❌ Source data loading failed: {e}")
            return False
    
    def _load_correlation_data(self):
        """Load correlation analysis data."""
        try:
            corr_path = self.data_dir / 'data' / 'lag_correlations.csv'
            if corr_path.exists():
                self.correlation_data = pd.read_csv(corr_path)
                logger.info(f"✅ Correlation data: {len(self.correlation_data)} points")
                return True
            else:
                logger.warning("⚠️ Correlation data not found")
                return False
        except Exception as e:
            logger.error(f"❌ Correlation data loading failed: {e}")
            return False
    
    def _load_forecast_data(self):
        """Load forecast data."""
        try:
            forecast_path = self.data_dir / 'data' / 'forecast.csv'
            if forecast_path.exists():
                self.forecast_data = pd.read_csv(forecast_path)
                self.forecast_data['date'] = pd.to_datetime(self.forecast_data['date'])
                logger.info(f"✅ Forecast data: {len(self.forecast_data)} points")
                return True
            else:
                logger.warning("⚠️ Forecast data not found")
                return False
        except Exception as e:
            logger.error(f"❌ Forecast data loading failed: {e}")
            return False
    
    def create_all_visualizations(self):
        """Create all visualizations with comprehensive error handling."""
        logger.info("🎨 Creating comprehensive visualizations...")
        
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly not available - cannot create visualizations")
            return False
        
        # Load data first
        if not self.load_all_data():
            print("❌ No data loaded - creating sample visualizations instead")
            return self._create_sample_visualizations()
        
        # Create visualizations
        visualizations = [
            ("Executive Dashboard", self._create_executive_dashboard),
            ("Time Series Analysis", self._create_time_series_plots),
            ("Source Analysis", self._create_source_plots),
            ("Model Performance", self._create_model_plots),
            ("Correlation Analysis", self._create_correlation_plots),
            ("Data Quality Dashboard", self._create_quality_plots),
            ("Forecast Visualization", self._create_forecast_plots)
        ]
        
        for name, create_func in visualizations:
            try:
                logger.info(f"📊 Creating {name}...")
                if create_func():
                    self.created_plots.append(name)
                    logger.info(f"✅ {name} completed")
                else:
                    self.failed_plots.append(name)
                    logger.warning(f"⚠️ {name} failed")
            except Exception as e:
                self.failed_plots.append(name)
                logger.error(f"❌ {name} error: {e}")
        
        # Create summary
        self._create_summary_page()
        
        # Open results
        self._open_results()
        
        # Print summary
        self._print_results_summary()
        
        return len(self.created_plots) > 0
    
    def _save_plot_with_fallbacks(self, fig, filename, title="Plot"):
        """Save plot with HTML and PNG fallbacks."""
        base_path = self.output_dir / filename
        html_path = base_path.with_suffix('.html')
        
        try:
            # Try to save HTML
            fig.write_html(
                html_path,
                include_plotlyjs=True,
                config=PLOT_CONFIG,
                div_id=f"plot_{filename.replace('.', '_')}"
            )
            logger.info(f"📄 HTML saved: {html_path.name}")
            
            # Try to save PNG backup
            if PNG_EXPORT_AVAILABLE:
                png_path = PNGFallback.save_figure_as_png(fig, base_path)
                if png_path:
                    logger.info(f"🖼️ PNG backup: {png_path.name}")
            
            return html_path
            
        except Exception as e:
            logger.error(f"❌ Plot saving failed: {e}")
            
            # Try PNG-only fallback
            if PNG_EXPORT_AVAILABLE:
                logger.info("📸 Attempting PNG-only fallback...")
                png_path = PNGFallback.save_figure_as_png(fig, base_path)
                if png_path:
                    logger.info(f"✅ PNG fallback successful: {png_path.name}")
                    return png_path
            
            return None
    
    def _create_executive_dashboard(self):
        """Create executive dashboard."""
        try:
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '📧 Mail Volume Over Time',
                    '📞 Call Volume Over Time',
                    '📊 Weekly Pattern Analysis',
                    '📈 Key Metrics Summary'
                ),
                specs=[
                    [{'type': 'scatter'}, {'type': 'scatter'}],
                    [{'type': 'bar'}, {'type': 'table'}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Mail volume
            if 'mail_volume_total' in self.combined_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=self.combined_data['mail_volume_total'],
                        name='Mail Volume',
                        line=dict(color=COLORS['primary'], width=3),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            # Call volume with quality distinction
            if 'call_count' in self.combined_data.columns:
                actual_data = self.combined_data[
                    self.combined_data.get('data_quality', 'actual') == 'actual'
                ]
                fig.add_trace(
                    go.Scatter(
                        x=actual_data['date'],
                        y=actual_data['call_count'],
                        name='Calls (Actual)',
                        line=dict(color=COLORS['success'], width=3),
                        mode='lines+markers',
                        marker=dict(size=5)
                    ),
                    row=1, col=2
                )
                
                augmented_data = self.combined_data[
                    self.combined_data.get('data_quality', 'none') == 'augmented'
                ]
                if len(augmented_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=augmented_data['date'],
                            y=augmented_data['call_count'],
                            name='Calls (Augmented)',
                            line=dict(color=COLORS['warning'], width=2, dash='dot'),
                            mode='lines',
                            opacity=0.7
                        ),
                        row=1, col=2
                    )
            
            # Weekly pattern
            if 'day_of_week' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                weekly_data = self.combined_data.groupby('day_of_week')['call_count'].mean()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                fig.add_trace(
                    go.Bar(
                        x=day_names,
                        y=weekly_data.values,
                        name='Average Daily Calls',
                        marker_color=COLORS['info'],
                        text=[f'{v:.0f}' for v in weekly_data.values],
                        textposition='outside'
                    ),
                    row=2, col=1
                )
            
            # Summary table
            summary_data = self._calculate_summary_metrics()
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Metric', 'Value'],
                        fill_color=COLORS['primary'],
                        font=dict(color='white', size=14),
                        align='left'
                    ),
                    cells=dict(
                        values=[
                            list(summary_data.keys()),
                            list(summary_data.values())
                        ],
                        fill_color='white',
                        align='left',
                        font=dict(size=12)
                    )
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=900,
                title={
                    'text': "📊 Executive Analytics Dashboard",
                    'x': 0.5,
                    'font': {'size': 24, 'color': COLORS['primary']}
                },
                template='plotly_white',
                showlegend=True
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'executive_dashboard', 'Executive Dashboard')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Executive dashboard error: {e}")
            return False
    
    def _calculate_summary_metrics(self):
        """Calculate summary metrics for display."""
        metrics = {}
        
        try:
            if self.combined_data is not None:
                metrics['📅 Total Days'] = len(self.combined_data)
                
                if 'mail_volume_total' in self.combined_data.columns:
                    metrics['📧 Total Mail'] = f"{self.combined_data['mail_volume_total'].sum():,.0f}"
                    metrics['📧 Avg Daily Mail'] = f"{self.combined_data['mail_volume_total'].mean():.0f}"
                
                if 'call_count' in self.combined_data.columns:
                    metrics['📞 Total Calls'] = f"{self.combined_data['call_count'].sum():,.0f}"
                    metrics['📞 Avg Daily Calls'] = f"{self.combined_data['call_count'].mean():.0f}"
                
                if 'data_quality' in self.combined_data.columns:
                    completeness = (self.combined_data['data_quality'] == 'actual').mean() * 100
                    metrics['✅ Data Quality'] = f"{completeness:.1f}%"
            
            if self.evaluation_results:
                best_model = self.evaluation_results[0]
                metrics['🤖 Best Model'] = best_model['name']
                metrics['🎯 Model MAE'] = f"{best_model['mae']:.1f}"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'Error': 'Could not calculate metrics'}
    def _create_time_series_plots(self):
        """Create time series analysis plots."""
        try:
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    '📈 Mail and Call Volume Trends',
                    '📊 Moving Averages and Patterns',
                    '🔗 Correlation Analysis'
                ),
                vertical_spacing=0.1
            )
            
            # Main trends
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
                        line=dict(color=COLORS['success'], width=2),
                        yaxis='y2'
                    ),
                    row=1, col=1
                )
            
            # Moving averages
            if 'call_count' in self.combined_data.columns and len(self.combined_data) > 7:
                ma_7 = self.combined_data['call_count'].rolling(7, center=True).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.combined_data['date'],
                        y=ma_7,
                        name='7-day MA',
                        line=dict(color=COLORS['secondary'], width=3)
                    ),
                    row=2, col=1
                )
                
                if len(self.combined_data) > 30:
                    ma_30 = self.combined_data['call_count'].rolling(30, center=True).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=self.combined_data['date'],
                            y=ma_30,
                            name='30-day MA',
                            line=dict(color=COLORS['info'], width=3)
                        ),
                        row=2, col=1
                    )
            
            # Correlation analysis
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
                height=1200,
                title="📈 Time Series Analysis Dashboard",
                template='plotly_white'
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'time_series_analysis', 'Time Series Analysis')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Time series plots error: {e}")
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
                logger.warning("No mail source columns found")
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '📧 Mail Volume by Source Over Time',
                    '📊 Source Volume Distribution',
                    '🎯 Source Performance Metrics',
                    '📅 Source Activity Heatmap'
                )
            )
            
            # Time series by source
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                color = COLORS.get(source_name.upper(), f'hsl({i*60}, 70%, 50%)')
                
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
            
            # Box plots for distribution
            for i, col in enumerate(mail_source_cols):
                source_name = col.replace('mail_', '').title()
                source_data = self.combined_data[self.combined_data[col] > 0][col]
                
                if len(source_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=source_data,
                            name=f'{source_name}',
                            marker_color=COLORS.get(source_name.upper(), f'hsl({i*60}, 70%, 50%)')
                        ),
                        row=1, col=2
                    )
            
            # Performance metrics
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
            
            # Activity heatmap
            if mail_source_cols and 'day_of_week' in self.combined_data.columns:
                col = mail_source_cols[0]
                source_name = col.replace('mail_', '').title()
                
                # Create day x week heatmap data
                heatmap_data = self.combined_data.pivot_table(
                    index=self.combined_data['date'].dt.isocalendar().week,
                    columns='day_of_week',
                    values=col,
                    aggfunc='mean',
                    fill_value=0
                ).iloc[-12:]  # Last 12 weeks
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data.values,
                        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        y=[f'W{w}' for w in heatmap_data.index],
                        colorscale='Blues',
                        name=f'{source_name} Activity'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=900,
                title="🎯 Source Analysis Dashboard",
                template='plotly_white'
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'source_analysis', 'Source Analysis')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Source plots error: {e}")
            return False
    
    def _create_model_plots(self):
        """Create model performance plots."""
        try:
            if not self.evaluation_results:
                logger.warning("No evaluation results available")
                return False
            
            models = self.evaluation_results[:8]  # Top 8 models
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '📏 Mean Absolute Error (Lower = Better)',
                    '📐 R-Squared Score (Higher = Better)',
                    '🎯 MAPE Percentage (Lower = Better)',
                    '🤖 Model Type Distribution'
                )
            )
            
            # Extract data
            names = [m['name'] for m in models]
            mae_vals = [m['mae'] for m in models]
            r2_vals = [m['r2'] for m in models]
            mape_vals = [m['mape'] for m in models]
            types = [m.get('type', 'unknown') for m in models]
            
            # MAE comparison
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
            
            # R² comparison
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
            
            # MAPE comparison
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
            
            # Model type pie chart
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
                height=900,
                title="🤖 Model Performance Dashboard",
                template='plotly_white'
            )
            
            # Rotate x-axis labels for readability
            fig.update_xaxes(tickangle=45)
            
            saved_path = self._save_plot_with_fallbacks(fig, 'model_performance', 'Model Performance')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Model plots error: {e}")
            return False
    
    def _create_correlation_plots(self):
        """Create correlation analysis plots."""
        try:
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '🔗 Mail vs Calls Scatter Plot',
                    '⏰ Lag Correlation Analysis',
                    '🎯 Source Correlations',
                    '📅 Weekly Pattern Correlation'
                )
            )
            
            # Scatter plot: Mail vs Calls
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
                    try:
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
                    except:
                        pass  # Skip trend line if calculation fails
            
            # Lag correlation
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
            
            # Source correlations
            mail_source_cols = [col for col in self.combined_data.columns 
                               if col.startswith('mail_') and col != 'mail_volume_total']
            
            if mail_source_cols and 'call_count' in self.combined_data.columns:
                source_corrs = {}
                for col in mail_source_cols:
                    source_name = col.replace('mail_', '').title()
                    valid_mask = (self.combined_data[col] > 0) & (self.combined_data['call_count'] > 0)
                    if valid_mask.sum() > 10:
                        try:
                            corr = self.combined_data.loc[valid_mask, col].corr(
                                self.combined_data.loc[valid_mask, 'call_count']
                            )
                            source_corrs[source_name] = corr
                        except:
                            pass
                
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
            
            # Day-of-week pattern
            if 'day_of_week' in self.combined_data.columns and 'call_count' in self.combined_data.columns:
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
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
                height=900,
                title="🔗 Correlation Analysis Dashboard",
                template='plotly_white'
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'correlation_analysis', 'Correlation Analysis')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Correlation plots error: {e}")
            return False
    
    def _create_quality_plots(self):
        """Create data quality dashboard."""
        try:
            if self.combined_data is None:
                return False
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '✅ Data Quality Over Time',
                    '📊 Missing Data Analysis',
                    '🎯 Completeness by Source',
                    '📈 Quality Score'
                ),
                specs=[
                    [{'type': 'scatter'}, {'type': 'bar'}],
                    [{'type': 'bar'}, {'type': 'indicator'}]
                ]
            )
            
            # Data quality over time
            if 'data_quality' in self.combined_data.columns:
                quality_counts = self.combined_data.groupby(['date', 'data_quality']).size().unstack(fill_value=0)
                
                if 'actual' in quality_counts.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=quality_counts.index,
                            y=quality_counts['actual'],
                            name='Actual Data',
                            line=dict(color=COLORS['success'], width=2),
                            stackgroup='one'
                        ),
                        row=1, col=1
                    )
                
                if 'augmented' in quality_counts.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=quality_counts.index,
                            y=quality_counts['augmented'],
                            name='Augmented Data',
                            line=dict(color=COLORS['warning'], width=2),
                            stackgroup='one'
                        ),
                        row=1, col=1
                    )
            
            # Missing data analysis
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
            
            # Completeness by source
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
            
            # Overall quality score
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
                height=900,
                title="🔍 Data Quality Dashboard",
                template='plotly_white'
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'data_quality', 'Data Quality')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Quality plots error: {e}")
            return False
    
    def _create_forecast_plots(self):
        """Create forecast visualization."""
        try:
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
                    line=dict(color=COLORS['success'], width=3),
                    marker=dict(size=6)
                ))
            
            # Augmented historical data
            augmented_recent = recent_data[recent_data.get('data_quality', 'none') == 'augmented']
            if len(augmented_recent) > 0:
                fig.add_trace(go.Scatter(
                    x=augmented_recent['date'],
                    y=augmented_recent['call_count'],
                    mode='lines+markers',
                    name='Historical (Augmented)',
                    line=dict(color=COLORS['warning'], width=2, dash='dot'),
                    marker=dict(size=4, symbol='diamond'),
                    opacity=0.7
                ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=self.forecast_data['date'],
                y=self.forecast_data['predicted_calls'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=COLORS['forecast'], width=4, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Confidence interval (if available)
            if 'upper_bound' in self.forecast_data.columns and 'lower_bound' in self.forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.forecast_data['date'].tolist() + self.forecast_data['date'].tolist()[::-1],
                    y=self.forecast_data['upper_bound'].tolist() + self.forecast_data['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor=COLORS['confidence'],
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
            
            # Add vertical line at forecast start
            last_date = self.combined_data['date'].max()
            fig.add_vline(
                x=last_date,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top"
            )
            
            # Calculate stats
            avg_forecast = self.forecast_data['predicted_calls'].mean()
            
            fig.update_layout(
                title={
                    'text': f"🔮 Call Volume Forecast<br><sub>Average forecast: {avg_forecast:.0f} calls/day</sub>",
                    'x': 0.5,
                    'font': {'size': 20}
                },
                xaxis_title='Date',
                yaxis_title='Call Volume',
                height=700,
                template='plotly_white',
                hovermode='x unified'
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'forecast_visualization', 'Forecast')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Forecast plots error: {e}")
            return False
    
    def _create_sample_visualizations(self):
        """Create sample visualizations when no data is available."""
        logger.info("📊 Creating sample visualizations...")
        
        try:
            # Generate sample data
            dates = pd.date_range('2024-01-01', periods=90, freq='D')
            np.random.seed(42)
            
            mail_data = np.random.randint(100, 500, 90) + np.sin(np.arange(90) * 2 * np.pi / 7) * 50
            call_data = mail_data * 0.3 + np.random.randint(-20, 20, 90)
            call_data = np.maximum(call_data, 0)
            
            # Create sample dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '📧 Sample Mail Volume',
                    '📞 Sample Call Volume',
                    '📊 Sample Weekly Pattern',
                    '🎯 Sample Metrics'
                )
            )
            
            # Mail volume
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=mail_data,
                    name='Mail Volume',
                    line=dict(color=COLORS['primary'], width=3)
                ),
                row=1, col=1
            )
            
            # Call volume
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=call_data,
                    name='Call Volume',
                    line=dict(color=COLORS['success'], width=3)
                ),
                row=1, col=2
            )
            
            # Weekly pattern
            weekly_pattern = [150, 180, 170, 165, 160, 120, 100]
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig.add_trace(
                go.Bar(
                    x=day_names,
                    y=weekly_pattern,
                    name='Weekly Pattern',
                    marker_color=COLORS['info']
                ),
                row=2, col=1
            )
            
            # Sample metrics table
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'], fill_color=COLORS['primary']),
                    cells=dict(values=[
                        ['Total Days', 'Avg Mail', 'Avg Calls', 'Correlation'],
                        ['90', '250', '75', '0.65']
                    ])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title="📊 Sample Analytics Dashboard (No Data Available)",
                template='plotly_white'
            )
            
            saved_path = self._save_plot_with_fallbacks(fig, 'sample_dashboard', 'Sample Dashboard')
            return saved_path is not None
            
        except Exception as e:
            logger.error(f"Sample visualization error: {e}")
            return False
    
    def _create_summary_page(self):
        """Create HTML summary page with links to all plots."""
        try:
            # Calculate summary stats
            stats = self._calculate_summary_metrics()
            
            # Get list of created plot files
            plot_files = []
            for file in self.output_dir.glob("*.html"):
                if file.name != 'summary.html':
                    plot_files.append(file.name)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>📊 Analytics Dashboard Summary</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
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
                    .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
                    .content {{ padding: 40px; }}
                    .plots-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        margin: 30px 0;
                    }}
                    .plot-card {{
                        background: {COLORS['light']};
                        border-radius: 10px
                        padding: 20px;
                        text-align: center;
                        transition: all 0.3s;
                        border: 2px solid transparent;
                    }}
                    .plot-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                        border-color: {COLORS['primary']};
                    }}
                    .plot-icon {{ font-size: 3em; margin-bottom: 15px; }}
                    .plot-title {{ font-size: 1.3em; font-weight: 600; margin-bottom: 10px; color: {COLORS['primary']}; }}
                    .plot-link {{
                        display: inline-block;
                        background: {COLORS['primary']};
                        color: white;
                        padding: 12px 24px;
                        text-decoration: none;
                        border-radius: 25px;
                        transition: all 0.3s;
                        font-weight: 500;
                    }}
                    .plot-link:hover {{
                        background: {COLORS['secondary']};
                        transform: translateY(-2px);
                    }}
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin: 20px 0;
                    }}
                    .stat-card {{
                        background: {COLORS['light']};
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        border-left: 4px solid {COLORS['primary']};
                    }}
                    .stat-value {{ font-size: 2em; font-weight: bold; color: {COLORS['primary']}; }}
                    .footer {{ background: {COLORS['dark']}; color: white; text-align: center; padding: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Analytics Dashboard</h1>
                        <p>Interactive Visualizations Summary</p>
                        <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    </div>
                    
                    <div class="content">
                        <h2>📈 Key Statistics</h2>
                        <div class="stats-grid">
                            {self._generate_stats_html(stats)}
                        </div>
                        
                        <h2>🎯 Interactive Visualizations</h2>
                        <div class="plots-grid">
                            {self._generate_plot_cards_html(plot_files)}
                        </div>
                        
                        <h2>ℹ️ Usage Instructions</h2>
                        <ul>
                            <li><strong>Interactive:</strong> All plots are fully interactive - hover, zoom, and explore</li>
                            <li><strong>Export:</strong> Use plot toolbars to download PNG/PDF versions</li>
                            <li><strong>Mobile:</strong> All visualizations are mobile-responsive</li>
                            <li><strong>Filtering:</strong> Click legend items to show/hide data series</li>
                        </ul>
                    </div>
                    
                    <div class="footer">
                        <p>🚀 Enhanced Analytics System | Created with Plotly & Python</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save summary page
            summary_path = self.output_dir / 'summary.html'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"📄 Summary page created: {summary_path.name}")
            return summary_path
            
        except Exception as e:
            logger.error(f"Summary page creation failed: {e}")
            return None
    
    def _generate_stats_html(self, stats):
        """Generate HTML for statistics cards."""
        stats_html = ""
        for key, value in stats.items():
            stats_html += f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div>{key}</div>
            </div>
            """
        return stats_html
    
    def _generate_plot_cards_html(self, plot_files):
        """Generate HTML for plot cards."""
        plot_configs = {
            'executive_dashboard.html': ('📊', 'Executive Dashboard', 'High-level overview and key metrics'),
            'time_series_analysis.html': ('📈', 'Time Series Analysis', 'Temporal trends and patterns'),
            'source_analysis.html': ('🎯', 'Source Analysis', 'Performance by mail source'),
            'model_performance.html': ('🤖', 'Model Performance', 'Prediction model comparison'),
            'correlation_analysis.html': ('🔗', 'Correlation Analysis', 'Relationship analysis'),
            'data_quality.html': ('🔍', 'Data Quality', 'Data completeness assessment'),
            'forecast_visualization.html': ('🔮', 'Forecast', 'Future predictions'),
            'sample_dashboard.html': ('📊', 'Sample Dashboard', 'Example visualization')
        }
        
        cards_html = ""
        for filename in plot_files:
            if filename in plot_configs:
                icon, title, description = plot_configs[filename]
                cards_html += f"""
                <div class="plot-card">
                    <div class="plot-icon">{icon}</div>
                    <div class="plot-title">{title}</div>
                    <p>{description}</p>
                    <a href="{filename}" class="plot-link" target="_blank">Open Plot</a>
                </div>
                """
            else:
                # Generic card for unknown files
                name = filename.replace('.html', '').replace('_', ' ').title()
                cards_html += f"""
                <div class="plot-card">
                    <div class="plot-icon">📊</div>
                    <div class="plot-title">{name}</div>
                    <p>Interactive visualization</p>
                    <a href="{filename}" class="plot-link" target="_blank">Open Plot</a>
                </div>
                """
        
        return cards_html
    
    def _open_results(self):
        """Open the results with comprehensive fallback methods."""
        print("\n🌐 Opening visualization results...")
        
        # Try to find the best file to open
        priority_files = [
            'summary.html',
            'executive_dashboard.html', 
            'sample_dashboard.html'
        ]
        
        target_file = None
        for filename in priority_files:
            file_path = self.output_dir / filename
            if file_path.exists():
                target_file = file_path
                break
        
        if not target_file:
            # Find any HTML file
            html_files = list(self.output_dir.glob("*.html"))
            if html_files:
                target_file = html_files[0]
        
        if target_file:
            print(f"🎯 Opening: {target_file.name}")
            BrowserOpener.open_file(target_file)
        else:
            print("❌ No HTML files found to open")
    
    def _print_results_summary(self):
        """Print comprehensive results summary."""
        print("\n" + "="*80)
        print("🎉 VISUALIZATION RESULTS SUMMARY")
        print("="*80)
        
        if self.created_plots:
            print(f"✅ Successfully created {len(self.created_plots)} visualizations:")
            for plot in self.created_plots:
                print(f"   📊 {plot}")
        
        if self.failed_plots:
            print(f"\n⚠️ Failed to create {len(self.failed_plots)} visualizations:")
            for plot in self.failed_plots:
                print(f"   ❌ {plot}")
        
        print(f"\n📁 Results location: {self.output_dir}")
        
        # List all created files
        html_files = list(self.output_dir.glob("*.html"))
        png_files = list(self.output_dir.glob("*.png"))
        
        if html_files:
            print(f"\n🌐 HTML Files ({len(html_files)}):")
            for file in html_files:
                print(f"   📄 {file.name}")
        
        if png_files:
            print(f"\n🖼️ PNG Files ({len(png_files)}):")
            for file in png_files:
                print(f"   🎨 {file.name}")
        
        # Print manual access instructions
        print(f"\n💡 Manual Access:")
        print(f"   1. Navigate to: {self.output_dir}")
        print(f"   2. Double-click any .html file")
        print(f"   3. Or copy file path to browser address bar")
        
        if html_files:
            main_file = self.output_dir / 'summary.html'
            if main_file.exists():
                print(f"\n🚀 Start here: {main_file}")
            else:
                print(f"\n🚀 Start here: {html_files[0]}")
        
        print("="*80)

# ============================================================================
# STANDALONE EXECUTION AND TESTING
# ============================================================================

def test_environment():
    """Test the environment thoroughly."""
    print("🧪 Testing environment...")
    
    issues = []
    
    # Test Plotly
    if not PLOTLY_AVAILABLE:
        issues.append("Plotly not available")
    else:
        try:
            fig = go.Figure(data=go.Bar(x=['Test'], y=[1]))
            print("✅ Plotly basic functionality works")
        except Exception as e:
            issues.append(f"Plotly error: {e}")
    
    # Test PNG export
    if not PNG_EXPORT_AVAILABLE:
        issues.append("PNG export not available (install kaleido)")
    
    # Test data directory
    if not DEFAULT_DATA_DIR.exists():
        issues.append(f"Data directory not found: {DEFAULT_DATA_DIR}")
    
    # Test file system permissions
    try:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        test_file = DEFAULT_OUTPUT_DIR / 'test_write.txt'
        test_file.write_text('test')
        test_file.unlink()
        print("✅ File system permissions OK")
    except Exception as e:
        issues.append(f"File system error: {e}")
    
    if issues:
        print("\n⚠️ Environment issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Environment test passed!")
        return True

def create_test_visualization():
    """Create a simple test visualization."""
    print("🎨 Creating test visualization...")
    
    try:
        # Create test plotter
        plotter = RobustAnalyticsPlotter()
        
        # Create sample dashboard
        if plotter._create_sample_visualizations():
            print("✅ Test visualization created successfully!")
            
            # Try to open it
            test_file = plotter.output_dir / 'sample_dashboard.html'
            if test_file.exists():
                BrowserOpener.open_file(test_file)
            
            return True
        else:
            print("❌ Test visualization failed")
            return False
            
    except Exception as e:
        print(f"❌ Test visualization error: {e}")
        return False

def main():
    """Main execution function with comprehensive error handling."""
    print("🚀 ROBUST ANALYTICS PLOTTING SYSTEM")
    print("="*60)
    
    try:
        # Test environment first
        if not test_environment():
            print("\n💡 Fix the issues above and try again")
            print("💡 Common fixes:")
            print("   pip install plotly pandas numpy kaleido")
            return False
        
        # Check for analysis results
        if not DEFAULT_DATA_DIR.exists():
            print(f"\n⚠️ Analysis results not found: {DEFAULT_DATA_DIR}")
            print("🎯 Creating sample visualization instead...")
            return create_test_visualization()
        
        # Create full plotter
        print(f"\n📊 Initializing robust plotter...")
        plotter = RobustAnalyticsPlotter()
        
        # Create all visualizations
        success = plotter.create_all_visualizations()
        
        if success:
            print("\n🎉 SUCCESS! Visualizations created successfully!")
        else:
            print("\n⚠️ Some visualizations failed, but results may still be available")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"📋 Full error: {traceback.format_exc()}")
        
        # Try emergency sample creation
        print("\n🆘 Attempting emergency sample creation...")
        try:
            return create_test_visualization()
        except:
            print("❌ Emergency creation also failed")
            return False

def quick_start():
    """Quick start function for immediate results."""
    print("⚡ QUICK START MODE")
    print("-" * 30)
    
    try:
        # Just create a simple test plot
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Add sample data
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            values = np.random.randint(50, 200, 30)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Sample Data',
                line=dict(color='#1f77b4', width=3)
            ))
            
            fig.update_layout(
                title="🚀 Quick Start Test Plot",
                xaxis_title="Date",
                yaxis_title="Value",
                template='plotly_white'
            )
            
            # Save and open
            output_dir = Path("quick_start_plots")
            output_dir.mkdir(exist_ok=True)
            
            file_path = output_dir / "quick_test.html"
            fig.write_html(file_path, include_plotlyjs=True)
            
            print(f"✅ Quick test plot created: {file_path}")
            BrowserOpener.open_file(file_path)
            
            return True
        else:
            print("❌ Plotly not available for quick start")
            return False
            
    except Exception as e:
        print(f"❌ Quick start failed: {e}")
        return False

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['-h', '--help', 'help']:
            print("""
🎨 Robust Analytics Plotting System

Usage:
    python robust_plotting.py              # Full analysis
    python robust_plotting.py test         # Test environment
    python robust_plotting.py quick        # Quick start test
    python robust_plotting.py sample       # Create sample plots

Features:
- Multiple browser opening fallbacks
- PNG backup generation
- Cross-platform compatibility
- Comprehensive error handling
- Sample data when analysis results missing

Requirements:
    pip install plotly pandas numpy kaleido
            """)
            sys.exit(0)
            
        elif arg in ['test', 'check']:
            test_environment()
            sys.exit(0)
            
        elif arg in ['quick', 'fast']:
            quick_start()
            sys.exit(0)
            
        elif arg in ['sample', 'demo']:
            create_test_visualization()
            sys.exit(0)
    
    # Default: run full analysis
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
