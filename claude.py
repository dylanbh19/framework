"""
intent_explainability.py - Comprehensive Explainability Framework
================================================================
Production-ready explainability analysis for intent augmentation models.
Generates detailed reports, visualizations, and explanations for stakeholders.

Requirements:
pip install pandas numpy matplotlib seaborn plotly scikit-learn shap lime wordcloud

Usage:
python intent_explainability.py --input augmentation_results_pro/best_augmented_data.csv
"""

import warnings
warnings.filterwarnings('ignore')

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Optional but recommended for advanced explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    
try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# Set visualization styles
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-whitegrid')


class IntentExplainabilityAnalyzer:
    """
    Comprehensive explainability analyzer for intent augmentation results.
    Generates multiple types of explanations, visualizations, and reports.
    """
    
    def __init__(self, data_path: str, output_dir: str = "explainability_results"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        for subdir in ['plots', 'reports', 'interactive', 'data']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # Load data
        log.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.original_size = len(self.df)
        
        # Initialize results storage
        self.explanations = {}
        self.metrics = {}
        self.visualizations = []
        
    def run_full_analysis(self):
        """Run complete explainability analysis pipeline."""
        log.info("Starting comprehensive explainability analysis...")
        
        # 1. Data quality and coverage analysis
        self.analyze_data_quality()
        
        # 2. Intent distribution analysis
        self.analyze_intent_distributions()
        
        # 3. Confidence analysis
        self.analyze_confidence_scores()
        
        # 4. Feature importance analysis
        self.analyze_feature_importance()
        
        # 5. Error analysis
        self.analyze_errors_and_edge_cases()
        
        # 6. Model behavior patterns
        self.analyze_model_patterns()
        
        # 7. Activity sequence analysis
        self.analyze_activity_sequences()
        
        # 8. Generate individual case explanations
        self.generate_case_explanations()
        
        # 9. Create interactive dashboards
        self.create_interactive_dashboards()
        
        # 10. Generate comprehensive report
        self.generate_executive_report()
        
        log.info(f"Analysis complete! Results saved to {self.output_dir}")
        
    def analyze_data_quality(self):
        """Analyze data quality and augmentation coverage."""
        log.info("Analyzing data quality and coverage...")
        
        quality_metrics = {
            'total_records': len(self.df),
            'original_intents': (self.df.get('intent_base', pd.Series()) != 'Unknown').sum(),
            'augmented_intents': (self.df.get('intent_augmented', pd.Series()) != 'Unknown').sum(),
            'improvement_rate': 0,
            'confidence_stats': {},
            'missing_data_analysis': {}
        }
        
        # Calculate improvement
        if 'intent_base' in self.df.columns:
            orig_unknown = (self.df['intent_base'] == 'Unknown').sum()
            aug_unknown = (self.df['intent_augmented'] == 'Unknown').sum()
            quality_metrics['improvement_rate'] = (orig_unknown - aug_unknown) / orig_unknown if orig_unknown > 0 else 0
        
        # Confidence statistics
        if 'intent_confidence' in self.df.columns:
            quality_metrics['confidence_stats'] = {
                'mean': self.df['intent_confidence'].mean(),
                'median': self.df['intent_confidence'].median(),
                'std': self.df['intent_confidence'].std(),
                'quartiles': self.df['intent_confidence'].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        # Missing data analysis
        for col in ['activity_sequence', 'first_activity', 'last_activity']:
            if col in self.df.columns:
                quality_metrics['missing_data_analysis'][col] = {
                    'missing_count': self.df[col].isna().sum(),
                    'missing_percentage': (self.df[col].isna().sum() / len(self.df)) * 100
                }
        
        self.metrics['data_quality'] = quality_metrics
        
        # Visualization
        self._plot_data_quality_dashboard()
        
    def analyze_intent_distributions(self):
        """Analyze intent distribution patterns."""
        log.info("Analyzing intent distributions...")
        
        # Intent frequency analysis
        intent_counts = self.df['intent_augmented'].value_counts()
        
        # Before/after comparison if available
        if 'intent_base' in self.df.columns:
            before_after = pd.DataFrame({
                'Before': self.df['intent_base'].value_counts(),
                'After': self.df['intent_augmented'].value_counts()
            }).fillna(0)
            
            # Calculate changes
            before_after['Change'] = before_after['After'] - before_after['Before']
            before_after['Change_Pct'] = (before_after['Change'] / before_after['Before'] * 100).round(2)
            
            self.metrics['intent_distribution'] = {
                'before_after': before_after.to_dict(),
                'top_intents': intent_counts.head(10).to_dict(),
                'rare_intents': intent_counts.tail(10).to_dict()
            }
        
        # Visualizations
        self._plot_intent_distributions(intent_counts, before_after if 'intent_base' in self.df.columns else None)
        
    def analyze_confidence_scores(self):
        """Analyze model confidence patterns."""
        log.info("Analyzing confidence scores...")
        
        if 'intent_confidence' not in self.df.columns:
            log.warning("No confidence scores found in data")
            return
            
        # Confidence by intent
        conf_by_intent = self.df.groupby('intent_augmented')['intent_confidence'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        # Confidence distribution analysis
        confidence_bins = pd.cut(self.df['intent_confidence'], 
                                bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        self.metrics['confidence_analysis'] = {
            'by_intent': conf_by_intent.to_dict(),
            'distribution': confidence_bins.value_counts().to_dict(),
            'low_confidence_cases': len(self.df[self.df['intent_confidence'] < 0.7]),
            'high_confidence_cases': len(self.df[self.df['intent_confidence'] >= 0.85])
        }
        
        # Visualizations
        self._plot_confidence_analysis(conf_by_intent, confidence_bins)
        
    def analyze_feature_importance(self):
        """Analyze which features drive intent predictions."""
        log.info("Analyzing feature importance...")
        
        # Activity sequence analysis
        if 'activity_sequence' in self.df.columns:
            # Extract activity patterns
            activity_patterns = defaultdict(list)
            
            for idx, row in self.df.iterrows():
                if pd.notna(row.get('activity_sequence')):
                    activities = str(row['activity_sequence']).split('|')
                    intent = row['intent_augmented']
                    activity_patterns[intent].extend(activities)
            
            # Find most common activities per intent
            intent_signatures = {}
            for intent, activities in activity_patterns.items():
                activity_counts = Counter(activities)
                intent_signatures[intent] = {
                    'top_activities': dict(activity_counts.most_common(10)),
                    'unique_activities': len(set(activities)),
                    'total_activities': len(activities)
                }
            
            self.metrics['feature_importance'] = {
                'intent_signatures': intent_signatures,
                'distinctive_patterns': self._find_distinctive_patterns(activity_patterns)
            }
        
        # TF-IDF analysis for text features
        self._analyze_text_features()
        
        # Visualizations
        self._plot_feature_importance()
        
    def analyze_errors_and_edge_cases(self):
        """Identify and analyze potential errors and edge cases."""
        log.info("Analyzing errors and edge cases...")
        
        edge_cases = {
            'low_confidence': self.df[self.df.get('intent_confidence', 1) < 0.5] if 'intent_confidence' in self.df.columns else pd.DataFrame(),
            'unknown_results': self.df[self.df['intent_augmented'] == 'Unknown'],
            'short_sequences': pd.DataFrame(),
            'conflicting_patterns': pd.DataFrame()
        }
        
        # Short activity sequences
        if 'activity_sequence' in self.df.columns:
            seq_lengths = self.df['activity_sequence'].fillna('').str.split('|').str.len()
            edge_cases['short_sequences'] = self.df[seq_lengths <= 1]
        
        # Analyze edge case patterns
        edge_case_analysis = {}
        for case_type, case_df in edge_cases.items():
            if not case_df.empty and len(case_df) > 0:
                edge_case_analysis[case_type] = {
                    'count': len(case_df),
                    'percentage': (len(case_df) / len(self.df)) * 100,
                    'common_patterns': self._analyze_edge_case_patterns(case_df)
                }
        
        self.metrics['edge_cases'] = edge_case_analysis
        
        # Generate edge case report
        self._generate_edge_case_report(edge_cases)
        
    def analyze_model_patterns(self):
        """Analyze model behavior patterns and decision boundaries."""
        log.info("Analyzing model behavior patterns...")
        
        # Pattern analysis
        patterns = {
            'intent_transitions': self._analyze_intent_transitions(),
            'confidence_patterns': self._analyze_confidence_patterns(),
            'activity_intent_correlation': self._analyze_activity_intent_correlation()
        }
        
        self.metrics['model_patterns'] = patterns
        
        # Visualizations
        self._plot_model_patterns(patterns)
        
    def analyze_activity_sequences(self):
        """Deep dive into activity sequence patterns."""
        log.info("Analyzing activity sequences...")
        
        if 'activity_sequence' not in self.df.columns:
            log.warning("No activity sequences found")
            return
            
        # Sequence length analysis
        self.df['seq_length'] = self.df['activity_sequence'].fillna('').str.split('|').str.len()
        
        # N-gram analysis
        ngram_analysis = self._analyze_ngrams()
        
        # Sequential pattern mining
        sequential_patterns = self._mine_sequential_patterns()
        
        self.metrics['sequence_analysis'] = {
            'length_stats': self.df['seq_length'].describe().to_dict(),
            'ngrams': ngram_analysis,
            'sequential_patterns': sequential_patterns
        }
        
        # Visualizations
        self._plot_sequence_analysis()
        
    def generate_case_explanations(self, sample_size: int = 100):
        """Generate detailed explanations for individual cases."""
        log.info(f"Generating explanations for {sample_size} sample cases...")
        
        # Sample cases across different confidence levels
        if 'intent_confidence' in self.df.columns:
            samples = pd.concat([
                self.df[self.df['intent_confidence'] < 0.5].sample(min(20, len(self.df[self.df['intent_confidence'] < 0.5]))),
                self.df[(self.df['intent_confidence'] >= 0.5) & (self.df['intent_confidence'] < 0.8)].sample(min(40, len(self.df[(self.df['intent_confidence'] >= 0.5) & (self.df['intent_confidence'] < 0.8)]))),
                self.df[self.df['intent_confidence'] >= 0.8].sample(min(40, len(self.df[self.df['intent_confidence'] >= 0.8])))
            ])
        else:
            samples = self.df.sample(min(sample_size, len(self.df)))
        
        explanations = []
        for idx, row in samples.iterrows():
            explanation = self._generate_single_explanation(row)
            explanations.append(explanation)
        
        # Save explanations
        pd.DataFrame(explanations).to_csv(
            self.output_dir / 'data' / 'case_explanations.csv',
            index=False
        )
        
        self.explanations['sample_cases'] = explanations
        
    def create_interactive_dashboards(self):
        """Create interactive Plotly dashboards."""
        log.info("Creating interactive dashboards...")
        
        # 1. Overview Dashboard
        self._create_overview_dashboard()
        
        # 2. Intent Deep Dive Dashboard
        self._create_intent_dashboard()
        
        # 3. Confidence Analysis Dashboard
        self._create_confidence_dashboard()
        
        # 4. Pattern Explorer
        self._create_pattern_explorer()
        
    def generate_executive_report(self):
        """Generate comprehensive executive report."""
        log.info("Generating executive report...")
        
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': self._generate_executive_summary(),
            'key_findings': self._generate_key_findings(),
            'recommendations': self._generate_recommendations(),
            'technical_details': self.metrics,
            'visualizations': [str(f) for f in self.visualizations]
        }
        
        # Save JSON report
        with open(self.output_dir / 'reports' / 'executive_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        # Generate PDF-ready markdown
        self._generate_markdown_report(report)
        
    # ========== Helper Methods ==========
    
    def _plot_data_quality_dashboard(self):
        """Create data quality visualization dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality and Coverage Analysis', fontsize=16)
        
        # 1. Coverage improvement
        if 'intent_base' in self.df.columns:
            coverage_data = pd.DataFrame({
                'Before': [
                    (self.df['intent_base'] != 'Unknown').sum(),
                    (self.df['intent_base'] == 'Unknown').sum()
                ],
                'After': [
                    (self.df['intent_augmented'] != 'Unknown').sum(),
                    (self.df['intent_augmented'] == 'Unknown').sum()
                ]
            }, index=['Known', 'Unknown'])
            
            coverage_data.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Intent Coverage: Before vs After')
            axes[0, 0].set_ylabel('Number of Records')
            axes[0, 0].tick_params(axis='x', rotation=0)
        
        # 2. Confidence distribution
        if 'intent_confidence' in self.df.columns:
            self.df['intent_confidence'].hist(bins=50, ax=axes[0, 1], edgecolor='black')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(self.df['intent_confidence'].mean(), color='red', 
                              linestyle='--', label=f'Mean: {self.df["intent_confidence"].mean():.3f}')
            axes[0, 1].legend()
        
        # 3. Missing data analysis
        missing_data = []
        for col in ['activity_sequence', 'first_activity', 'last_activity']:
            if col in self.df.columns:
                missing_data.append({
                    'Column': col,
                    'Missing %': (self.df[col].isna().sum() / len(self.df)) * 100
                })
        
        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            missing_df.plot(x='Column', y='Missing %', kind='bar', ax=axes[1, 0], legend=False)
            axes[1, 0].set_title('Missing Data by Column')
            axes[1, 0].set_ylabel('Missing Percentage')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Augmentation method distribution
        if 'aug_method' in self.df.columns:
            method_counts = self.df['aug_method'].value_counts()
            method_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
            axes[1, 1].set_title('Augmentation Methods Used')
            axes[1, 1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'data_quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('data_quality_dashboard.png')
        
    def _plot_intent_distributions(self, intent_counts, before_after_df):
        """Create intent distribution visualizations."""
        # 1. Top intents bar chart
        plt.figure(figsize=(12, 6))
        top_intents = intent_counts.head(15)
        
        ax = top_intents.plot(kind='barh')
        ax.set_xlabel('Number of Records')
        ax.set_title('Top 15 Intent Categories (After Augmentation)')
        
        # Add value labels
        for i, v in enumerate(top_intents.values):
            ax.text(v + 10, i, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'top_intents.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Before/After comparison
        if before_after_df is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Stacked bar chart
            before_after_df[['Before', 'After']].plot(kind='bar', ax=ax1)
            ax1.set_title('Intent Distribution: Before vs After Augmentation')
            ax1.set_xlabel('Intent')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            
            # Change visualization
            changes = before_after_df['Change'].sort_values()
            colors = ['red' if x < 0 else 'green' for x in changes]
            changes.plot(kind='barh', ax=ax2, color=colors)
            ax2.set_title('Change in Intent Counts')
            ax2.set_xlabel('Change in Count')
            ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'intent_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.visualizations.append('intent_comparison.png')
        
        self.visualizations.append('top_intents.png')
        
    def _plot_confidence_analysis(self, conf_by_intent, confidence_bins):
        """Create confidence analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confidence by intent (box plot)
        if 'intent_confidence' in self.df.columns:
            intent_conf_data = []
            for intent in conf_by_intent.index[:15]:  # Top 15 intents
                intent_data = self.df[self.df['intent_augmented'] == intent]['intent_confidence']
                intent_conf_data.append(intent_data)
            
            axes[0, 0].boxplot(intent_conf_data, labels=conf_by_intent.index[:15])
            axes[0, 0].set_title('Confidence Distribution by Intent (Top 15)')
            axes[0, 0].set_ylabel('Confidence Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Threshold')
            
        # 2. Confidence distribution pie chart
        confidence_bins.value_counts().plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
        axes[0, 1].set_title('Confidence Level Distribution')
        axes[0, 1].set_ylabel('')
        
        # 3. Confidence vs frequency scatter
        conf_freq = pd.DataFrame({
            'mean_confidence': conf_by_intent['mean'],
            'count': conf_by_intent['count']
        })
        
        axes[1, 0].scatter(conf_freq['count'], conf_freq['mean_confidence'], alpha=0.6)
        axes[1, 0].set_xlabel('Frequency (Count)')
        axes[1, 0].set_ylabel('Mean Confidence')
        axes[1, 0].set_title('Intent Frequency vs Mean Confidence')
        axes[1, 0].axhline(0.7, color='red', linestyle='--', alpha=0.5)
        
        # Add intent labels for outliers
        for idx, row in conf_freq.iterrows():
            if row['mean_confidence'] < 0.6 or row['count'] > conf_freq['count'].quantile(0.9):
                axes[1, 0].annotate(idx, (row['count'], row['mean_confidence']), 
                                   fontsize=8, alpha=0.7)
        
        # 4. Confidence over time (if timestamp available)
        if any(col in self.df.columns for col in ['timestamp', 'date', 'created_at']):
            # Placeholder for temporal analysis
            axes[1, 1].text(0.5, 0.5, 'Temporal Analysis\n(Requires timestamp data)', 
                           ha='center', va='center', fontsize=12)
        else:
            # Confidence density plot
            if 'intent_confidence' in self.df.columns:
                self.df['intent_confidence'].plot(kind='density', ax=axes[1, 1])
                axes[1, 1].set_title('Confidence Score Density')
                axes[1, 1].set_xlabel('Confidence Score')
                axes[1, 1].axvline(self.df['intent_confidence'].mean(), 
                                  color='red', linestyle='--', 
                                  label=f'Mean: {self.df["intent_confidence"].mean():.3f}')
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('confidence_analysis.png')
        
    def _find_distinctive_patterns(self, activity_patterns):
        """Find patterns that are distinctive for each intent."""
        distinctive = {}
        
        # Calculate activity frequencies across all intents
        global_activity_freq = Counter()
        for activities in activity_patterns.values():
            global_activity_freq.update(activities)
        
        # Find distinctive activities for each intent
        for intent, activities in activity_patterns.items():
            intent_freq = Counter(activities)
            distinctive_score = {}
            
            for activity, count in intent_freq.items():
                # Calculate distinctiveness score (TF-IDF like)
                tf = count / len(activities) if activities else 0
                idf = np.log(len(activity_patterns) / sum(1 for v in activity_patterns.values() if activity in v))
                distinctive_score[activity] = tf * idf
            
            # Get top distinctive activities
            top_distinctive = sorted(distinctive_score.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            distinctive[intent] = {
                'activities': dict(top_distinctive),
                'unique_to_intent': [a for a in intent_freq if global_activity_freq[a] == intent_freq[a]]
            }
        
        return distinctive
        
    def _analyze_text_features(self):
        """Analyze text features using TF-IDF."""
        if 'activity_sequence' not in self.df.columns:
            return
            
        log.info("Performing TF-IDF analysis...")
        
        # Prepare text data
        texts = self.df['activity_sequence'].fillna('')
        
        # TF-IDF analysis
        tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        tfidf_matrix = tfidf.fit_transform(texts)
        
        # Get feature importance by intent
        feature_names = tfidf.get_feature_names_out()
        intent_features = {}
        
        for intent in self.df['intent_augmented'].unique():
            intent_mask = self.df['intent_augmented'] == intent
            if intent_mask.sum() > 0:
                intent_tfidf = tfidf_matrix[intent_mask].mean(axis=0).A1
                top_features_idx = intent_tfidf.argsort()[-10:][::-1]
                intent_features[intent] = {
                    feature_names[i]: float(intent_tfidf[i]) 
                    for i in top_features_idx
                }
        
        self.metrics['text_features'] = intent_features
        
    def _plot_feature_importance(self):
        """Create feature importance visualizations."""
        if 'intent_signatures' not in self.metrics.get('feature_importance', {}):
            return
            
        # 1. Activity heatmap
        signatures = self.metrics['feature_importance']['intent_signatures']
        
        # Create activity-intent matrix
        all_activities = set()
        for data in signatures.values():
            all_activities.update(data['top_activities'].keys())
        
        activity_intent_matrix = pd.DataFrame(
            index=list(signatures.keys())[:15],  # Top 15 intents
            columns=list(all_activities)[:20]     # Top 20 activities
        )
        
        for intent in activity_intent_matrix.index:
            for activity in activity_intent_matrix.columns:
                activity_intent_matrix.loc[intent, activity] = \
                    signatures.get(intent, {}).get('top_activities', {}).get(activity, 0)
        
        activity_intent_matrix = activity_intent_matrix.fillna(0).astype(float)
        
        # Normalize by row (intent)
        activity_intent_matrix = activity_intent_matrix.div(
            activity_intent_matrix.sum(axis=1), axis=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(activity_intent_matrix, cmap='YlOrRd', cbar_kws={'label': 'Relative Frequency'})
        plt.title('Activity-Intent Association Heatmap')
        plt.xlabel('Activities')
        plt.ylabel('Intents')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'activity_intent_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distinctive patterns visualization
        distinctive = self.metrics['feature_importance'].get('distinctive_patterns', {})
        if distinctive:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Prepare data for visualization
            intent_activities = []
            for intent, data in list(distinctive.items())[:10]:
                for activity, score in list(data['activities'].items())[:3]:
                    intent_activities.append({
                        'Intent': intent,
                        'Activity': activity,
                        'Distinctiveness': score
                    })
            
            if intent_activities:
                df_distinctive = pd.DataFrame(intent_activities)
                pivot = df_distinctive.pivot(index='Intent', columns='Activity', values='Distinctiveness')
                pivot.fillna(0).plot(kind='barh', stacked=True, ax=ax)
                ax.set_title('Most Distinctive Activities by Intent')
                ax.set_xlabel('Distinctiveness Score')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'plots' / 'distinctive_patterns.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                self.visualizations.append('distinctive_patterns.png')
        
        self.visualizations.append('activity_intent_heatmap.png')
        
    def _analyze_edge_case_patterns(self, edge_df):
        """Analyze patterns in edge cases."""
        patterns = {}
        
        if len(edge_df) == 0:
            return patterns
            
        # Activity sequence patterns
        if 'activity_sequence' in edge_df.columns:
            sequences = edge_df['activity_sequence'].fillna('').str.split('|')
            all_activities = []
            for seq in sequences:
                all_activities.extend([a for a in seq if a])
            
            if all_activities:
                activity_counts = Counter(all_activities)
                patterns['common_activities'] = dict(activity_counts.most_common(10))
                patterns['avg_sequence_length'] = np.mean([len(seq) for seq in sequences])
        
        # Intent distribution in edge cases
        if 'intent_augmented' in edge_df.columns:
            patterns['intent_distribution'] = edge_df['intent_augmented'].value_counts().to_dict()
        
        # Confidence distribution
        if 'intent_confidence' in edge_df.columns:
            patterns['confidence_stats'] = {
                'mean': edge_df['intent_confidence'].mean(),
                'std': edge_df['intent_confidence'].std(),
                'min': edge_df['intent_confidence'].min(),
                'max': edge_df['intent_confidence'].max()
            }
        
        return patterns
        
    def _generate_edge_case_report(self, edge_cases):
        """Generate detailed edge case report."""
        report_path = self.output_dir / 'reports' / 'edge_case_analysis.txt'
        
        with open(report_path, 'w') as f:
            f.write("EDGE CASE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for case_type, case_df in edge_cases.items():
                if len(case_df) > 0:
                    f.write(f"\n{case_type.upper().replace('_', ' ')}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Count: {len(case_df)}\n")
                    f.write(f"Percentage: {(len(case_df) / len(self.df)) * 100:.2f}%\n")
                    
                    # Sample cases
                    f.write("\nSample Cases:\n")
                    sample_cols = ['intent_augmented', 'intent_confidence', 'activity_sequence']
                    sample_cols = [c for c in sample_cols if c in case_df.columns]
                    
                    if sample_cols:
                        sample = case_df[sample_cols].head(5)
                        f.write(sample.to_string() + "\n")
                    
                    f.write("\n")
        
        # Also create a CSV with all edge cases
        all_edge_cases = []
        for case_type, case_df in edge_cases.items():
            if len(case_df) > 0:
                case_df_copy = case_df.copy()
                case_df_copy['edge_case_type'] = case_type
                all_edge_cases.append(case_df_copy)
        
        if all_edge_cases:
            pd.concat(all_edge_cases).to_csv(
                self.output_dir / 'data' / 'all_edge_cases.csv',
                index=False
            )
        
    def _analyze_intent_transitions(self):
        """Analyze how intents change or transition."""
        transitions = {}
        
        if 'intent_base' in self.df.columns and 'intent_augmented' in self.df.columns:
            # Create transition matrix
            transition_df = pd.crosstab(
                self.df['intent_base'], 
                self.df['intent_augmented'],
                normalize='index'
            )
            
            transitions['matrix'] = transition_df.to_dict()
            
            # Find major transitions
            major_transitions = []
            for from_intent in transition_df.index:
                for to_intent in transition_df.columns:
                    if from_intent != to_intent and transition_df.loc[from_intent, to_intent] > 0.1:
                        major_transitions.append({
                            'from': from_intent,
                            'to': to_intent,
                            'probability': transition_df.loc[from_intent, to_intent]
                        })
            
            transitions['major_transitions'] = sorted(
                major_transitions, 
                key=lambda x: x['probability'], 
                reverse=True
            )[:20]
        
        return transitions
        
    def _analyze_confidence_patterns(self):
        """Analyze patterns in confidence scores."""
        patterns = {}
        
        if 'intent_confidence' not in self.df.columns:
            return patterns
            
        # Confidence by sequence length
        if 'activity_sequence' in self.df.columns:
            self.df['_seq_len'] = self.df['activity_sequence'].fillna('').str.split('|').str.len()
            conf_by_length = self.df.groupby('_seq_len')['intent_confidence'].agg(['mean', 'std'])
            patterns['by_sequence_length'] = conf_by_length.to_dict()
            self.df.drop('_seq_len', axis=1, inplace=True)
        
        # Confidence by method
        if 'aug_method' in self.df.columns:
            conf_by_method = self.df.groupby('aug_method')['intent_confidence'].agg(['mean', 'std', 'count'])
            patterns['by_method'] = conf_by_method.to_dict()
        
        return patterns
        
    def _analyze_activity_intent_correlation(self):
        """Analyze correlation between activities and intents."""
        correlations = {}
        
        if 'activity_sequence' not in self.df.columns:
            return correlations
            
        # Create binary matrix for activities
        all_activities = set()
        for seq in self.df['activity_sequence'].fillna(''):
            if seq:
                all_activities.update(seq.split('|'))
        
        # Limit to top 50 activities
        activity_counts = Counter()
        for seq in self.df['activity_sequence'].fillna(''):
            if seq:
                activity_counts.update(seq.split('|'))
        
        top_activities = [act for act, _ in activity_counts.most_common(50)]
        
        # Create binary features
        for activity in top_activities:
            self.df[f'_has_{activity}'] = self.df['activity_sequence'].fillna('').str.contains(
                activity, regex=False
            ).astype(int)
        
        # Calculate correlations
        for intent in self.df['intent_augmented'].value_counts().head(10).index:
            intent_mask = (self.df['intent_augmented'] == intent).astype(int)
            activity_correlations = {}
            
            for activity in top_activities:
                corr = intent_mask.corr(self.df[f'_has_{activity}'])
                if abs(corr) > 0.1:  # Only significant correlations
                    activity_correlations[activity] = corr
            
            correlations[intent] = dict(sorted(
                activity_correlations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10])
        
        # Clean up temporary columns
        for activity in top_activities:
            self.df.drop(f'_has_{activity}', axis=1, inplace=True)
        
        return correlations
        
    def _plot_model_patterns(self, patterns):
        """Create model pattern visualizations."""
        # 1. Intent transition heatmap
        if 'matrix' in patterns.get('intent_transitions', {}):
            transition_df = pd.DataFrame(patterns['intent_transitions']['matrix'])
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(transition_df, annot=True, fmt='.2f', cmap='Blues', 
                       cbar_kws={'label': 'Transition Probability'})
            plt.title('Intent Transition Matrix (From Base to Augmented)')
            plt.xlabel('Augmented Intent')
            plt.ylabel('Base Intent')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'intent_transitions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.visualizations.append('intent_transitions.png')
        
        # 2. Confidence patterns
        if patterns.get('confidence_patterns'):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Confidence by sequence length
            if 'by_sequence_length' in patterns['confidence_patterns']:
                seq_conf = pd.DataFrame(patterns['confidence_patterns']['by_sequence_length']).T
                if 'mean' in seq_conf.columns:
                    seq_conf['mean'].plot(kind='line', ax=axes[0], marker='o')
                    axes[0].fill_between(seq_conf.index, 
                                       seq_conf['mean'] - seq_conf.get('std', 0),
                                       seq_conf['mean'] + seq_conf.get('std', 0),
                                       alpha=0.3)
                    axes[0].set_title('Confidence by Sequence Length')
                    axes[0].set_xlabel('Sequence Length')
                    axes[0].set_ylabel('Mean Confidence')
                    axes[0].grid(True, alpha=0.3)
            
            # Confidence by method
            if 'by_method' in patterns['confidence_patterns']:
                method_conf = pd.DataFrame(patterns['confidence_patterns']['by_method']).T
                if 'mean' in method_conf.columns:
                    method_conf['mean'].sort_values().plot(kind='barh', ax=axes[1], xerr=method_conf.get('std'))
                    axes[1].set_title('Confidence by Augmentation Method')
                    axes[1].set_xlabel('Mean Confidence')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'confidence_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.visualizations.append('confidence_patterns.png')
        
    def _analyze_ngrams(self, n=3):
        """Analyze n-gram patterns in activity sequences."""
        ngrams = defaultdict(Counter)
        
        if 'activity_sequence' not in self.df.columns:
            return ngrams
            
        for idx, row in self.df.iterrows():
            if pd.notna(row.get('activity_sequence')):
                activities = row['activity_sequence'].split('|')
                intent = row['intent_augmented']
                
                # Extract n-grams
                for i in range(len(activities) - n + 1):
                    ngram = tuple(activities[i:i+n])
                    ngrams[intent][ngram] += 1
        
        # Convert to regular dict with top n-grams
        ngram_dict = {}
        for intent, counter in ngrams.items():
            ngram_dict[intent] = {
                ' -> '.join(gram): count 
                for gram, count in counter.most_common(5)
            }
        
        return ngram_dict
        
    def _mine_sequential_patterns(self):
        """Mine frequent sequential patterns."""
        patterns = {}
        
        if 'activity_sequence' not in self.df.columns:
            return patterns
            
        # Find frequent subsequences by intent
        for intent in self.df['intent_augmented'].value_counts().head(10).index:
            intent_sequences = self.df[
                self.df['intent_augmented'] == intent
            ]['activity_sequence'].fillna('').tolist()
            
            # Find common subsequences
            subsequence_counts = Counter()
            for seq in intent_sequences:
                if seq:
                    activities = seq.split('|')
                    # Generate all subsequences of length 2-4
                    for length in range(2, min(5, len(activities) + 1)):
                        for i in range(len(activities) - length + 1):
                            subseq = tuple(activities[i:i+length])
                            subsequence_counts[subseq] += 1
            
            # Get top patterns
            patterns[intent] = {
                ' -> '.join(pattern): count
                for pattern, count in subsequence_counts.most_common(5)
                if count >= 5  # Minimum support
            }
        
        return patterns
        
    def _plot_sequence_analysis(self):
        """Create sequence analysis visualizations."""
        if 'seq_length' not in self.df.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sequence length distribution
        self.df['seq_length'].hist(bins=30, ax=axes[0, 0], edgecolor='black')
        axes[0, 0].set_title('Distribution of Activity Sequence Lengths')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['seq_length'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.df["seq_length"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Sequence length by intent
        seq_by_intent = self.df.groupby('intent_augmented')['seq_length'].mean().sort_values(ascending=False).head(15)
        seq_by_intent.plot(kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Average Sequence Length by Intent (Top 15)')
        axes[0, 1].set_xlabel('Average Sequence Length')
        
        # 3. Word cloud of activities
        if 'activity_sequence' in self.df.columns:
            all_activities = ' '.join(self.df['activity_sequence'].fillna('').str.replace('|', ' '))
            if all_activities.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_activities)
                axes[1, 0].imshow(wordcloud, interpolation='bilinear')
                axes[1, 0].set_title('Activity Word Cloud')
                axes[1, 0].axis('off')
        
        # 4. N-gram frequency plot
        if 'ngrams' in self.metrics.get('sequence_analysis', {}):
            # Aggregate top n-grams across intents
            all_ngrams = Counter()
            for intent_ngrams in self.metrics['sequence_analysis']['ngrams'].values():
                all_ngrams.update(intent_ngrams)
            
            top_ngrams = dict(all_ngrams.most_common(10))
            if top_ngrams:
                pd.Series(top_ngrams).plot(kind='barh', ax=axes[1, 1])
                axes[1, 1].set_title('Top 10 Activity Sequences (3-grams)')
                axes[1, 1].set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'sequence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('sequence_analysis.png')
        
    def _generate_single_explanation(self, row):
        """Generate explanation for a single case."""
        explanation = {
            'intent': row.get('intent_augmented', 'Unknown'),
            'confidence': row.get('intent_confidence', 0),
            'method': row.get('aug_method', 'Unknown'),
            'activity_sequence': row.get('activity_sequence', ''),
            'explanation': ''
        }
        
        # Build explanation based on available data
        reasons = []
        
        # Check confidence level
        conf = row.get('intent_confidence', 0)
        if conf >= 0.9:
            reasons.append(f"High confidence prediction ({conf:.2%})")
        elif conf >= 0.7:
            reasons.append(f"Moderate confidence prediction ({conf:.2%})")
        else:
            reasons.append(f"Low confidence prediction ({conf:.2%})")
        
        # Check activity patterns
        if pd.notna(row.get('activity_sequence')):
            activities = row['activity_sequence'].split('|')
            if len(activities) > 5:
                reasons.append(f"Long activity sequence ({len(activities)} activities)")
            elif len(activities) < 2:
                reasons.append(f"Short activity sequence ({len(activities)} activities)")
            
            # Check for key activities
            key_activities = {
                'Transfer': ['transfer', 'acat', 'dtc'],
                'Sell': ['sell', 'liquidate', 'redemption'],
                'Fraud Assistance': ['fraud', 'unauthorized', 'dispute']
            }
            
            for intent, keywords in key_activities.items():
                if any(kw in ' '.join(activities).lower() for kw in keywords):
                    if row.get('intent_augmented') == intent:
                        reasons.append(f"Contains key activities for {intent}")
                    else:
                        reasons.append(f"Contains {intent}-related activities but classified as {row.get('intent_augmented')}")
        
        explanation['explanation'] = ' | '.join(reasons)
        return explanation
        
    def _create_overview_dashboard(self):
        """Create interactive overview dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Intent Distribution', 'Confidence Distribution', 
                          'Coverage Improvement', 'Method Performance'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'pie'}, {'type': 'scatter'}]]
        )
        
        # 1. Intent distribution
        intent_counts = self.df['intent_augmented'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=intent_counts.index, y=intent_counts.values, name='Intent Count'),
            row=1, col=1
        )
        
        # 2. Confidence histogram
        if 'intent_confidence' in self.df.columns:
            fig.add_trace(
                go.Histogram(x=self.df['intent_confidence'], nbinsx=30, name='Confidence'),
                row=1, col=2
            )
        
        # 3. Coverage pie chart
        if 'intent_base' in self.df.columns:
            coverage = pd.Series({
                'Originally Known': (self.df['intent_base'] != 'Unknown').sum(),
                'Augmented': ((self.df['intent_base'] == 'Unknown') & 
                             (self.df['intent_augmented'] != 'Unknown')).sum(),
                'Still Unknown': (self.df['intent_augmented'] == 'Unknown').sum()
            })
            fig.add_trace(
                go.Pie(labels=coverage.index, values=coverage.values),
                row=2, col=1
            )
        
        # 4. Method performance scatter
        if 'aug_method' in self.df.columns and 'intent_confidence' in self.df.columns:
            method_stats = self.df.groupby('aug_method').agg({
                'intent_confidence': ['mean', 'count']
            }).round(3)
            
            fig.add_trace(
                go.Scatter(
                    x=method_stats[('intent_confidence', 'count')],
                    y=method_stats[('intent_confidence', 'mean')],
                    mode='markers+text',
                    text=method_stats.index,
                    textposition='top center',
                    marker=dict(size=10)
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Intent Augmentation Overview Dashboard")
        fig.write_html(self.output_dir / 'interactive' / 'overview_dashboard.html')
        
    def _create_intent_dashboard(self):
        """Create interactive intent-specific dashboard."""
        # Select top intents for detailed analysis
        top_intents = self.df['intent_augmented'].value_counts().head(10).index
        
        # Create subplot for each intent
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[f'{intent} Analysis' for intent in top_intents],
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}]] * 5
        )
        
        for i, intent in enumerate(top_intents):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            intent_data = self.df[self.df['intent_augmented'] == intent]
            
            # Confidence distribution for this intent
            if 'intent_confidence' in intent_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=intent_data['intent_confidence'],
                        name=intent,
                        nbinsx=20,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=1200, title_text="Intent-Specific Confidence Analysis")
        fig.write_html(self.output_dir / 'interactive' / 'intent_dashboard.html')
        
    def _create_confidence_dashboard(self):
        """Create interactive confidence analysis dashboard."""
        if 'intent_confidence' not in self.df.columns:
            return
            
        # Create confidence bands
        self.df['confidence_band'] = pd.cut(
            self.df['intent_confidence'],
            bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Sunburst chart of confidence by intent
        fig = px.sunburst(
            self.df.groupby(['confidence_band', 'intent_augmented']).size().reset_index(name='count'),
            path=['confidence_band', 'intent_augmented'],
            values='count',
            title='Confidence Distribution by Intent (Sunburst View)'
        )
        
        fig.update_layout(height=800)
        fig.write_html(self.output_dir / 'interactive' / 'confidence_sunburst.html')
        
        # Clean up temporary column
        self.df.drop('confidence_band', axis=1, inplace=True)
        
    def _create_pattern_explorer(self):
        """Create interactive pattern exploration tool."""
        # Activity sequence pattern visualization
        if 'activity_sequence' not in self.df.columns:
            return
            
        # Create co-occurrence matrix
        activity_pairs = defaultdict(int)
        
        for seq in self.df['activity_sequence'].fillna('').head(1000):  # Sample for performance
            if seq:
                activities = seq.split('|')
                for i in range(len(activities) - 1):
                    pair = (activities[i], activities[i+1])
                    activity_pairs[pair] += 1
        
        # Convert to network graph format
        edges = []
        for (source, target), weight in activity_pairs.items():
            if weight > 5:  # Minimum threshold
                edges.append({
                    'source': source,
                    'target': target,
                    'weight': weight
                })
        
        # Create interactive network graph (simplified for plotly)
        if edges:
            # Note: For a full network graph, consider using networkx + pyvis
            edge_trace = []
            node_trace = []
            
            # This is a simplified visualization
            activities = list(set([e['source'] for e in edges] + [e['target'] for e in edges]))
            
            fig = go.Figure()
            
            # Add edges
            for edge in edges[:50]:  # Limit for visualization
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[activities.index(edge['source']), activities.index(edge['target'])],
                    mode='lines',
                    line=dict(width=edge['weight']/10, color='gray'),
                    showlegend=False
                ))
            
            fig.update_layout(
                title='Activity Transition Patterns (Sample)',
                height=600,
                showlegend=False
            )
            
            fig.write_html(self.output_dir / 'interactive' / 'pattern_explorer.html')
        
    def _generate_executive_summary(self):
        """Generate executive summary of findings."""
        summary = {
            'overview': {
                'total_records_analyzed': len(self.df),
                'improvement_rate': self.metrics.get('data_quality', {}).get('improvement_rate', 0),
                'methods_used': list(set(self.df['aug_method'].unique())) if 'aug_method' in self.df.columns else [],
                'average_confidence': self.df['intent_confidence'].mean() if 'intent_confidence' in self.df.columns else None
            },
            'key_achievements': [],
            'areas_of_concern': [],
            'data_quality': {
                'completeness': 1 - (self.df.get('activity_sequence', pd.Series()).isna().sum() / len(self.df)),
                'augmentation_coverage': 1 - ((self.df['intent_augmented'] == 'Unknown').sum() / len(self.df))
            }
        }
        
        # Key achievements
        if summary['overview']['improvement_rate'] > 0.5:
            summary['key_achievements'].append(f"Achieved {summary['overview']['improvement_rate']:.1%} improvement in intent coverage")
        
        if summary['overview']['average_confidence'] and summary['overview']['average_confidence'] > 0.8:
            summary['key_achievements'].append(f"High average confidence score: {summary['overview']['average_confidence']:.3f}")
        
        # Areas of concern
        unknown_rate = (self.df['intent_augmented'] == 'Unknown').sum() / len(self.df)
        if unknown_rate > 0.1:
            summary['areas_of_concern'].append(f"Still {unknown_rate:.1%} of records have unknown intent")
        
        low_conf = (self.df.get('intent_confidence', pd.Series(1)) < 0.7).sum()
        if low_conf > len(self.df) * 0.2:
            summary['areas_of_concern'].append(f"{low_conf} records ({low_conf/len(self.df):.1%}) have low confidence scores")
        
        return summary
        
    def _generate_key_findings(self):
        """Generate key findings from the analysis."""
        findings = []
        
        # Finding 1: Most improved intents
        if 'intent_distribution' in self.metrics and 'before_after' in self.metrics['intent_distribution']:
            ba = self.metrics['intent_distribution']['before_after']
            if 'Change' in ba:
                top_improved = sorted(
                    [(k, v) for k, v in ba['Change'].items() if v > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                if top_improved:
                    findings.append({
                        'title': 'Most Improved Intent Categories',
                        'description': f"Top improved intents: {', '.join([f'{intent} (+{count})' for intent, count in top_improved])}",
                        'impact': 'high'
                    })
        
        # Finding 2: Confidence patterns
        if 'confidence_analysis' in self.metrics:
            conf_by_intent = self.metrics['confidence_analysis'].get('by_intent', {})
            if conf_by_intent and 'mean' in conf_by_intent:
                low_conf_intents = [
                    intent for intent, stats in conf_by_intent['mean'].items()
                    if stats < 0.7
                ]
                if low_conf_intents:
                    findings.append({
                        'title': 'Low Confidence Intent Categories',
                        'description': f"Intents with low average confidence: {', '.join(low_conf_intents[:5])}",
                        'impact': 'medium',
                        'recommendation': 'Consider additional training data or alternative methods for these intents'
                    })
        
        # Finding 3: Distinctive patterns
        if 'feature_importance' in self.metrics and 'distinctive_patterns' in self.metrics['feature_importance']:
            findings.append({
                'title': 'Distinctive Activity Patterns Identified',
                'description': 'Successfully identified unique activity patterns for each intent category',
                'impact': 'high',
                'details': 'See activity-intent heatmap for detailed patterns'
            })
        
        return findings
        
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Recommendation 1: Data quality
        if 'data_quality' in self.metrics:
            missing_rates = self.metrics['data_quality'].get('missing_data_analysis', {})
            high_missing = [
                col for col, stats in missing_rates.items()
                if stats.get('missing_percentage', 0) > 20
            ]
            if high_missing:
                recommendations.append({
                    'priority': 'high',
                    'category': 'data_quality',
                    'recommendation': f"Address missing data in columns: {', '.join(high_missing)}",
                    'expected_impact': 'Improved model accuracy and coverage'
                })
        
        # Recommendation 2: Low confidence cases
        low_conf_count = (self.df.get('intent_confidence', pd.Series(1)) < 0.7).sum()
        if low_conf_count > len(self.df) * 0.15:
            recommendations.append({
                'priority': 'high',
                'category': 'model_improvement',
                'recommendation': 'Implement ensemble methods or collect more training data for low-confidence cases',
                'expected_impact': f'Could improve confidence for {low_conf_count:,} records'
            })
        
        # Recommendation 3: Edge cases
        if 'edge_cases' in self.metrics:
            total_edge_cases = sum(
                stats.get('count', 0) 
                for stats in self.metrics['edge_cases'].values()
            )
            if total_edge_cases > len(self.df) * 0.05:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'edge_case_handling',
                    'recommendation': 'Develop specialized rules or models for edge cases',
                    'expected_impact': f'Better handling of {total_edge_cases:,} edge cases'
                })
        
        return recommendations
        
    def _generate_html_report(self, report_data):
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Intent Augmentation Explainability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #0066cc; }
                .finding { margin: 20px 0; padding: 15px; border-left: 4px solid #0066cc; background: #f9f9f9; }
                .recommendation { margin: 20px 0; padding: 15px; border-left: 4px solid #ff6600; background: #fff5f0; }
                .high-priority { border-left-color: #ff0000; }
                .chart { margin: 20px 0; text-align: center; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Intent Augmentation Explainability Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div>Total Records</div>
                    <div class="metric-value">{total_records:,}</div>
                </div>
                <div class="metric">
                    <div>Improvement Rate</div>
                    <div class="metric-value">{improvement_rate:.1%}</div>
                </div>
                <div class="metric">
                    <div>Average Confidence</div>
                    <div class="metric-value">{avg_confidence:.3f}</div>
                </div>
            </div>
            
            <h2>Key Findings</h2>
            {findings_html}
            
            <h2>Recommendations</h2>
            {recommendations_html}
            
            <h2>Visualizations</h2>
            {visualizations_html}
            
            <h2>Interactive Dashboards</h2>
            <ul>
                <li><a href="../interactive/overview_dashboard.html">Overview Dashboard</a></li>
                <li><a href="../interactive/intent_dashboard.html">Intent Analysis Dashboard</a></li>
                <li><a href="../interactive/confidence_sunburst.html">Confidence Distribution</a></li>
                <li><a href="../interactive/pattern_explorer.html">Pattern Explorer</a></li>
            </ul>
        </body>
        </html>
        """
        
        # Format findings
        findings_html = ""
        for finding in report_data['key_findings']:
            findings_html += f"""
            <div class="finding">
                <h3>{finding['title']}</h3>
                <p>{finding['description']}</p>
            </div>
            """
        
        # Format recommendations  
        recommendations_html = ""
        for rec in report_data['recommendations']:
            priority_class = "high-priority" if rec['priority'] == 'high' else ""
            recommendations_html += f"""
            <div class="recommendation {priority_class}">
                <h3>{rec['recommendation']}</h3>
                <p>Expected Impact: {rec['expected_impact']}</p>
            </div>
            """
        
        # Format visualizations
        visualizations_html = ""
        for viz in report_data['visualizations']:
            visualizations_html += f"""
            <div class="chart">
                <img src="../plots/{viz}" alt="{viz}">
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            timestamp=report_data['generated_at'],
            total_records=report_data['summary']['overview']['total_records_analyzed'],
            improvement_rate=report_data['summary']['overview']['improvement_rate'],
            avg_confidence=report_data['summary']['overview'].get('average_confidence', 0),
            findings_html=findings_html,
            recommendations_html=recommendations_html,
            visualizations_html=visualizations_html
        )
        
        with open(self.output_dir / 'reports' / 'explainability_report.html', 'w') as f:
            f.write(html_content)
            
    def _generate_markdown_report(self, report_data):
        """Generate markdown report for documentation."""
        md_content = f"""
# Intent Augmentation Explainability Report

Generated: {report_data['generated_at']}

## Executive Summary

- **Total Records Analyzed**: {report_data['summary']['overview']['total_records_analyzed']:,}
- **Improvement Rate**: {report_data['summary']['overview']['improvement_rate']:.1%}
- **Average Confidence**: {report_data['summary']['overview'].get('average_confidence', 0):.3f}

### Key Achievements
{chr(10).join(f"- {achievement}" for achievement in report_data['summary']['key_achievements'])}

### Areas of Concern
{chr(10).join(f"- {concern}" for concern in report_data['summary']['areas_of_concern'])}

## Key Findings

"""
        
        for finding in report_data['key_findings']:
            md_content += f"""
### {finding['title']}
{finding['description']}

"""
        
        md_content += """
## Recommendations

"""
        
        for rec in report_data['recommendations']:
            md_content += f"""
### {rec['recommendation']} (Priority: {rec['priority']})
Expected Impact: {rec['expected_impact']}

"""
        
        md_content += """
## Visualization Gallery

"""
        
        for viz in report_data['visualizations']:
            md_content += f"![{viz}](../plots/{viz})\n\n"
        
        with open(self.output_dir / 'reports' / 'explainability_report.md', 'w') as f:
            f.write(md_content)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive explainability analysis for intent augmentation"
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to augmented data CSV file"
    )
    parser.add_argument(
        "--output",
        default="explainability_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = IntentExplainabilityAnalyzer(args.input, args.output)
    analyzer.run_full_analysis()
    
    log.info("Explainability analysis complete!")
    log.info(f"Results saved to: {args.output}")
    log.info("Key outputs:")
    log.info("  - Executive report: reports/explainability_report.html")
    log.info("  - Interactive dashboards: interactive/")
    log.info("  - Visualizations: plots/")
    log.info("  - Raw data: data/")


if __name__ == "__main__":
    main()