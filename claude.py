"""
intent_explainability_data_only.py - Explainability Without Model Files
======================================================================
Comprehensive explainability analysis using only the augmented data CSV.
No model files or retraining required.

Requirements:
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn

Usage:
python intent_explainability_data_only.py --input augmentation_results_pro/best_augmented_data.csv
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
from sklearn.preprocessing import StandardScaler

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


class DataOnlyExplainabilityAnalyzer:
    """
    Explainability analyzer that works purely with output data.
    No model files required - analyzes patterns directly from results.
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
        
        # Check for comparison results CSV if available
        comparison_path = self.data_path.parent / 'method_comparison.csv'
        if comparison_path.exists():
            self.comparison_df = pd.read_csv(comparison_path, index_col=0)
            log.info("Found method comparison data")
        else:
            self.comparison_df = None
            
        # Initialize results storage
        self.explanations = {}
        self.metrics = {}
        self.visualizations = []
        
    def run_full_analysis(self):
        """Run complete explainability analysis using only data."""
        log.info("Starting data-based explainability analysis...")
        
        # 1. Data overview and quality
        self.analyze_data_overview()
        
        # 2. Intent patterns from results
        self.analyze_intent_patterns()
        
        # 3. Confidence-based analysis
        self.analyze_confidence_patterns()
        
        # 4. Activity sequence patterns
        self.analyze_activity_patterns()
        
        # 5. Rule-based pattern discovery
        self.discover_decision_rules()
        
        # 6. Error and edge case analysis
        self.analyze_edge_cases()
        
        # 7. Method comparison (if available)
        self.analyze_method_performance()
        
        # 8. Generate case-by-case explanations
        self.generate_data_driven_explanations()
        
        # 9. Create visualizations
        self.create_comprehensive_visualizations()
        
        # 10. Generate reports
        self.generate_reports()
        
        log.info(f"Analysis complete! Results saved to {self.output_dir}")
        
    def analyze_data_overview(self):
        """Analyze basic data statistics and quality."""
        log.info("Analyzing data overview...")
        
        overview = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
        
        # Intent coverage analysis
        if 'intent_augmented' in self.df.columns:
            overview['intent_coverage'] = {
                'unique_intents': self.df['intent_augmented'].nunique(),
                'unknown_count': (self.df['intent_augmented'] == 'Unknown').sum(),
                'unknown_rate': (self.df['intent_augmented'] == 'Unknown').mean(),
                'most_common': self.df['intent_augmented'].value_counts().head(5).to_dict()
            }
        
        # Original vs augmented comparison
        if 'intent_base' in self.df.columns and 'intent_augmented' in self.df.columns:
            overview['improvement'] = {
                'original_unknown': (self.df['intent_base'] == 'Unknown').sum(),
                'augmented_unknown': (self.df['intent_augmented'] == 'Unknown').sum(),
                'records_improved': ((self.df['intent_base'] == 'Unknown') & 
                                   (self.df['intent_augmented'] != 'Unknown')).sum(),
                'improvement_rate': 0
            }
            if overview['improvement']['original_unknown'] > 0:
                overview['improvement']['improvement_rate'] = (
                    overview['improvement']['records_improved'] / 
                    overview['improvement']['original_unknown']
                )
        
        # Data completeness
        overview['completeness'] = {}
        for col in ['activity_sequence', 'first_activity', 'last_activity', 'intent_confidence']:
            if col in self.df.columns:
                overview['completeness'][col] = {
                    'non_null_count': self.df[col].notna().sum(),
                    'completeness_rate': self.df[col].notna().mean()
                }
        
        self.metrics['overview'] = overview
        self._visualize_overview()
        
    def analyze_intent_patterns(self):
        """Analyze patterns in intent predictions."""
        log.info("Analyzing intent patterns...")
        
        if 'intent_augmented' not in self.df.columns:
            log.warning("No intent_augmented column found")
            return
            
        patterns = {
            'distribution': self.df['intent_augmented'].value_counts().to_dict(),
            'proportions': self.df['intent_augmented'].value_counts(normalize=True).to_dict()
        }
        
        # Transition analysis if we have before/after
        if 'intent_base' in self.df.columns:
            transitions = pd.crosstab(
                self.df['intent_base'], 
                self.df['intent_augmented']
            )
            patterns['transitions'] = {
                'matrix': transitions.to_dict(),
                'major_changes': self._find_major_transitions(transitions)
            }
        
        # Intent by method if available
        if 'aug_method' in self.df.columns:
            patterns['by_method'] = {}
            for method in self.df['aug_method'].unique():
                method_df = self.df[self.df['aug_method'] == method]
                patterns['by_method'][method] = {
                    'count': len(method_df),
                    'top_intents': method_df['intent_augmented'].value_counts().head(5).to_dict()
                }
        
        self.metrics['intent_patterns'] = patterns
        self._visualize_intent_patterns()
        
    def analyze_confidence_patterns(self):
        """Analyze confidence score patterns."""
        log.info("Analyzing confidence patterns...")
        
        if 'intent_confidence' not in self.df.columns:
            log.warning("No confidence scores found")
            return
            
        confidence_analysis = {
            'overall_stats': {
                'mean': self.df['intent_confidence'].mean(),
                'median': self.df['intent_confidence'].median(),
                'std': self.df['intent_confidence'].std(),
                'min': self.df['intent_confidence'].min(),
                'max': self.df['intent_confidence'].max()
            },
            'distribution': {
                'very_low': (self.df['intent_confidence'] < 0.5).sum(),
                'low': ((self.df['intent_confidence'] >= 0.5) & 
                       (self.df['intent_confidence'] < 0.7)).sum(),
                'medium': ((self.df['intent_confidence'] >= 0.7) & 
                          (self.df['intent_confidence'] < 0.85)).sum(),
                'high': ((self.df['intent_confidence'] >= 0.85) & 
                        (self.df['intent_confidence'] < 0.95)).sum(),
                'very_high': (self.df['intent_confidence'] >= 0.95).sum()
            }
        }
        
        # Confidence by intent
        conf_by_intent = self.df.groupby('intent_augmented')['intent_confidence'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).round(3)
        confidence_analysis['by_intent'] = conf_by_intent.to_dict()
        
        # Find low/high confidence intents
        confidence_analysis['insights'] = {
            'high_confidence_intents': conf_by_intent[conf_by_intent['mean'] > 0.85].index.tolist(),
            'low_confidence_intents': conf_by_intent[conf_by_intent['mean'] < 0.7].index.tolist(),
            'most_variable_intents': conf_by_intent.nlargest(5, 'std').index.tolist()
        }
        
        self.metrics['confidence'] = confidence_analysis
        self._visualize_confidence_patterns()
        
    def analyze_activity_patterns(self):
        """Analyze activity sequence patterns."""
        log.info("Analyzing activity patterns...")
        
        if 'activity_sequence' not in self.df.columns:
            log.warning("No activity sequences found")
            return
            
        # Basic sequence statistics
        self.df['seq_length'] = self.df['activity_sequence'].fillna('').str.split('|').str.len()
        
        activity_analysis = {
            'sequence_stats': {
                'mean_length': self.df['seq_length'].mean(),
                'median_length': self.df['seq_length'].median(),
                'max_length': self.df['seq_length'].max(),
                'empty_sequences': (self.df['seq_length'] == 0).sum()
            }
        }
        
        # Activity frequency analysis
        all_activities = []
        for seq in self.df['activity_sequence'].fillna(''):
            if seq:
                all_activities.extend(seq.split('|'))
        
        activity_counts = Counter(all_activities)
        activity_analysis['top_activities'] = dict(activity_counts.most_common(20))
        activity_analysis['unique_activities'] = len(activity_counts)
        
        # Intent-specific activity patterns
        intent_activities = {}
        for intent in self.df['intent_augmented'].value_counts().head(10).index:
            intent_df = self.df[self.df['intent_augmented'] == intent]
            intent_acts = []
            for seq in intent_df['activity_sequence'].fillna(''):
                if seq:
                    intent_acts.extend(seq.split('|'))
            
            if intent_acts:
                intent_activity_counts = Counter(intent_acts)
                intent_activities[intent] = {
                    'top_activities': dict(intent_activity_counts.most_common(10)),
                    'unique_count': len(intent_activity_counts),
                    'total_activities': len(intent_acts)
                }
        
        activity_analysis['by_intent'] = intent_activities
        
        # N-gram patterns
        activity_analysis['ngrams'] = self._extract_ngrams()
        
        self.metrics['activities'] = activity_analysis
        self._visualize_activity_patterns()
        
    def discover_decision_rules(self):
        """Discover apparent decision rules from the data."""
        log.info("Discovering decision rules from patterns...")
        
        rules = {'discovered_rules': [], 'pattern_statistics': {}}
        
        if 'activity_sequence' not in self.df.columns or 'intent_augmented' not in self.df.columns:
            log.warning("Cannot discover rules without activity sequences and intents")
            return
            
        # Find strong activity-intent associations
        for intent in self.df['intent_augmented'].value_counts().head(15).index:
            if intent == 'Unknown':
                continue
                
            intent_df = self.df[self.df['intent_augmented'] == intent]
            other_df = self.df[self.df['intent_augmented'] != intent]
            
            # Find activities that appear frequently for this intent
            intent_activities = []
            for seq in intent_df['activity_sequence'].fillna(''):
                if seq:
                    intent_activities.extend(seq.split('|'))
            
            other_activities = []
            for seq in other_df['activity_sequence'].fillna(''):
                if seq:
                    other_activities.extend(seq.split('|'))
            
            intent_act_freq = Counter(intent_activities)
            other_act_freq = Counter(other_activities)
            
            # Calculate distinctiveness scores
            for activity, count in intent_act_freq.most_common(20):
                intent_rate = count / len(intent_df) if len(intent_df) > 0 else 0
                other_rate = other_act_freq.get(activity, 0) / len(other_df) if len(other_df) > 0 else 0
                
                if intent_rate > 0.3 and intent_rate > other_rate * 2:
                    rules['discovered_rules'].append({
                        'rule': f"IF activity_sequence CONTAINS '{activity}' THEN intent = '{intent}'",
                        'confidence': intent_rate,
                        'support': count,
                        'lift': intent_rate / (other_rate + 0.001)
                    })
            
            # Look for activity combinations
            if 'intent_confidence' in self.df.columns:
                high_conf_intent = intent_df[intent_df['intent_confidence'] > 0.9]
                if len(high_conf_intent) > 10:
                    # Find common patterns in high confidence predictions
                    common_patterns = self._find_common_subsequences(
                        high_conf_intent['activity_sequence'].fillna('').tolist()
                    )
                    for pattern, freq in common_patterns[:3]:
                        if freq > 0.5:
                            rules['discovered_rules'].append({
                                'rule': f"IF activity_sequence CONTAINS pattern '{' -> '.join(pattern)}' THEN intent = '{intent}'",
                                'confidence': freq,
                                'support': int(freq * len(high_conf_intent)),
                                'pattern_type': 'sequence'
                            })
        
        # Sort rules by confidence
        rules['discovered_rules'] = sorted(
            rules['discovered_rules'], 
            key=lambda x: x['confidence'], 
            reverse=True
        )[:50]
        
        # Pattern statistics
        rules['pattern_statistics'] = {
            'total_rules_discovered': len(rules['discovered_rules']),
            'high_confidence_rules': sum(1 for r in rules['discovered_rules'] if r['confidence'] > 0.8),
            'sequence_patterns': sum(1 for r in rules['discovered_rules'] if r.get('pattern_type') == 'sequence')
        }
        
        self.metrics['rules'] = rules
        self._save_rules_report(rules)
        
    def analyze_edge_cases(self):
        """Analyze edge cases and potential errors."""
        log.info("Analyzing edge cases...")
        
        edge_cases = {
            'categories': {},
            'statistics': {},
            'samples': {}
        }
        
        # 1. Low confidence predictions
        if 'intent_confidence' in self.df.columns:
            low_conf = self.df[self.df['intent_confidence'] < 0.5]
            edge_cases['categories']['low_confidence'] = {
                'count': len(low_conf),
                'percentage': len(low_conf) / len(self.df) * 100,
                'intent_distribution': low_conf['intent_augmented'].value_counts().to_dict() if 'intent_augmented' in low_conf.columns else {}
            }
            if len(low_conf) > 0:
                edge_cases['samples']['low_confidence'] = low_conf.head(10).to_dict('records')
        
        # 2. Unknown predictions
        if 'intent_augmented' in self.df.columns:
            unknown = self.df[self.df['intent_augmented'] == 'Unknown']
            edge_cases['categories']['unknown_predictions'] = {
                'count': len(unknown),
                'percentage': len(unknown) / len(self.df) * 100
            }
            if len(unknown) > 0 and 'activity_sequence' in unknown.columns:
                # Analyze what's common in unknown predictions
                unknown_activities = []
                for seq in unknown['activity_sequence'].fillna(''):
                    if seq:
                        unknown_activities.extend(seq.split('|'))
                edge_cases['categories']['unknown_predictions']['common_activities'] = dict(
                    Counter(unknown_activities).most_common(10)
                )
        
        # 3. Very short sequences
        if 'activity_sequence' in self.df.columns:
            self.df['_temp_seq_len'] = self.df['activity_sequence'].fillna('').str.split('|').str.len()
            short_seq = self.df[self.df['_temp_seq_len'] <= 1]
            edge_cases['categories']['short_sequences'] = {
                'count': len(short_seq),
                'percentage': len(short_seq) / len(self.df) * 100,
                'avg_confidence': short_seq['intent_confidence'].mean() if 'intent_confidence' in short_seq.columns else None
            }
            self.df.drop('_temp_seq_len', axis=1, inplace=True)
        
        # 4. Conflicting patterns (if we have base intents)
        if 'intent_base' in self.df.columns and 'intent_augmented' in self.df.columns:
            conflicts = self.df[
                (self.df['intent_base'] != 'Unknown') & 
                (self.df['intent_augmented'] != 'Unknown') &
                (self.df['intent_base'] != self.df['intent_augmented'])
            ]
            edge_cases['categories']['conflicting_predictions'] = {
                'count': len(conflicts),
                'percentage': len(conflicts) / len(self.df) * 100,
                'conflict_pairs': pd.crosstab(
                    conflicts['intent_base'], 
                    conflicts['intent_augmented']
                ).to_dict() if len(conflicts) > 0 else {}
            }
        
        # Overall statistics
        total_edge_cases = sum(
            cat['count'] for cat in edge_cases['categories'].values()
        )
        edge_cases['statistics'] = {
            'total_edge_cases': total_edge_cases,
            'edge_case_rate': total_edge_cases / len(self.df) * 100,
            'categories_analyzed': len(edge_cases['categories'])
        }
        
        self.metrics['edge_cases'] = edge_cases
        self._visualize_edge_cases()
        
    def analyze_method_performance(self):
        """Analyze performance across different methods if available."""
        log.info("Analyzing method performance...")
        
        method_analysis = {}
        
        # From augmented data
        if 'aug_method' in self.df.columns:
            method_stats = self.df.groupby('aug_method').agg({
                'intent_augmented': lambda x: (x != 'Unknown').sum(),
                'intent_confidence': ['mean', 'std', 'min', 'max'] if 'intent_confidence' in self.df.columns else []
            })
            
            method_analysis['from_data'] = {
                'usage_counts': self.df['aug_method'].value_counts().to_dict(),
                'statistics': method_stats.to_dict() if not method_stats.empty else {}
            }
        
        # From comparison file if available
        if self.comparison_df is not None:
            method_analysis['from_comparison'] = {
                'unknown_rates': self.comparison_df['unknown_rate'].to_dict() if 'unknown_rate' in self.comparison_df.columns else {},
                'improvements': self.comparison_df['improved'].to_dict() if 'improved' in self.comparison_df.columns else {},
                'processing_times': self.comparison_df['time_s'].to_dict() if 'time_s' in self.comparison_df.columns else {}
            }
        
        self.metrics['methods'] = method_analysis
        self._visualize_method_performance()
        
    def generate_data_driven_explanations(self):
        """Generate explanations based on data patterns."""
        log.info("Generating data-driven explanations...")
        
        # Sample diverse cases
        samples = []
        
        # High confidence cases
        if 'intent_confidence' in self.df.columns:
            high_conf = self.df[self.df['intent_confidence'] > 0.9].sample(
                min(30, len(self.df[self.df['intent_confidence'] > 0.9]))
            )
            samples.extend(high_conf.index)
            
            # Low confidence cases
            low_conf = self.df[self.df['intent_confidence'] < 0.6].sample(
                min(30, len(self.df[self.df['intent_confidence'] < 0.6]))
            )
            samples.extend(low_conf.index)
            
            # Medium confidence
            med_conf = self.df[
                (self.df['intent_confidence'] >= 0.6) & 
                (self.df['intent_confidence'] <= 0.9)
            ].sample(min(40, len(self.df[(self.df['intent_confidence'] >= 0.6) & (self.df['intent_confidence'] <= 0.9)])))
            samples.extend(med_conf.index)
        else:
            # Random sample if no confidence
            samples = self.df.sample(min(100, len(self.df))).index
        
        explanations = []
        for idx in samples:
            row = self.df.loc[idx]
            explanation = self._generate_single_explanation(row)
            explanations.append(explanation)
        
        # Save explanations
        pd.DataFrame(explanations).to_csv(
            self.output_dir / 'data' / 'case_explanations.csv',
            index=False
        )
        
        self.explanations['sample_cases'] = explanations
        
    def create_comprehensive_visualizations(self):
        """Create all visualizations."""
        log.info("Creating comprehensive visualizations...")
        
        # Create interactive dashboards
        self._create_overview_dashboard()
        self._create_intent_explorer()
        self._create_confidence_analyzer()
        self._create_pattern_viewer()
        
    def generate_reports(self):
        """Generate final reports."""
        log.info("Generating reports...")
        
        # Generate executive summary
        summary = self._generate_executive_summary()
        
        # Generate detailed findings
        findings = self._generate_key_findings()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create report structure
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': str(self.data_path),
            'summary': summary,
            'key_findings': findings,
            'recommendations': recommendations,
            'metrics': self.metrics,
            'visualizations': self.visualizations
        }
        
        # Save JSON report
        with open(self.output_dir / 'reports' / 'explainability_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
    # ========== Helper Methods ==========
    
    def _visualize_overview(self):
        """Create overview visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Overview and Quality Analysis', fontsize=16)
        
        # 1. Intent distribution
        if 'intent_augmented' in self.df.columns:
            intent_counts = self.df['intent_augmented'].value_counts().head(10)
            intent_counts.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Top 10 Intent Categories')
            axes[0, 0].set_xlabel('Intent')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Coverage improvement
        if 'improvement' in self.metrics['overview']:
            imp = self.metrics['overview']['improvement']
            coverage_data = pd.Series({
                'Original Unknown': imp['original_unknown'],
                'Improved': imp['records_improved'],
                'Still Unknown': imp['augmented_unknown']
            })
            coverage_data.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
            axes[0, 1].set_title('Coverage Improvement')
            axes[0, 1].set_ylabel('')
        
        # 3. Data completeness
        if 'completeness' in self.metrics['overview']:
            comp_data = pd.DataFrame([
                {'Column': col, 'Completeness': stats['completeness_rate'] * 100}
                for col, stats in self.metrics['overview']['completeness'].items()
            ])
            if not comp_data.empty:
                comp_data.plot(x='Column', y='Completeness', kind='bar', ax=axes[1, 0], legend=False)
                axes[1, 0].set_title('Data Completeness by Column')
                axes[1, 0].set_ylabel('Completeness %')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].axhline(y=80, color='r', linestyle='--', alpha=0.5)
        
        # 4. Confidence distribution
        if 'intent_confidence' in self.df.columns:
            self.df['intent_confidence'].hist(bins=30, ax=axes[1, 1], edgecolor='black')
            axes[1, 1].set_title('Confidence Score Distribution')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(self.df['intent_confidence'].mean(), color='red', 
                              linestyle='--', label=f'Mean: {self.df["intent_confidence"].mean():.3f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'overview_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('overview_analysis.png')
        
    def _visualize_intent_patterns(self):
        """Visualize intent pattern analysis."""
        if 'intent_patterns' not in self.metrics:
            return
            
        # 1. Intent transition heatmap (if available)
        if 'transitions' in self.metrics['intent_patterns']:
            plt.figure(figsize=(12, 10))
            transition_df = pd.DataFrame(self.metrics['intent_patterns']['transitions']['matrix'])
            
            # Normalize by row for better visualization
            transition_norm = transition_df.div(transition_df.sum(axis=1), axis=0).fillna(0)
            
            sns.heatmap(transition_norm, annot=True, fmt='.2f', cmap='Blues',
                       cbar_kws={'label': 'Transition Probability'})
            plt.title('Intent Transition Matrix (Base → Augmented)')
            plt.xlabel('Augmented Intent')
            plt.ylabel('Base Intent')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'intent_transitions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.visualizations.append('intent_transitions.png')
        
        # 2. Intent distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Distribution chart
        dist = pd.Series(self.metrics['intent_patterns']['distribution']).head(15)
        dist.plot(kind='barh', ax=ax1)
        ax1.set_title('Intent Distribution (Top 15)')
        ax1.set_xlabel('Count')
        
        # Proportions pie chart
        props = pd.Series(self.metrics['intent_patterns']['proportions']).head(10)
        props.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Intent Proportions (Top 10)')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'intent_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('intent_distribution.png')
        
    def _visualize_confidence_patterns(self):
        """Visualize confidence pattern analysis."""
        if 'confidence' not in self.metrics or 'intent_confidence' not in self.df.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Confidence Analysis', fontsize=16)
        
        # 1. Confidence by intent (top 15)
        conf_by_intent = pd.DataFrame(self.metrics['confidence']['by_intent']).T
        if 'mean' in conf_by_intent.columns:
            top_intents = conf_by_intent.nlargest(15, 'count')
            top_intents['mean'].sort_values().plot(kind='barh', ax=axes[0, 0], xerr=top_intents['std'])
            axes[0, 0].set_title('Mean Confidence by Intent (Top 15 by frequency)')
            axes[0, 0].set_xlabel('Mean Confidence Score')
            axes[0, 0].axvline(0.7, color='red', linestyle='--', alpha=0.5)
        
        # 2. Confidence distribution bands
        dist_data = pd.Series(self.metrics['confidence']['distribution'])
        dist_data.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence Band')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Confidence density plot
        self.df['intent_confidence'].plot(kind='density', ax=axes[1, 0])
        axes[1, 0].set_title('Confidence Score Density')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].axvline(self.df['intent_confidence'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.df["intent_confidence"].mean():.3f}')
        axes[1, 0].axvline(self.df['intent_confidence'].median(), color='green', 
                          linestyle='--', label=f'Median: {self.df["intent_confidence"].median():.3f}')
        axes[1, 0].legend()
        
        # 4. Box plot by intent
        intent_conf_data = []
        intent_labels = []
        for intent in conf_by_intent.nlargest(10, 'count').index:
            intent_data = self.df[self.df['intent_augmented'] == intent]['intent_confidence']
            if len(intent_data) > 0:
                intent_conf_data.append(intent_data)
                intent_labels.append(intent[:15])  # Truncate long names
        
        if intent_conf_data:
            axes[1, 1].boxplot(intent_conf_data, labels=intent_labels)
            axes[1, 1].set_title('Confidence Distribution by Intent (Top 10)')
            axes[1, 1].set_ylabel('Confidence Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].axhline(0.7, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('confidence_analysis.png')
        
    def _visualize_activity_patterns(self):
        """Visualize activity pattern analysis."""
        if 'activities' not in self.metrics:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Activity Pattern Analysis', fontsize=16)
        
        # 1. Top activities bar chart
        top_acts = pd.Series(self.metrics['activities']['top_activities']).head(15)
        top_acts.plot(kind='barh', ax=axes[0, 0])
        axes[0, 0].set_title('Top 15 Most Common Activities')
        axes[0, 0].set_xlabel('Frequency')
        
        # 2. Sequence length distribution
        if 'seq_length' in self.df.columns:
            self.df['seq_length'].hist(bins=30, ax=axes[0, 1], edgecolor='black')
            axes[0, 1].set_title('Activity Sequence Length Distribution')
            axes[0, 1].set_xlabel('Sequence Length')
            axes[0, 1].set_ylabel('Frequency')
            mean_len = self.metrics['activities']['sequence_stats']['mean_length']
            axes[0, 1].axvline(mean_len, color='red', linestyle='--', 
                              label=f'Mean: {mean_len:.1f}')
            axes[0, 1].legend()
        
        # 3. Activity word cloud
        if self.metrics['activities']['top_activities']:
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(
                self.metrics['activities']['top_activities']
            )
            axes[1, 0].imshow(wordcloud, interpolation='bilinear')
            axes[1, 0].set_title('Activity Word Cloud')
            axes[1, 0].axis('off')
        
        # 4. Activities per intent
        if 'by_intent' in self.metrics['activities']:
            intent_activity_counts = {
                intent: data['unique_count'] 
                for intent, data in self.metrics['activities']['by_intent'].items()
            }
            if intent_activity_counts:
                pd.Series(intent_activity_counts).head(10).plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Unique Activities per Intent (Top 10)')
                axes[1, 1].set_ylabel('Unique Activity Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'activity_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('activity_patterns.png')
        
        # Create activity-intent heatmap
        self._create_activity_intent_heatmap()
        
    def _create_activity_intent_heatmap(self):
        """Create heatmap showing activity-intent associations."""
        if 'by_intent' not in self.metrics.get('activities', {}):
            return
            
        # Build matrix
        intents = list(self.metrics['activities']['by_intent'].keys())[:10]
        all_activities = set()
        for data in self.metrics['activities']['by_intent'].values():
            all_activities.update(data['top_activities'].keys())
        
        activities = sorted(all_activities, 
                          key=lambda x: sum(self.metrics['activities']['by_intent'].get(i, {}).get('top_activities', {}).get(x, 0) 
                                          for i in intents), 
                          reverse=True)[:20]
        
        matrix = pd.DataFrame(index=intents, columns=activities)
        for intent in intents:
            for activity in activities:
                matrix.loc[intent, activity] = self.metrics['activities']['by_intent'].get(
                    intent, {}
                ).get('top_activities', {}).get(activity, 0)
        
        matrix = matrix.fillna(0).astype(float)
        
        # Normalize by row
        matrix_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix_norm, cmap='YlOrRd', cbar_kws={'label': 'Relative Frequency'})
        plt.title('Activity-Intent Association Heatmap')
        plt.xlabel('Activities')
        plt.ylabel('Intents')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'activity_intent_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('activity_intent_heatmap.png')
        
    def _visualize_edge_cases(self):
        """Visualize edge case analysis."""
        if 'edge_cases' not in self.metrics:
            return
            
        # Summary visualization
        categories = self.metrics['edge_cases']['categories']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Edge case distribution
        edge_case_counts = pd.Series({
            cat: data['count'] 
            for cat, data in categories.items()
        })
        
        if not edge_case_counts.empty:
            edge_case_counts.plot(kind='bar', ax=ax1)
            ax1.set_title('Edge Case Distribution')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Edge case percentages
        edge_case_pcts = pd.Series({
            cat: data['percentage'] 
            for cat, data in categories.items()
        })
        
        if not edge_case_pcts.empty:
            edge_case_pcts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
            ax2.set_title('Edge Case Proportions')
            ax2.set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'edge_case_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('edge_case_analysis.png')
        
    def _visualize_method_performance(self):
        """Visualize method performance comparison."""
        if 'methods' not in self.metrics:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Method Performance Analysis', fontsize=16)
        
        # From augmented data
        if 'from_data' in self.metrics['methods']:
            # Method usage
            usage = pd.Series(self.metrics['methods']['from_data']['usage_counts'])
            if not usage.empty:
                usage.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%')
                axes[0, 0].set_title('Method Usage Distribution')
                axes[0, 0].set_ylabel('')
        
        # From comparison file
        if 'from_comparison' in self.metrics['methods']:
            comp = self.metrics['methods']['from_comparison']
            
            # Unknown rates
            if comp.get('unknown_rates'):
                unknown_rates = pd.Series(comp['unknown_rates']) * 100
                unknown_rates.sort_values().plot(kind='barh', ax=axes[0, 1])
                axes[0, 1].set_title('Unknown Rate by Method')
                axes[0, 1].set_xlabel('Unknown %')
            
            # Processing times
            if comp.get('processing_times'):
                times = pd.Series(comp['processing_times'])
                times.sort_values().plot(kind='barh', ax=axes[1, 0])
                axes[1, 0].set_title('Processing Time by Method')
                axes[1, 0].set_xlabel('Time (seconds)')
            
            # Improvements
            if comp.get('improvements'):
                improvements = pd.Series(comp['improvements'])
                improvements.sort_values(ascending=False).plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Records Improved by Method')
                axes[1, 1].set_ylabel('Records Improved')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'method_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.visualizations.append('method_performance.png')
        
    def _extract_ngrams(self, n=3):
        """Extract n-gram patterns from sequences."""
        ngrams = defaultdict(int)
        
        if 'activity_sequence' not in self.df.columns:
            return {}
            
        for seq in self.df['activity_sequence'].fillna('').head(5000):  # Sample for performance
            if seq:
                activities = seq.split('|')
                for i in range(len(activities) - n + 1):
                    ngram = ' -> '.join(activities[i:i+n])
                    ngrams[ngram] += 1
        
        # Return top n-grams
        return dict(Counter(ngrams).most_common(20))
        
    def _find_common_subsequences(self, sequences, min_length=2, max_length=4):
        """Find common subsequences in activity sequences."""
        subsequence_counts = Counter()
        
        for seq in sequences:
            if seq:
                activities = seq.split('|')
                for length in range(min_length, min(max_length + 1, len(activities) + 1)):
                    for i in range(len(activities) - length + 1):
                        subseq = tuple(activities[i:i+length])
                        subsequence_counts[subseq] += 1
        
        total = len(sequences)
        common_patterns = [
            (pattern, count / total) 
            for pattern, count in subsequence_counts.most_common(10)
            if count >= total * 0.1  # At least 10% support
        ]
        
        return common_patterns
        
    def _find_major_transitions(self, transition_matrix):
        """Find major intent transitions."""
        major_transitions = []
        
        for from_intent in transition_matrix.index:
            for to_intent in transition_matrix.columns:
                count = transition_matrix.loc[from_intent, to_intent]
                if from_intent != to_intent and count > 10:  # Significant transitions
                    major_transitions.append({
                        'from': from_intent,
                        'to': to_intent,
                        'count': int(count),
                        'percentage': count / transition_matrix.loc[from_intent].sum() * 100
                    })
        
        return sorted(major_transitions, key=lambda x: x['count'], reverse=True)[:20]
        
    def _save_rules_report(self, rules):
        """Save discovered rules to a readable report."""
        report_path = self.output_dir / 'reports' / 'discovered_rules.txt'
        
        with open(report_path, 'w') as f:
            f.write("DISCOVERED DECISION RULES\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total rules discovered: {len(rules['discovered_rules'])}\n\n")
            
            f.write("TOP RULES BY CONFIDENCE:\n")
            f.write("-" * 60 + "\n\n")
            
            for i, rule in enumerate(rules['discovered_rules'][:30], 1):
                f.write(f"{i}. {rule['rule']}\n")
                f.write(f"   Confidence: {rule['confidence']:.2%}\n")
                f.write(f"   Support: {rule['support']} cases\n")
                if 'lift' in rule:
                    f.write(f"   Lift: {rule['lift']:.2f}\n")
                f.write("\n")
                
        # Also save as CSV for analysis
        pd.DataFrame(rules['discovered_rules']).to_csv(
            self.output_dir / 'data' / 'discovered_rules.csv',
            index=False
        )
        
    def _generate_single_explanation(self, row):
        """Generate explanation for a single prediction."""
        explanation = {
            'intent': row.get('intent_augmented', 'Unknown'),
            'confidence': row.get('intent_confidence', 'N/A'),
            'method': row.get('aug_method', 'Unknown'),
            'base_intent': row.get('intent_base', 'N/A'),
            'activity_sequence': row.get('activity_sequence', ''),
            'sequence_length': len(row.get('activity_sequence', '').split('|')) if pd.notna(row.get('activity_sequence')) else 0,
            'reasoning': []
        }
        
        # Build reasoning
        reasons = []
        
        # Confidence-based reasoning
        if 'intent_confidence' in row and pd.notna(row['intent_confidence']):
            conf = row['intent_confidence']
            if conf >= 0.9:
                reasons.append(f"Very high confidence ({conf:.2%})")
            elif conf >= 0.7:
                reasons.append(f"Good confidence ({conf:.2%})")
            else:
                reasons.append(f"Low confidence ({conf:.2%}) - consider manual review")
        
        # Method-based reasoning
        if 'aug_method' in row and pd.notna(row['aug_method']):
            method = row['aug_method']
            method_explanations = {
                'rule': "Rule-based pattern matching",
                'ml': "Machine learning model prediction",
                'semantic': "Semantic similarity analysis",
                'fuzzy': "Fuzzy string matching",
                'zeroshot': "Zero-shot classification",
                'bert': "BERT fine-tuned model",
                'ensemble': "Ensemble voting from multiple methods"
            }
            reasons.append(f"Method: {method_explanations.get(method, method)}")
        
        # Activity-based reasoning
        if pd.notna(row.get('activity_sequence')):
            activities = row['activity_sequence'].split('|')
            
            # Check for key activities based on discovered patterns
            key_patterns = {
                'Transfer': ['transfer', 'acat', 'dtc'],
                'Sell': ['sell', 'liquidate', 'redemption'],
                'Fraud Assistance': ['fraud', 'unauthorized', 'dispute'],
                'Dividend': ['dividend', 'reinvest', 'drip'],
                'Tax': ['tax', '1099', 'withholding']
            }
            
            matched_patterns = []
            for intent, keywords in key_patterns.items():
                if any(kw in ' '.join(activities).lower() for kw in keywords):
                    if row.get('intent_augmented') == intent:
                        matched_patterns.append(f"Contains {intent}-related activities")
                    else:
                        matched_patterns.append(f"Note: Contains {intent}-related activities but classified as {row.get('intent_augmented')}")
            
            if matched_patterns:
                reasons.extend(matched_patterns)
            
            # Sequence length reasoning
            if len(activities) > 10:
                reasons.append(f"Long activity sequence ({len(activities)} activities)")
            elif len(activities) < 3:
                reasons.append(f"Short activity sequence ({len(activities)} activities) - may affect accuracy")
        
        # Change reasoning
        if 'intent_base' in row and pd.notna(row['intent_base']):
            if row['intent_base'] == 'Unknown' and row.get('intent_augmented') != 'Unknown':
                reasons.append("Successfully augmented from Unknown")
            elif row['intent_base'] != row.get('intent_augmented') and row['intent_base'] != 'Unknown':
                reasons.append(f"Changed from original: {row['intent_base']} → {row.get('intent_augmented')}")
        
        explanation['reasoning'] = ' | '.join(reasons) if reasons else 'No specific reasoning available'
        
        return explanation
        
    def _create_overview_dashboard(self):
        """Create interactive overview dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Intent Distribution', 'Confidence Scores', 
                          'Data Coverage', 'Method Usage'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'pie'}, {'type': 'scatter'}]]
        )
        
        # 1. Intent distribution
        if 'intent_augmented' in self.df.columns:
            intent_counts = self.df['intent_augmented'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=intent_counts.index, y=intent_counts.values, name='Count'),
                row=1, col=1
            )
        
        # 2. Confidence histogram
        if 'intent_confidence' in self.df.columns:
            fig.add_trace(
                go.Histogram(x=self.df['intent_confidence'], nbinsx=30, name='Confidence'),
                row=1, col=2
            )
        
        # 3. Coverage pie
        if 'overview' in self.metrics and 'improvement' in self.metrics['overview']:
            imp = self.metrics['overview']['improvement']
            coverage = pd.Series({
                'Originally Known': len(self.df) - imp['original_unknown'],
                'Augmented': imp['records_improved'],
                'Still Unknown': imp['augmented_unknown']
            })
            fig.add_trace(
                go.Pie(labels=coverage.index, values=coverage.values),
                row=2, col=1
            )
        
        # 4. Method scatter (if available)
        if 'aug_method' in self.df.columns and 'intent_confidence' in self.df.columns:
            method_stats = self.df.groupby('aug_method').agg({
                'intent_confidence': ['mean', 'count']
            })
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
            fig.update_xaxes(title_text="Count", row=2, col=2)
            fig.update_yaxes(title_text="Mean Confidence", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Intent Augmentation Overview Dashboard")
        fig.write_html(self.output_dir / 'interactive' / 'overview_dashboard.html')
        
    def _create_intent_explorer(self):
        """Create interactive intent exploration tool."""
        # Sunburst visualization of intents
        if 'intent_augmented' in self.df.columns:
            # Prepare hierarchical data
            intent_data = []
            
            # Add confidence bands if available
            if 'intent_confidence' in self.df.columns:
                self.df['_conf_band'] = pd.cut(
                    self.df['intent_confidence'],
                    bins=[0, 0.5, 0.7, 0.85, 1.0],
                    labels=['Low', 'Medium', 'High', 'Very High']
                )
                
                for _, row in self.df.iterrows():
                    intent_data.append({
                        'confidence': row['_conf_band'],
                        'intent': row['intent_augmented'],
                        'value': 1
                    })
                
                self.df.drop('_conf_band', axis=1, inplace=True)
            else:
                for intent in self.df['intent_augmented'].value_counts().index:
                    intent_data.append({
                        'intent': intent,
                        'value': (self.df['intent_augmented'] == intent).sum()
                    })
            
            df_sunburst = pd.DataFrame(intent_data)
            
            if 'confidence' in df_sunburst.columns:
                fig = px.sunburst(
                    df_sunburst,
                    path=['confidence', 'intent'],
                    values='value',
                    title='Intent Distribution by Confidence Level'
                )
            else:
                fig = px.treemap(
                    df_sunburst,
                    path=['intent'],
                    values='value',
                    title='Intent Distribution'
                )
            
            fig.update_layout(height=700)
            fig.write_html(self.output_dir / 'interactive' / 'intent_explorer.html')
            
    def _create_confidence_analyzer(self):
        """Create interactive confidence analysis tool."""
        if 'intent_confidence' not in self.df.columns:
            return
            
        # Multi-dimensional confidence analysis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Confidence Distribution by Intent', 'Confidence Trends'),
            row_heights=[0.6, 0.4]
        )
        
        # Box plots for top intents
        top_intents = self.df['intent_augmented'].value_counts().head(15).index
        
        for intent in top_intents:
            intent_data = self.df[self.df['intent_augmented'] == intent]['intent_confidence']
            fig.add_trace(
                go.Box(y=intent_data, name=intent, showlegend=False),
                row=1, col=1
            )
        
        # Confidence histogram with overlay
        fig.add_trace(
            go.Histogram(
                x=self.df['intent_confidence'],
                nbinsx=50,
                name='Distribution',
                histnorm='probability'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=900, title_text="Confidence Analysis Dashboard")
        fig.update_yaxes(title_text="Confidence Score", row=1, col=1)
        fig.update_xaxes(title_text="Intent", row=1, col=1)
        fig.update_xaxes(title_text="Confidence Score", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1)
        
        fig.write_html(self.output_dir / 'interactive' / 'confidence_analyzer.html')
        
    def _create_pattern_viewer(self):
        """Create interactive pattern visualization."""
        if 'activity_sequence' not in self.df.columns:
            return
            
        # Create activity flow visualization
        # Sample data for performance
        sample_df = self.df.sample(min(1000, len(self.df)))
        
        # Extract transitions
        transitions = defaultdict(int)
        for seq in sample_df['activity_sequence'].fillna(''):
            if seq:
                activities = seq.split('|')
                for i in range(len(activities) - 1):
                    transitions[(activities[i], activities[i+1])] += 1
        
        # Create Sankey diagram
        top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:50]
        
        sources = []
        targets = []
        values = []
        
        for (source, target), value in top_transitions:
            sources.append(source)
            targets.append(target)
            values.append(value)
        
        # Create node list
        nodes = list(set(sources + targets))
        node_dict = {node: i for i, node in enumerate(nodes)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes
            ),
            link=dict(
                source=[node_dict[s] for s in sources],
                target=[node_dict[t] for t in targets],
                value=values
            )
        )])
        
        fig.update_layout(
            title_text="Activity Flow Patterns (Sample)",
            height=800
        )
        
        fig.write_html(self.output_dir / 'interactive' / 'pattern_viewer.html')
        
    def _generate_executive_summary(self):
        """Generate executive summary."""
        summary = {
            'overview': {
                'total_records': len(self.df),
                'data_source': str(self.data_path.name),
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            },
            'key_metrics': {},
            'highlights': []
        }
        
        # Key metrics
        if 'intent_augmented' in self.df.columns:
            summary['key_metrics']['unique_intents'] = self.df['intent_augmented'].nunique()
            summary['key_metrics']['unknown_rate'] = (self.df['intent_augmented'] == 'Unknown').mean()
        
        if 'intent_confidence' in self.df.columns:
            summary['key_metrics']['avg_confidence'] = self.df['intent_confidence'].mean()
            summary['key_metrics']['high_confidence_rate'] = (self.df['intent_confidence'] > 0.85).mean()
        
        if 'overview' in self.metrics and 'improvement' in self.metrics['overview']:
            imp_rate = self.metrics['overview']['improvement']['improvement_rate']
            summary['key_metrics']['improvement_rate'] = imp_rate
            
        # Highlights
        if summary['key_metrics'].get('improvement_rate', 0) > 0.5:
            summary['highlights'].append(
                f"Achieved {summary['key_metrics']['improvement_rate']:.1%} reduction in unknown intents"
            )
            
        if summary['key_metrics'].get('avg_confidence', 0) > 0.8:
            summary['highlights'].append(
                f"High average confidence of {summary['key_metrics']['avg_confidence']:.3f}"
            )
            
        if 'rules' in self.metrics and self.metrics['rules']['discovered_rules']:
            high_conf_rules = sum(1 for r in self.metrics['rules']['discovered_rules'] if r['confidence'] > 0.8)
            summary['highlights'].append(
                f"Discovered {high_conf_rules} high-confidence decision rules"
            )
            
        return summary
        
    def _generate_key_findings(self):
        """Generate key findings."""
        findings = []
        
        # Finding 1: Most confident intents
        if 'confidence' in self.metrics and 'insights' in self.metrics['confidence']:
            high_conf = self.metrics['confidence']['insights']['high_confidence_intents']
            if high_conf:
                findings.append({
                    'title': 'High Confidence Intent Categories',
                    'description': f"These intents show consistently high confidence: {', '.join(high_conf[:5])}",
                    'impact': 'Highly reliable predictions for these categories',
                    'recommendation': 'Can be used for automation with minimal review'
                })
        
        # Finding 2: Problem areas
        if 'confidence' in self.metrics and 'insights' in self.metrics['confidence']:
            low_conf = self.metrics['confidence']['insights']['low_confidence_intents']
            if low_conf:
                findings.append({
                    'title': 'Low Confidence Intent Categories',
                    'description': f"These intents need improvement: {', '.join(low_conf[:5])}",
                    'impact': 'May require manual review',
                    'recommendation': 'Collect more training data or refine rules for these intents'
                })
        
        # Finding 3: Edge cases
        if 'edge_cases' in self.metrics:
            total_edge = self.metrics['edge_cases']['statistics']['total_edge_cases']
            edge_rate = self.metrics['edge_cases']['statistics']['edge_case_rate']
            if edge_rate > 5:
                findings.append({
                    'title': 'Significant Edge Cases Detected',
                    'description': f"{total_edge:,} edge cases found ({edge_rate:.1f}% of data)",
                    'impact': 'May affect overall system reliability',
                    'recommendation': 'Develop specialized handling for edge cases'
                })
        
        # Finding 4: Activity patterns
        if 'activities' in self.metrics and 'by_intent' in self.metrics['activities']:
            distinctive_intents = []
            for intent, data in self.metrics['activities']['by_intent'].items():
                if data['unique_count'] > 20 and data['total_activities'] > 100:
                    distinctive_intents.append(intent)
            
            if distinctive_intents:
                findings.append({
                    'title': 'Distinctive Activity Patterns Identified',
                    'description': f"Clear activity patterns found for: {', '.join(distinctive_intents[:5])}",
                    'impact': 'Strong evidence for rule-based classification',
                    'recommendation': 'Can create specific rules for these intents'
                })
        
        # Finding 5: Method performance
        if self.comparison_df is not None and 'unknown_rate' in self.comparison_df.columns:
            best_method = self.comparison_df['unknown_rate'].idxmin()
            best_rate = self.comparison_df.loc[best_method, 'unknown_rate']
            findings.append({
                'title': 'Best Performing Method',
                'description': f"{best_method} achieved the lowest unknown rate ({best_rate:.2%})",
                'impact': 'Optimal method identified',
                'recommendation': f'Consider using {best_method} as primary method'
            })
        
        return findings
        
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Recommendation 1: Data quality
        if 'overview' in self.metrics and 'completeness' in self.metrics['overview']:
            low_completeness = [
                col for col, stats in self.metrics['overview']['completeness'].items()
                if stats['completeness_rate'] < 0.8
            ]
            if low_completeness:
                recommendations.append({
                    'priority': 'high',
                    'category': 'data_quality',
                    'title': 'Improve Data Completeness',
                    'description': f"Address missing data in: {', '.join(low_completeness)}",
                    'expected_impact': 'Could improve predictions for incomplete records',
                    'effort': 'medium'
                })
        
        # Recommendation 2: Low confidence handling
        if 'confidence' in self.metrics:
            low_conf_count = self.metrics['confidence']['distribution'].get('very_low', 0)
            low_conf_count += self.metrics['confidence']['distribution'].get('low', 0)
            
            if low_conf_count > len(self.df) * 0.2:
                recommendations.append({
                    'priority': 'high',
                    'category': 'model_improvement',
                    'title': 'Address Low Confidence Predictions',
                    'description': f"{low_conf_count:,} records have low confidence scores",
                    'expected_impact': f'Could improve {low_conf_count:,} predictions',
                    'effort': 'high',
                    'suggested_actions': [
                        'Implement ensemble methods',
                        'Collect more training data',
                        'Add human-in-the-loop for low confidence cases'
                    ]
                })
        
        # Recommendation 3: Rule implementation
        if 'rules' in self.metrics and self.metrics['rules']['discovered_rules']:
            high_conf_rules = [r for r in self.metrics['rules']['discovered_rules'] if r['confidence'] > 0.85]
            if len(high_conf_rules) >= 10:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'rule_implementation',
                    'title': 'Implement High-Confidence Rules',
                    'description': f"Found {len(high_conf_rules)} rules with >85% confidence",
                    'expected_impact': 'Fast, interpretable predictions for common cases',
                    'effort': 'low',
                    'suggested_actions': [
                        'Implement top rules in production',
                        'Create rule-based pre-filter',
                        'Monitor rule performance over time'
                    ]
                })
        
        # Recommendation 4: Edge case handling
        if 'edge_cases' in self.metrics:
            edge_rate = self.metrics['edge_cases']['statistics']['edge_case_rate']
            if edge_rate > 10:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'edge_case_handling',
                    'title': 'Develop Edge Case Strategy',
                    'description': f"{edge_rate:.1f}% of data are edge cases",
                    'expected_impact': 'Improved handling of difficult cases',
                    'effort': 'medium',
                    'suggested_actions': [
                        'Create specialized models for edge cases',
                        'Implement fallback strategies',
                        'Flag edge cases for manual review'
                    ]
                })
        
        # Recommendation 5: Monitoring
        recommendations.append({
            'priority': 'low',
            'category': 'monitoring',
            'title': 'Implement Continuous Monitoring',
            'description': 'Track model performance over time',
            'expected_impact': 'Early detection of model drift',
            'effort': 'low',
            'suggested_actions': [
                'Monitor confidence score distributions',
                'Track unknown rate trends',
                'Set up alerts for anomalies'
            ]
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
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 { 
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 { 
                    color: #34495e;
                    margin-top: 40px;
                    border-bottom: 1px solid #ecf0f1;
                    padding-bottom: 10px;
                }
                h3 { color: #7f8c8d; }
                .summary { 
                    background: #ecf0f1; 
                    padding: 25px; 
                    border-radius: 8px;
                    margin: 20px 0;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .metric { 
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-value { 
                    font-size: 32px; 
                    font-weight: bold; 
                    color: #3498db;
                    margin: 10px 0;
                }
                .metric-label {
                    color: #7f8c8d;
                    font-size: 14px;
                }
                .finding { 
                    margin: 25px 0; 
                    padding: 20px; 
                    border-left: 4px solid #3498db; 
                    background: #f8f9fa;
                    border-radius: 4px;
                }
                .recommendation { 
                    margin: 25px 0; 
                    padding: 20px; 
                    border-left: 4px solid #e74c3c; 
                    background: #fff5f5;
                    border-radius: 4px;
                }
                .high-priority { border-left-color: #e74c3c; background: #ffe5e5; }
                .medium-priority { border-left-color: #f39c12; background: #fff9e5; }
                .low-priority { border-left-color: #27ae60; background: #e8f8f5; }
                .chart { 
                    margin: 30px 0; 
                    text-align: center;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                img { 
                    max-width: 100%; 
                    height: auto;
                    border-radius: 4px;
                }
                .highlight {
                    background: #f39c12;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-weight: bold;
                }
                .dashboard-links {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }
                .dashboard-links ul {
                    list-style: none;
                    padding: 0;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                }
                .dashboard-links li {
                    background: white;
                    padding: 15px;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .dashboard-links a {
                    color: #3498db;
                    text-decoration: none;
                    font-weight: 500;
                }
                .dashboard-links a:hover {
                    text-decoration: underline;
                }
                .footer {
                    margin-top: 50px;
                    padding: 20px;
                    text-align: center;
                    color: #7f8c8d;
                    border-top: 1px solid #ecf0f1;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Intent Augmentation Explainability Report</h1>
                <p style="color: #7f8c8d;">Generated: {timestamp} | Data Source: {data_source}</p>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-label">Total Records</div>
                            <div class="metric-value">{total_records:,}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Improvement Rate</div>
                            <div class="metric-value">{improvement_rate:.1%}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Average Confidence</div>
                            <div class="metric-value">{avg_confidence:.3f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Unknown Rate</div>
                            <div class="metric-value">{unknown_rate:.1%}</div>
                        </div>
                    </div>
                    
                    <h3>Key Highlights</h3>
                    <ul>
                        {highlights}
                    </ul>
                </div>
                
                <h2>Key Findings</h2>
                {findings_html}
                
                <h2>Recommendations</h2>
                {recommendations_html}
                
                <h2>Visualizations</h2>
                {visualizations_html}
                
                <div class="dashboard-links">
                    <h2>Interactive Dashboards</h2>
                    <ul>
                        <li><a href="../interactive/overview_dashboard.html">📊 Overview Dashboard</a></li>
                        <li><a href="../interactive/intent_explorer.html">🔍 Intent Explorer</a></li>
                        <li><a href="../interactive/confidence_analyzer.html">📈 Confidence Analyzer</a></li>
                        <li><a href="../interactive/pattern_viewer.html">🔗 Pattern Viewer</a></li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>This report provides comprehensive analysis of the intent augmentation results.<br>
                    For questions or additional analysis, please refer to the technical documentation.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Format data
        summary = report_data['summary']
        
        # Format highlights
        highlights_html = '\n'.join(
            f'<li>{highlight}</li>' 
            for highlight in summary.get('highlights', [])
        )
        
        # Format findings
        findings_html = ""
        for finding in report_data['key_findings']:
            findings_html += f"""
            <div class="finding">
                <h3>{finding['title']}</h3>
                <p><strong>Finding:</strong> {finding['description']}</p>
                <p><strong>Impact:</strong> {finding.get('impact', 'N/A')}</p>
                {f"<p><strong>Recommendation:</strong> {finding.get('recommendation', '')}</p>" if finding.get('recommendation') else ''}
            </div>
            """
        
        # Format recommendations  
        recommendations_html = ""
        for rec in report_data['recommendations']:
            priority_class = f"{rec['priority']}-priority"
            actions_html = ""
            if 'suggested_actions' in rec:
                actions_html = "<ul>" + "".join(f"<li>{action}</li>" for action in rec['suggested_actions']) + "</ul>"
            
            recommendations_html += f"""
            <div class="recommendation {priority_class}">
                <h3>{rec['title']} <span class="highlight">{rec['priority'].upper()} PRIORITY</span></h3>
                <p>{rec['description']}</p>
                <p><strong>Expected Impact:</strong> {rec['expected_impact']}</p>
                <p><strong>Effort:</strong> {rec.get('effort', 'Unknown')}</p>
                {actions_html}
            </div>
            """
        
        # Format visualizations
        visualizations_html = ""
        for viz in report_data['visualizations']:
            viz_title = viz.replace('_', ' ').replace('.png', '').title()
            visualizations_html += f"""
            <div class="chart">
                <h3>{viz_title}</h3>
                <img src="../plots/{viz}" alt="{viz_title}">
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            timestamp=report_data['generated_at'],
            data_source=report_data['data_source'],
            total_records=summary['overview']['total_records'],
            improvement_rate=summary['key_metrics'].get('improvement_rate', 0),
            avg_confidence=summary['key_metrics'].get('avg_confidence', 0),
            unknown_rate=summary['key_metrics'].get('unknown_rate', 0),
            highlights=highlights_html,
            findings_html=findings_html,
            recommendations_html=recommendations_html,
            visualizations_html=visualizations_html
        )
        
        with open(self.output_dir / 'reports' / 'explainability_report.html', 'w') as f:
            f.write(html_content)
            
    def _generate_markdown_report(self, report_data):
        """Generate markdown report for documentation."""
        summary = report_data['summary']
        
        md_content = f"""# Intent Augmentation Explainability Report

**Generated:** {report_data['generated_at']}  
**Data Source:** {report_data['data_source']}

## Executive Summary

### Key Metrics
- **Total Records Analyzed:** {summary['overview']['total_records']:,}
- **Improvement Rate:** {summary['key_metrics'].get('improvement_rate', 0):.1%}
- **Average Confidence:** {summary['key_metrics'].get('avg_confidence', 0):.3f}
- **Unknown Rate:** {summary['key_metrics'].get('unknown_rate', 0):.1%}

### Highlights
"""
        
        for highlight in summary.get('highlights', []):
            md_content += f"- {highlight}\n"
        
        md_content += "\n## Key Findings\n\n"
        
        for i, finding in enumerate(report_data['key_findings'], 1):
            md_content += f"""### {i}. {finding['title']}

**Finding:** {finding['description']}

**Impact:** {finding.get('impact', 'N/A')}

"""
            if finding.get('recommendation'):
                md_content += f"**Recommendation:** {finding['recommendation']}\n\n"
        
        md_content += "## Recommendations\n\n"
        
        for i, rec in enumerate(report_data['recommendations'], 1):
            md_content += f"""### {i}. {rec['title']} (Priority: {rec['priority'].upper()})

{rec['description']}

- **Expected Impact:** {rec['expected_impact']}
- **Effort Required:** {rec.get('effort', 'Unknown')}

"""
            if 'suggested_actions' in rec:
                md_content += "**Suggested Actions:**\n"
                for action in rec['suggested_actions']:
                    md_content += f"- {action}\n"
                md_content += "\n"
        
        md_content += """## Visualizations

The following visualizations have been generated:

"""
        
        for viz in report_data['visualizations']:
            viz_title = viz.replace('_', ' ').replace('.png', '').title()
            md_content += f"### {viz_title}\n![{viz_title}](../plots/{viz})\n\n"
        
        md_content += """## Interactive Dashboards

Explore the data interactively through:
- [Overview Dashboard](../interactive/overview_dashboard.html)
- [Intent Explorer](../interactive/intent_explorer.html)
- [Confidence Analyzer](../interactive/confidence_analyzer.html)
- [Pattern Viewer](../interactive/pattern_viewer.html)

## Technical Details

For detailed metrics and technical analysis, refer to:
- `reports/discovered_rules.txt` - Discovered decision rules
- `data/case_explanations.csv` - Individual case explanations
- `reports/explainability_report.json` - Complete metrics in JSON format
"""
        
        with open(self.output_dir / 'reports' / 'explainability_report.md', 'w') as f:
            f.write(md_content)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate explainability analysis from augmented data (no models required)"
    )
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to augmented data CSV file (e.g., best_augmented_data.csv)"
    )
    parser.add_argument(
        "--output",
        default="explainability_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DataOnlyExplainabilityAnalyzer(args.input, args.output)
    analyzer.run_full_analysis()
    
    log.info("=" * 60)
    log.info("Explainability analysis complete!")
    log.info(f"Results saved to: {args.output}")
    log.info("")
    log.info("Key outputs:")
    log.info("  - Executive report: reports/explainability_report.html")
    log.info("  - Technical documentation: reports/explainability_report.md")
    log.info("  - Discovered rules: reports/discovered_rules.txt")
    log.info("  - Interactive dashboards: interactive/")
    log.info("  - Visualizations: plots/")
    log.info("  - Case explanations: data/case_explanations.csv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()