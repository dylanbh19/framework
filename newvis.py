    # ────────────────────────────────────────────────────────────────
    # 1️⃣  Utility inside the class (place near the top of the class)
    # ────────────────────────────────────────────────────────────────
    def _flatten_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with single-level columns; keep the first level if MultiIndex."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df
        
        
        
    # ────────────────────────────────────────────────────────────────
    # 2️⃣  analyze_method_performance  (≈ original line 1150)
    # ────────────────────────────────────────────────────────────────
    @_safe
    def analyze_method_performance(self):
        """
        Summarise how each augmentation method performed.
        Robust to missing confidence column and MultiIndex 'count' errors.
        """
        if 'aug_method' not in self.df.columns:
            log.warning("No aug_method column – skipping method performance step")
            return

        # ---------- aggregate ------------------------------------------------
        agg_dict = {'intent_augmented': lambda x: (x != 'Unknown').sum()}
        if 'intent_confidence' in self.df.columns:
            agg_dict['intent_confidence'] = ['mean', 'std', 'min', 'max']

        method_stats = self.df.groupby('aug_method').agg(agg_dict)
        method_stats = self._flatten_cols(method_stats)

        # ensure a plain 'count' column exists
        if 'count' not in method_stats.columns:
            method_stats['count'] = self.df.groupby('aug_method').size()

        # ---------- store & visualise ----------------------------------------
        self.metrics['methods'] = {
            'table': method_stats.to_dict(),
            'usage': self.df['aug_method'].value_counts().to_dict()
        }
        self._visualize_method_performance()
        
        
        
        
        
    # ────────────────────────────────────────────────────────────────
    # 3️⃣  _visualize_confidence_patterns  (≈ original line 1750)
    # ────────────────────────────────────────────────────────────────
    @_safe
    def _visualize_confidence_patterns(self):
        """Create the four-panel confidence analysis figure."""
        if 'intent_confidence' not in self.df.columns or 'confidence' not in self.metrics:
            log.warning("Confidence data missing – skipping confidence plots")
            return

        # -------- rebuild & flatten -----------------------------------------
        conf_by_intent = pd.DataFrame(self.metrics['confidence']['by_intent']).T
        conf_by_intent = self._flatten_cols(conf_by_intent)

        # ‘count’ can disappear if previous step failed; recreate if needed
        if 'count' not in conf_by_intent.columns:
            conf_by_intent['count'] = (
                self.df.groupby('intent_augmented')['intent_confidence'].size()
            )

        # -------- plotting ---------------------------------------------------
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Confidence Analysis', fontsize=16)

            # 1 ▸ barh of mean confidence (top 15 by count)
            top = conf_by_intent.nlargest(15, 'count')
            top['mean'].sort_values().plot(kind='barh',
                                           xerr=top['std'],
                                           ax=axes[0, 0])
            axes[0, 0].set_title('Mean Confidence (top-15)')
            axes[0, 0].axvline(0.7, ls='--', c='red', alpha=.5)

            # 2 ▸ band histogram
            bands = pd.cut(self.df['intent_confidence'],
                           bins=[0, .5, .7, .85, 1],
                           labels=['<0.5', '0.5-0.7', '0.7-0.85', '0.85-1'])
            bands.value_counts().reindex(['<0.5','0.5-0.7','0.7-0.85','0.85-1']) \
                 .plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Confidence Bands')

            # 3 ▸ density
            self.df['intent_confidence'].plot(kind='density', ax=axes[1, 0])
            axes[1, 0].set_title('Confidence Density')
            axes[1, 0].axvline(self.df['intent_confidence'].mean(),
                               ls='--', c='red', label='mean')
            axes[1, 0].legend()

            # 4 ▸ boxplot by intent (top 10 by volume)
            vol_top = (self.df['intent_augmented'].value_counts()
                                          .head(10).index)
            data = [self.df.loc[self.df['intent_augmented'] == i,
                                'intent_confidence'] for i in vol_top]
            axes[1, 1].boxplot(data, labels=[i[:15] for i in vol_top])
            axes[1, 1].set_title('Confidence per Intent (top-10)')
            axes[1, 1].axhline(0.7, ls='--', c='red', alpha=.5)
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            out = self.output_dir / 'plots' / 'confidence_analysis.png'
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            self.visualizations.append(out.name)

        except Exception as e:
            log.warning("Confidence plot failed: %s", e)