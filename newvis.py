    # ------------------------------------------------------------------
    #  REPLACE the whole existing _visualize_confidence_patterns()
    # ------------------------------------------------------------------
    def _visualize_confidence_patterns(self):
        """Visualize statistics derived from intent_confidence column."""
        # ---- safety checks -------------------------------------------------
        if 'intent_confidence' not in self.df.columns:
            logging.warning("Skip confidence plots – column missing")
            return

        # ensure we have confidence metrics
        if 'confidence' not in self.metrics:
            # compute a minimal version on-the-fly
            conf_by_intent = (self.df
                              .groupby('intent_augmented')['intent_confidence']
                              .agg(['mean', 'median', 'std', 'min', 'max', 'count']))
        else:
            conf_by_intent = pd.DataFrame(
                self.metrics['confidence']['by_intent']).T

        # ---- flatten any MultiIndex columns --------------------------------
        conf_by_intent.columns = [c if isinstance(c, str) else c[0]
                                  for c in conf_by_intent.columns]

        # guard against completely empty frame
        if conf_by_intent.empty:
            logging.warning("Confidence stats empty – skipping plots")
            return

        # ---- plotting ------------------------------------------------------
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Confidence Analysis', fontsize=16)

        # 1) barh of mean confidence for 15 most frequent intents
        top_intents = conf_by_intent.nlargest(15, 'count')
        (top_intents['mean']
         .sort_values()
         .plot(kind='barh', ax=axes[0, 0],
               xerr=top_intents['std'], color='#4c72b0'))
        axes[0, 0].set_title('Mean confidence (top-15 intents)')
        axes[0, 0].set_xlabel('Mean confidence')
        axes[0, 0].axvline(0.7, ls='--', c='red', alpha=.6)

        # 2) band distribution
        bands = pd.cut(self.df['intent_confidence'],
                       bins=[0, .5, .7, .85, 1],
                       labels=['<0.5', '0.5-0.7', '0.7-0.85', '0.85-1'])
        bands.value_counts().reindex(['<0.5','0.5-0.7','0.7-0.85','0.85-1']) \
             .plot(kind='bar', ax=axes[0, 1], color='#55a868')
        axes[0, 1].set_title('Confidence bands')
        axes[0, 1].set_ylabel('Count')

        # 3) density
        self.df['intent_confidence'].plot(kind='density', ax=axes[1, 0])
        axes[1, 0].set_title('Density of confidence scores')
        axes[1, 0].axvline(self.df['intent_confidence'].mean(),
                           ls='--', c='red', label='mean')
        axes[1, 0].legend()

        # 4) boxplot for top-10 intents by volume
        vol_top = (self.df['intent_augmented']
                   .value_counts()
                   .head(10)).index
        data = [self.df.loc[self.df['intent_augmented'] == intent,
                            'intent_confidence']
                for intent in vol_top]
        axes[1, 1].boxplot(data, labels=[i[:15] for i in vol_top])
        axes[1, 1].set_title('Confidence per intent (top-10)')
        axes[1, 1].axhline(0.7, ls='--', c='red', alpha=.6)
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        out = self.output_dir / 'plots' / 'confidence_analysis.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

        self.visualizations.append(out.name)