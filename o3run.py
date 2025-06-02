# run_explainability_defaults.py
"""
Quick-start wrapper for intent_explainability_data_only_refactored.py
--------------------------------------------------------------------
• Looks for 'best_augmented_data.csv' in the current folder.
• Writes outputs to './explainability_results'.
• Prints a helpful message if the CSV isn’t found.
"""

from pathlib import Path
from intent_explainability_data_only_refactored import (
    AnalyzerConfig,
    DataOnlyExplainabilityAnalyzer,
    log,   # reuse the same logger for consistency
)

def main() -> None:
    input_csv  = Path("best_augmented_data.csv")
    output_dir = Path("explainability_results")

    if not input_csv.exists():
        log.error("Expected CSV '%s' was not found. "
                  "Place the file alongside this launcher or "
                  "edit 'input_csv' in run_explainability_defaults.py.", input_csv)
        return

    cfg = AnalyzerConfig(input_csv, output_dir)
    DataOnlyExplainabilityAnalyzer(cfg).run()
    log.info("✅ Finished!  See '%s' for reports & visuals.", output_dir)

if __name__ == "__main__":
    main()