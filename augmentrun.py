"""
Run Intent-Augmentation Framework
=================================

This is a *launcher* for the IntentAugmentationFramework when you
don’t want to pass command-line arguments.

How it works
------------
1.  It looks in the same folder for:
        •  intent_augmentation_comparison.py   (the framework you just saved)
        •  merged_call_data.csv                (output of your feature pipeline)

2.  It creates an output folder called  ./augmentation_results  (unless you
    change the variable below).

3.  It runs **all** augmentation methods and prints a one-line summary at the
    end.

Customise the three variables under “USER SETTINGS” as needed and run:

    python run_intent_augmentation.py
"""

# ------------------------------------------------------------------
# USER SETTINGS  – edit these three lines if you need different paths
# ------------------------------------------------------------------
INPUT_CSV  = "merged_call_data.csv"      # feature-pipeline output
OUTPUT_DIR = "augmentation_results"      # where reports / plots go
METHODS    = None                        # None → run all.  Or e.g. ["rule_based", "ml"]

# ------------------------------------------------------------------
# No user edits needed below
# ------------------------------------------------------------------
import importlib
import sys
from pathlib import Path

FRAMEWORK_MODULE = "intent_augmentation_comparison"

# Add current folder to sys.path so we can import the module
sys.path.append(str(Path(__file__).resolve().parent))

try:
    framework_mod = importlib.import_module(FRAMEWORK_MODULE)
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"❌  Could not import {FRAMEWORK_MODULE}.py.\n"
        "    Make sure it is in the same folder as this launcher."
    ) from exc

# Instantiate and run
print("\n=== INTENT-AUGMENTATION LAUNCHER ===")
print(f"Input file :  {INPUT_CSV}")
print(f"Output dir :  {OUTPUT_DIR}")
print(f"Methods    :  {'all' if METHODS is None else METHODS}\n")

framework = framework_mod.IntentAugmentationFramework(
    data_path=INPUT_CSV,
    output_dir=OUTPUT_DIR,
)
framework.run(METHODS)

print(
    "\n✅  Done!  "
    "Check the output folder for 'method_comparison.csv', "
    "'best_augmented_data.csv', plots and reports.\n"
)