"""
Launcher for Intent-Augmentation Framework
=========================================
Put this file in the same folder as
  • intent_augmentation_comparison.py
  • merged_call_data.csv

Then just run:  python run_intent_augmentation.py
(No command-line flags required.)
"""

from pathlib import Path
import importlib, sys, time

# ---------- USER CONFIG ----------
INPUT_CSV  = "merged_call_data.csv"      # feature-pipeline output
OUTPUT_DIR = "augmentation_results"      # where results go
METHODS    = None                        # None → run all; or e.g. ["rule","ml"]
# -----------------------------------

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

print("\n🟢  INTENT-AUGMENTATION LAUNCHER")
print(f"   input   : {INPUT_CSV}")
print(f"   output  : {OUTPUT_DIR}")
print(f"   methods : {'all' if METHODS is None else METHODS}\n")

try:
    framework_mod = importlib.import_module("intent_augmentation_comparison")
except ModuleNotFoundError:
    sys.exit("❌  Missing intent_augmentation_comparison.py in this folder.")

start = time.time()
framework_mod.IntentAugmentationFramework(INPUT_CSV, OUTPUT_DIR).run(METHODS)
print(f"\n✅  Completed in {time.time() - start:.1f} seconds\n")