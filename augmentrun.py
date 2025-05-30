"""
Launcher – no CLI flags needed
==============================
Put this file, intent_augmentation_comparison.py and merged_call_data.csv
in the same folder, then run:

    python run_intent_augmentation.py
"""
from pathlib import Path
import importlib, sys, time

# ---------- USER SETTINGS ----------
INPUT_CSV  = "merged_call_data.csv"
OUTPUT_DIR = "augmentation_results"
METHODS    = None                 # None → run all, or ["rule","ml",...]
# -----------------------------------

print("\n🟢  INTENT-AUGMENTATION LAUNCHER")
print(f"   input  : {INPUT_CSV}")
print(f"   output : {OUTPUT_DIR}")
print(f"   methods: {'all' if METHODS is None else METHODS}")

HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE))
try:
    fw=importlib.import_module("intent_augmentation_comparison")
except ModuleNotFoundError:
    sys.exit("❌  intent_augmentation_comparison.py not in this folder.")

start=time.time()
fw.IntentAugmentationFramework(INPUT_CSV,OUTPUT_DIR).run(METHODS)
print(f"\n✅  Finished in {time.time()-start:.1f}s")