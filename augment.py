"""
Intent Augmentation – Comparison Framework
=========================================

Benchmarks multiple techniques for replacing “Unknown” intents in the
merged call dataset created by your feature-engineering pipeline.

Dependencies
------------
pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm
pip install sentence-transformers rapidfuzz modAL-python
# optional for GPU/BERT
pip install torch transformers

Typical run
-----------
python intent_augmentation_comparison.py \
       --input  merged_call_data.csv \
       --output augmentation_results \
       --methods all
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm           # <<< NEW progress bars
tqdm.pandas()

# ------------------------------------------------------------------
# Optional heavy libs (gracefully skipped if missing)
# ------------------------------------------------------------------
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ------------------------------------------------------------------
# Framework
# ------------------------------------------------------------------
class IntentAugmentationFramework:
    """Run each augmentation method, track progress and export results."""

    # ==============================================================
    # Initialise
    # ==============================================================
    def __init__(self, data_path: str, output_dir: str = "augmentation_results"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        for sub in ("plots", "reports", "models"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        print(f"📄  Loading data: {self.data_path}")
        self.df: pd.DataFrame = pd.read_csv(self.data_path, low_memory=False)
        self.original_df = self.df.copy()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Any] = {}

        # ------------------- constants -------------------
        self.STANDARD_INTENTS = [
            "Fraud Assistance", "Escheatment", "Balance/Value", "Sell",
            "Repeat Caller", "Name Change", "Buy Stock", "Statement",
            "Recent Activity", "Corporate Action", "Data Protection",
            "Press and Media", "Privacy Breach", "Consolidation",
            "Proxy Inquiry", "Complaint Call", "General Inquiry",
            "Tax Information", "Banking Details", "Dividend Payment",
            "Address Change", "Check Replacement", "Stock Quote",
            "Beneficiary Information", "Dividend Reinvestment",
            "Certificate Issuance", "Transfer",
            "Existing IC User Login Problem",
            "New IC User Login Problem", "Fulfillment", "Enrolment",
            "Associate", "Lost Certificate", "Blank", "Unknown",
        ]
        self.LETTER_TO_INTENT = {
            "a": "Balance/Value", "b": "Sell", "c": "Repeat Caller",
            "d": "Name Change",  "e": "Buy Stock", "f": "Statement",
            "g": "Recent Activity", "h": "Tax Information",
            "i": "Banking Details", "j": "Dividend Payment",
            "k": "Address Change", "l": "Check Replacement",
            "m": "Stock Quote",   "n": "Beneficiary Information",
            "o": "Dividend Reinvestment", "p": "Certificate Issuance",
            "q": "Transfer", "r": "Existing IC User Login Problem",
            "s": "New IC User Login Problem", "t": "Fulfillment",
            "u": "Enrolment", "w": "Associate", "x": "Lost Certificate",
            "y": "Blank", "z": "Unknown",
        }

    # ==============================================================
    # Helper: standardise intent strings
    # ==============================================================
    def _standardize_intent(self, raw):
        if pd.isna(raw):
            return "Unknown"
        txt = str(raw).strip()
        if txt.lower() in self.LETTER_TO_INTENT:
            return self.LETTER_TO_INTENT[txt.lower()]
        return txt if txt in self.STANDARD_INTENTS else "Unknown"

    # ==============================================================
    # Baseline
    # ==============================================================
    def prepare_baseline(self):
        print("\n▶ BASELINE")
        cand = [c for c in self.df.columns if "intent" in c.lower()]
        self.intent_col = cand[0] if cand else "Intent"
        if self.intent_col not in self.df.columns:
            self.df[self.intent_col] = "Unknown"

        self.df["intent_baseline"] = self.df[self.intent_col].apply(
            self._standardize_intent
        )
        rate = (self.df["intent_baseline"] == "Unknown").mean()
        self.results["baseline"] = {
            "stats": {
                "unknown_rate": float(rate),
                "unknown_count": int(rate * len(self.df)),
            }
        }
        print(f"   Unknown rate: {rate:.2%}")

    # ==============================================================
    # Method 1 – rule-based
    # ==============================================================
    def method_1_rule_based(self):
        print("\n▶ RULE-BASED")
        t0 = time.time()

        activity_map = {
            "Sell": "Sell",
            "Tax Information": "Tax Information",
            "Dividend Payment": "Dividend Payment",
            "Transfer": "Transfer",
            "Address Change": "Address Change",
            "Check Replacement": "Check Replacement",
            "Name Change": "Name Change",
            "Banking Details": "Banking Details",
        }
        keyword_patterns = {
            "Fraud Assistance": ["fraud", "unauthorized"],
            "Tax Information": ["tax", "irs", "1099"],
            "Dividend Payment": ["dividend"],
            "Transfer": ["transfer", "acat", "dtc"],
            "Sell": ["sell", "liquidate"],
        }

        def infer(row):
            if row["intent_baseline"] != "Unknown":
                return row["intent_baseline"], 1.0, "baseline"

            # direct activity mapping
            seq = str(row.get("activity_sequence", "")).split("|")
            for act in seq:
                act = act.strip()
                if act in activity_map:
                    return activity_map[act], 0.9, "activity_direct"

            text = "|".join(seq).lower()
            for intent, kws in keyword_patterns.items():
                if any(k in text for k in kws):
                    return intent, 0.7, "keyword_match"

            return "Unknown", 0.0, "unknown"

        res = self.df.progress_apply(infer, axis=1, result_type="expand")
        res.columns = ["intent_rule", "conf_rule", "src_rule"]
        self.df = pd.concat([self.df, res], axis=1)

        self._store_metrics("rule", "intent_rule", t0)

    # ==============================================================
    # Method 2 – ML (TF-IDF + Random Forest + optional XGB)
    # ==============================================================
    def method_2_ml_classification(self):
        print("\n▶ ML CLASSIFIER")
        t0 = time.time()

        train_mask = (self.df["intent_rule"] != "Unknown") & (self.df["conf_rule"] >= 0.7)
        if train_mask.sum() < 200:
            print("   not enough labelled rows – skipped")
            return

        tfidf = TfidfVectorizer(max_features=900, ngram_range=(1, 2))
        X_train = tfidf.fit_transform(
            self.df.loc[train_mask, "activity_sequence"].fillna("")
        )
        y_train = self.df.loc[train_mask, "intent_rule"]

        rf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
        cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_macro")
        rf.fit(X_train, y_train)

        # XGB blend
        if XGB_AVAILABLE:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.3,
                objective="multi:softprob",
                random_state=42,
            )
            xgb_clf.fit(X_train, y_train)

        unk = self.df["intent_baseline"] == "Unknown"
        if unk.sum():
            X_unk = tfidf.transform(self.df.loc[unk, "activity_sequence"].fillna(""))
            rf_pred = rf.predict_proba(X_unk)
            best_pred = rf.classes_[rf_pred.argmax(1)]
            best_conf = rf_pred.max(1)

            if XGB_AVAILABLE:
                xgb_pred = xgb_clf.predict_proba(X_unk)
                xgb_best = xgb_clf.classes_[xgb_pred.argmax(1)]
                xgb_conf = xgb_pred.max(1)

                use_xgb = xgb_conf > best_conf
                best_pred[use_xgb] = xgb_best[use_xgb]
                best_conf[use_xgb] = xgb_conf[use_xgb]

            self.df.loc[unk, "intent_ml"] = best_pred
            self.df.loc[unk, "conf_ml"] = best_conf

        self.df.loc[~unk, "intent_ml"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_ml"] = 1.0

        self.models.update({"tfidf": tfidf, "rf": rf})
        if XGB_AVAILABLE:
            self.models["xgb"] = xgb_clf

        self._store_metrics("ml", "intent_ml", t0, {"cv_f1": float(cv.mean())})

    # ==============================================================
    # Method 3 – semantic similarity (SentenceTransformer)
    # ==============================================================
    def method_3_semantic_similarity(self):
        print("\n▶ SEMANTIC SIMILARITY")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("   sentence-transformers not installed – skipped")
            return
        t0 = time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")

        intent_emb = {i: model.encode(i) for i in self.STANDARD_INTENTS if i != "Unknown"}

        unk = self.df["intent_baseline"] == "Unknown"
        if unk.sum() == 0:
            self._store_metrics("semantic", "intent_rule", t0)
            return

        texts = (
            self.df.loc[unk, "activity_sequence"].fillna("")
            + " "
            + self.df.loc[unk, "first_activity"].fillna("")
            + " "
            + self.df.loc[unk, "last_activity"].fillna("")
        ).tolist()

        embeddings = model.encode(
            texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True
        )

        preds, confs = [], []
        for emb in tqdm(embeddings, desc="‣ semantic ‣ scoring", unit="row"):
            sims = {k: cosine_similarity([emb], [v])[0, 0] for k, v in intent_emb.items()}
            best, score = max(sims.items(), key=lambda x: x[1])
            preds.append(best if score >= 0.25 else "Unknown")
            confs.append(float(score))

        self.df.loc[unk, "intent_sem"] = preds
        self.df.loc[unk, "conf_sem"] = confs
        self.df.loc[~unk, "intent_sem"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_sem"] = 1.0

        self.models["semantic_model"] = model
        self._store_metrics("semantic", "intent_sem", t0)

    # ==============================================================
    # Method 4 – fuzzy matching (RapidFuzz)
    # ==============================================================
    def method_4_fuzzy_matching(self):
        print("\n▶ FUZZY MATCHING")
        if not RAPIDFUZZ_AVAILABLE:
            print("   rapidfuzz not installed – skipped")
            return
        t0 = time.time()

        keywords = {
            "Sell": ["sell", "liquidate"],
            "Tax Information": ["tax", "irs", "1099"],
            "Transfer": ["transfer", "acat"],
            "Balance/Value": ["balance", "value"],
        }

        def fuzzy(txt: str):
            txt = txt.lower()
            best, best_sc = "Unknown", 0.0
            for intent, kws in keywords.items():
                for kw in kws:
                    sc = fuzz.partial_ratio(kw, txt) / 100
                    if sc > best_sc:
                        best, best_sc = intent, sc
            return best if best_sc >= 0.7 else "Unknown", best_sc

        unk = self.df["intent_baseline"] == "Unknown"
        res = (
            self.df.loc[unk, "activity_sequence"]
            .fillna("")
            .progress_apply(fuzzy)
        )
        self.df.loc[unk, "intent_fuzzy"] = res.str[0]
        self.df.loc[unk, "conf_fuzzy"] = res.str[1]
        self.df.loc[~unk, "intent_fuzzy"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_fuzzy"] = 1.0

        self._store_metrics("fuzzy", "intent_fuzzy", t0)

    # ==============================================================
    # Method 5 – ensemble vote
    # ==============================================================
    def method_5_ensemble(self):
        print("\n▶ ENSEMBLE VOTE")
        t0 = time.time()
        cols = [
            ("intent_rule", "conf_rule"),
            ("intent_ml", "conf_ml"),
            ("intent_sem", "conf_sem"),
            ("intent_fuzzy", "conf_fuzzy"),
        ]
        cols = [c for c in cols if c[0] in self.df.columns]
        if len(cols) < 2:
            print("   need ≥ 2 component methods – skipped")
            return

        def vote(row):
            scores: Dict[str, float] = {}
            for ic, cc in cols:
                i, c = row[ic], row[cc]
                if i != "Unknown":
                    scores[i] = scores.get(i, 0.0) + c
            if not scores:
                return "Unknown", 0.0
            best = max(scores, key=scores.get)
            return best, scores[best] / sum(scores.values())

        res = self.df.progress_apply(vote, axis=1, result_type="expand")
        self.df["intent_ensemble"] = res[0]
        self.df["conf_ensemble"] = res[1]
        self._store_metrics("ensemble", "intent_ensemble", t0)

    # ==============================================================
    # Metrics helper
    # ==============================================================
    def _store_metrics(self, key: str, col: str, start: float, extra: Dict[str, Any] | None = None):
        unknown_rate = (self.df[col] == "Unknown").mean()
        improved = ((self.df["intent_baseline"] == "Unknown") & (self.df[col] != "Unknown")).sum()
        self.results[key] = {
            "stats": {
                "unknown_rate": float(unknown_rate),
                "improved": int(improved),
                "time_s": float(time.time() - start),
                **(extra or {}),
            }
        }
        print(f"   → unknown {unknown_rate:.2%} | improved {improved:,} | {time.time() - start:.1f}s")

    # ==============================================================
    # Run orchestrator
    # ==============================================================
    def run(self, methods: List[str] | None):
        all_methods = {
            "rule": self.method_1_rule_based,
            "ml": self.method_2_ml_classification,
            "semantic": self.method_3_semantic_similarity,
            "fuzzy": self.method_4_fuzzy_matching,
            "ensemble": self.method_5_ensemble,
        }
        if methods is None:
            methods = list(all_methods.keys())

        self.prepare_baseline()

        for m in methods:
            fn = all_methods.get(m)
            if not fn:
                continue
            print(f"\n=== START {m.upper()} ===")
            try:
                fn()
            except Exception as exc:
                print(f"🚫  {m} failed: {exc}")
            print(f"===  END  {m.upper()}  ===")

        # choose best
        best = min(
            (k for k in self.results if k != "baseline" and "stats" in self.results[k]),
            key=lambda x: self.results[x]["stats"]["unknown_rate"],
        )
        print(f"\n⭐  Best method: {best}  (unknown {self.results[best]['stats']['unknown_rate']:.2%})")

        # export
        self._export(best)

    # ==============================================================
    # Export results & report
    # ==============================================================
    def _export(self, best_method: str):
        best_col = f"intent_{best_method}"
        self.original_df["intent_augmented"] = self.df[best_col]
        self.original_df["augmentation_method"] = best_method
        self.original_df.to_csv(self.output_dir / "best_augmented_data.csv", index=False)

        summary = pd.DataFrame(
            {k: v["stats"] for k, v in self.results.items() if "stats" in v}
        ).T
        summary.to_csv(self.output_dir / "method_comparison.csv")

        print("\n📊  Outputs written:")
        print("   best dataset :", self.output_dir / "best_augmented_data.csv")
        print("   comparison   :", self.output_dir / "method_comparison.csv")

        # quick bar chart
        plt.figure(figsize=(8, 4))
        plt.bar(summary.index, summary["unknown_rate"] * 100)
        plt.ylabel("Unknown rate (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "unknown_rates.png", dpi=300)
        plt.close()

        # short text report
        with open(self.output_dir / "reports" / "summary.txt", "w") as fh:
            fh.write(
                f"Generated {datetime.now():%Y-%m-%d %H:%M}\n"
                f"Input file : {self.data_path}\n"
                f"Best method: {best_method}\n\n"
                + summary.to_string()
            )
        print("   report      :", self.output_dir / "reports" / "summary.txt")


# ======================================================================
# CLI
# ======================================================================
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Compare intent-augmentation methods")
    ap.add_argument("--input", required=True, help="merged_call_data.csv")
    ap.add_argument("--output", default="augmentation_results", help="output folder")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "rule", "ml", "semantic", "fuzzy", "ensemble"],
        help="subset of methods to run",
    )
    args = ap.parse_args()
    m = None if "all" in args.methods else args.methods
    IntentAugmentationFramework(args.input, args.output).run(m)


if __name__ == "__main__":
    main()