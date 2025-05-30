"""
Intent Augmentation Comparison Framework
========================================

Benchmarks several advanced techniques that replace “Unknown” intents in
the merged call-data file you produced with the feature-engineering
pipeline.

--------------------------------------------------------------
Quick start
-----------
1.  Finish your feature-engineering pipeline so it writes
    `merged_call_data.csv` (with columns like activity_sequence,
    first_activity, last_activity, etc.).

2.  Install extra libraries (besides scikit-learn, xgboost, matplotlib):

    pip install sentence-transformers rapidfuzz modAL-python

   •  If you plan to try BERT on GPU, also install torch + transformers.

3.  Run *every* method and export results:

    python intent_augmentation_comparison.py \
           --input  merged_call_data.csv \
           --output augmentation_results \
           --methods all
--------------------------------------------------------------
"""

# ------------------------------------------------------------------
# Standard imports
# ------------------------------------------------------------------
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ------------------------------------------------------------------
# Optional-dependency guards
# ------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

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

try:
    from modAL.models import ActiveLearner  # noqa: F401

    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


# ------------------------------------------------------------------
# Main framework class
# ------------------------------------------------------------------
class IntentAugmentationFramework:
    """Run and compare multiple augmentation approaches."""

    # ----------------------------------------
    # Construction
    # ----------------------------------------
    def __init__(self, data_path: str, output_dir: str = "augmentation_results"):
        self.data_path = Path(data_path)
        self.df: pd.DataFrame = pd.read_csv(self.data_path)
        self.original_df = self.df.copy()

        self.output_dir = Path(output_dir)
        for sub in ("plots", "reports", "models"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Any] = {}

        # Standard categories (26 + 8 misc from earlier work)
        self.STANDARD_INTENTS = [
            "Fraud Assistance",
            "Escheatment",
            "Balance/Value",
            "Sell",
            "Repeat Caller",
            "Name Change",
            "Buy Stock",
            "Statement",
            "Recent Activity",
            "Corporate Action",
            "Data Protection",
            "Press and Media",
            "Privacy Breach",
            "Consolidation",
            "Proxy Inquiry",
            "Complaint Call",
            "General Inquiry",
            "Tax Information",
            "Banking Details",
            "Dividend Payment",
            "Address Change",
            "Check Replacement",
            "Stock Quote",
            "Beneficiary Information",
            "Dividend Reinvestment",
            "Certificate Issuance",
            "Transfer",
            "Existing IC User Login Problem",
            "New IC User Login Problem",
            "Fulfillment",
            "Enrolment",
            "Associate",
            "Lost Certificate",
            "Blank",
            "Unknown",
        ]

        self.LETTER_TO_INTENT = {
            "a": "Balance/Value",
            "b": "Sell",
            "c": "Repeat Caller",
            "d": "Name Change",
            "e": "Buy Stock",
            "f": "Statement",
            "g": "Recent Activity",
            "h": "Tax Information",
            "i": "Banking Details",
            "j": "Dividend Payment",
            "k": "Address Change",
            "l": "Check Replacement",
            "m": "Stock Quote",
            "n": "Beneficiary Information",
            "o": "Dividend Reinvestment",
            "p": "Certificate Issuance",
            "q": "Transfer",
            "r": "Existing IC User Login Problem",
            "s": "New IC User Login Problem",
            "t": "Fulfillment",
            "u": "Enrolment",
            "w": "Associate",
            "x": "Lost Certificate",
            "y": "Blank",
            "z": "Unknown",
        }

    # ==================================================================
    # BASELINE
    # ==================================================================
    def prepare_baseline(self) -> None:
        print("\n============= BASELINE =============")
        intent_cols = [c for c in self.df.columns if "intent" in c.lower()]

        self.intent_col = intent_cols[0] if intent_cols else "Intent"
        if self.intent_col not in self.df.columns:
            self.df[self.intent_col] = "Unknown"

        self.df["intent_baseline"] = self.df[self.intent_col].apply(
            self._standardize_intent
        )

        stats = {
            "unknown_count": (self.df["intent_baseline"] == "Unknown").sum(),
            "unknown_rate": (self.df["intent_baseline"] == "Unknown").mean(),
            "distribution": self.df["intent_baseline"].value_counts().to_dict(),
        }
        self.results["baseline"] = {"stats": stats}
        print(f"Baseline unknown rate: {stats['unknown_rate']:.2%}")

    def _standardize_intent(self, raw):
        if pd.isna(raw):
            return "Unknown"
        txt = str(raw).strip()
        if txt.lower() in self.LETTER_TO_INTENT:
            return self.LETTER_TO_INTENT[txt.lower()]
        return txt if txt in self.STANDARD_INTENTS else "Unknown"

    # ==================================================================
    # METHOD 1 — rule-based
    # ==================================================================
    def method_1_rule_based(self) -> None:
        print("\n============= RULE-BASED ===========")
        begin = time.time()

        # Simple map (extend as needed)
        activity_to_intent = {
            "Sell": "Sell",
            "Tax Information": "Tax Information",
            "Dividend Payment": "Dividend Payment",
            "Certificate Issuance": "Certificate Issuance",
            "Transfer": "Transfer",
            "Address Change": "Address Change",
            "Check Replacement": "Check Replacement",
            "Stock Quote": "Stock Quote",
            "Beneficiary Information": "Beneficiary Information",
            "Name Change": "Name Change",
            "Banking Details": "Banking Details",
        }

        keyword_patterns = {
            "Fraud Assistance": ["fraud", "unauthorized"],
            "Tax Information": ["tax", "1099", "irs"],
            "Dividend Payment": ["dividend", "distribution"],
            "Transfer": ["transfer", "acat", "dtc"],
            "Sell": ["sell", "liquidate"],
        }

        def infer(row):
            if row["intent_baseline"] != "Unknown":
                return row["intent_baseline"], 0.9, "baseline"

            seq = str(row.get("activity_sequence", "")).split("|")
            for act in seq:
                if act.strip() in activity_to_intent:
                    return activity_to_intent[act.strip()], 0.9, "activity_direct"

            text = "|".join(seq).lower()
            for intent, kw in keyword_patterns.items():
                if any(k in text for k in kw):
                    return intent, 0.7, "keyword_match"

            return "Unknown", 0.0, "unknown"

        res = self.df.apply(infer, axis=1)
        self.df["intent_rule_based"] = res.str[0]
        self.df["confidence_rule_based"] = res.str[1]
        self.df["source_rule_based"] = res.str[2]

        self._store_metrics("rule_based", "intent_rule_based", begin)

    # ==================================================================
    # METHOD 2 — ML (RF + XGB, TF-IDF)
    # ==================================================================
    def method_2_ml_classification(self) -> None:
        print("\n============= ML CLASSIFIER =========")
        begin = time.time()

        # labelled rows
        train_mask = (self.df["intent_rule_based"] != "Unknown") & (
            self.df["confidence_rule_based"] >= 0.7
        )
        if train_mask.sum() < 200:
            print("Not enough labelled rows – skipping ML.")
            return

        tfidf = TfidfVectorizer(max_features=800, ngram_range=(1, 2))
        X_train = tfidf.fit_transform(
            self.df.loc[train_mask, "activity_sequence"].fillna("")
        )
        y_train = self.df.loc[train_mask, "intent_rule_based"]

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.3,
            objective="multi:softprob",
            random_state=42,
        )

        cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"RF CV-F1: {cv.mean():.3f}")

        rf.fit(X_train, y_train)
        xgb_clf.fit(X_train, y_train)

        # predict unknowns
        unk_mask = self.df["intent_baseline"] == "Unknown"
        if unk_mask.sum() > 0:
            X_unk = tfidf.transform(self.df.loc[unk_mask, "activity_sequence"].fillna(""))

            rf_pred = rf.predict_proba(X_unk)
            xgb_pred = xgb_clf.predict_proba(X_unk)

            # choose model with higher prob row-by-row
            best_idx = rf_pred.max(1) < xgb_pred.max(1)
            final_pred = rf.predict(X_unk)
            final_conf = rf_pred.max(1)
            final_pred[best_idx] = xgb_clf.classes_[xgb_pred.argmax(1)][best_idx]
            final_conf[best_idx] = xgb_pred.max(1)[best_idx]

            self.df.loc[unk_mask, "intent_ml"] = final_pred
            self.df.loc[unk_mask, "confidence_ml"] = final_conf

        # keep existing for known rows
        self.df.loc[~unk_mask, "intent_ml"] = self.df.loc[~unk_mask, "intent_rule_based"]
        self.df.loc[~unk_mask, "confidence_ml"] = 1.0

        self.models["ml_tfidf"] = tfidf
        self.models["ml_rf"] = rf
        self.models["ml_xgb"] = xgb_clf
        self._store_metrics("ml", "intent_ml", begin, extra={"cv_f1": cv.mean()})

    # ==================================================================
    # METHOD 3 — semantic similarity (Sentence-Transformers)
    # ==================================================================
    def method_3_semantic_similarity(self) -> None:
        print("\n============= SEMANTIC SIMILARITY ===")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("sentence-transformers not installed.")
            return
        begin = time.time()

        model = SentenceTransformer("all-MiniLM-L6-v2")
        intent_text = {i: i for i in self.STANDARD_INTENTS if i != "Unknown"}
        intent_emb = {i: model.encode(txt) for i, txt in intent_text.items()}

        unk_mask = self.df["intent_baseline"] == "Unknown"
        if unk_mask.sum() == 0:
            self._store_metrics("semantic", None, begin)
            return

        texts = self.df.loc[unk_mask, "activity_sequence"].fillna("").tolist()
        batch_emb = model.encode(texts, batch_size=128, convert_to_numpy=True)

        preds, confs = [], []
        for e in batch_emb:
            sims = {k: cosine_similarity([e], [v])[0, 0] for k, v in intent_emb.items()}
            best = max(sims, key=sims.get)
            preds.append(best)
            confs.append(sims[best])

        self.df.loc[unk_mask, "intent_semantic"] = preds
        self.df.loc[unk_mask, "confidence_semantic"] = confs
        self.df.loc[~unk_mask, "intent_semantic"] = self.df.loc[
            ~unk_mask, "intent_rule_based"
        ]
        self.df.loc[~unk_mask, "confidence_semantic"] = 1.0

        self.models["semantic_model"] = model
        self._store_metrics("semantic", "intent_semantic", begin)

    # ==================================================================
    # METHOD 4 — fuzzy string match (rapidfuzz)
    # ==================================================================
    def method_4_fuzzy_matching(self) -> None:
        print("\n============= FUZZY MATCHING ========")
        if not RAPIDFUZZ_AVAILABLE:
            print("rapidfuzz not installed.")
            return
        begin = time.time()

        keywords = {
            "Sell": ["sell", "liquidate"],
            "Tax Information": ["tax", "irs", "1099"],
            "Dividend Payment": ["dividend"],
            "Transfer": ["transfer", "acat"],
            "Balance/Value": ["balance", "value"],
        }

        def match(txt: str):
            txt = txt.lower()
            best_int, best_sc = "Unknown", 0.0
            for intent, kws in keywords.items():
                for k in kws:
                    sc = fuzz.partial_ratio(k, txt) / 100.0
                    if sc > best_sc:
                        best_int, best_sc = intent, sc
            return (best_int, best_sc) if best_sc >= 0.7 else ("Unknown", 0.0)

        unk_mask = self.df["intent_baseline"] == "Unknown"
        preds, confs = zip(
            *self.df.loc[unk_mask, "activity_sequence"]
            .fillna("")
            .apply(match)
            .tolist()
        )
        self.df.loc[unk_mask, "intent_fuzzy"] = preds
        self.df.loc[unk_mask, "confidence_fuzzy"] = confs
        self.df.loc[~unk_mask, "intent_fuzzy"] = self.df.loc[
            ~unk_mask, "intent_rule_based"
        ]
        self.df.loc[~unk_mask, "confidence_fuzzy"] = 1.0

        self._store_metrics("fuzzy", "intent_fuzzy", begin)

    # ==================================================================
    # METHOD 5 — ensemble vote
    # ==================================================================
    def method_5_ensemble(self) -> None:
        print("\n============= ENSEMBLE VOTE =========")
        begin = time.time()

        source_cols = {
            "rule_based": ("intent_rule_based", "confidence_rule_based"),
            "ml": ("intent_ml", "confidence_ml"),
            "semantic": ("intent_semantic", "confidence_semantic"),
            "fuzzy": ("intent_fuzzy", "confidence_fuzzy"),
        }
        avail = {k: v for k, v in source_cols.items() if v[0] in self.df.columns}
        if len(avail) < 2:
            print("Not enough methods available for ensemble.")
            return

        def vote(row):
            tallies: Dict[str, float] = {}
            for intent_col, conf_col in avail.values():
                intent, conf = row[intent_col], row.get(conf_col, 0.5)
                if intent != "Unknown":
                    tallies[intent] = tallies.get(intent, 0.0) + conf
            if not tallies:
                return "Unknown", 0.0
            best = max(tallies, key=tallies.get)
            return best, tallies[best] / sum(tallies.values())

        res = self.df.apply(vote, axis=1)
        self.df["intent_ensemble"], self.df["confidence_ensemble"] = res.str
        self._store_metrics("ensemble", "intent_ensemble", begin)

    # ==================================================================
    # METHOD 6 — BERT placeholder (GPU only)
    # ==================================================================
    def method_6_bert_classification(self) -> None:
        print("\n============= BERT (GPU) ============")
        if not TRANSFORMERS_AVAILABLE or not torch.cuda.is_available():
            print("BERT skipped – Transformers or GPU not available.")
            return
        # full fine-tuning would be >100 lines – skipped in this demo
        self.results["bert"] = {"error": "Not implemented in demo"}

    # ==================================================================
    # Convenience: compute + store metrics
    # ==================================================================
    def _store_metrics(
        self,
        key: str,
        intent_col: str | None,
        start_time: float,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        intent_col = intent_col or f"intent_{key}"
        if intent_col not in self.df.columns:
            return

        unknown_rate = (self.df[intent_col] == "Unknown").mean()
        improved = ((self.df["intent_baseline"] == "Unknown") & (self.df[intent_col] != "Unknown")).sum()  # type: ignore[operator]  # noqa: E501
        self.results[key] = {
            "stats": {
                "unknown_rate": float(unknown_rate),
                "improved_count": int(improved),
                "execution_time": time.time() - start_time,
                **(extra or {}),
            }
        }
        print(f"{key} unknown rate: {unknown_rate:.2%}  |  improved: {improved:,}")

    # ==================================================================
    # Comparison + reporting
    # ==================================================================
    def compare_and_export(self):
        print("\n============= EXPORT & REPORT =======")

        # summary CSV
        summary = []
        for k, v in self.results.items():
            if "stats" in v:
                summary.append({"method": k, **v["stats"]})
        pd.DataFrame(summary).to_csv(self.output_dir / "method_comparison.csv", index=False)

        # merged dataset (best column wins)
        best = min(
            (k for k in self.results if "stats" in self.results[k]),
            key=lambda x: self.results[x]["stats"]["unknown_rate"],
        )
        best_col = f"intent_{best}"
        self.original_df["intent_augmented"] = self.df[best_col]
        self.original_df["augmentation_method"] = best
        self.original_df.to_csv(self.output_dir / "best_augmented_data.csv", index=False)
        print(f"Best method: {best}  |  results saved to best_augmented_data.csv")

        # text report
        with open(self.output_dir / "reports" / "summary.txt", "w") as fh:
            fh.write(
                f"Generated {datetime.now():%Y-%m-%d %H:%M}\n"
                f"Input: {self.data_path}\n\n"
                + pd.DataFrame(summary).to_string(index=False)
            )
        print("Report written.")

    # ==================================================================
    # Orchestrator
    # ==================================================================
    def run(self, method_list: List[str] | None = None):
        self.prepare_baseline()

        mapping = {
            "rule_based": self.method_1_rule_based,
            "ml": self.method_2_ml_classification,
            "semantic": self.method_3_semantic_similarity,
            "fuzzy": self.method_4_fuzzy_matching,
            "ensemble": self.method_5_ensemble,
            "bert": self.method_6_bert_classification,
        }

        method_seq = method_list or list(mapping.keys())
        for m in method_seq:
            if m in mapping:
                try:
                    mapping[m]()
                except Exception as exc:
                    print(f"Method {m} failed: {exc}")

        self.compare_and_export()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Compare intent augmentation methods")
    ap.add_argument("--input", required=True, help="merged_call_data.csv")
    ap.add_argument("--output", default="augmentation_results", help="output folder")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "rule_based", "ml", "semantic", "fuzzy", "ensemble", "bert"],
        help="subset of methods to run",
    )
    args = ap.parse_args()

    if args.methods == ["all"]:
        methods = None
    else:
        methods = args.methods

    IntentAugmentationFramework(args.input, args.output).run(methods)


if __name__ == "__main__":
    main()
    
    
    
    
    





# 1.  Ensure merged_call_data.csv exists (from your feature pipeline).

# 2.  Install extra libs (pick what you need):
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
pip install sentence-transformers rapidfuzz modAL-python   # semantic / fuzzy / active-learning
pip install torch transformers                             # only if you want the BERT placeholder

# 3.  Run every method and write results to ./augmentation_results
python intent_augmentation_comparison.py --input merged_call_data.csv --output augmentation_results --methods all