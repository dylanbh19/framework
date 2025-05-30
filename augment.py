"""
intent_augmentation_pro.py  –  2025-05-30
=========================================
End-to-end intent augmentation with rich logging and progress bars.

Minimal requirements
--------------------
pip install pandas numpy tqdm seaborn matplotlib scikit-learn xgboost rapidfuzz

Extra (optional) for better NLP
-------------------------------
pip install sentence-transformers           # semantic
pip install torch transformers              # zeroshot + BERT (GPU)

"""

from __future__ import annotations
import warnings, logging, time, json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
tqdm.pandas()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ------------------------------------------------------------------------------
# Optional heavy libraries
# ------------------------------------------------------------------------------
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from rapidfuzz import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              pipeline, Trainer, TrainingArguments)
    HAS_TRF = True
except ImportError:
    HAS_TRF = False

# ------------------------------------------------------------------------------
# Logging config
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
class IntentAugmentor:
    # --------------------------------------------------------------------------
    def __init__(self, data_path: str, output_dir: str = "augmentation_results"):
        self.in_path  = Path(data_path)
        self.out_dir  = Path(output_dir)
        for sub in ("plots", "reports", "models"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)

        log.info("Loading %s …", self.in_path.name)
        self.df       = pd.read_csv(self.in_path, low_memory=False)
        self.orig_df  = self.df.copy()
        self.results  : Dict[str, Dict[str, Any]] = {}
        self.models   : Dict[str, Any]            = {}

        # canonical intents
        self.STD = [
            "Fraud Assistance","Escheatment","Balance/Value","Sell","Repeat Caller",
            "Name Change","Buy Stock","Statement","Recent Activity","Corporate Action",
            "Data Protection","Press and Media","Privacy Breach","Consolidation",
            "Proxy Inquiry","Complaint Call","General Inquiry","Tax Information",
            "Banking Details","Dividend Payment","Address Change","Check Replacement",
            "Stock Quote","Beneficiary Information","Dividend Reinvestment",
            "Certificate Issuance","Transfer","Existing IC User Login Problem",
            "New IC User Login Problem","Fulfillment","Enrolment","Associate",
            "Lost Certificate","Blank","Unknown"
        ]
        self.LTR = {      # letter-to-intent mapping
            "a":"Balance/Value","b":"Sell","c":"Repeat Caller","d":"Name Change",
            "e":"Buy Stock","f":"Statement","g":"Recent Activity","h":"Tax Information",
            "i":"Banking Details","j":"Dividend Payment","k":"Address Change",
            "l":"Check Replacement","m":"Stock Quote","n":"Beneficiary Information",
            "o":"Dividend Reinvestment","p":"Certificate Issuance","q":"Transfer",
            "r":"Existing IC User Login Problem","s":"New IC User Login Problem",
            "t":"Fulfillment","u":"Enrolment","w":"Associate","x":"Lost Certificate",
            "y":"Blank","z":"Unknown",
        }

    # --------------------------------------------------------------------------
    # Helpers
    def _std(self, raw):
        if pd.isna(raw):
            return "Unknown"
        txt = str(raw).strip()
        if txt.lower() in self.LTR:
            return self.LTR[txt.lower()]
        return txt if txt in self.STD else "Unknown"

    def _store_stats(self, key: str, col: str, t0: float, extra=None):
        unk = (self.df[col] == "Unknown").mean()
        imp = ((self.df["intent_base"] == "Unknown") & (self.df[col] != "Unknown")).sum()
        self.results[key] = {"stats": {
            "unknown_rate": round(float(unk), 4),
            "improved": int(imp),
            "time_s": round(time.time() - t0, 1),
            **(extra or {})
        }}
        log.info("→ %-9s unknown %5.2f%%  | improved %6d  | %5.1fs",
                 key, unk * 100, imp, time.time() - t0)

    # ==========================================================================
    # 0. Baseline
    # ==========================================================================
    def baseline(self):
        col = next((c for c in self.df.columns if "intent" in c.lower()), "Intent")
        if col not in self.df.columns:
            self.df[col] = "Unknown"
        self.df["intent_base"] = self.df[col].apply(self._std)
        rate = (self.df["intent_base"] == "Unknown").mean()
        self.results["baseline"] = {"stats": {"unknown_rate": round(float(rate), 4)}}
        log.info("Baseline unknown %.2f%%", rate * 100)

    # ==========================================================================
    # 1. Rule-based
    # ==========================================================================
    def rule(self):
        log.info("Running rule-based …")
        t0 = time.time()

        activity_map = {"Sell": "Sell", "Transfer": "Transfer"}
        keywords = {
            "Fraud Assistance": ["fraud", "unauthorized"],
            "Sell": ["sell", "liquidate"],
            "Transfer": ["transfer", "acat"],
        }

        def infer(row):
            if row["intent_base"] != "Unknown":
                return row["intent_base"], 1.0
            seq = str(row.get("activity_sequence", "")).split("|")
            for act in seq:
                if act.strip() in activity_map:
                    return activity_map[act.strip()], 0.9
            text = "|".join(seq).lower()
            for intent, words in keywords.items():
                if any(w in text for w in words):
                    return intent, 0.7
            return "Unknown", 0.0

        res = self.df.progress_apply(infer, axis=1, result_type="expand")
        self.df["intent_rule"], self.df["conf_rule"] = res[0], res[1]
        self._store_stats("rule", "intent_rule", t0)

    # ==========================================================================
    # 2. ML  (TF-IDF + RandomForest)
    # ==========================================================================
    def ml(self):
        log.info("Running ML (TF-IDF + RF) …")
        t0 = time.time()

        mask = (self.df["intent_rule"] != "Unknown") & (self.df["conf_rule"] >= 0.7)
        if mask.sum() < 300:
            log.warning("ML skipped – not enough labelled rows")
            return

        subs = self.df[mask].sample(min(60_000, mask.sum()), random_state=1)
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = tfidf.fit_transform(subs["activity_sequence"].fillna(""))
        y = subs["intent_rule"]

        rf = RandomForestClassifier(n_estimators=180, n_jobs=-1, random_state=42, verbose=1)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2,
                                              stratify=y, random_state=42)
        rf.fit(Xtr, ytr)
        f1 = f1_score(yva, rf.predict(Xva), average="macro")

        # predict unknown
        unk = self.df["intent_base"] == "Unknown"
        if unk.any():
            Xunk = tfidf.transform(self.df.loc[unk, "activity_sequence"].fillna(""))
            self.df.loc[unk, "intent_ml"] = rf.predict(Xunk)
            self.df.loc[unk, "conf_ml"]   = rf.predict_proba(Xunk).max(1)

        self.df.loc[~unk, "intent_ml"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_ml"]   = 1.0

        # save feature importance for explainability
        imp = rf.feature_importances_
        top = np.argsort(imp)[-25:][::-1]
        pd.DataFrame({
            "token": tfidf.get_feature_names_out()[top],
            "importance": imp[top]
        }).to_csv(self.out_dir / "rf_top_tokens.csv", index=False)

        self.models.update({"rf": rf, "tfidf": tfidf})
        self._store_stats("ml", "intent_ml", t0, {"f1": round(f1, 3)})

    # ==========================================================================
    # 3. Sentence-BERT semantic similarity
    # ==========================================================================
    def semantic(self):
        if not HAS_SBERT:
            log.warning("semantic skipped – sentence-transformers not installed")
            return
        log.info("Running semantic (MiniLM) …")
        t0 = time.time()

        model = SentenceTransformer("all-MiniLM-L6-v2")
        intent_emb = {i: model.encode(i) for i in self.STD if i != "Unknown"}

        unk = self.df["intent_base"] == "Unknown"
        if not unk.any():
            self._store_stats("semantic", "intent_rule", t0)
            return

        texts = (
            self.df.loc[unk, "activity_sequence"].fillna("") + " " +
            self.df.loc[unk, "first_activity"].fillna("") + " " +
            self.df.loc[unk, "last_activity"].fillna("")
        ).tolist()

        emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                           show_progress_bar=True)
        preds, confs = [], []
        for vec in tqdm(emb, unit="row"):
            sims = {i: cosine_similarity([vec], [e])[0, 0] for i, e in intent_emb.items()}
            best, score = max(sims.items(), key=lambda x: x[1])
            preds.append(best if score >= 0.25 else "Unknown")
            confs.append(float(score))

        self.df.loc[unk, "intent_sem"] = preds
        self.df.loc[unk, "conf_sem"]   = confs
        self.df.loc[~unk, "intent_sem"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_sem"]   = 1.0

        self._store_stats("semantic", "intent_sem", t0)

    # ==========================================================================
    # 4. Fuzzy matching
    # ==========================================================================
    def fuzzy(self):
        if not HAS_FUZZY:
            log.warning("fuzzy skipped – rapidfuzz not installed")
            return
        log.info("Running fuzzy …")
        t0 = time.time()

        kw = {"Transfer": ["transfer", "dtc"], "Sell": ["sell", "liquidate"]}

        def fuzz_match(txt: str):
            txt = txt.lower()
            best, score = "Unknown", 0
            for intent, words in kw.items():
                for w in words:
                    s = fuzz.partial_ratio(w, txt)
                    if s > score:
                        best, score = intent, s
            return best if score >= 70 else "Unknown", score / 100

        unk = self.df["intent_base"] == "Unknown"
        pairs = self.df.loc[unk, "activity_sequence"].fillna("").progress_apply(fuzz_match)
        self.df.loc[unk, "intent_fuzzy"] = pairs.str[0]
        self.df.loc[unk, "conf_fuzzy"]   = pairs.str[1]
        self.df.loc[~unk, "intent_fuzzy"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_fuzzy"]   = 1.0

        self._store_stats("fuzzy", "intent_fuzzy", t0)

    # ==========================================================================
    # 5. Zero-shot (BART-MNLI) – CPU friendly
    # ==========================================================================
        # ==========================================================================
    # 5. ZERO-SHOT  (fast CPU/GPU – batched, distilled model)
    # ==========================================================================
    def zeroshot(self):
        """
        Zero-shot classification using a distilled MNLI model
        (valhalla/distilbart-mnli-12-3).  Batched inference for speed.
        """
        if not HAS_TRF:
            log.warning("zeroshot skipped – transformers not installed")
            return

        t0 = time.time()
        log.info("Running zero-shot (distilbart-mnli-12-3, batched) …")

        # ── choose compact model & pipeline
        zpipe = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",  # ~50 % size of bart-large
            device=-1                               # -1 = CPU   (GPU index otherwise)
        )

        candidate_labels = [i for i in self.STD if i != "Unknown"]

        # ── grab only rows that still need a label
        unk_mask = self.df["intent_base"] == "Unknown"
        total_unk = int(unk_mask.sum())
        if total_unk == 0:
            self._store_stats("zeroshot", "intent_rule", t0)
            return
        log.info("Zero-shot will process %d unknown rows", total_unk)

        # ── shorten text to 256 chars (speed!)
        texts = (
            self.df.loc[unk_mask, "activity_sequence"].fillna("") + " " +
            self.df.loc[unk_mask, "first_activity"].fillna("") + " " +
            self.df.loc[unk_mask, "last_activity"].fillna("")
        ).str.slice(0, 256).tolist()   # truncate

        batch_size = 16          # tweak to your RAM / CPU
        preds, confs, expl = [], [], []

        # ── batch inference with tqdm
        for i in tqdm(range(0, total_unk, batch_size), unit="batch"):
            batch = texts[i:i + batch_size]
            outputs = zpipe(batch,
                            candidate_labels=candidate_labels,
                            multi_label=False,
                            batch_size=batch_size,
                            truncation=True,  # in case text>512
                            top_k=1)          # only best label
            for out in outputs:
                preds.append(out["labels"][0])
                confs.append(float(out["scores"][0]))
                # crude explanation = 3 longest tokens (optional)
                tk = sorted(out["sequence"].split(), key=len, reverse=True)[:3]
                expl.append(" | ".join(tk))

        # ── write back
        self.df.loc[unk_mask, "intent_zeroshot"]   = preds
        self.df.loc[unk_mask, "conf_zeroshot"]     = confs
        self.df.loc[unk_mask, "explain_zeroshot"]  = expl

        # copy over for rows already labelled
        self.df.loc[~unk_mask, ["intent_zeroshot"]]  = self.df.loc[~unk_mask, "intent_rule"].values
        self.df.loc[~unk_mask, ["conf_zeroshot"]]    = 1.0
        self.df.loc[~unk_mask, ["explain_zeroshot"]] = ""

        # store pipeline for potential reuse
        self.models["zeroshot_pipeline"] = zpipe

        self._store_stats("zeroshot", "intent_zeroshot", t0)

    # ==========================================================================
    # 6. BERT fine-tune (GPU only)
    # ==========================================================================
    def bert(self):
        if not HAS_TRF:
            log.warning("bert skipped – transformers not installed")
            return
        if not torch.cuda.is_available():
            log.warning("bert skipped – GPU not detected")
            return
        log.info("Running BERT fine-tune …")
        t0 = time.time()

        mask = (self.df["intent_rule"] != "Unknown") & (self.df["conf_rule"] >= 0.9)
        if mask.sum() < 800:
            log.warning("bert skipped – not enough high-confidence labels")
            return

        labs = self.df.loc[mask, "intent_rule"].unique().tolist()
        lab2id = {l: i for i, l in enumerate(labs)}

        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tok(self.df.loc[mask, "activity_sequence"].fillna("").tolist(),
                     truncation=True, padding=True)

        class DS(torch.utils.data.Dataset):
            def __init__(self, enc, y):
                self.enc = enc
                self.y   = [lab2id[i] for i in y]

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.y[idx])
                return item

        ds = DS(inputs, self.df.loc[mask, "intent_rule"])
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(labs)).cuda()
        trainer = Trainer(
            model,
            TrainingArguments(output_dir=self.out_dir/"bert_tmp",
                              per_device_train_batch_size=16,
                              num_train_epochs=1,
                              report_to=None,
                              logging_steps=50),
            train_dataset=ds
        )
        trainer.train()

        # predict unknown
        unk = self.df["intent_base"] == "Unknown"
        if unk.sum():
            enc2 = tok(self.df.loc[unk, "activity_sequence"].fillna("").tolist(),
                       truncation=True, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                probs = model(**enc2).logits.softmax(dim=1).cpu().numpy()
            pred = [labs[i] for i in probs.argmax(1)]
            self.df.loc[unk, "intent_bert"] = pred
            self.df.loc[unk, "conf_bert"]   = probs.max(1)

        self.df.loc[~unk, "intent_bert"] = self.df.loc[~unk, "intent_rule"]
        self.df.loc[~unk, "conf_bert"]   = 1.0

        self._store_stats("bert", "intent_bert", t0)

    # ==========================================================================
    # 7. Ensemble vote
    # ==========================================================================
    def ensemble(self):
        log.info("Running ensemble …")
        t0 = time.time()

        pairs = [(i, i.replace("intent_", "conf_"))
                 for i in self.df.columns
                 if i.startswith("intent_") and i not in ("intent_base", "intent_ensemble")]

        pairs = [p for p in pairs if p[1] in self.df.columns]
        if len(pairs) < 2:
            log.warning("ensemble skipped – need ≥2 prediction sources")
            return

        def voter(row):
            scores = {}
            for i_col, c_col in pairs:
                intent = row[i_col]
                conf   = row[c_col]
                if intent != "Unknown":
                    scores[intent] = scores.get(intent, 0) + conf
            if not scores:
                return "Unknown", 0.0
            best = max(scores, key=scores.get)
            total = sum(scores.values())
            return best, scores[best] / total if total else 0.0

        res = self.df.progress_apply(voter, axis=1, result_type="expand")
        self.df["intent_ensemble"], self.df["conf_ensemble"] = res[0], res[1]

        self._store_stats("ensemble", "intent_ensemble", t0,
                          {"sources": [p[0] for p in pairs]})

    # ==========================================================================
    # Main orchestrator
    # ==========================================================================
    def run(self, methods: List[str] | None):
        self.baseline()
        registry = {
            "rule": self.rule, "ml": self.ml, "semantic": self.semantic,
            "fuzzy": self.fuzzy, "zeroshot": self.zeroshot,
            "bert": self.bert, "ensemble": self.ensemble
        }
        todo = methods or list(registry.keys())
        for m in todo:
            log.info("========== %s ==========", m.upper())
            try:
                registry[m]()
            except Exception as e:
                log.error("%s failed: %s", m, e, exc_info=True)

        best = min((k for k in self.results if k != "baseline"),
                   key=lambda k: self.results[k]["stats"]["unknown_rate"])
        log.info("★ BEST METHOD: %s  (unknown %.2f%%)",
                 best, self.results[best]["stats"]["unknown_rate"] * 100)
        self.export(best)

    # ==========================================================================
    # Export
    # ==========================================================================
    def export(self, best: str):
        out_best = self.out_dir / "best_augmented_data.csv"
        out_cmp  = self.out_dir / "method_comparison.csv"

        self.orig_df["intent_augmented"]  = self.df[f"intent_{best}"]
        self.orig_df["intent_confidence"] = self.df.get(f"conf_{best}", 1.0)
        self.orig_df["aug_method"]        = best
        self.orig_df.to_csv(out_best, index=False)

        pd.DataFrame({k: v["stats"] for k, v in self.results.items() if "stats" in v}) \
          .T.to_csv(out_cmp)

        # quick bar plot
        summary = pd.read_csv(out_cmp)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=summary.index, y=summary["unknown_rate"] * 100)
        plt.ylabel("Unknown (%)"); plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.out_dir / "plots" / "unknown_rates.png", dpi=300)
        plt.close()

        # summary text
        with open(self.out_dir / "reports" / "summary.txt", "w") as f:
            f.write(f"Generated {datetime.now():%Y-%m-%d %H:%M}\nBest={best}\n\n"
                    + summary.to_string())

        log.info("Outputs saved in %s", self.out_dir)

# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Intent augmentation pipeline (pro version)")
    ap.add_argument("--input", required=True, help="merged_call_data.csv")
    ap.add_argument("--output", default="augmentation_results_pro")
    ap.add_argument("--methods", nargs="+", default=["all"],
                    choices=["all", "rule", "ml", "semantic", "fuzzy",
                             "zeroshot", "bert", "ensemble"])
    args = ap.parse_args()
    selected = None if "all" in args.methods else args.methods
    IntentAugmentor(args.input, args.output).run(selected)

if __name__ == "__main__":
    main()