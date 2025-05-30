# ╔══════════════════════════════════════════════════════════════╗
# ║  intent_augmentation_pro.py  –  v2025-05-30                 ║
# ║  • 7 augmentation methods (rule, ML, semantic, fuzzy, BERT)  ║
# ║  • tqdm progress bars, rich logging                          ║
# ║  • GPU-aware BERT; auto-skips on CPU                         ║
# ║  • automatic unknown/improved metrics                        ║
# ║  • exports scores, plots, best-dataset                       ║
# ╚══════════════════════════════════════════════════════════════╝
"""
Dependencies
------------
pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm rapidfuzz
pip install sentence-transformers
# (optional for GPU / BERT)
pip install torch transformers

Run
---
python intent_augmentation_pro.py --input merged_call_data.csv
"""

from __future__ import annotations
import json, pickle, time, warnings, logging, os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
tqdm.pandas()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ─── optional heavy libs ────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

try:
    from rapidfuzz import fuzz
    RAPID_OK = True
except ImportError:
    RAPID_OK = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import Trainer, TrainingArguments
    BERT_OK = True
except ImportError:
    BERT_OK = False

# ─── logging ────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
class IntentAugmentor:
    """Compare multiple augmentation methods, pick the best, export."""
    # ── init ────────────────────────────────────────────────────
    def __init__(self,
                 data_path: str,
                 output_dir: str = "augmentation_results"):
        self.data_path = Path(data_path)
        self.out_dir   = Path(output_dir)
        for sub in ("plots", "reports", "models"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)

        log.info("Loading %s", self.data_path.name)
        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.orig_df = self.df.copy()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.models : Dict[str, Any] = {}

        self.STD_INTENTS = [
            # (same 34 intents as earlier)
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
        self.LTR = {
            "a":"Balance/Value","b":"Sell","c":"Repeat Caller","d":"Name Change",
            "e":"Buy Stock","f":"Statement","g":"Recent Activity","h":"Tax Information",
            "i":"Banking Details","j":"Dividend Payment","k":"Address Change",
            "l":"Check Replacement","m":"Stock Quote","n":"Beneficiary Information",
            "o":"Dividend Reinvestment","p":"Certificate Issuance","q":"Transfer",
            "r":"Existing IC User Login Problem","s":"New IC User Login Problem",
            "t":"Fulfillment","u":"Enrolment","w":"Associate","x":"Lost Certificate",
            "y":"Blank","z":"Unknown",
        }

    # ── utility ────────────────────────────────────────────────
    def _std_intent(self, raw):
        if pd.isna(raw):
            return "Unknown"
        txt = str(raw).strip()
        if txt.lower() in self.LTR:
            return self.LTR[txt.lower()]
        return txt if txt in self.STD_INTENTS else "Unknown"

    def _metric(self, key, col, start, extra=None):
        unk = (self.df[col] == "Unknown").mean()
        imp = ((self.df["intent_base"] == "Unknown") &
               (self.df[col] != "Unknown")).sum()
        self.results[key] = {
            "stats":{
                "unknown_rate": float(unk),
                "improved": int(imp),
                "time_s": round(time.time()-start,2),
                **(extra or {})
            }
        }
        log.info("→ %-9s  unknown %5.2f%% | improved %6d | %5.1fs",
                 key, unk*100, imp, time.time()-start)

    # ── baseline ------------------------------------------------
    def baseline(self):
        log.info("▶ baseline")
        col = next((c for c in self.df.columns if "intent" in c.lower()), "Intent")
        if col not in self.df.columns:
            self.df[col] = "Unknown"
        self.df["intent_base"] = self.df[col].apply(self._std_intent)
        self.results["baseline"]={"stats":{
            "unknown_rate": float((self.df["intent_base"]=="Unknown").mean())
        }}

    # ── method: RULE -------------------------------------------
    def rule(self):
        log.info("▶ rule-based")
        t0 = time.time()
        act_map = {"Sell":"Sell","Tax Information":"Tax Information",
                   "Dividend Payment":"Dividend Payment","Transfer":"Transfer"}
        kw = {"Fraud Assistance":["fraud","unauthorized"],
              "Sell":["sell","liquidate"],"Transfer":["transfer","acat"]}
        def fn(r):
            if r["intent_base"]!="Unknown":
                return r["intent_base"],1.0
            seq = str(r.get("activity_sequence","")).split("|")
            for a in seq:
                if a.strip() in act_map:
                    return act_map[a.strip()],0.9
            text = "|".join(seq).lower()
            for intent, words in kw.items():
                if any(w in text for w in words):
                    return intent,0.7
            return "Unknown",0.0
        res = self.df.progress_apply(fn, axis=1, result_type="expand")
        self.df["intent_rule"], self.df["conf_rule"] = res[0], res[1]
        self._metric("rule","intent_rule",t0)

    # ── method: ML TF-IDF + RF + optional XGB -------------------
    def ml(self):
        log.info("▶ ML (TF-IDF + RF)")
        t0 = time.time()
        mask = (self.df["intent_rule"]!="Unknown")&(self.df["conf_rule"]>=0.7)
        if mask.sum()<300:
            log.warning("Not enough labelled rows – skip ML"); return
        MAX = 50_000
        mask_sub = mask.sample(MAX, random_state=1) if mask.sum()>MAX else mask
        tfidf = TfidfVectorizer(max_features=800, ngram_range=(1,2))
        X = tfidf.fit_transform(self.df.loc[mask_sub,"activity_sequence"].fillna(""))
        y = self.df.loc[mask_sub,"intent_rule"]
        rf = RandomForestClassifier(n_estimators=150,n_jobs=-1,random_state=42,verbose=1)
        Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
        rf.fit(Xtr,ytr)
        f1 = f1_score(yva, rf.predict(Xva), average="macro")
        # predict unknowns
        unk = self.df["intent_base"]=="Unknown"
        if unk.any():
            Xunk = tfidf.transform(self.df.loc[unk,"activity_sequence"].fillna(""))
            self.df.loc[unk,"intent_ml"] = rf.predict(Xunk)
            self.df.loc[unk,"conf_ml"]   = rf.predict_proba(Xunk).max(1)
        self.df.loc[~unk,"intent_ml"] = self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_ml"]   = 1.0
        self.models.update({"ml_tfidf":tfidf,"ml_rf":rf})
        self._metric("ml","intent_ml",t0,{"val_f1":round(f1,3)})

    # ── method: SEMANTIC (Sentence-BERT) -------------------------
    def semantic(self):
        if not SBERT_OK:
            log.warning("skip semantic – sentence_transformers not installed"); return
        log.info("▶ semantic")
        t0=time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_int = {i:model.encode(i) for i in self.STD_INTENTS if i!="Unknown"}
        unk = self.df["intent_base"]=="Unknown"
        if not unk.any():
            self._metric("semantic","intent_rule",t0); return
        texts = (self.df.loc[unk,"activity_sequence"].fillna("")+" "+
                 self.df.loc[unk,"first_activity"].fillna("")+" "+
                 self.df.loc[unk,"last_activity"].fillna("")).tolist()
        emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                           show_progress_bar=True)
        preds,conf=[] ,[]
        for e in tqdm(emb,unit="row",desc="score"):
            sims={k:cosine_similarity([e],[v])[0,0] for k,v in emb_int.items()}
            best,score=max(sims.items(), key=lambda x:x[1])
            preds.append(best if score>=0.25 else "Unknown")
            conf.append(float(score))
        self.df.loc[unk,"intent_sem"]=preds
        self.df.loc[unk,"conf_sem"]=conf
        self.df.loc[~unk,"intent_sem"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_sem"]=1.0
        self.models["semantic_model"]=model
        self._metric("semantic","intent_sem",t0)

    # ── method: FUZZY -------------------------------------------
    def fuzzy(self):
        if not RAPID_OK:
            log.warning("skip fuzzy – rapidfuzz not installed"); return
        log.info("▶ fuzzy")
        t0=time.time()
        kw = {"Sell":["sell","liquidate"],"Transfer":["transfer","acat"]}
        def fm(txt:str):
            txt=txt.lower(); best,bs="Unknown",0
            for intent, words in kw.items():
                for w in words:
                    sc = fuzz.partial_ratio(w,txt)
                    if sc>bs: best,bs=intent,sc
            return best if bs>=70 else "Unknown", bs/100
        unk=self.df["intent_base"]=="Unknown"
        pairs = self.df.loc[unk,"activity_sequence"].fillna("").progress_apply(fm)
        self.df.loc[unk,"intent_fuzzy"]=pairs.str[0]
        self.df.loc[unk,"conf_fuzzy"]=pairs.str[1]
        self.df.loc[~unk,"intent_fuzzy"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_fuzzy"]=1.0
        self._metric("fuzzy","intent_fuzzy",t0)

    # ── method: BERT fine-tune (optional GPU) --------------------
    def bert(self):
        if not BERT_OK:
            log.warning("skip BERT – transformers/torch not installed"); return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device=="cpu":
            log.warning("BERT skipped – GPU not available"); return
        log.info("▶ BERT (distilbert-base-uncased)")
        t0=time.time()
        # label encoding
        mask = (self.df["intent_rule"]!="Unknown")&(self.df["conf_rule"]>=0.9)
        if mask.sum()<500:
            log.warning("Not enough high-confidence labels – skip BERT")
            return
        lbl = self.df["intent_rule"][mask].unique().tolist()
        le = {l:i for i,l in enumerate(lbl)}
        num2lab = {v:k for k,v in le.items()}
        # dataset
        texts = self.df.loc[mask,"activity_sequence"].fillna("").tolist()
        y     = [le[i] for i in self.df.loc[mask,"intent_rule"]]
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        enc = tokenizer(texts, truncation=True, padding=True)
        class DS(torch.utils.data.Dataset):
            def __init__(s,e,y): s.e=e; s.y=y
            def __len__(s): return len(s.y)
            def __getitem__(s,i):
                item={k:torch.tensor(v[i]) for k,v in s.e.items()}
                item["labels"]=torch.tensor(s.y[i])
                return item
        ds = DS(enc, y)
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(lbl)).to(device)
        args = TrainingArguments(
            output_dir=self.out_dir/"bert_tmp", per_device_train_batch_size=16,
            num_train_epochs=1, logging_steps=20, report_to=None)
        Trainer(model,args,train_dataset=ds).train()
        # predict unknown
        unk=self.df["intent_base"]=="Unknown"
        if unk.sum():
            texts_unk=self.df.loc[unk,"activity_sequence"].fillna("").tolist()
            enc2=tokenizer(texts_unk, truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits=model(**{k:v.to(device) for k,v in enc2.items()}).logits
            probs=torch.nn.functional.softmax(logits,dim=1).cpu().numpy()
            pred_ids=probs.argmax(1)
            self.df.loc[unk,"intent_bert"]=[num2lab[i] for i in pred_ids]
            self.df.loc[unk,"conf_bert"]=probs.max(1)
        self.df.loc[~unk,"intent_bert"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_bert"]=1.0
        self.models["bert"]=model; self.models["bert_tokenizer"]=tokenizer
        self._metric("bert","intent_bert",t0)

    # ── method: ENSEMBLE ----------------------------------------
    def ensemble(self):
        log.info("▶ ensemble")
        t0=time.time()
        cols=[c for c in self.df.columns if c.startswith("intent_") and c!="intent_base"]
        conf_cols=[c.replace("intent_","conf_") for c in cols]
        cols=[c for c,cc in zip(cols,conf_cols) if cc in self.df.columns]
        conf_cols=[cc for cc in conf_cols if cc in self.df.columns]
        if len(cols)<2:
            log.warning("not enough predictors – skip ensemble"); return
        def vote(row):
            scores={}
            for i_col,c_col in zip(cols,conf_cols):
                i=row[i_col]; c=row[c_col]
                if i!="Unknown": scores[i]=scores.get(i,0)+c
            if not scores: return "Unknown",0.0
            best=max(scores,key=scores.get)
            return best, scores[best]/sum(scores.values())
        res=self.df.progress_apply(vote,axis=1,result_type="expand")
        self.df["intent_ens"],self.df["conf_ens"]=res[0],res[1]
        self._metric("ensemble","intent_ens",t0,{"models":cols})

    # ── run orchestrator ----------------------------------------
    def run(self, methods: List[str]|None):
        self.baseline()
        all_methods = {
            "rule":self.rule, "ml":self.ml, "semantic":self.semantic,
            "fuzzy":self.fuzzy, "bert":self.bert, "ensemble":self.ensemble
        }
        todo = methods or list(all_methods.keys())
        for m in todo:
            fn = all_methods.get(m)
            if fn:
                log.info("==========  %s  ==========", m.upper())
                try: fn()
                except Exception as e: log.error("Method %s failed: %s", m, e, exc_info=True)
        # choose best
        best = min((k for k in self.results if k!="baseline"),
                   key=lambda k:self.results[k]["stats"]["unknown_rate"])
        log.info("★ Best method: %s (unknown %.2f%%)",
                 best, self.results[best]["stats"]["unknown_rate"]*100)
        self.export(best)

    # ── export ---------------------------------------------------
    def export(self, best):
        self.orig_df["intent_augmented"]=self.df[f"intent_{best}"]
        self.orig_df["intent_confidence"]=self.df.get(f"conf_{best}",1.0)
        self.orig_df["aug_method"]=best
        self.orig_df.to_csv(self.out_dir/"best_augmented_data.csv", index=False)
        pd.DataFrame(
            {k:v["stats"] for k,v in self.results.items() if "stats" in v}
        ).T.to_csv(self.out_dir/"method_comparison.csv")
        # quick bar
        smry = pd.read_csv(self.out_dir/"method_comparison.csv")
        plt.figure(figsize=(8,4))
        sns.barplot(x=smry.index, y=smry["unknown_rate"]*100, palette="husl")
        plt.ylabel("Unknown (%)"); plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(self.out_dir/"plots"/"unknown_rates.png",dpi=300)
        # report
        with open(self.out_dir/"reports"/"summary.txt","w") as f:
            f.write(f"Generated {datetime.now():%Y-%m-%d %H:%M}\nBest: {best}\n\n"+
                     smry.to_string())
        log.info("All outputs saved in %s", self.out_dir)

# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="augmentation_results")
    ap.add_argument("--methods", nargs="+", default=["all"],
                    choices=["all","rule","ml","semantic","fuzzy","bert","ensemble"])
    args = ap.parse_args()
    todo = None if "all" in args.methods else args.methods
    IntentAugmentor(args.input,args.output).run(todo)

if __name__ == "__main__":
    main()