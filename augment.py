# ╔══════════════════════════════════════════════════════════════════╗
# ║  Intent Augmentation – Comparison Framework (full version)      ║
# ║  2025-05-30                                                     ║
# ║  Adds progress bars (tqdm) + faster RF classifier               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""
Run *after* your feature-engineering pipeline creates merged_call_data.csv.

Minimal deps
------------
pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm

Optional extras
---------------
pip install sentence-transformers rapidfuzz modAL-python
pip install torch transformers          # only if you later enable BERT
"""
from __future__ import annotations
import json, pickle, time, warnings
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
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")

# ─── optional heavy libs ───────────────────────────────────────────
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
# (BERT placeholder omitted – can be added later)

# ═══════════════════════════════════════════════════════════════════
class IntentAugmentationFramework:
    """Compare several augmentation methods & export best result."""
    # ── init ───────────────────────────────────────────────────────
    def __init__(self, data_path: str, output_dir: str = "augmentation_results"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        for sub in ("plots", "reports", "models"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        print(f"📄  Loading {self.data_path}")
        self.df: pd.DataFrame = pd.read_csv(self.data_path, low_memory=False)
        self.orig_df = self.df.copy()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Any] = {}

        # canonical intents
        self.STANDARD_INTENTS = [
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
        self.LETTER_TO_INTENT = {
            "a":"Balance/Value","b":"Sell","c":"Repeat Caller","d":"Name Change",
            "e":"Buy Stock","f":"Statement","g":"Recent Activity","h":"Tax Information",
            "i":"Banking Details","j":"Dividend Payment","k":"Address Change",
            "l":"Check Replacement","m":"Stock Quote","n":"Beneficiary Information",
            "o":"Dividend Reinvestment","p":"Certificate Issuance","q":"Transfer",
            "r":"Existing IC User Login Problem","s":"New IC User Login Problem",
            "t":"Fulfillment","u":"Enrolment","w":"Associate","x":"Lost Certificate",
            "y":"Blank","z":"Unknown"
        }

    # ── helpers ────────────────────────────────────────────────────
    def _std(self, raw):
        if pd.isna(raw): return "Unknown"
        raw=str(raw).strip()
        if raw.lower() in self.LETTER_TO_INTENT:
            return self.LETTER_TO_INTENT[raw.lower()]
        return raw if raw in self.STANDARD_INTENTS else "Unknown"

    def _store(self, key:str, col:str, t0:float, extra:dict|None=None):
        unk = (self.df[col]=="Unknown").mean()
        imp = ((self.df["intent_baseline"]=="Unknown")&(self.df[col]!="Unknown")).sum()
        self.results[key] = {"stats":{
            "unknown_rate":float(unk),
            "improved":int(imp),
            "time_s":float(time.time()-t0),
            **(extra or {})
        }}
        print(f"   → unknown {unk:.2%} | improved {imp:,} | {time.time()-t0:.1f}s")

    # ── baseline ───────────────────────────────────────────────────
    def baseline(self):
        print("\n▶ BASELINE")
        col=next((c for c in self.df.columns if "intent" in c.lower()),"Intent")
        if col not in self.df.columns: self.df[col]="Unknown"
        self.df["intent_baseline"]=self.df[col].apply(self._std)
        rate=(self.df["intent_baseline"]=="Unknown").mean()
        self.results["baseline"]={"stats":{"unknown_rate":float(rate)}}
        print(f"   Unknown rate {rate:.2%}")

    # ── rule-based ────────────────────────────────────────────────
    def method_rule(self):
        print("\n▶ RULE-BASED")
        t0=time.time()
        activity_map={
            "Sell":"Sell","Tax Information":"Tax Information","Dividend Payment":"Dividend Payment",
            "Transfer":"Transfer","Address Change":"Address Change","Check Replacement":"Check Replacement",
            "Name Change":"Name Change","Banking Details":"Banking Details"
        }
        kw={"Fraud Assistance":["fraud","unauthorized"],
            "Tax Information":["tax","irs","1099"],
            "Dividend Payment":["dividend"],
            "Transfer":["transfer","acat","dtc"],
            "Sell":["sell","liquidate"]}

        def infer(r):
            if r["intent_baseline"]!="Unknown":
                return r["intent_baseline"],1.0
            seq=str(r.get("activity_sequence","")).split("|")
            for a in seq:
                a=a.strip()
                if a in activity_map:
                    return activity_map[a],0.9
            text="|".join(seq).lower()
            for i,ws in kw.items():
                if any(w in text for w in ws):
                    return i,0.7
            return "Unknown",0.0

        res=self.df.progress_apply(infer,axis=1,result_type="expand")
        self.df["intent_rule"],self.df["conf_rule"]=res[0],res[1]
        self._store("rule","intent_rule",t0)

    # ── ML fast ───────────────────────────────────────────────────
    def method_ml(self):
        print("\n▶ ML CLASSIFIER (fast)")
        t0=time.time()

        labelled = (self.df["intent_rule"]!="Unknown")&(self.df["conf_rule"]>=0.7)
        n_lab=labelled.sum()
        if n_lab<200:
            print("   not enough labelled rows – skipped"); return
        print(f"   labelled rows: {n_lab:,}")

        MAX_TRAIN=40_000
        if n_lab>MAX_TRAIN:
            idx=self.df[labelled].sample(MAX_TRAIN,random_state=42).index
            train_mask=self.df.index.isin(idx)
            print(f"   down-sampled to {MAX_TRAIN:,}")
        else:
            train_mask=labelled

        tfidf=TfidfVectorizer(max_features=600,ngram_range=(1,2))
        X=tfidf.fit_transform(self.df.loc[train_mask,"activity_sequence"].fillna(""))
        y=self.df.loc[train_mask,"intent_rule"]

        rf=RandomForestClassifier(
            n_estimators=120, n_jobs=-1, random_state=42, verbose=1
        )
        X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.2,
                                               stratify=y,random_state=42)
        rf.fit(X_tr,y_tr)
        val_f1=f1_score(y_val,rf.predict(X_val),average="macro")
        print(f"   hold-out F1 {val_f1:.3f}")

        unk=self.df["intent_baseline"]=="Unknown"
        if unk.sum():
            X_unk=tfidf.transform(self.df.loc[unk,"activity_sequence"].fillna(""))
            pred=rf.predict(X_unk)
            conf=rf.predict_proba(X_unk).max(1)
            self.df.loc[unk,["intent_ml","conf_ml"]]=np.column_stack([pred,conf])

        self.df.loc[~unk,"intent_ml"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_ml"]=1.0

        self.models.update({"ml_tfidf":tfidf,"ml_rf":rf})
        self._store("ml","intent_ml",t0,{"val_f1":float(val_f1),"labelled":int(n_lab)})

    # ── semantic ───────────────────────────────────────────────────
    def method_semantic(self):
        print("\n▶ SEMANTIC SIMILARITY")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("   sentence-transformers missing – skipped"); return
        t0=time.time()
        model=SentenceTransformer("all-MiniLM-L6-v2")
        intent_emb={i:model.encode(i) for i in self.STANDARD_INTENTS if i!="Unknown"}

        unk=self.df["intent_baseline"]=="Unknown"
        if not unk.any():
            self._store("semantic","intent_rule",t0); return

        texts=(
            self.df.loc[unk,"activity_sequence"].fillna("")+" "+
            self.df.loc[unk,"first_activity"].fillna("")+" "+
            self.df.loc[unk,"last_activity"].fillna("")
        ).tolist()
        emb=model.encode(texts,batch_size=128,convert_to_numpy=True,show_progress_bar=True)

        preds,confs=[],[]
        for e in tqdm(emb,desc="‣ semantic ‣ scoring",unit="row"):
            sims={k:cosine_similarity([e],[v])[0,0] for k,v in intent_emb.items()}
            best,score=max(sims.items(),key=lambda x:x[1])
            preds.append(best if score>=0.25 else "Unknown")
            confs.append(float(score))

        self.df.loc[unk,"intent_sem"]=preds
        self.df.loc[unk,"conf_sem"]=confs
        self.df.loc[~unk,"intent_sem"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_sem"]=1.0

        self.models["semantic_model"]=model
        self._store("semantic","intent_sem",t0)

    # ── fuzzy ──────────────────────────────────────────────────────
    def method_fuzzy(self):
        print("\n▶ FUZZY MATCHING")
        if not RAPIDFUZZ_AVAILABLE:
            print("   rapidfuzz missing – skipped"); return
        t0=time.time()
        kw={"Sell":["sell","liquidate"],
            "Tax Information":["tax","irs","1099"],
            "Transfer":["transfer","acat"]}

        def match(txt:str):
            txt=txt.lower(); best,bs="Unknown",0.0
            for intent,words in kw.items():
                for w in words:
                    sc=fuzz.partial_ratio(w,txt)/100
                    if sc>bs: best,bs=intent,sc
            return best if bs>=0.7 else "Unknown",bs

        unk=self.df["intent_baseline"]=="Unknown"
        pairs=self.df.loc[unk,"activity_sequence"].fillna("").progress_apply(match)
        self.df.loc[unk,"intent_fuzzy"]=pairs.str[0]
        self.df.loc[unk,"conf_fuzzy"]=pairs.str[1]
        self.df.loc[~unk,"intent_fuzzy"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_fuzzy"]=1.0
        self._store("fuzzy","intent_fuzzy",t0)

    # ── ensemble ───────────────────────────────────────────────────
    def method_ensemble(self):
        print("\n▶ ENSEMBLE VOTE")
        t0=time.time()
        cols=[("intent_rule","conf_rule"),("intent_ml","conf_ml"),
              ("intent_sem","conf_sem"),("intent_fuzzy","conf_fuzzy")]
        cols=[c for c in cols if c[0] in self.df.columns]
        if len(cols)<2:
            print("   need ≥2 methods – skipped"); return

        def vote(row):
            score={}
            for ic,cc in cols:
                it,co=row[ic],row[cc]
                if it!="Unknown": score[it]=score.get(it,0)+co
            if not score:
                return "Unknown",0.0
            best=max(score,key=score.get)
            return best,score[best]/sum(score.values())

        res=self.df.progress_apply(vote,axis=1,result_type="expand")
        self.df["intent_ens"],self.df["conf_ens"]=res[0],res[1]
        self._store("ensemble","intent_ens",t0)

    # ── run orchestrator ───────────────────────────────────────────
    def run(self, methods:List[str]|None):
        self.baseline()
        mapping={
            "rule":self.method_rule,"ml":self.method_ml,
            "semantic":self.method_semantic,"fuzzy":self.method_fuzzy,
            "ensemble":self.method_ensemble
        }
        todo=methods or list(mapping.keys())
        for m in todo:
            fn=mapping.get(m); 
            if not fn: continue
            print(f"\n=== START {m.upper()} ==="); s=time.time()
            try: fn()
            except Exception as exc: print(f"🚫  {m} failed: {exc}")
            print(f"===  END  {m.upper()} ({time.time()-s:.1f}s) ===")

        best=min((k for k in self.results if k!="baseline" and "stats" in self.results[k]),
                 key=lambda k:self.results[k]["stats"]["unknown_rate"])
        print(f"\n⭐  Best method: {best} "
              f"(unknown {self.results[best]['stats']['unknown_rate']:.2%})")
        self._export(best)

    # ── export ─────────────────────────────────────────────────────
    def _export(self,best:str):
        best_col=f"intent_{best}"
        self.orig_df["intent_augmented"]=self.df[best_col]
        self.orig_df["augmentation_method"]=best
        self.orig_df.to_csv(self.output_dir/"best_augmented_data.csv",index=False)

        summary=pd.DataFrame(
            {k:v["stats"] for k,v in self.results.items() if "stats" in v}).T
        summary.to_csv(self.output_dir/"method_comparison.csv")

        plt.figure(figsize=(8,4))
        plt.bar(summary.index,summary["unknown_rate"]*100)
        plt.ylabel("Unknown (%)"); plt.xticks(rotation=45,ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir/"plots"/"unknown_rates.png",dpi=300)
        plt.close()

        with open(self.output_dir/"reports"/"summary.txt","w") as f:
            f.write(f"Generated {datetime.now():%Y-%m-%d %H:%M}\n"
                    f"Input : {self.data_path}\nBest  : {best}\n\n"+
                    summary.to_string())
        print("\n📊  Outputs:")
        print("   best data :", self.output_dir/"best_augmented_data.csv")
        print("   comparison:", self.output_dir/"method_comparison.csv")
        print("   plot      :", self.output_dir/"plots"/"unknown_rates.png")
        print("   report    :", self.output_dir/"reports"/"summary.txt")

# ═══════════════════════════════════════════════════════════════════
def main():
    import argparse
    ap=argparse.ArgumentParser("Intent augmentation")
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",default="augmentation_results")
    ap.add_argument("--methods",nargs="+",default=["all"],
                    choices=["all","rule","ml","semantic","fuzzy","ensemble"])
    args=ap.parse_args()
    m=None if "all" in args.methods else args.methods
    IntentAugmentationFramework(args.input,args.output).run(m)

if __name__=="__main__":
    main()