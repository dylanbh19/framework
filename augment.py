# ╔══════════════════════════════════════════════════════════════╗
# ║  intent_augmentation_pro.py  –  v2025-05-30                 ║
# ║  End-to-end intent augmentation + logging + progress bars    ║
# ║  Methods:                                                   ║
# ║     rule, ml, semantic, fuzzy, zeroshot, bert, ensemble      ║
# ║  Requirements:                                              ║
# ║     pandas numpy tqdm seaborn matplotlib scikit-learn        ║
# ║     xgboost rapidfuzz sentence-transformers                  ║
# ║     (optional) torch transformers  – for BERT on GPU         ║
# ╚══════════════════════════════════════════════════════════════╝
"""
Run
----
python intent_augmentation_pro.py \
       --input  merged_call_data.csv \
       --output augmentation_results_pro \
       --methods all          # or a subset e.g. rule ml zeroshot ensemble
"""
from __future__ import annotations
import json, pickle, time, warnings, logging
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

# ─── optional libs ──────────────────────────────────────────────
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
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              pipeline, Trainer, TrainingArguments)
    BERT_OK      = True
    ZSHOT_OK     = True     # zero-shot uses transformers pipeline too
except ImportError:
    BERT_OK = ZSHOT_OK = False

# ─── logging conf ───────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s %(levelname)s | %(message)s",
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
class IntentAugmentor:
    def __init__(self, data_path: str, out_dir: str = "augmentation_results"):
        self.data_path = Path(data_path)
        self.out_dir   = Path(out_dir)
        for sub in ("plots", "reports", "models"):
            (self.out_dir/sub).mkdir(parents=True, exist_ok=True)

        log.info("Loading %s", self.data_path.name)
        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.orig_df = self.df.copy()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.models : Dict[str, Any]            = {}

        self.STD_INTENTS = [
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
            "y":"Blank","z":"Unknown"
        }

    # ── helpers ─────────────────────────────────────────────────
    def _std(self, raw):
        if pd.isna(raw): return "Unknown"
        txt = str(raw).strip()
        return self.LTR.get(txt.lower(), txt if txt in self.STD_INTENTS else "Unknown")

    def _metric(self, key, col, start, extra=None):
        unk = (self.df[col] == "Unknown").mean()
        imp = ((self.df["intent_base"] == "Unknown") & (self.df[col]!="Unknown")).sum()
        self.results[key] = {"stats":{
            "unknown_rate": round(float(unk),4),
            "improved": int(imp),
            "time_s": round(time.time()-start,1),
            **(extra or {})
        }}
        log.info("→ %-9s unknown %5.2f%% | improved %6d | %5.1fs",
                 key, unk*100, imp, time.time()-start)

    # ────────────────────────────────────────────────────────────
    # BASELINE
    def baseline(self):
        col = next((c for c in self.df.columns if "intent" in c.lower()), "Intent")
        if col not in self.df.columns:
            self.df[col] = "Unknown"
        self.df["intent_base"] = self.df[col].apply(self._std)
        self.results["baseline"] = {"stats":{
            "unknown_rate": round(float((self.df['intent_base']=="Unknown").mean()),4)
        }}
        log.info("baseline unknown %.2f%%", self.results["baseline"]["stats"]["unknown_rate"]*100)

    # ────────────────────────────────────────────────────────────
    # RULE
    def rule(self):
        t0=time.time(); log.info("rule-based …")
        act_map={"Sell":"Sell","Tax Information":"Tax Information","Transfer":"Transfer"}
        kw={"Sell":["sell","liquidate"],"Transfer":["transfer","acat"],
            "Fraud Assistance":["fraud","unauthorized"]}
        def f(row):
            if row["intent_base"]!="Unknown": return row["intent_base"],1.0
            seq=str(row.get("activity_sequence","")).split("|")
            for a in seq:
                if a.strip() in act_map: return act_map[a.strip()],0.9
            txt="|".join(seq).lower()
            for i,ws in kw.items():
                if any(w in txt for w in ws): return i,0.7
            return "Unknown",0.0
        res=self.df.progress_apply(f,axis=1,result_type="expand")
        self.df["intent_rule"], self.df["conf_rule"] = res[0],res[1]
        self._metric("rule","intent_rule",t0)

    # ML (TF-IDF + RF) ------------------------------------------
    def ml(self):
        t0=time.time(); log.info("ML (TF-IDF+RF) …")
        mask=(self.df["intent_rule"]!="Unknown")&(self.df["conf_rule"]>=0.7)
        if mask.sum()<300: log.warning("insufficient labels"); return
        MAX=60_000
        subs = self.df[mask].sample(MAX, random_state=1) if mask.sum()>MAX else self.df[mask]
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        X=tfidf.fit_transform(subs["activity_sequence"].fillna(""))
        y=subs["intent_rule"]
        rf=RandomForestClassifier(n_estimators=180,n_jobs=-1,random_state=42,verbose=1)
        Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        rf.fit(Xtr,ytr)
        f1=f1_score(yva,rf.predict(Xva),average="macro")
        unk=self.df["intent_base"]=="Unknown"
        if unk.any():
            preds=rf.predict(tfidf.transform(self.df.loc[unk,"activity_sequence"].fillna("")))
            prob =rf.predict_proba(tfidf.transform(self.df.loc[unk,"activity_sequence"].fillna(""))).max(1)
            self.df.loc[unk,"intent_ml"]=preds
            self.df.loc[unk,"conf_ml"]=prob
        self.df.loc[~unk,"intent_ml"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_ml"]=1.0
        self.models.update({"rf":rf,"tfidf":tfidf})
        # save top tokens for explainability
        fi=rf.feature_importances_; top=np.argsort(fi)[-25:][::-1]
        pd.DataFrame({"token":tfidf.get_feature_names_out()[top],"importance":fi[top]})\
          .to_csv(self.out_dir/"rf_top_tokens.csv",index=False)
        self._metric("ml","intent_ml",t0,{"f1":round(f1,3)})

    # SEMANTIC (Sentence-BERT) -----------------------------------
    def semantic(self):
        if not SBERT_OK: log.warning("sentence-transformers missing"); return
        t0=time.time(); log.info("semantic (MiniLM)")
        model=SentenceTransformer("all-MiniLM-L6-v2")
        emb_int={i:model.encode(i) for i in self.STD_INTENTS if i!="Unknown"}
        unk=self.df["intent_base"]=="Unknown"
        if not unk.any(): self._metric("semantic","intent_rule",t0); return
        texts=(self.df.loc[unk,"activity_sequence"].fillna("")+" "+
               self.df.loc[unk,"first_activity"].fillna("")+" "+
               self.df.loc[unk,"last_activity"].fillna("")).tolist()
        emb=model.encode(texts,batch_size=128,convert_to_numpy=True,show_progress_bar=True)
        pred,conf=[],[]
        for e in tqdm(emb,unit="row"):
            sims={k:cosine_similarity([e],[v])[0,0] for k,v in emb_int.items()}
            best,sc=max(sims.items(),key=lambda x:x[1])
            pred.append(best if sc>=0.25 else "Unknown"); conf.append(float(sc))
        self.df.loc[unk,"intent_sem"]=pred; self.df.loc[unk,"conf_sem"]=conf
        self.df.loc[~unk,"intent_sem"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_sem"]=1.0
        self._metric("semantic","intent_sem",t0)

    # FUZZY -------------------------------------------------------
    def fuzzy(self):
        if not RAPID_OK: log.warning("rapidfuzz missing"); return
        t0=time.time(); log.info("fuzzy …")
        kw={"Transfer":["transfer","dtc"],"Sell":["sell","liquidate"]}
        def f(txt):
            txt=txt.lower(); best,bs="Unknown",0
            for i,words in kw.items():
                for w in words:
                    sc=fuzz.partial_ratio(w,txt)
                    if sc>bs: best,bs=i,sc
            return best if bs>=70 else "Unknown",bs/100
        unk=self.df["intent_base"]=="Unknown"
        pairs=self.df.loc[unk,"activity_sequence"].fillna("").progress_apply(f)
        self.df.loc[unk,"intent_fuzzy"]=pairs.str[0]; self.df.loc[unk,"conf_fuzzy"]=pairs.str[1]
        self.df.loc[~unk,"intent_fuzzy"]=self.df.loc[~unk,"intent_rule"]; self.df.loc[~unk,"conf_fuzzy"]=1.0
        self._metric("fuzzy","intent_fuzzy",t0)

    # ZERO-SHOT (CPU OK) -----------------------------------------
    def zeroshot(self):
        if not ZSHOT_OK: log.warning("transformers missing"); return
        t0=time.time(); log.info("zero-shot BART-MNLI … (CPU)")
        pipe=pipeline("zero-shot-classification",model="facebook/bart-large-mnli",device=-1)
        cand=[i for i in self.STD_INTENTS if i!="Unknown"]
        unk=self.df["intent_base"]=="Unknown"
        if not unk.any(): self._metric("zeroshot","intent_rule",t0); return
        texts=(self.df.loc[unk,"activity_sequence"].fillna("")+" "+
               self.df.loc[unk,"first_activity"].fillna("")+" "+
               self.df.loc[unk,"last_activity"].fillna("")).tolist()
        preds,confs,exps=[],[],[]
        for txt in tqdm(texts,unit="call"):
            out=pipe(txt,candidate_labels=cand,multi_label=False)
            preds.append(out["labels"][0]); confs.append(float(out["scores"][0]))
            tok=sorted(txt.split(),key=len,reverse=True)[:3]
            exps.append(" | ".join(tok))
        self.df.loc[unk,"intent_zeroshot"]=preds; self.df.loc[unk,"conf_zeroshot"]=confs
        self.df.loc[unk,"explain_zeroshot"]=exps
        self.df.loc[~unk,"intent_zeroshot"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_zeroshot"]=1.0;  self.df.loc[~unk,"explain_zeroshot"]=""
        self._metric("zeroshot","intent_zeroshot",t0)

    # BERT fine-tune (GPU) ---------------------------------------
    def bert(self):
        if not BERT_OK: log.warning("transformers/torch missing"); return
        if not torch.cuda.is_available(): log.warning("GPU not available – skip BERT"); return
        t0=time.time(); log.info("BERT (fine-tune) …")
        mask=(self.df["intent_rule"]!="Unknown")&(self.df["conf_rule"]>=0.9)
        if mask.sum()<800: log.warning("not enough labels"); return
        labs=self.df.loc[mask,"intent_rule"].unique().tolist(); le={l:i for i,l in enumerate(labs)}
        tok=AutoTokenizer.from_pretrained("distilbert-base-uncased")
        enc=tok(self.df.loc[mask,"activity_sequence"].fillna("").tolist(),
                truncation=True,padding=True)
        class DS(torch.utils.data.Dataset):
            def __init__(s,e,y): s.e=e;s.y=[le[i] for i in y]
            def __len__(s): return len(s.y)
            def __getitem__(s,i): d={k:torch.tensor(v[i]) for k,v in s.e.items()}
            d["labels"]=torch.tensor(s.y[i]); return d
        ds=DS(enc,self.df.loc[mask,"intent_rule"])
        model=AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",num_labels=len(labs)).cuda()
        tr=Trainer(model,TrainingArguments(output_dir=self.out_dir/"bert_tmp",
                                           per_device_train_batch_size=16,
                                           num_train_epochs=1,
                                           logging_steps=50,report_to=None),
                   train_dataset=ds)
        tr.train()
        # predict unknowns
        unk=self.df["intent_base"]=="Unknown"
        if unk.sum():
            enc2=tok(self.df.loc[unk,"activity_sequence"].fillna("").tolist(),
                     truncation=True,padding=True,return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits=model(**enc2).logits.softmax(dim=1).cpu().numpy()
            pred=np.argmax(logits,1); conf=np.max(logits,1)
            self.df.loc[unk,"intent_bert"]=[labs[i] for i in pred]
            self.df.loc[unk,"conf_bert"]=conf
        self.df.loc[~unk,"intent_bert"]=self.df.loc[~unk,"intent_rule"]
        self.df.loc[~unk,"conf_bert"]=1.0
        self._metric("bert","intent_bert",t0)

    # ENSEMBLE ---------------------------------------------------
    def ensemble(self):
        t0=time.time(); log.info("ensemble …")
        pairs=[(i,c.replace("intent_","conf_"))
               for i in self.df.columns if i.startswith("intent_")
               for c in (i,)]
        pairs=[p for p in pairs if p[1] in self.df.columns and p[0]!="intent_ensemble"]
        if len(pairs)<2: log.warning("need ≥2 models"); return
        def vote(row):
            sc={}
            for i,c in pairs:
                if row[i]!="Unknown": sc[row[i]]=sc.get(row[i],0)+row[c]
            if not sc: return "Unknown",0.0
            best=max(sc,key=sc.get); tot=sum(sc.values())
            return best,sc[best]/tot
        res=self.df.progress_apply(vote,axis=1,result_type="expand")
        self.df["intent_ensemble"],self.df["conf_ensemble"]=res[0],res[1]
        self._metric("ensemble","intent_ensemble",t0,{"sources":[p[0] for p in pairs]})

    # RUN orchestrator -------------------------------------------
    def run(self, methods:List[str]|None):
        self.baseline()
        table={
            "rule":self.rule, "ml":self.ml, "semantic":self.semantic,
            "fuzzy":self.fuzzy, "zeroshot":self.zeroshot,
            "bert":self.bert, "ensemble":self.ensemble
        }
        for m in (methods or table.keys()):
            fn=table[m]; log.info("===== %s =====", m.upper())
            try: fn()
            except Exception as e: log.error("%s failed: %s", m,e,exc_info=True)
        # choose best
        best=min((k for k in self.results if k!="baseline"),
                 key=lambda k:self.results[k]["stats"]["unknown_rate"])
        log.info("★ best %s (unknown %.2f%%)",
                 best, self.results[best]["stats"]["unknown_rate"]*100)
        self.export(best)

    # EXPORT ------------------------------------------------------
    def export(self,best):
        self.orig_df["intent_augmented"]=self.df[f"intent_{best}"]
        self.orig_df["intent_confidence"]=self.df.get(f"conf_{best}",1.0)
        self.orig_df["aug_method"]=best
        self.orig_df.to_csv(self.out_dir/"best_augmented_data.csv",index=False)
        pd.DataFrame({k:v["stats"] for k,v in self.results.items() if "stats" in v})\
          .T.to_csv(self.out_dir/"method_comparison.csv")
        # bar plot
        summ=pd.read_csv(self.out_dir/"method_comparison.csv")
        plt.figure(figsize=(8,4))
        sns.barplot(x=summ.index, y=summ["unknown_rate"]*100, palette="husl")
        plt.ylabel("Unknown (%)"); plt.xticks(rotation=45,ha="right")
        plt.tight_layout(); plt.savefig(self.out_dir/"plots"/"unknown_rates.png",dpi=300)
        # summary text
        with open(self.out_dir/"reports"/"summary.txt","w") as f:
            f.write(f"Generated {datetime.now():%Y-%m-%d %H:%M}\nBest={best}\n\n"+
                     summ.to_string())
        log.info("everything saved under %s", self.out_dir)

# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",default="augmentation_results")
    ap.add_argument("--methods",nargs="+",default=["all"],
                    choices=["all","rule","ml","semantic","fuzzy",
                             "zeroshot","bert","ensemble"])
    a=ap.parse_args()
    todo=None if "all" in a.methods else a.methods
    IntentAugmentor(a.input,a.output).run(todo)

if __name__=="__main__":
    main()