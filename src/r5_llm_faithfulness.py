#!/usr/bin/env python3
"""R5 — LLM-interpretation faithfulness. For N test predictions,
generate the real SHAP->Groq/Llama-3.3-70b explanation and quantify whether the
LLM stays grounded in the SHAP features it was given (anti-hallucination), cites
the true top drivers, and states directions consistent with SHAP signs. base env."""
import os, re, sys, json, csv, time, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from protbind.predictor import ProtBind
from protbind.ai_explain import ai_explain_prediction
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY=os.environ["GROQ_API_KEY"]; MODEL=os.environ.get("GROQ_MODEL","llama-3.3-70b-versatile")
# named features derived from pb.all_feat_cols after model load (includes proto_/interface)
POS=re.compile(r"increas|higher|favor|strong|positive|driv|promot|boost|enhanc|elevat",re.I)
NEG=re.compile(r"decreas|lower|weak|reduc|negativ|against|unfavor|disrupt|hinder|impair|poor",re.I)

def cited(text, named):
    t=text.lower()
    out=set()
    for f in named:
        if f.lower() in t or f.replace("_"," ").lower() in t: out.add(f)
    return out
def direction_in_window(text,feat):
    # find mentions of feat, look at +/-120 char window for direction sentiment
    t=text; idxs=[m.start() for m in re.finditer(re.escape(feat),t,re.I)]
    for i in idxs:
        w=t[max(0,i-120):i+120]
        p,n=bool(POS.search(w)),bool(NEG.search(w))
        if p and not n: return +1
        if n and not p: return -1
    return 0

pb=ProtBind()
named=set(c for c in pb.all_feat_cols if not (c.startswith("aac_") or c.startswith("dpc_")))
print("citeable named features:",len(named),"(incl proto_*:",[c for c in named if c.startswith("proto")],")")
rows=list(csv.DictReader(open("data/external/r5_test_instances.csv")))
agg=dict(ground_num=0,ground_den=0,hall=0,top1_hit=0,top3_cov=0,top3_den=0,dir_ok=0,dir_den=0)
recs=[]
for r in rows:
    try:
        res=pb.predict(r["sequence"], r["target"])
        ex=pb.explain(res, top_n=10)
        tf=ex["top_features"]                       # [(name,shap,val)]
        provided=[n for n,s,v in tf]; signs={n:np.sign(s) for n,s,v in tf}
        text=ai_explain_prediction(sequence=r["sequence"],target=r["target"],result=res,
                top_features=tf, all_feat_cols=pb.all_feat_cols, feat_vec=res["_feat_vec"],
                api_key=KEY, model=MODEL)
        c=cited(text, named)
        valid=c                                     # all in `named` are valid feature names
        grounded=c & set(provided)
        hall=c - set(provided)                      # cited a real feature NOT in the provided top-10
        agg["ground_num"]+=len(grounded); agg["ground_den"]+=len(valid)
        agg["hall"]+=len(hall)
        top3=provided[:3]
        agg["top1_hit"]+= int(provided[0] in c); 
        agg["top3_cov"]+=len(set(top3)&c); agg["top3_den"]+=3
        for f in grounded:
            d=direction_in_window(text,f)
            if d!=0: agg["dir_den"]+=1; agg["dir_ok"]+= int(d==signs[f])
        recs.append(dict(target=r["target"],label=r["binding_label"],prob=round(res["probability"],3),
            n_cited=len(valid),n_grounded=len(grounded),n_hall=len(hall),top1=provided[0],top1_cited=provided[0] in c))
        time.sleep(0.3)
    except Exception as e:
        print("  skip:",repr(e)[:120])
import pandas as pd
df=pd.DataFrame(recs); 
print(df.to_string(index=False))
gp=agg["ground_num"]/max(agg["ground_den"],1)
print("\n================ R5 LLM FAITHFULNESS (n=%d) ================"%len(recs))
print(f"  Feature-grounding precision = {gp:.3f}  (cited features that ARE in the provided SHAP top-10)")
print(f"  Hallucinated-feature rate   = {agg['hall']/max(agg['ground_den'],1):.3f}  ({agg['hall']} features cited outside the provided set)")
print(f"  Top-1 SHAP feature mentioned= {agg['top1_hit']}/{len(recs)} = {agg['top1_hit']/max(len(recs),1):.3f}")
print(f"  Top-3 SHAP coverage         = {agg['top3_cov']/max(agg['top3_den'],1):.3f}")
print(f"  Direction consistency       = {agg['dir_ok']/max(agg['dir_den'],1):.3f}  (n={agg['dir_den']} directional mentions; approx regex)")
df.to_csv("outputs/r5_llm_faithfulness.csv",index=False)
json.dump({k:(round(v,4) if isinstance(v,float) else v) for k,v in {
 "n":len(recs),"grounding_precision":gp,"hallucination_rate":agg['hall']/max(agg['ground_den'],1),
 "top1_mention_rate":agg['top1_hit']/max(len(recs),1),"top3_coverage":agg['top3_cov']/max(agg['top3_den'],1),
 "direction_consistency":agg['dir_ok']/max(agg['dir_den'],1)}.items()}, open("outputs/r5_metrics.json","w"),indent=1)
print("saved outputs/r5_llm_faithfulness.csv + r5_metrics.json")
