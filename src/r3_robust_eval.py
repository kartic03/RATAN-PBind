#!/usr/bin/env python3
"""R3 — robust evaluation (CPU, LightGBM). Reconstructs base+prototype model on
the standard train/test split and reports: bootstrap 95% CIs on headline metrics,
per-target reliability with CIs (fixes small-n AUROC=1.000), single-vs-shared
models, and leave-one-design-method-out generalization."""
import os, json, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.RandomState(42)
ZERO={"human-serum-albumin","human-tnfa","human-orm2","human-gm2a"}
LGB=dict(objective="binary",n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.9,
         colsample_bytree=0.8,min_child_samples=10,reg_lambda=1.0,is_unbalance=True,verbosity=-1,n_jobs=8)

fm=pd.read_parquet(os.path.join(ROOT,"features/feature_matrix.parquet"))
base_cols=list(pd.read_csv(os.path.join(ROOT,"features/feature_columns.csv"))["column"])
emb=np.load(os.path.join(ROOT,"features/esm2_embeddings.npy"))
pids=np.load(os.path.join(ROOT,"features/esm2_protein_ids.npy"),allow_pickle=True)
p2r={str(p):i for i,p in enumerate(pids)}
E=np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)
y=fm["binding_label"].values.astype(int); tgt=fm["target"].values; split=fm["split"].values
tr=(split=="train")|(split=="val"); te=split=="test"

def proto_feats(E,tgt,y,trmask):
    """leakage-safe: prototypes from TRAIN only."""
    eps=1e-8; out=np.zeros((len(E),7),np.float32)
    for t in np.unique(tgt):
        m=tgt==t; trm=m&trmask
        pos=E[trm&(y==1)]; neg=E[trm&(y==0)]
        pp=pos.mean(0) if len(pos) else np.zeros(E.shape[1]); pn=neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff=pp-pn; dn=np.linalg.norm(diff)+eps
        ei=E[m]
        cp=(ei@pp)/((np.linalg.norm(ei,axis=1)*np.linalg.norm(pp))+eps)
        cn=(ei@pn)/((np.linalg.norm(ei,axis=1)*np.linalg.norm(pn))+eps)
        l2=np.linalg.norm(ei-pp,axis=1); dproj=(ei@diff)/dn; ratio=cp/(cn+eps)
        out[m]=np.stack([cp,cn,l2,dproj,ratio,np.full(m.sum(),len(pos)),np.full(m.sum(),len(neg))],1)
    return out

P=proto_feats(E,tgt,y,tr)
X=np.hstack([fm[base_cols].values.astype(np.float32),P])  # base 463 + proto 7 = 470
m=lgb.LGBMClassifier(**LGB); m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]
yte=y[te]; Xte=X[te]; tgt_te=tgt[te]   # test-aligned (p has length n_test)

def metrics(yy,pp,thr=0.5):
    pr=(pp>=thr).astype(int)
    return dict(auroc=roc_auc_score(yy,pp),auprc=average_precision_score(yy,pp),
                f1=f1_score(yy,pr,zero_division=0),mcc=matthews_corrcoef(yy,pr))
def boot_ci(yy,pp,fn,n=1000):
    vals=[]
    idx=np.arange(len(yy))
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(yy[b]))<2: continue
        try: vals.append(fn(yy[b],pp[b]))
        except: pass
    return np.percentile(vals,[2.5,97.5])

print("="*70); print("R3.1 HEADLINE TEST METRICS (base+proto, n=%d) with 95%% bootstrap CI"%te.sum()); print("="*70)
M=metrics(yte,p)
for k,fn in [("auroc",roc_auc_score),("auprc",average_precision_score)]:
    lo,hi=boot_ci(yte,p,fn); print(f"  {k.upper():6s} = {M[k]:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
print(f"  F1     = {M['f1']:.3f}   MCC = {M['mcc']:.3f}")

print("\n"+"="*70); print("R3.2 PER-TARGET TEST AUROC with 95%% CI (fixes small-n AUROC=1.000)"); print("="*70)
rows=[]
for t in sorted(set(tgt)-ZERO):
    mm=tgt_te==t
    if mm.sum()<2 or len(np.unique(yte[mm]))<2: continue
    yy,pp=yte[mm],p[mm]; au=roc_auc_score(yy,pp); lo,hi=boot_ci(yy,pp,roc_auc_score,500)
    rel="OK" if mm.sum()>=20 else ("low-n" if mm.sum()>=8 else "UNRELIABLE")
    rows.append((t,int(mm.sum()),int(yy.sum()),round(au,3),round(lo,3),round(hi,3),rel))
rdf=pd.DataFrame(rows,columns=["target","n","binders","auroc","ci_lo","ci_hi","reliability"]).sort_values("n",ascending=False)
print(rdf.to_string(index=False))
nbad=(rdf.reliability=="UNRELIABLE").sum()
print(f"\n  -> {nbad} targets have UNRELIABLE AUROC (n<8); their CI spans ~[0,1]. Report only n>=20 as reliable.")
rdf.to_csv(os.path.join(ROOT,"outputs/r3_per_target_ci.csv"),index=False)

print("\n"+"="*70); print("R3.3 SINGLE-TARGET vs SHARED model (answers 'separate models would be better')"); print("="*70)
print(f"  {'target':24s} {'n_te':>4s} {'shared':>7s} {'single':>7s} {'Δ(sh-si)':>8s}")
cmp=[]
for t in sorted(set(tgt)-ZERO):
    mm=tgt_te==t
    if mm.sum()<15 or len(np.unique(yte[mm]))<2: continue
    sh=roc_auc_score(yte[mm],p[mm])
    mt=tgt==t
    if (tr&mt&(y==1)).sum()<3: continue
    ms=lgb.LGBMClassifier(**LGB)
    ms.fit(X[tr&mt],y[tr&mt]); ps=ms.predict_proba(Xte[mm])[:,1]; si=roc_auc_score(yte[mm],ps)
    cmp.append((t,int(mm.sum()),round(sh,3),round(si,3),round(sh-si,3)))
    print(f"  {t:24s} {int(mm.sum()):4d} {sh:7.3f} {si:7.3f} {sh-si:+8.3f}")
if cmp:
    d=np.mean([c[4] for c in cmp]); print(f"\n  mean Δ(shared-single) = {d:+.3f}  -> {'SHARED helps' if d>0.01 else ('SINGLE helps' if d<-0.01 else 'roughly equal')}")
pd.DataFrame(cmp,columns=["target","n_te","shared_auroc","single_auroc","delta"]).to_csv(os.path.join(ROOT,"outputs/r3_single_vs_shared.csv"),index=False)

print("\n"+"="*70); print("R3.4 LEAVE-ONE-DESIGN-METHOD-OUT generalization"); print("="*70)
_pw=pd.read_parquet(os.path.join(ROOT,"data/pairs_with_splits.parquet"))[["protein_id","target","design_method"]]
dm=fm.merge(_pw,on=["protein_id","target"],how="left")["design_method"].fillna("unknown").values
top=pd.Series(dm).value_counts(); top=top[top>=40].index.tolist()[:8]
print(f"  {'design_method':32s} {'n':>5s} {'binders':>7s} {'AUROC':>6s}")
lmo=[]
for meth in top:
    hm=dm==meth
    if len(np.unique(y[hm]))<2 or hm.sum()<20: continue
    mo=lgb.LGBMClassifier(**LGB); mo.fit(X[~hm],y[~hm]); pm=mo.predict_proba(X[hm])[:,1]
    au=roc_auc_score(y[hm],pm); lmo.append((meth,int(hm.sum()),int(y[hm].sum()),round(au,3)))
    print(f"  {meth[:32]:32s} {int(hm.sum()):5d} {int(y[hm].sum()):7d} {au:6.3f}")
if lmo: print(f"\n  mean leave-one-method-out AUROC = {np.mean([l[3] for l in lmo]):.3f}  (vs in-distribution {M['auroc']:.3f})")
pd.DataFrame(lmo,columns=["design_method","n","binders","auroc"]).to_csv(os.path.join(ROOT,"outputs/r3_leave_method_out.csv"),index=False)
print("\nSaved: outputs/r3_{per_target_ci,single_vs_shared,leave_method_out}.csv")
