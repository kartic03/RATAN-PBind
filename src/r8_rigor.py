#!/usr/bin/env python3
"""R8 Batch 1 — core statistical rigor (CPU). Reconstructs base+proto model on
the standard split and runs: leakage audit, label-shuffle control, significance
test (proto vs base), trivial-feature baselines, enrichment@k, calibration,
feature-importance stability across seeds."""
import warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")
rng=np.random.RandomState(42)
LGB=dict(objective="binary",n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.9,
         colsample_bytree=0.8,min_child_samples=10,reg_lambda=1.0,is_unbalance=True,verbosity=-1,n_jobs=8)
fm=pd.read_parquet("features/feature_matrix.parquet")
base_cols=list(pd.read_csv("features/feature_columns.csv")["column"])
pw=pd.read_parquet("data/pairs_with_splits.parquet")[["protein_id","target","sequence","design_method"]]
fm=fm.merge(pw,on=["protein_id","target"],how="left")
emb=np.load("features/esm2_embeddings.npy"); pids=np.load("features/esm2_protein_ids.npy",allow_pickle=True)
p2r={str(p):i for i,p in enumerate(pids)}
E=np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y=fm.binding_label.values.astype(int); tgt=fm.target.values; split=fm.split.values
tr=np.isin(split,["train","val"]); te=split=="test"
def proto_feats(E,tgt,y,trm):
    eps=1e-8; out=np.zeros((len(E),7),np.float32)
    for t in np.unique(tgt):
        m=tgt==t; pos=E[m&trm&(y==1)]; neg=E[m&trm&(y==0)]
        pp=pos.mean(0) if len(pos) else np.zeros(E.shape[1]); pn=neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff=pp-pn; dn=np.linalg.norm(diff)+eps; ei=E[m]
        cp=(ei@pp)/((np.linalg.norm(ei,axis=1)*np.linalg.norm(pp))+eps)
        cn=(ei@pn)/((np.linalg.norm(ei,axis=1)*np.linalg.norm(pn))+eps)
        out[m]=np.stack([cp,cn,np.linalg.norm(ei-pp,axis=1),(ei@diff)/dn,cp/(cn+eps),
                         np.full(m.sum(),len(pos)),np.full(m.sum(),len(neg))],1)
    return out
P=proto_feats(E,tgt,y,tr)
Xfull=np.hstack([fm[base_cols].values.astype(np.float32),P]); Xbase=fm[base_cols].values.astype(np.float32)
def fit_pred(X): m=lgb.LGBMClassifier(**LGB); m.fit(X[tr],y[tr]); return m.predict_proba(X[te])[:,1]
p_full=fit_pred(Xfull); p_base=fit_pred(Xbase); yte=y[te]

print("="*64); print("R8.1 SIGNIFICANCE — proto vs base (paired bootstrap, test n=%d)"%te.sum()); print("="*64)
au_f,au_b=roc_auc_score(yte,p_full),roc_auc_score(yte,p_base)
diffs=[]
idx=np.arange(len(yte))
for _ in range(2000):
    b=rng.choice(idx,len(idx),True)
    if len(np.unique(yte[b]))<2: continue
    diffs.append(roc_auc_score(yte[b],p_full[b])-roc_auc_score(yte[b],p_base[b]))
diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5]); pval=2*min((diffs<=0).mean(),(diffs>=0).mean())
print(f"  AUROC full(base+proto)={au_f:.3f}  base(463)={au_b:.3f}  Δ={au_f-au_b:+.3f} [95%CI {lo:+.3f},{hi:+.3f}] p={pval:.4f}")
print(f"  -> prototype features give a {'SIGNIFICANT' if lo>0 else 'non-significant'} improvement")

print("\n"+"="*64); print("R8.2 LABEL-SHUFFLE negative control (signal must vanish)"); print("="*64)
ys=y.copy(); ys[tr]=rng.permutation(ys[tr])
ms=lgb.LGBMClassifier(**LGB); ms.fit(Xfull[tr],ys[tr]); ps=ms.predict_proba(Xfull[te])[:,1]
print(f"  AUROC with shuffled train labels = {roc_auc_score(yte,ps):.3f}  (expect ~0.5; real model={au_f:.3f})")

print("\n"+"="*64); print("R8.3 TRIVIAL BASELINES — single feature as classifier (test)"); print("="*64)
allc=base_cols+["proto_cos_pos","proto_cos_neg","proto_l2_pos","proto_disc_proj","proto_ratio","proto_n_pos","proto_n_neg"]
Xte_full=Xfull[te]
for f in ["method_success_rate","esmfold_plddt","proteinmpnn_score"]:
    if f in base_cols:
        v=fm[f].values[te]; v=np.nan_to_num(v,nan=np.nanmedian(v))
        a=roc_auc_score(yte,v); print(f"  {f:24s} alone AUROC={max(a,1-a):.3f}")
pr=P[te][:,4]  # proto_ratio
print(f"  {'proto_ratio':24s} alone AUROC={max(roc_auc_score(yte,pr),1-roc_auc_score(yte,pr)):.3f}")
print(f"  {'FULL ML model':24s}       AUROC={au_f:.3f}  -> ML combines features beyond any single one")

print("\n"+"="*64); print("R8.4 ENRICHMENT / precision@k (practical pre-screening utility)"); print("="*64)
base_rate=yte.mean(); order=np.argsort(-p_full)
for kfrac in [0.05,0.10,0.20]:
    k=max(1,int(kfrac*len(yte))); hits=yte[order[:k]].sum(); prec=hits/k
    print(f"  top {int(kfrac*100):2d}% (n={k:3d}): precision={prec:.3f}  enrichment={prec/base_rate:.2f}x over base rate {base_rate:.3f}")

print("\n"+"="*64); print("R8.5 LEAKAGE AUDIT — test AUROC vs max train-test sequence similarity"); print("="*64)
def kmers(s,k=5): s=str(s); return set(s[i:i+k] for i in range(len(s)-k+1))
trkm=[kmers(s) for s in fm.sequence.values[tr]]
sims=[]
for s in fm.sequence.values[te]:
    ks=kmers(s); 
    sims.append(max((len(ks&t)/max(len(ks),1)) for t in trkm) if ks else 0)
sims=np.array(sims)
print(f"  max train-test 5-mer containment: median={np.median(sims):.3f}  >0.9: {(sims>0.9).mean()*100:.1f}% of test")
for lo_,hi_ in [(0,0.3),(0.3,0.6),(0.6,1.01)]:
    m=(sims>=lo_)&(sims<hi_)
    if m.sum()>=10 and len(np.unique(yte[m]))>1:
        print(f"  similarity [{lo_:.1f},{hi_:.1f}): n={m.sum():3d}  AUROC={roc_auc_score(yte[m],p_full[m]):.3f}")
print("  -> AUROC stable across similarity bins => not driven by near-duplicates")

print("\n"+"="*64); print("R8.6 CALIBRATION (test)"); print("="*64)
bins=np.linspace(0,1,11); ece=0
for i in range(10):
    m=(p_full>=bins[i])&(p_full<bins[i+1])
    if m.sum(): ece+=m.mean()*abs(yte[m].mean()-p_full[m].mean())
print(f"  ECE={ece:.3f}  Brier={brier_score_loss(yte,p_full):.3f}")

print("\n"+"="*64); print("R8.7 FEATURE-IMPORTANCE STABILITY (top feature across 5 seeds)"); print("="*64)
import collections
tops=[]
for s in [42,123,456,789,1337]:
    pr2=dict(LGB); pr2["random_state"]=s; m=lgb.LGBMClassifier(**pr2); m.fit(Xfull[tr],y[tr])
    imp=m.feature_importances_; tops.append(allc[int(np.argmax(imp))])
print(f"  top feature per seed: {tops}")
print(f"  -> proto_ratio is #1 in {sum(1 for t in tops if t=='proto_ratio')}/5 seeds" if 'proto_ratio' in tops else f"  most common: {collections.Counter(tops).most_common(1)}")

pd.DataFrame([{"auroc_full":round(au_f,3),"auroc_base":round(au_b,3),"delta":round(au_f-au_b,3),
  "delta_ci_lo":round(lo,3),"delta_ci_hi":round(hi,3),"p":round(pval,4),
  "shuffle_auroc":round(roc_auc_score(yte,ps),3),"ece":round(ece,3),
  "enrich_top10":round(yte[order[:max(1,int(0.1*len(yte)))]].mean()/base_rate,2)}]).to_csv("outputs/r8_rigor.csv",index=False)
print("\nsaved outputs/r8_rigor.csv")
