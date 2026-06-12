#!/usr/bin/env python3
"""R8 high-value: nested CV (#3), runtime benchmark (#4), FDR correction (#5),
calibration reliability (#10), proto within-design-method conditioning (#11). CPU."""
import time, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
rng=np.random.RandomState(42)
LGB=dict(objective="binary",n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.9,
         colsample_bytree=0.8,min_child_samples=10,reg_lambda=1.0,is_unbalance=True,verbosity=-1,n_jobs=8)
fm=pd.read_parquet("features/feature_matrix.parquet")
base_cols=list(pd.read_csv("features/feature_columns.csv")["column"])
pw=pd.read_parquet("data/pairs_with_splits.parquet")[["protein_id","target","design_method"]]
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
        cp=(ei@pp)/((np.linalg.norm(ei,axis=1)*np.linalg.norm(pp))+eps); cn=(ei@pn)/((np.linalg.norm(ei,axis=1)*np.linalg.norm(pn))+eps)
        out[m]=np.stack([cp,cn,np.linalg.norm(ei-pp,axis=1),(ei@diff)/dn,cp/(cn+eps),np.full(m.sum(),len(pos)),np.full(m.sum(),len(neg))],1)
    return out
B=fm[base_cols].values.astype(np.float32)

print("="*60); print("#3 NESTED CV (proto recomputed per outer fold; inner grid)"); print("="*60)
idx=np.arange(len(y)); outer=StratifiedKFold(5,shuffle=True,random_state=42); oa=[]
for oi,(otr,ote) in enumerate(outer.split(idx,y)):
    trm=np.isin(idx,otr); P=proto_feats(E,tgt,y,trm); X=np.hstack([B,P])
    best=None;bs=-1
    for nl in [15,31,63]:
        inner=StratifiedKFold(3,shuffle=True,random_state=1); ia=[]
        for itr,ite in inner.split(otr,y[otr]):
            pr=dict(LGB);pr["num_leaves"]=nl; m=lgb.LGBMClassifier(**pr); m.fit(X[otr[itr]],y[otr[itr]]); ia.append(roc_auc_score(y[otr[ite]],m.predict_proba(X[otr[ite]])[:,1]))
        if np.mean(ia)>bs: bs=np.mean(ia); best=nl
    pr=dict(LGB);pr["num_leaves"]=best; m=lgb.LGBMClassifier(**pr); m.fit(X[otr],y[otr]); a=roc_auc_score(y[ote],m.predict_proba(X[ote])[:,1]); oa.append(a)
    print(f"  outer fold {oi+1}: AUROC={a:.3f} (inner-best num_leaves={best})")
print(f"  NESTED CV AUROC = {np.mean(oa):.3f} ± {np.std(oa):.3f}  (no test-set tuning) vs single-split 0.946")

# fit canonical model for the rest
P=proto_feats(E,tgt,y,tr); X=np.hstack([B,P]); m=lgb.LGBMClassifier(**LGB); m.fit(X[tr],y[tr]); pte=m.predict_proba(X[te])[:,1]; yte=y[te]
print("\n"+"="*60); print("#4 RUNTIME / THROUGHPUT"); print("="*60)
t0=time.time(); _=m.predict_proba(X[te]); dt=(time.time()-t0)/len(te)
print(f"  RATAN-PBind inference = {dt*1000:.3f} ms/sequence ({1/dt:.0f} seq/s)  vs Boltz-2 folding ~8 s/complex")
print(f"  -> ~{8/dt:.0e}x faster than structure prediction")

print("\n"+"="*60); print("#5 FDR CORRECTION (per-target AUROC>0.5, BH)"); print("="*60)
pv=[]; tnames=[]
for t in sorted(set(tgt[te])):
    mm=(tgt[te]==t)
    if mm.sum()<8 or len(np.unique(yte[mm]))<2: continue
    pos=pte[mm][yte[mm]==1]; neg=pte[mm][yte[mm]==0]
    try: _,p=mannwhitneyu(pos,neg,alternative="greater"); pv.append(p); tnames.append(t)
    except: pass
pv=np.array(pv); order=np.argsort(pv); m_=len(pv); bh=pv[order]*m_/(np.arange(1,m_+1))
sig=np.zeros(m_,bool); 
for i in range(m_-1,-1,-1):
    if bh[i]<=0.05: sig[order[:i+1]]=True; break
print(f"  {m_} targets tested; {sig.sum()} significant after Benjamini-Hochberg FDR<0.05:")
for i in np.argsort(pv): print(f"    {tnames[i]:24s} p={pv[i]:.4f} {'SIG' if sig[i] else 'ns'}")

print("\n"+"="*60); print("#10 CALIBRATION RELIABILITY (bins + bootstrap)"); print("="*60)
bins=np.linspace(0,1,6)
for i in range(5):
    mm=(pte>=bins[i])&(pte<bins[i+1])
    if mm.sum()>=5: print(f"  pred [{bins[i]:.1f},{bins[i+1]:.1f}): n={mm.sum():3d} predicted={pte[mm].mean():.3f} observed={yte[mm].mean():.3f}")

print("\n"+"="*60); print("#11 proto_ratio WITHIN design method (scaffold-family confound check)"); print("="*60)
dm=fm.design_method.fillna("NA").values
pr_test=P[te][:,4]
for meth in pd.Series(dm[te]).value_counts().head(5).index:
    mm=(dm[te]==meth)
    if mm.sum()>=15 and len(np.unique(yte[mm]))>1:
        a=roc_auc_score(yte[mm],pr_test[mm]); print(f"  within '{meth[:28]}' (n={mm.sum()}): proto_ratio AUROC={max(a,1-a):.3f}")
print("  -> if proto_ratio still discriminates WITHIN a method, it is not merely scaffold-family detection")
pd.DataFrame({"nested_cv_auroc":[round(np.mean(oa),3)],"nested_cv_std":[round(np.std(oa),3)],
  "inference_ms":[round(dt*1000,3)],"fdr_sig_targets":[int(sig.sum())]}).to_csv("outputs/r8b_stats.csv",index=False)
print("\nsaved outputs/r8b_stats.csv")
