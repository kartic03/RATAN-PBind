#!/usr/bin/env python3
"""R8 Batch 2 — few-shot recovery curve. For a held-out target (LOTO), build its
prototype from k known binders+non-binders ('shots'), predict the remaining pairs.
Shows AUROC recovery vs k => deployment guideline for new targets. CPU."""
import warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
LGB=dict(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=31,subsample=0.9,
         colsample_bytree=0.8,is_unbalance=True,verbosity=-1,n_jobs=8)
fm=pd.read_parquet("features/feature_matrix.parquet")
base_cols=list(pd.read_csv("features/feature_columns.csv")["column"])
emb=np.load("features/esm2_embeddings.npy"); pids=np.load("features/esm2_protein_ids.npy",allow_pickle=True)
p2r={str(p):i for i,p in enumerate(pids)}
E=np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y=fm.binding_label.values.astype(int); tgt=fm.target.values
Xb=fm[base_cols].values.astype(np.float32); eps=1e-8
def proto_row(ei,pp,pn,npos,nneg):
    cp=(ei@pp)/((np.linalg.norm(ei)*np.linalg.norm(pp))+eps); cn=(ei@pn)/((np.linalg.norm(ei)*np.linalg.norm(pn))+eps)
    diff=pp-pn; dn=np.linalg.norm(diff)+eps
    return [cp,cn,np.linalg.norm(ei-pp),(ei@diff)/dn,cp/(cn+eps),npos,nneg]
def full_proto(idx_mask, prot):  # prot: dict target->(pp,pn,npos,nneg)
    out=np.zeros((len(E),7),np.float32)
    for t,(pp,pn,a,b) in prot.items():
        m=(tgt==t)&idx_mask
        for i in np.where(m)[0]: out[i]=proto_row(E[i],pp,pn,a,b)
    return out
# targets with enough binders+nonbinders for k up to 10
cand=[t for t in np.unique(tgt) if ((tgt==t)&(y==1)).sum()>=15 and ((tgt==t)&(y==0)).sum()>=15]
print("few-shot targets:",cand)
KS=[0,1,2,5,10]; rngs=range(5)
results={k:[] for k in KS}
for held in cand:
    hm=tgt==held; trm=~hm
    # build prototypes for training targets from their full data
    prot_tr={}
    for t in np.unique(tgt[trm]):
        m=(tgt==t)&trm; pos=E[m&(y==1)]; neg=E[m&(y==0)]
        if len(pos)<1 or len(neg)<1: continue
        prot_tr[t]=(pos.mean(0),neg.mean(0),len(pos),len(neg))
    Ptr=full_proto(trm,prot_tr)
    Xtr=np.hstack([Xb,Ptr])[trm]; ytr=y[trm]
    m=lgb.LGBMClassifier(**LGB); m.fit(Xtr,ytr)
    posidx=np.where(hm&(y==1))[0]; negidx=np.where(hm&(y==0))[0]
    for k in KS:
        aucs=[]
        for r in rngs:
            rs=np.random.RandomState(r)
            if k==0:
                pp=np.zeros(1280); pn=np.zeros(1280); shots=set()
            else:
                sp=rs.choice(posidx,min(k,len(posidx)),False); sn=rs.choice(negidx,min(k,len(negidx)),False)
                shots=set(sp)|set(sn); pp=E[sp].mean(0); pn=E[sn].mean(0)
            evalidx=[i for i in np.where(hm)[0] if i not in shots]
            ev=np.array(evalidx)
            if len(np.unique(y[ev]))<2: continue
            Pev=np.array([proto_row(E[i],pp,pn,k,k) for i in ev])
            Xev=np.hstack([Xb[ev],Pev]); aucs.append(roc_auc_score(y[ev],m.predict_proba(Xev)[:,1]))
        if aucs: results[k].append(np.mean(aucs))
print("\n=== FEW-SHOT RECOVERY (mean AUROC over %d targets) ==="%len(cand))
print(f"  {'k (known binders+nonbinders)':32s} {'mean AUROC':>10s}")
rows=[]
for k in KS:
    mu=np.mean(results[k]); print(f"  {('k=%d'%k):32s} {mu:10.3f}"); rows.append({"k":k,"mean_auroc":round(mu,3)})
print(f"\n  k=0 (cold start) = {np.mean(results[0]):.3f}  ->  k=10 = {np.mean(results[10]):.3f}  (recovery +{np.mean(results[10])-np.mean(results[0]):.3f})")
pd.DataFrame(rows).to_csv("outputs/r8_fewshot.csv",index=False); print("saved outputs/r8_fewshot.csv")
