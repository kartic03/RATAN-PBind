#!/usr/bin/env python3
"""#6 few-shot recovery WITH bootstrap CIs over (target x shot-draw). CPU."""
import warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
LGB=dict(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=31,subsample=0.9,colsample_bytree=0.8,is_unbalance=True,verbosity=-1,n_jobs=8)
fm=pd.read_parquet("features/feature_matrix.parquet"); base_cols=list(pd.read_csv("features/feature_columns.csv")["column"])
emb=np.load("features/esm2_embeddings.npy"); pids=np.load("features/esm2_protein_ids.npy",allow_pickle=True)
p2r={str(p):i for i,p in enumerate(pids)}; E=np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y=fm.binding_label.values.astype(int); tgt=fm.target.values; Xb=fm[base_cols].values.astype(np.float32); eps=1e-8
def prow(ei,pp,pn,a,b):
    cp=(ei@pp)/((np.linalg.norm(ei)*np.linalg.norm(pp))+eps); cn=(ei@pn)/((np.linalg.norm(ei)*np.linalg.norm(pn))+eps); d=pp-pn
    return [cp,cn,np.linalg.norm(ei-pp),(ei@d)/(np.linalg.norm(d)+eps),cp/(cn+eps),a,b]
cand=[t for t in np.unique(tgt) if ((tgt==t)&(y==1)).sum()>=15 and ((tgt==t)&(y==0)).sum()>=15]
KS=[0,1,2,5,10]; percell={k:[] for k in KS}  # store per (target,draw) AUROC
for held in cand:
    hm=tgt==held; trm=~hm; prot={}
    for t in np.unique(tgt[trm]):
        mm=(tgt==t)&trm; pos=E[mm&(y==1)]; neg=E[mm&(y==0)]
        if len(pos)and len(neg): prot[t]=(pos.mean(0),neg.mean(0),len(pos),len(neg))
    Ptr=np.zeros((len(E),7),np.float32)
    for t,(pp,pn,a,b) in prot.items():
        for i in np.where((tgt==t)&trm)[0]: Ptr[i]=prow(E[i],pp,pn,a,b)
    m=lgb.LGBMClassifier(**LGB); m.fit(np.hstack([Xb,Ptr])[trm],y[trm])
    pidx=np.where(hm&(y==1))[0]; nidx=np.where(hm&(y==0))[0]
    for k in KS:
        for r in range(10):
            rs=np.random.RandomState(r)
            if k==0: pp=np.zeros(1280);pn=np.zeros(1280);shots=set()
            else:
                sp=rs.choice(pidx,min(k,len(pidx)),False);sn=rs.choice(nidx,min(k,len(nidx)),False);shots=set(sp)|set(sn);pp=E[sp].mean(0);pn=E[sn].mean(0)
            ev=np.array([i for i in np.where(hm)[0] if i not in shots])
            if len(np.unique(y[ev]))<2: continue
            Pe=np.array([prow(E[i],pp,pn,k,k) for i in ev])
            percell[k].append(roc_auc_score(y[ev],m.predict_proba(np.hstack([Xb[ev],Pe]))[:,1]))
print(f"few-shot targets (n={len(cand)}): {cand}")
print(f"\n  {'k':>3s} {'mean AUROC':>10s} {'95% CI':>18s}")
rows=[]
for k in KS:
    v=np.array(percell[k]); boot=[np.mean(np.random.RandomState(s).choice(v,len(v),True)) for s in range(2000)]
    lo,hi=np.percentile(boot,[2.5,97.5]); print(f"  {k:3d} {v.mean():10.3f}   [{lo:.3f}, {hi:.3f}]"); rows.append({"k":k,"auroc":round(v.mean(),3),"ci_lo":round(lo,3),"ci_hi":round(hi,3)})
pd.DataFrame(rows).to_csv("outputs/r8b_fewshot_ci.csv",index=False); print("saved outputs/r8b_fewshot_ci.csv")
