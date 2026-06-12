#!/usr/bin/env python3
"""R6 - feature engineering: (1) frequency-normalised DPC
(observed/expected, removing AA-composition bias) tested vs raw DPC; (2) sequence
redundancy / effective-sample-size analysis. CPU/LightGBM. research env."""
import warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
AAs="ACDEFGHIKLMNPQRSTVWY"; AIDX={a:i for i,a in enumerate(AAs)}
def aac(seq):
    seq="".join(c for c in seq.upper() if c in AAs); n=max(len(seq),1)
    v=np.zeros(20)
    for c in seq: v[AIDX[c]]+=1
    return v/n
def dpc_raw(seq):
    seq="".join(c for c in seq.upper() if c in AAs); m=max(len(seq)-1,1)
    v=np.zeros(400)
    for i in range(len(seq)-1): v[AIDX[seq[i]]*20+AIDX[seq[i+1]]]+=1
    return v/m
def dpc_norm(seq):
    """observed/expected: log((obs+eps)/(f_i*f_j+eps)) - removes AA-frequency bias."""
    a=aac(seq); d=dpc_raw(seq); exp=np.outer(a,a).ravel(); eps=1e-4
    return np.log((d+eps)/(exp+eps))

d=pd.read_parquet("data/pairs_with_splits.parquet")
d=d[['protein_id','target','sequence','binding_label','split']].dropna(subset=['sequence'])
y=d.binding_label.values.astype(int); tr=d.split.isin(['train','val']).values; te=(d.split=='test').values
seqs=d.sequence.tolist()
AACm=np.stack([aac(s) for s in seqs]); RAW=np.stack([dpc_raw(s) for s in seqs]); NORM=np.stack([dpc_norm(s) for s in seqs])
L=np.array([[len(s)] for s in seqs])
P=dict(objective="binary",n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.9,
       colsample_bytree=0.8,is_unbalance=True,verbosity=-1,n_jobs=8)
def evalset(X,name):
    m=lgb.LGBMClassifier(**P); m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]
    au=roc_auc_score(y[te],p); ap=average_precision_score(y[te],p)
    # 5-fold CV on train for stability
    cv=[]; skf=StratifiedKFold(5,shuffle=True,random_state=42)
    Xt,yt=X[tr],y[tr]
    for a,b in skf.split(Xt,yt):
        mm=lgb.LGBMClassifier(**P); mm.fit(Xt[a],yt[a]); cv.append(roc_auc_score(yt[b],mm.predict_proba(Xt[b])[:,1]))
    print(f"  {name:28s} test AUROC={au:.3f} AUPRC={ap:.3f} | 5-fold CV={np.mean(cv):.3f}±{np.std(cv):.3f}")
    return au
print("=== R6.1 DPC frequency-normalisation ===")
print("  feature set = AAC(20) + DPC(400) + length, sequence-only:")
a1=evalset(np.hstack([AACm,RAW,L]),"raw DPC (current)")
a2=evalset(np.hstack([AACm,NORM,L]),"frequency-normalised DPC")
print(f"  -> Δ AUROC (norm - raw) = {a2-a1:+.3f}")

print("\n=== R6.2 sequence redundancy / effective sample size ===")
uniq=d.drop_duplicates('sequence'); dup=len(d)-len(uniq)
print(f"  total pairs={len(d)}  unique binder sequences={d.sequence.nunique()}  exact-duplicate rows={dup}")
print(f"  {'target':24s} {'pairs':>5s} {'uniq_seq':>8s} {'redundancy%':>11s}")
for t,g in d.groupby('target'):
    if len(g)>=20:
        u=g.sequence.nunique(); print(f"  {t:24s} {len(g):5d} {u:8d} {100*(1-u/len(g)):11.1f}")
pd.DataFrame([{"feature_set":"raw_DPC","test_auroc":round(a1,3)},
              {"feature_set":"freq_norm_DPC","test_auroc":round(a2,3)}]).to_csv("outputs/r6_dpc_norm.csv",index=False)
print("\nsaved outputs/r6_dpc_norm.csv")
