#!/usr/bin/env python3
"""R8 Batch 3 - cheap ML vs expensive structure prediction. On Overath (which has
precomputed AF2/AF3/Boltz/ColabFold interface-confidence metrics + binary labels),
compare each structural metric (single-feature classifier) vs a cheap sequence-feature
ML model (5-fold CV) for predicting experimental binding success. CPU."""
import re, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
d=pd.read_csv("data/external/overath_prepared_training_dataset.csv", low_memory=False)
y=d.binder.values.astype(int)
print(f"Overath: {len(d)} binders, {y.sum()} positive ({100*y.mean():.1f}%)")
# structural metrics (expensive folding) as single-feature classifiers
mets=["colab_iptm_avg","boltz1_iptm_avg","af3_iptm_avg","colab_actifptm_avg",
      "boltz1_ipSAE_avg","af3_ipSAE_avg","af2_pae_interaction","boltz1_complex_iplddt_avg"]
print("\n=== Expensive structure-prediction metrics (single-feature AUROC) ===")
rows=[]
for m in mets:
    if m not in d.columns: continue
    v=pd.to_numeric(d[m],errors="coerce").values; ok=~np.isnan(v)
    if ok.sum()<100 or len(np.unique(y[ok]))<2: continue
    a=roc_auc_score(y[ok],v[ok]); a=max(a,1-a)
    print(f"  {m:28s} AUROC={a:.3f}  (n={ok.sum()})"); rows.append((m,round(a,3),int(ok.sum())))
# cheap sequence-feature ML
AAs="ACDEFGHIKLMNPQRSTVWY"; AID={a:i for i,a in enumerate(AAs)}
aa3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def parse(ss):
    if not isinstance(ss,str): return None
    r={}
    for tok in ss.split(":"):
        mm=re.match(r'([A-Z]{3})_(\d+)=',tok)
        if mm: r[int(mm.group(2))]=aa3.get(mm.group(1),'X')
    return "".join(r[i] for i in sorted(r)) if r else None
seqs=d.input_pymol_binder_ss.map(parse)
def feat(s):
    s="".join(c for c in str(s).upper() if c in AAs); n=max(len(s),1)
    aac=np.zeros(20); dpc=np.zeros(400)
    for c in s: aac[AID[c]]+=1
    for i in range(len(s)-1): dpc[AID[s[i]]*20+AID[s[i+1]]]+=1
    return np.concatenate([aac/n,dpc/max(n-1,1),[n]])
ok=seqs.notna().values
X=np.stack([feat(s) for s in seqs[ok]]); yo=y[ok]
P=dict(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=31,is_unbalance=True,verbosity=-1,n_jobs=8)
cv=[]; skf=StratifiedKFold(5,shuffle=True,random_state=42)
for a,b in skf.split(X,yo):
    m=lgb.LGBMClassifier(**P); m.fit(X[a],yo[a]); cv.append(roc_auc_score(yo[b],m.predict_proba(X[b])[:,1]))
mlauc=np.mean(cv)
print(f"\n=== Cheap sequence-feature ML (5-fold CV, no folding) ===")
print(f"  AAC+DPC LightGBM AUROC={mlauc:.3f}±{np.std(cv):.3f}  (n={ok.sum()})")
best_struct=max(r[1] for r in rows) if rows else 0
print(f"\n  Best expensive structural metric = {best_struct:.3f}  |  cheap sequence ML = {mlauc:.3f}")
print(f"  -> sequence ML is {'COMPETITIVE WITH / BEATS' if mlauc>=best_struct-0.02 else 'below'} structure prediction, at ~0.5s vs minutes-hours/complex")
pd.DataFrame(rows+[("sequence_ML_5foldCV",round(mlauc,3),int(ok.sum()))],columns=["method","auroc","n"]).to_csv("outputs/r8_benchmark.csv",index=False)
print("saved outputs/r8_benchmark.csv")
