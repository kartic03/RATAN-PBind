#!/usr/bin/env python3
"""SKEMPI 2.0 natural-PPI transfer analysis.
Tests whether the physicochemical-feature philosophy underlying RATAN-PBind
transfers to natural protein-protein interactions: predict binding-abolishing
mutations from physicochemical change + interface location. GroupKFold by complex."""
import re, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
d=pd.read_csv("data/external/skempi_v2.csv",sep=";",low_memory=False)
# AA physicochemical scales
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
VOL={'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,
'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
CHG={'D':-1,'E':-1,'K':1,'R':1,'H':0.5}
def chg(a): return CHG.get(a,0)
aw=pd.to_numeric(d['Affinity_wt_parsed'],errors='coerce').values; am=pd.to_numeric(d['Affinity_mut_parsed'],errors='coerce').values
mut=d['Mutation(s)_cleaned'].astype(str).values; loc=d['iMutation_Location(s)'].astype(str).values; pdbv=d['#Pdb'].astype(str).values
def parse_mut(m):
    mm=re.match(r'^([A-Z])[A-Z]?(\d+)([A-Z])$',m)
    if mm: return mm.group(1),mm.group(3)
    return None,None
rows=[]
for i in range(len(d)):
    if mut[i].count(",")>0: continue
    w,mt=parse_mut(mut[i])
    if w is None or w not in KD or mt not in KD: continue
    if np.isnan(am[i]) or np.isnan(aw[i]): continue
    abolish=1 if am[i]>1e-5 else (0 if am[i]<1e-7 else None)
    if abolish is None: continue
    rows.append(dict(pdb=pdbv[i].split("_")[0], y=abolish,
        d_hydro=KD[mt]-KD[w], d_vol=VOL[mt]-VOL[w], d_chg=chg(mt)-chg(w),
        abs_dhydro=abs(KD[mt]-KD[w]), abs_dvol=abs(VOL[mt]-VOL[w]),
        to_gly_pro=int(mt in "GP"), from_hydrophobic=int(w in "AILMFWV"),
        core=int("COR" in loc[i]), rim=int("RIM" in loc[i]), support=int("SUP" in loc[i])))
df=pd.DataFrame(rows)
print(f"single-mutation binary task: n={len(df)}  abolishing={df.y.sum()} ({100*df.y.mean():.0f}%)  complexes={df.pdb.nunique()}")
feats=["d_hydro","d_vol","d_chg","abs_dhydro","abs_dvol","to_gly_pro","from_hydrophobic","core","rim","support"]
X=df[feats].values; y=df.y.values; g=df.pdb.values
aucs=[]; gkf=GroupKFold(5)
for a,b in gkf.split(X,y,g):
    m=lgb.LGBMClassifier(objective="binary",n_estimators=300,learning_rate=0.05,num_leaves=15,
        is_unbalance=True,verbosity=-1,n_jobs=8); m.fit(X[a],y[a]); aucs.append(roc_auc_score(y[b],m.predict_proba(X[b])[:,1]))
print(f"\n=== Physicochemical features predict binding-abolishing mutations (natural PPI) ===")
print(f"  GroupKFold-by-complex AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
# single-feature: interface-core location alone
print(f"  (core-location alone AUROC = {max(roc_auc_score(y,df.core.values),1-roc_auc_score(y,df.core.values)):.3f}; |Δhydrophobicity| alone = {max(roc_auc_score(y,df.abs_dhydro.values),1-roc_auc_score(y,df.abs_dhydro.values)):.3f})")
print(f"\n  -> the physicochemical feature philosophy {'DOES' if np.mean(aucs)>0.6 else 'does NOT'} transfer to natural PPI binding determinants")
pd.DataFrame({"analysis":["skempi_binding_abolishing"],"n":[len(df)],"auroc":[round(np.mean(aucs),3)],"complexes":[df.pdb.nunique()]}).to_csv("outputs/r8_skempi.csv",index=False)
print("saved outputs/r8_skempi.csv")
