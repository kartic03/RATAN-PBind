#!/usr/bin/env python3
"""R8 Batch 4 - robustness checks:
1 k-NN baseline (is proto just nearest-neighbour?)  2 expression confound
3 affinity/strength ranking  4 grouped leakage (leave-author/method-out)
5 fair structural benchmark  6 feature-count reconciliation. CPU."""
import warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
LGB=dict(objective="binary",n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.9,
         colsample_bytree=0.8,min_child_samples=10,reg_lambda=1.0,is_unbalance=True,verbosity=-1,n_jobs=8)
fm=pd.read_parquet("features/feature_matrix.parquet")
base_cols=list(pd.read_csv("features/feature_columns.csv")["column"])
pw=pd.read_parquet("data/pairs_with_splits.parquet")[["protein_id","target","author","design_method"]]
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
P=proto_feats(E,tgt,y,tr)
Xfull=np.hstack([fm[base_cols].values.astype(np.float32),P])
m=lgb.LGBMClassifier(**LGB); m.fit(Xfull[tr],y[tr]); pful=m.predict_proba(Xfull[te])[:,1]; yte=y[te]
au_full=roc_auc_score(yte,pful)
print(f"FULL base+proto (470 feat) test AUROC={au_full:.3f}")

print("\n=== 1. k-NN / nearest-centroid baseline (is proto just k-NN?) ===")
# nearest-centroid: predict by proto_cos_pos (cosine to target's train binders) alone
knn_score=P[te][:,0]  # proto_cos_pos
print(f"  ESM-2 nearest-centroid (proto_cos_pos alone) AUROC={roc_auc_score(yte,knn_score):.3f}")
print(f"  proto_ratio alone AUROC={roc_auc_score(yte,P[te][:,4]):.3f}")
print(f"  -> FULL ML model ({au_full:.3f}) {'beats' if au_full>roc_auc_score(yte,knn_score)+0.02 else 'is close to'} naive k-NN")

print("\n=== 2. EXPRESSION CONFOUND (binding vs expression) ===")
expr=fm.expressed.fillna(True).astype(bool).values
sub=te & expr  # expressed test pairs only
if sub.sum()>30 and len(np.unique(y[sub]))>1:
    pe=m.predict_proba(Xfull[sub])[:,1]
    print(f"  binding AUROC on EXPRESSED-only test (n={sub.sum()}): {roc_auc_score(y[sub],pe):.3f}  (vs all-test {au_full:.3f})")
# does the model predict expression itself?
me=lgb.LGBMClassifier(**LGB); me.fit(Xfull[tr], expr[tr].astype(int)); pe2=me.predict_proba(Xfull[te])[:,1]
print(f"  same features predicting EXPRESSION: AUROC={roc_auc_score(expr[te].astype(int),pe2):.3f} (if ~binding AUROC, confound risk)")

print("\n=== 3. AFFINITY/STRENGTH RANKING (predicts binding, not just expression) ===")
bs=fm.binding_strength.values  # 0=none,1=weak,2=med,3=strong
bm=te & (bs>0)  # binders only, with strength
if bm.sum()>20:
    rho,p_=spearmanr(pful[bm[te]] if False else m.predict_proba(Xfull[bm])[:,1], bs[bm])
    print(f"  Spearman(pred prob, binding strength) among binders n={bm.sum()}: rho={rho:.3f} p={p_:.3g}")
    strong=bm&(bs==3); weak=bm&(bs==1)
    if strong.sum()>3 and weak.sum()>3:
        ys=np.concatenate([np.ones(strong.sum()),np.zeros(weak.sum())]); ps=np.concatenate([m.predict_proba(Xfull[strong])[:,1],m.predict_proba(Xfull[weak])[:,1]])
        print(f"  Strong-vs-Weak discrimination AUROC={roc_auc_score(ys,ps):.3f} (n_strong={strong.sum()},n_weak={weak.sum()})")

print("\n=== 4. GROUPED LEAKAGE - leave-author-out & leave-design-method-out (5-fold GroupKFold) ===")
for gcol in ["author","design_method"]:
    g=fm[gcol].fillna("NA").astype(str).values
    aucs=[]; gkf=GroupKFold(5)
    for a,b in gkf.split(Xfull,y,g):
        Pa=proto_feats(E,tgt,y,np.isin(np.arange(len(y)),a))
        Xa=np.hstack([fm[base_cols].values.astype(np.float32),Pa])
        if len(np.unique(y[b]))<2: continue
        mm=lgb.LGBMClassifier(**LGB); mm.fit(Xa[a],y[a]); aucs.append(roc_auc_score(y[b],mm.predict_proba(Xa[b])[:,1]))
    print(f"  leave-{gcol}-out GroupKFold AUROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  ({fm[gcol].nunique()} groups)  vs standard {au_full:.3f}")

print("\n=== 5. FAIR structural benchmark (Overath, BOTH CV-trained logistic) ===")
import re
ov=pd.read_csv("data/external/overath_prepared_training_dataset.csv",low_memory=False); yo=ov.binder.values.astype(int)
from sklearn.linear_model import LogisticRegression
skf=StratifiedKFold(5,shuffle=True,random_state=42)
def cvauc(X1):
    X1=np.nan_to_num(X1.reshape(-1,1)); c=[]
    for a,b in skf.split(X1,yo):
        lr=LogisticRegression(class_weight="balanced",max_iter=200).fit(X1[a],yo[a]); c.append(roc_auc_score(yo[b],lr.predict_proba(X1[b])[:,1]))
    return np.mean(c)
for met in ["af3_ipSAE_avg","boltz1_iptm_avg","colab_iptm_avg"]:
    if met in ov.columns: print(f"  {met:20s} (CV logistic) AUROC={cvauc(pd.to_numeric(ov[met],errors='coerce').values):.3f}")
print("  (compare to sequence-ML 5-fold CV 0.730 from R8.3 - now both are CV-trained, fair)")

print("\n=== 6. FEATURE COUNT reconciliation ===")
print(f"  headline reproduction model = base({len(base_cols)}) + proto(7) = {len(base_cols)+7} features (NOT 509)")
print(f"  -> manuscript must state 470 (interface features are a separate Phase-6a row), or the 509 model must be the one evaluated")
