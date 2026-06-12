#!/usr/bin/env python3
"""R1.5a - LOTO test of target-aware INTERACTION features (LightGBM).
A: target-blind 463 base. B: + [cos,L2,dot](binder,target) + PCA32(binder⊙target).
research env. Portable ROOT. -> outputs/r1_loto_targetaware.csv"""
import os, json, time, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
ZERO={"human-serum-albumin","human-tnfa","human-orm2","human-gm2a"}
LGB=dict(objective="binary",n_estimators=400,learning_rate=0.05,num_leaves=31,
         subsample=0.9,colsample_bytree=0.8,min_child_samples=10,reg_lambda=1.0,
         is_unbalance=True,verbosity=-1,n_jobs=8)

def load():
    fm=pd.read_parquet(os.path.join(ROOT,"features/feature_matrix.parquet"))
    base_cols=list(pd.read_csv(os.path.join(ROOT,"features/feature_columns.csv"))["column"])
    emb=np.load(os.path.join(ROOT,"features/esm2_embeddings.npy"))
    pids=np.load(os.path.join(ROOT,"features/esm2_protein_ids.npy"),allow_pickle=True)
    p2r={str(p):i for i,p in enumerate(pids)}
    te=np.load(os.path.join(ROOT,"models/target_emb_v2_whole.npy"))
    sl=json.load(open(os.path.join(ROOT,"models/target_emb_v2_slugs.json")))
    s2v={s:te[i] for i,s in enumerate(sl)}
    B=np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]])
    G=np.stack([s2v[t] for t in fm["target"]])
    return fm, base_cols, B, G

def inter(B,G):
    bn=B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-8); gn=G/(np.linalg.norm(G,axis=1,keepdims=True)+1e-8)
    return np.stack([(bn*gn).sum(1),(B*G).sum(1),np.linalg.norm(B-G,axis=1)],1)
def au(y,p): return roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan

def loto(fm,base_cols,B,G,use_t):
    y=fm["binding_label"].values.astype(int); Xb=fm[base_cols].values.astype(np.float32)
    rows=[]
    for t in sorted(x for x in fm["target"].unique() if x not in ZERO):
        te=(fm["target"]==t).values; tr=~te
        if len(np.unique(y[te]))<2: continue
        Xtr=[Xb[tr]]; Xte=[Xb[te]]
        if use_t:
            pca=PCA(32,random_state=42).fit(B[tr]*G[tr])
            Xtr+=[inter(B[tr],G[tr]),pca.transform(B[tr]*G[tr])]
            Xte+=[inter(B[te],G[te]),pca.transform(B[te]*G[te])]
        m=lgb.LGBMClassifier(**LGB); m.fit(np.hstack(Xtr),y[tr]); p=m.predict_proba(np.hstack(Xte))[:,1]
        rows.append(dict(target=t,n=int(te.sum()),binders=int(y[te].sum()),auroc=au(y[te],p),auprc=average_precision_score(y[te],p)))
    return pd.DataFrame(rows)

def main():
    log("loading ..."); fm,bc,B,G=load()
    log(f"pairs={len(fm)} base={len(bc)} binder={B.shape} target={G.shape}")
    a=loto(fm,bc,B,G,False); b=loto(fm,bc,B,G,True)
    out=a.rename(columns={"auroc":"auroc_base","auprc":"auprc_base"}).merge(
        b.rename(columns={"auroc":"auroc_tgt","auprc":"auprc_tgt"})[["target","auroc_tgt","auprc_tgt"]],on="target")
    out["d_auroc"]=out["auroc_tgt"]-out["auroc_base"]; out=out.sort_values("n",ascending=False)
    os.makedirs(os.path.join(ROOT,"outputs"),exist_ok=True)
    out.to_csv(os.path.join(ROOT,"outputs/r1_loto_targetaware.csv"),index=False)
    print(out.to_string(index=False,formatters={c:(lambda x:f"{x:.3f}") for c in ["auroc_base","auprc_base","auroc_tgt","auprc_tgt","d_auroc"]}))
    print(f"\nSUMMARY (n_targets={len(out)}):")
    print(f"  A target-blind  mean AUROC={out.auroc_base.mean():.4f}  pair-weighted={np.average(out.auroc_base,weights=out.n):.4f}")
    print(f"  B target-aware  mean AUROC={out.auroc_tgt.mean():.4f}  pair-weighted={np.average(out.auroc_tgt,weights=out.n):.4f}")
    print(f"  mean delta (B-A)={out.d_auroc.mean():+.4f}   [manuscript reported proto-LOTO 0.658]")

if __name__=="__main__": main()
