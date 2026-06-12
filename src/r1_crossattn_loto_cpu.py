#!/usr/bin/env python3
"""R1.4/R1.5 - cross-attention binder x target LOTO, CPU-ONLY + lightweight.
Safe variant: never touches the display GPU (CUDA disabled), caps CPU threads,
and downsamples each target to <=TGT_CAP residues (strided) to bound runtime.
-> outputs/r1_xattn_loto_{pure,hybrid}_cpu.csv"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # force CPU, leave display GPU alone
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import json, time, gc, numpy as np, torch, torch.nn as nn, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_num_threads(8); torch.manual_seed(42); np.random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cpu"
ZERO = {"human-serum-albumin","human-tnfa","human-orm2","human-gm2a"}
TGT_CAP = 160          # max target residues per target (strided sample)
EPOCHS, PATIENCE, BSZ = 25, 6, 256
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

meta = pd.read_csv(os.path.join(ROOT,"data/targets/pairs_meta.csv"))
y = meta["binding_label"].values.astype(np.float32); tgt = meta["target"].values
base = np.load(os.path.join(ROOT,"data/targets/base_features.npy"))
emb = np.load(os.path.join(ROOT,"features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT,"features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p):i for i,p in enumerate(pids)}
B = np.stack([emb[p2r[str(p)]] for p in meta["protein_id"]]).astype(np.float32)

TPR = np.load(os.path.join(ROOT,"models/target_perres.npz"))
def cap(a):
    a = np.asarray(a, dtype=np.float32)
    if a.shape[0] <= TGT_CAP: return a
    idx = np.linspace(0, a.shape[0]-1, TGT_CAP).round().astype(int)
    return a[idx]
tgt_res = {k: torch.tensor(cap(TPR[k])) for k in TPR.files}
log(f"pairs={len(meta)} base={base.shape} binder={B.shape} targets={len(tgt_res)} "
    f"(capped to <= {TGT_CAP} res; max now {max(t.shape[0] for t in tgt_res.values())})")

class XAttn(nn.Module):
    def __init__(s,d=256,h=4,use_base=True,nb=463):
        super().__init__(); s.bp=nn.Linear(1280,d); s.tp=nn.Linear(1280,d)
        s.att=nn.MultiheadAttention(d,h,batch_first=True); s.use_base=use_base; comb=d*4
        if use_base: s.bn=nn.Sequential(nn.Linear(nb,d),nn.ReLU(),nn.LayerNorm(d)); comb+=d
        s.head=nn.Sequential(nn.LayerNorm(comb),nn.Linear(comb,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,1))
    def forward(s,b,tk,tm,bf=None):
        q=s.bp(b).unsqueeze(1); kv=s.tp(tk); ctx,_=s.att(q,kv,kv,key_padding_mask=tm)
        ctx=ctx.squeeze(1); qb=q.squeeze(1); f=[qb,ctx,qb*ctx,(qb-ctx).abs()]
        if s.use_base: f.append(s.bn(bf))
        return s.head(torch.cat(f,-1)).squeeze(-1)

def batches(idx,bsz=BSZ):
    for st in range(0,len(idx),bsz):
        j=idx[st:st+bsz]; slugs=[tgt[k] for k in j]; Lt=max(tgt_res[x].shape[0] for x in slugs)
        tk=torch.zeros(len(j),Lt,1280); tm=torch.ones(len(j),Lt,dtype=torch.bool)
        for r,x in enumerate(slugs):
            L=tgt_res[x].shape[0]; tk[r,:L]=tgt_res[x]; tm[r,:L]=False
        yield (torch.tensor(B[j]),tk,tm,torch.tensor(base[j]),torch.tensor(y[j]))

def train_eval(tr,te,use_base,mean,std):
    m=XAttn(use_base=use_base); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    pos=max(1,int(y[tr].sum())); pw=torch.tensor((len(tr)-pos)/pos,dtype=torch.float32)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    perm=np.random.RandomState(42).permutation(tr); c=int(0.9*len(perm)); fit,val=perm[:c],perm[c:]
    best=-1; bs=None; bad=0
    for ep in range(EPOCHS):
        m.train()
        for b,tk,tm,bf,yy in batches(np.random.permutation(fit)):
            bf=(bf-mean)/std; opt.zero_grad(); lf(m(b,tk,tm,bf),yy).backward(); opt.step()
        m.eval(); ps=[];ys=[]
        with torch.no_grad():
            for b,tk,tm,bf,yy in batches(val):
                bf=(bf-mean)/std; ps.append(torch.sigmoid(m(b,tk,tm,bf)).numpy()); ys.append(yy.numpy())
        va=roc_auc_score(np.concatenate(ys),np.concatenate(ps)) if len(np.unique(np.concatenate(ys)))>1 else 0.5
        if va>best: best=va; bs={k:v.detach().clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=PATIENCE: break
    m.load_state_dict(bs); m.eval(); ps=[]
    with torch.no_grad():
        for b,tk,tm,bf,yy in batches(te):
            bf=(bf-mean)/std; ps.append(torch.sigmoid(m(b,tk,tm,bf)).numpy())
    return np.concatenate(ps)

def loto(use_base):
    rows=[]
    for t in sorted(x for x in set(tgt) if x not in ZERO):
        te=np.where(tgt==t)[0]; tr=np.where(tgt!=t)[0]
        if len(np.unique(y[te]))<2: continue
        mean=torch.tensor(base[tr].mean(0)); std=torch.tensor(base[tr].std(0)+1e-6)
        p=train_eval(tr,te,use_base,mean,std)
        rows.append(dict(target=t,n=len(te),binders=int(y[te].sum()),
                         auroc=roc_auc_score(y[te],p),auprc=average_precision_score(y[te],p)))
        log(f"  {t:24s} n={len(te):4d} auroc={rows[-1]['auroc']:.3f}")
        gc.collect()
    return pd.DataFrame(rows)

def main():
    os.makedirs(os.path.join(ROOT,"outputs"),exist_ok=True)
    for use_base,name in [(False,"PURE"),(True,"HYBRID")]:
        log(f"=== LOTO {name} (use_base={use_base}) ===")
        df=loto(use_base).sort_values("n",ascending=False)
        df.to_csv(os.path.join(ROOT,f"outputs/r1_xattn_loto_{name.lower()}_cpu.csv"),index=False)
        log(f"{name}: mean AUROC={df.auroc.mean():.4f}  pair-weighted={np.average(df.auroc,weights=df.n):.4f}")

if __name__=="__main__": main()
