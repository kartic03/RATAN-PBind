import glob, os, subprocess, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
PY="/home/kartic/miniforge3/envs/mmgbsa/bin/python3"
res=[]
for d in sorted(glob.glob("outputs/boltz_set/boltz_results_boltz_set/predictions/*/")):
    name=os.path.basename(d.rstrip("/")); pdb=glob.glob(d+"*_model_0.pdb")
    if not pdb: continue
    try:
        out=subprocess.run([PY,"src/r8_mmgbsa_worker.py",pdb[0]],capture_output=True,text=True,timeout=600)
        line=[l for l in out.stdout.splitlines() if l.startswith("DG")]
        if line: g=float(line[0].split()[1]); res.append((name,name.rsplit("_",1)[0],round(g,1))); print(f"  {name:14s} dG={g:8.1f} kcal/mol",flush=True)
        else: print(f"  {name}: FAIL {out.stderr.strip().splitlines()[-1][:80] if out.stderr.strip() else 'no output'}",flush=True)
    except Exception as e: print(f"  {name}: {repr(e)[:70]}",flush=True)
df=pd.DataFrame(res,columns=["name","category","dg_bind"])
print("\n=== MM-GBSA dG_bind by category (more negative = stronger) ===")
print(df.groupby("category")["dg_bind"].agg(["mean","count"]).round(1).to_string())
b=df[df.category=="binder"]["dg_bind"]; nb=df[df.category=="nonbinder"]["dg_bind"]
if len(b) and len(nb):
    y=np.r_[np.ones(len(b)),np.zeros(len(nb))]; s=-np.r_[b.values,nb.values]
    print(f"\n  binder mean dG={b.mean():.1f} vs non-binder {nb.mean():.1f} kcal/mol | MM-GBSA AUROC={roc_auc_score(y,s):.3f} (n={len(y)})")
    dz=df[df.category=='designed']['dg_bind']; sc=df[df.category=='scrambled']['dg_bind']
    if len(dz) and len(sc): print(f"  designed dG={dz.iloc[0]:.1f} vs scrambled mean {sc.mean():.1f}")
df.to_csv("outputs/r8_mmgbsa.csv",index=False); print("saved outputs/r8_mmgbsa.csv")
