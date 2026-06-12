#!/usr/bin/env python3
"""#20 — REAL physics: MM-GBSA single-trajectory binding free energy on the R4
Boltz-2 complex structures (chain A=binder, B=target). GBn2 implicit solvent.
ΔG_bind = E_complex - E_binderAlone - E_targetAlone. Independent of Boltz. mmgbsa env."""
import glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pdbfixer import PDBFixer
from openmm import app, unit, LangevinIntegrator, Platform
ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
plat = Platform.getPlatformByName("CPU")
KCAL = unit.kilocalorie_per_mole
def E(top, pos, minimize=False):
    sys = ff.createSystem(top, nonbondedMethod=app.NoCutoff)
    sim = app.Simulation(top, sys, LangevinIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds), plat)
    sim.context.setPositions(pos)
    if minimize: sim.minimizeEnergy(maxIterations=300)
    st = sim.context.getState(getEnergy=True, getPositions=True)
    return st.getPotentialEnergy().value_in_unit(KCAL), st.getPositions()
def dg_bind(pdb):
    fx=PDBFixer(pdb); fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    Ec, minpos = E(fx.topology, fx.positions, minimize=True)
    def chainE(keep):
        m=app.Modeller(fx.topology, minpos); m.delete([c for c in list(m.topology.chains()) if c.id!=keep])
        e,_=E(m.topology, m.positions); return e
    return Ec - chainE("A") - chainE("B")
res=[]
for d in sorted(glob.glob("outputs/boltz_set/boltz_results_boltz_set/predictions/*/")):
    name=os.path.basename(d.rstrip("/")); pdb=glob.glob(d+"*_model_0.pdb")
    if not pdb: continue
    try:
        g=dg_bind(pdb[0]); cat=name.rsplit("_",1)[0]; res.append((name,cat,round(g,1))); print(f"  {name:14s} ΔG_bind={g:8.1f} kcal/mol", flush=True)
    except Exception as ex: print(f"  {name}: FAIL {repr(ex)[:90]}", flush=True)
df=pd.DataFrame(res,columns=["name","category","dg_bind"])
print("\n=== MM-GBSA ΔG_bind by category (more negative = stronger binding) ===")
print(df.groupby("category")["dg_bind"].agg(["mean","count"]).round(1).to_string())
from sklearn.metrics import roc_auc_score
b=df[df.category=="binder"]["dg_bind"]; nb=df[df.category=="nonbinder"]["dg_bind"]
if len(b) and len(nb):
    y=np.r_[np.ones(len(b)),np.zeros(len(nb))]; s=-np.r_[b.values,nb.values]
    print(f"\n  binder mean ΔG={b.mean():.1f}  vs  non-binder {nb.mean():.1f} kcal/mol")
    print(f"  MM-GBSA ΔG-as-classifier AUROC = {roc_auc_score(y,s):.3f} (n={len(y)})")
    dz=df[df.category=='designed']['dg_bind']; sc=df[df.category=='scrambled']['dg_bind']
    if len(dz): print(f"  designed ΔG={dz.iloc[0]:.1f}  vs  scrambled mean {sc.mean():.1f}")
df.to_csv("outputs/r8_mmgbsa.csv",index=False); print("saved outputs/r8_mmgbsa.csv")
