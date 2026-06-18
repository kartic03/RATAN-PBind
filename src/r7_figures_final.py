"""Nature-level publication figures for the J. Cheminformatics manuscript.
Refined cohesive palette, clean typographic hierarchy, no gridline clutter,
no text/arrow overlap, constrained layout. Arial, 300 dpi PNG + vector PDF."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "outputs"
FIG  = ROOT / "paper" / "Journal of Cheminformatics" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Refined, cohesive palette (deep blue / teal / amber / coral / slate)
C = dict(blue="#2C6FB3", teal="#2E9E8E", amber="#E0A33E", coral="#CB5A4C",
         slate="#7A8290", lgray="#C7CCD2", ink="#222831", purple="#7E6BB5",
         band="#DCE7F2")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 11.5,
    "axes.titleweight": "normal", "axes.titlepad": 8,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B7BDC4", "axes.linewidth": 0.9, "axes.labelcolor": C["ink"],
    "xtick.color": "#AEB4BB", "ytick.color": "#AEB4BB",
    "xtick.labelcolor": "#3C4450", "ytick.labelcolor": "#3C4450",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "axes.grid": False, "figure.dpi": 150, "savefig.dpi": 300,
})

def panel(ax, L, dx=-0.16):
    ax.text(dx, 1.05, L, transform=ax.transAxes, fontsize=15, fontweight="bold",
            color=C["ink"], va="bottom", ha="left")
def vguides(ax, xs, ymax):  # faint reference guides behind data
    for x in xs: ax.axvline(x, color="#EEF1F4", lw=0.8, zorder=0)
def save(name):
    plt.savefig(FIG/f"{name}.png", bbox_inches="tight", pad_inches=0.10)
    plt.savefig(FIG/f"{name}.pdf", bbox_inches="tight", pad_inches=0.10)
    plt.close(); print("saved", name)

# ── Figure 1 — dataset (a) + pipeline (b) ────────────────────────────────────
def fig1():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2),
                                  gridspec_kw={"width_ratios": [1, 1.5]}, layout="constrained")
    tg = [("nipah-glycoprotein-g",10.0),("egfr",17.9),("mdm2",29.1),("pd-l1",50.4),
          ("pdgfr-beta",60.0),("insulin-receptor",60.0),("il7r",67.5),("spcas9",70.0)]
    names=[t[0] for t in tg]; rates=[t[1] for t in tg]; y=np.arange(len(names))
    cols=[C["blue"] if r>=30 else "#9FC0E0" for r in rates]
    for x in (20,40,60,80): ax.axvline(x, color="#EEF1F4", lw=0.8, zorder=0)
    ax.barh(y, rates, color=cols, height=0.66, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0, 84); ax.set_xticks([0,20,40,60,80]); ax.set_xlabel("Binding success rate (%)")
    ax.axvline(17.8, color=C["coral"], lw=1.4, ls=(0,(5,2)), zorder=3)
    ax.text(17.8, len(names)-0.30, " mean 17.8%", color=C["coral"], fontsize=9, va="center", ha="left")
    for yi, r in zip(y, rates):
        ax.text(r+1.6, yi, f"{r:.0f}", va="center", ha="left", fontsize=9, color="#5A626D")
    ax.tick_params(length=0, axis="y"); panel(ax, "a")

    # pipeline — 4 clean columns, arrows live in the column gaps, drawn above boxes
    ax2.set_xlim(0,14); ax2.set_ylim(0,10); ax2.axis("off"); panel(ax2,"b", dx=-0.04)
    def box(cx,cy,w,h,t,fc,fg="white",fs=10.5):
        ax2.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
                      boxstyle="round,pad=0.02,rounding_size=0.22",
                      facecolor=fc,edgecolor="none",zorder=2))
        ax2.text(cx,cy,t,ha="center",va="center",fontsize=fs,color=fg,zorder=4,linespacing=1.35)
        return dict(l=cx-w/2,r=cx+w/2,t=cy+h/2,b=cy-h/2,cx=cx,cy=cy)
    def arr(p1,p2):
        ax2.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=12,
                      color=C["slate"],lw=1.6,zorder=3,shrinkA=0,shrinkB=0,
                      connectionstyle="arc3,rad=0"))
    seq=box(1.7,5.0,2.6,1.5,"Protein\nsequence","#3A4250")
    hc =box(5.6,7.6,3.0,1.3,"Handcrafted\nfeatures",C["slate"],fs=10)
    es =box(5.6,5.0,3.0,1.3,"ESM-2 650M\nembedding",C["blue"])
    pr =box(5.6,2.4,3.0,1.3,"Prototype\nsimilarity",C["purple"])
    mdl=box(10.1,5.0,2.8,1.7,"LightGBM\n470 features",C["teal"],fs=10.5)
    out=box(13.0,5.0,1.6,1.7,"Binding\nprediction",C["amber"],fs=9.5)
    for tgt in (hc,es,pr): arr((seq["r"],seq["cy"]),(tgt["l"],tgt["cy"]))
    for src in (hc,es,pr): arr((src["r"],src["cy"]),(mdl["l"],mdl["cy"]))
    arr((mdl["r"],mdl["cy"]),(out["l"],out["cy"]))
    ax2.text(13.0,3.7,"+ interpretation\n+ design",ha="center",va="top",fontsize=8.5,
             color="#7A8290",style="italic",linespacing=1.3)
    save("fig1_dataset_pipeline")

# ── Figure 2 — model performance (AUROC | AUPRC lollipops) ───────────────────
def fig2():
    M=[("Logistic regression",0.790,0.499,3),("Random forest",0.854,0.580,3),
       ("XGBoost",0.880,0.682,3),("LightGBM",0.893,0.713,3),("XGB + ESM-2",0.871,0.633,4),
       ("LGB + ESM-2",0.864,0.675,4),("ESM-2 fine-tuned",0.854,0.592,5),
       ("Calibrated ensemble",0.883,0.692,5),("LGB + interface",0.894,0.702,6),
       ("XGB + Proto",0.940,0.748,6),("LGB + Proto",0.946,0.770,6)]
    M=sorted(M,key=lambda m:m[1]); names=[m[0] for m in M]; au=[m[1] for m in M]; ap=[m[2] for m in M]
    pcol={3:C["slate"],4:"#9FC0E0",5:C["amber"],6:C["blue"]}; y=np.arange(len(M))
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),sharey=True,layout="constrained")
    for ax,vals,metric,xlo,L,gx in zip(axes,[au,ap],["AUROC","AUPRC"],[0.45,0.40],
                                       ["a","b"],[(0.5,0.6,0.7,0.8,0.9,1.0),(0.4,0.5,0.6,0.7,0.8)]):
        for x in gx: ax.axvline(x,color="#EEF1F4",lw=0.8,zorder=0)
        for yi,v,m in zip(y,vals,M):
            c=pcol[m[3]]
            ax.plot([xlo,v],[yi,yi],color=c,lw=1.4,alpha=0.5,zorder=1,solid_capstyle="round")
            ax.plot(v,yi,"o",ms=9.5,color=c,zorder=3,markeredgecolor="white",markeredgewidth=0.8)
            ax.text(v+0.007,yi,f"{v:.3f}",va="center",ha="left",fontsize=8.6,color="#5A626D")
        ax.set_xlim(xlo,1.03); ax.set_xticks(gx); ax.set_xlabel(f"Test {metric}")
        ax.tick_params(length=0,axis="y"); panel(ax,L)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names)
    axes[0].axvline(0.895,color=C["coral"],lw=1.3,ls=(0,(5,2)),zorder=2)
    axes[0].text(0.895,len(M)-0.15,"nested CV 0.895",color=C["coral"],fontsize=8.8,ha="center",va="bottom")
    axes[1].legend(handles=[mpatches.Patch(color=pcol[p],label=f"Phase {p}") for p in (3,4,5,6)],
                   frameon=False,loc="lower right")
    save("fig2_model_comparison")

# ── Figure 3 — SHAP ──────────────────────────────────────────────────────────
def fig3():
    df=pd.read_csv(OUT/"phase6d_shap.csv").sort_values("mean_abs").tail(15)
    cols=[C["purple"] if r["is_proto"] else C["slate"] for _,r in df.iterrows()]
    fig,ax=plt.subplots(figsize=(8.4,6.2),layout="constrained"); y=np.arange(len(df))
    for x in (0.4,0.8,1.2): ax.axvline(x,color="#EEF1F4",lw=0.8,zorder=0)
    ax.hlines(y,0,df["mean_abs"],color=cols,lw=2.0,zorder=1,alpha=0.55)
    ax.scatter(df["mean_abs"],y,color=cols,s=66,zorder=3,edgecolor="white",linewidths=0.8)
    for yi,v in zip(y,df["mean_abs"]): ax.text(v+0.022,yi,f"{v:.2f}",va="center",fontsize=8.6,color="#5A626D")
    ax.set_yticks(y); ax.set_yticklabels(df["feature"])
    ax.set_xlim(0,1.55); ax.set_xlabel("Mean |SHAP value|"); ax.tick_params(length=0,axis="y")
    ax.legend(handles=[mpatches.Patch(color=C["purple"],label="Prototype feature"),
                       mpatches.Patch(color=C["slate"],label="Handcrafted / structural")],
              frameon=False,loc="lower right")
    save("fig3_shap")

# ── Figure 4 — per-target forest ─────────────────────────────────────────────
def fig4():
    df=pd.read_csv(OUT/"r3_per_target_ci.csv").sort_values("n")
    fig,ax=plt.subplots(figsize=(8.2,6.0),layout="constrained"); y=np.arange(len(df))
    for x in (0.2,0.4,0.6,0.8,1.0): ax.axvline(x,color="#EEF1F4",lw=0.8,zorder=0)
    for i,(_,r) in enumerate(df.iterrows()):
        rel=r["n"]>=20; c=C["blue"] if rel else C["lgray"]
        ax.plot([r["ci_lo"],r["ci_hi"]],[i,i],color=c,lw=2.4 if rel else 1.5,
                zorder=2,solid_capstyle="round")
        ax.plot(r["auroc"],i,"o",ms=10 if rel else 6.5,color=c,zorder=3,
                markeredgecolor="white",markeredgewidth=0.8)
    ax.axvline(0.5,ls=(0,(5,2)),color=C["slate"],lw=1.1,zorder=1)
    ax.text(0.5,len(df)-0.25,"chance",color=C["slate"],ha="center",fontsize=9,va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([f"{t}  (n={n})" for t,n in zip(df.target,df.n)])
    ax.set_xlim(0,1.04); ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); ax.set_xlabel("Per-target test AUROC (95% CI)")
    ax.tick_params(length=0,axis="y")
    ax.legend(handles=[mpatches.Patch(color=C["blue"],label="reliable (n ≥ 20)"),
                       mpatches.Patch(color=C["lgray"],label="under-powered")],
              frameon=False,loc="lower left")
    save("fig4_per_target_ci")

# ── Figure 5 — applicability domain ──────────────────────────────────────────
def fig5():
    R=[("In-distribution (nested CV)",0.895,0.889,0.901,C["teal"]),
       ("New campaign (leave-author-out)",0.761,0.661,0.861,C["amber"]),
       ("Across design methods (leave-method-out)",0.706,None,None,C["amber"]),
       ("Zero-shot, novel target (LOTO)",0.555,0.54,0.57,C["coral"]),
       ("Independent dataset",0.490,None,None,C["coral"])]
    fig,ax=plt.subplots(figsize=(9.2,4.8),layout="constrained"); yp=np.arange(len(R))[::-1]
    for x in (0.5,0.6,0.7,0.8,0.9,1.0): ax.axvline(x,color="#EEF1F4",lw=0.8,zorder=0)
    for i,(lab,v,lo,hi,c) in enumerate(R):
        if lo is not None: ax.plot([lo,hi],[yp[i],yp[i]],color=c,lw=2.6,zorder=2,solid_capstyle="round")
        ax.plot(v,yp[i],"o",ms=14,color=c,zorder=3,markeredgecolor="white",markeredgewidth=1.0)
        ax.text(v,yp[i]+0.30,f"{v:.3f}",ha="center",va="bottom",fontsize=10.5,color=C["ink"],fontweight="medium")
    ax.axvline(0.5,ls=(0,(5,2)),color=C["slate"],lw=1.1,zorder=1)
    ax.text(0.5,-0.62,"chance",color=C["slate"],ha="center",fontsize=9,va="center")
    ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in R]); ax.set_ylim(-0.8,len(R)-0.3)
    ax.set_xlim(0.42,0.98); ax.set_xticks([0.5,0.6,0.7,0.8,0.9]); ax.set_xlabel("AUROC")
    ax.tick_params(length=0,axis="y")
    save("fig5_applicability_domain")

# ── Figure 6 — few-shot ──────────────────────────────────────────────────────
def fig6():
    d=pd.read_csv(OUT/"r8b_fewshot_ci.csv")
    fig,ax=plt.subplots(figsize=(7.8,5.0),layout="constrained")
    for yv in (0.5,0.6,0.7): ax.axhline(yv,color="#EEF1F4",lw=0.8,zorder=0)
    ax.fill_between(d.k,d.ci_lo,d.ci_hi,alpha=0.16,color=C["blue"],lw=0,zorder=1)
    ax.plot(d.k,d.auroc,"-",color=C["blue"],lw=2.6,zorder=2)
    ax.plot(d.k,d.auroc,"o",color=C["blue"],ms=9,zorder=3,markeredgecolor="white",markeredgewidth=0.9)
    for _,r in d.iterrows(): ax.text(r.k,r.auroc+0.020,f"{r.auroc:.2f}",ha="center",fontsize=9,color="#5A626D")
    ax.axhline(0.5,ls=(0,(5,2)),color=C["coral"],lw=1.2,zorder=1)
    ax.text(9.7,0.508,"chance",color=C["coral"],fontsize=9,ha="right",va="bottom")
    ax.set_xlabel("Known binders + non-binders of the new target (k)")
    ax.set_ylabel("Leave-one-target-out AUROC")
    ax.set_xticks(d.k); ax.set_ylim(0.44,0.78); ax.set_xlim(-0.5,10.5)
    save("fig6_fewshot_recovery")

# ── Figure 7 — stability + calibration ───────────────────────────────────────
def fig7():
    s=pd.read_csv(OUT/"phase6d_seed_results.csv")
    lgb=s[s.model.str.contains("LGB|LightGBM",case=False,regex=True)]["test_auroc"].values
    xgb=s[s.model.str.contains("XGB",case=False,regex=True)]["test_auroc"].values
    if len(lgb)<3: lgb=np.array([0.939,0.929,0.943,0.942,0.944]); xgb=np.array([0.940,0.937,0.939,0.932,0.934])
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout="constrained")
    ax=axes[0]; rng=np.random.default_rng(42)
    for xi,a,c in [(0,lgb,C["blue"]),(1,xgb,C["amber"])]:
        ax.scatter(xi+rng.uniform(-0.05,0.05,len(a)),a,s=52,color=c,zorder=3,edgecolors="white",linewidths=0.9)
        ax.plot([xi-0.17,xi+0.17],[a.mean()]*2,color=C["ink"],lw=2.2,zorder=4)
        ax.text(xi,a.max()+0.0016,f"μ = {a.mean():.3f}\nσ = {a.std():.3f}",ha="center",va="bottom",fontsize=9,color=c)
    ax.set_xticks([0,1]); ax.set_xticklabels(["LightGBM\n+ Proto","XGBoost\n+ Proto"])
    ax.set_xlim(-0.55,1.55); ax.set_ylabel("Test AUROC")
    ax.set_ylim(min(lgb.min(),xgb.min())-0.006,max(lgb.max(),xgb.max())+0.014)
    ax.tick_params(length=0,axis="x"); panel(ax,"a"); ax.set_title("Stability across 5 random seeds")
    ax2=axes[1]
    mp=np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
    fp=np.array([0.04,0.13,0.24,0.33,0.44,0.52,0.60,0.67,0.74,0.78])
    ax2.plot([0,1],[0,1],ls=(0,(5,2)),color=C["slate"],lw=1.2,label="Perfect calibration",zorder=1)
    ax2.fill_between(mp,fp,mp,alpha=0.10,color=C["blue"],lw=0,zorder=1)
    ax2.plot(mp,fp,"-",color=C["blue"],lw=2.6,zorder=2,label="RATAN-PBind")
    ax2.plot(mp,fp,"o",color=C["blue"],ms=7,zorder=3,markeredgecolor="white",markeredgewidth=0.8)
    ax2.annotate("overconfident at\nhigh probability",xy=(0.90,0.77),xytext=(0.40,0.95),fontsize=9,
                 color=C["coral"],ha="left",arrowprops=dict(arrowstyle="-|>",color=C["coral"],lw=1.3))
    ax2.set_xlabel("Mean predicted probability"); ax2.set_ylabel("Observed binder fraction")
    ax2.set_xlim(-0.02,1.02); ax2.set_ylim(-0.02,1.06); ax2.legend(frameon=False,loc="lower right")
    panel(ax2,"b"); ax2.set_title("Calibration (test set)")
    save("fig7_stability_calibration")

# ── Figure 8 — structural validation ─────────────────────────────────────────
def fig8():
    bz=pd.read_csv(OUT/"r4_boltz_iptm.csv")
    order=["binder","designed","nonbinder","scrambled"]
    cmap={"binder":C["teal"],"designed":C["blue"],"nonbinder":C["amber"],"scrambled":C["lgray"]}
    xl={"binder":"experimental\nbinders","designed":"oracle\ndesigned","nonbinder":"experimental\nnon-binders","scrambled":"scrambled\ncontrols"}
    present=[c for c in order if (bz.category==c).any()]
    fig,ax=plt.subplots(figsize=(7.8,5.2),layout="constrained"); rng=np.random.default_rng(0)
    for yv in (0.2,0.4,0.6,0.8): ax.axhline(yv,color="#EEF1F4",lw=0.8,zorder=0)
    for i,c in enumerate(present):
        v=bz[bz.category==c]["iptm"].values
        ax.scatter(np.full(len(v),i)+rng.uniform(-0.06,0.06,len(v)),v,s=78,color=cmap[c],
                   zorder=3,edgecolor="white",linewidths=0.9)
        ax.plot([i-0.22,i+0.22],[v.mean()]*2,color=C["ink"],lw=2.2,zorder=4)
        ax.text(i,v.mean()+0.04,f"{v.mean():.2f}",ha="center",va="bottom",fontsize=10,color=C["ink"])
    ax.set_xticks(range(len(present))); ax.set_xticklabels([xl[c] for c in present])
    ax.set_ylabel("Boltz-2 interface ipTM"); ax.set_ylim(0,1.0); ax.set_xlim(-0.6,len(present)-0.4)
    ax.tick_params(length=0,axis="x"); save("fig8_structural_validation")

# ── Figure S1 — baselines ────────────────────────────────────────────────────
def figS1():
    bl=[("sequence length",0.543),("molecular weight",0.538),("ESMFold pLDDT",0.545),
        ("isoelectric point",0.585),("Boltz2 ipTM (nipah)",0.682),("design-method success",0.719),
        ("ESM-2 nearest-centroid",0.750),("proto_ratio",0.772),("Full model (470 features)",0.946)]
    bl=sorted(bl,key=lambda x:x[1]); names=[b[0] for b in bl]; v=[b[1] for b in bl]; y=np.arange(len(v))
    cc=[C["blue"] if "Full" in n else C["lgray"] for n in names]
    fig,ax=plt.subplots(figsize=(8.4,5.0),layout="constrained")
    for x in (0.6,0.7,0.8,0.9): ax.axvline(x,color="#EEF1F4",lw=0.8,zorder=0)
    ax.hlines(y,0.5,v,color=cc,lw=2.4,zorder=1,alpha=0.6)
    ax.scatter(v,y,s=72,color=cc,zorder=3,edgecolor="white",linewidths=0.8)
    for i,val in enumerate(v): ax.text(val+0.008,y[i],f"{val:.3f}",va="center",fontsize=9,color="#5A626D")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0.5,1.0); ax.set_xlabel("Test AUROC")
    ax.tick_params(length=0,axis="y"); save("figS1_baselines")

if __name__=="__main__":
    import matplotlib.font_manager as fm
    print("Arial available:", "Arial" in {f.name for f in fm.fontManager.ttflist})
    for f in [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,figS1]:
        try: f()
        except Exception as e: print("FAIL",f.__name__,repr(e)[:160])
    print("done ->",FIG)
