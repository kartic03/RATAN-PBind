import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib import font_manager
# Arial (size 12), Liberation Sans metric-identical fallback
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
    "font.size":12,"axes.titlesize":13,"axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,
    "legend.fontsize":12,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight"})
avail=set(f.name for f in font_manager.fontManager.ttflist)
print("Arial available:", "Arial" in avail, "| using:", "Arial" if "Arial" in avail else "Liberation Sans")
OUT="paper/figures_v2"
C={"good":"#2c7fb8","mid":"#41ab5d","bad":"#e6550d","grey":"#737373"}

# Fig 1 — applicability-domain gradient (horizontal dot + error bars)
ax_labels=["In-distribution\n(held-out test)","Across design\nmethods","New campaign\n(leave-author-out)","Zero-shot to\nnovel targets (LOTO)","Independent\ndataset"]
vals=[0.946,0.817,0.771,0.555,0.49]; lo=[0.919,None,0.771-0.114,0.54,None]; hi=[0.968,None,0.771+0.114,0.57,None]
cols=[C["good"],C["mid"],C["mid"],C["bad"],C["bad"]]
fig,ax=plt.subplots(figsize=(9,5.2)); ypos=np.arange(len(vals))[::-1]
for i,(v,l,h) in enumerate(zip(vals,lo,hi)):
    xe=[[v-l],[h-v]] if l is not None else None
    ax.errorbar(v,ypos[i],xerr=xe,fmt="o",ms=13,color=cols[i],capsize=5,lw=2,zorder=10)
    ax.text(v,ypos[i]+0.22,f"{v:.3f}",ha="center",fontsize=12,zorder=11,
            bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.9))
ax.axvline(0.5,ls="--",color=C["grey"],lw=1.2,zorder=1); ax.text(0.5,len(vals)-0.4,"chance",color=C["grey"],fontsize=11,ha="center")
ax.set_yticks(ypos); ax.set_yticklabels(ax_labels); ax.set_xlim(0.40,1.0); ax.set_xlabel("AUROC")
ax.set_title("Applicability domain: a clear generalisation gradient"); ax.grid(axis="x",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_applicability_domain.png"); plt.close()

# Fig 2 — few-shot recovery curve with CI band
d=pd.read_csv("outputs/r8b_fewshot_ci.csv")
fig,ax=plt.subplots(figsize=(8,5))
ax.fill_between(d.k,d.ci_lo,d.ci_hi,alpha=0.2,color=C["good"])
ax.plot(d.k,d.auroc,"o-",color=C["good"],ms=10,lw=2.2,zorder=10)
for _,r in d.iterrows(): ax.text(r.k,r.auroc+0.02,f"{r.auroc:.2f}",ha="center",fontsize=12,zorder=11,bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=0.9))
ax.axhline(0.5,ls="--",color=C["grey"],lw=1.2); ax.text(8,0.51,"chance",color=C["grey"],fontsize=11)
ax.set_xlabel("known binders+non-binders of new target (k)"); ax.set_ylabel("LOTO AUROC")
ax.set_title("Few-shot recovery: 2 labels move a new target from chance to useful")
ax.set_xticks(d.k); ax.set_ylim(0.42,0.80); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_fewshot_recovery.png"); plt.close()

# Fig 3 — single-feature baselines vs full model (horizontal lollipop)
bl=[("seq length",0.543),("molecular weight",0.538),("ESMFold pLDDT",0.545),("Boltz2 ipTM (nipah)",0.682),
    ("method success rate",0.719),("ESM-2 nearest-centroid",0.750),("proto_ratio",0.772),("Full RATAN-PBind",0.946)]
bl=sorted(bl,key=lambda x:x[1]); names=[b[0] for b in bl]; v=[b[1] for b in bl]
fig,ax=plt.subplots(figsize=(8.5,5.2)); y=np.arange(len(v))
cc=[C["good"] if n=="Full RATAN-PBind" else C["grey"] for n in names]
ax.hlines(y,0.5,v,color=cc,lw=2.5,zorder=2); ax.plot(v,y,"o",ms=11,zorder=10)
for i,val in enumerate(v): ax.plot(v[i],y[i],"o",ms=11,color=cc[i],zorder=10); ax.text(val+0.008,y[i],f"{val:.3f}",va="center",fontsize=11,zorder=11)
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0.5,1.0); ax.set_xlabel("test AUROC")
ax.set_title("The model beats every single feature and naive retrieval"); ax.grid(axis="x",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_baselines.png"); plt.close()

# Fig 4 — per-target reliability (forest plot, reliable targets only)
pt=pd.read_csv("outputs/r3_per_target_ci.csv"); pt=pt[pt.n>=8].sort_values("n")
fig,ax=plt.subplots(figsize=(8.5,5.5)); y=np.arange(len(pt))
col=[C["good"] if r>=20 else C["bad"] for r in pt.n]
ax.errorbar(pt.auroc,y,xerr=[pt.auroc-pt.ci_lo,pt.ci_hi-pt.auroc],fmt="o",ms=9,capsize=4,lw=1.8,
            ecolor=C["grey"],mfc="none",zorder=10)
for i,(_,r) in enumerate(pt.iterrows()): ax.plot(r.auroc,i,"o",ms=9,color=col[i],zorder=11)
ax.axvline(0.5,ls="--",color=C["grey"],lw=1.2)
ax.set_yticks(y); ax.set_yticklabels([f"{t} (n={n})" for t,n in zip(pt.target,pt.n)]); ax.set_xlim(0,1.05)
ax.set_xlabel("per-target test AUROC (95% CI)")
ax.set_title("Only well-powered targets give reliable estimates (blue n>=20)"); ax.grid(axis="x",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_per_target_ci.png"); plt.close()

# Fig 5 — Boltz-2 design validation by category (dot/strip)
bz=pd.read_csv("outputs/r4_boltz_iptm.csv")
order=["binder","designed","nonbinder","scrambled"]; colmap={"binder":C["good"],"designed":C["mid"],"nonbinder":C["bad"],"scrambled":C["grey"]}
fig,ax=plt.subplots(figsize=(8,5))
for i,cat in enumerate(order):
    s=bz[bz.category==cat]["iptm"]; 
    if len(s): ax.scatter(np.full(len(s),i)+np.random.RandomState(0).uniform(-0.08,0.08,len(s)),s,s=80,color=colmap[cat],zorder=10,edgecolor="white"); ax.plot([i-0.2,i+0.2],[s.mean()]*2,color="black",lw=2,zorder=11)
ax.set_xticks(range(len(order))); ax.set_xticklabels(["experimental\nbinders","oracle\ndesigned","experimental\nnon-binders","scrambled\ncontrols"])
ax.set_ylabel("Boltz-2 interface ipTM"); ax.set_title("Independent structural check: ipTM separates binders from controls"); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_boltz_validation.png"); plt.close()
print("saved 5 figures to",OUT); import os; print(os.listdir(OUT))
