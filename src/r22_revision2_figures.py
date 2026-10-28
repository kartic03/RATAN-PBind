"""Figures rebuilt for the second revision, numbered as in the manuscript.

What changed against src/r7_figures_final.py, and why:

  Figure 1a  binding rates were per evaluation record (nipah 5,980 records),
             which disagreed with Table 1, Table S1 and Table S14. Now computed
             over the 2,630 modelled protein-target pairs, all 24 targets.
  Figure 2   the "LGB + interface" point carried 0.894/0.702, the Phase-6a
             XGBoost+IfaceHC run. Table 2 reports the LightGBM interface model
             with the binder-chain parse, 0.896/0.701 (r21 recompute).
  Figure 4   (file fig5_applicability_domain previously) the independent-dataset
             point was 0.490; the rebuilt per-target mean is 0.503 (Table S11).
             The 0.706 point was labelled leave-method-out, which is 0.702;
             both grouped estimates are now shown and labelled separately.
  Figure 5   x-axis said "Known binders + non-binders ... (k)", which reads as
             k in total; k is per class.
  Figure 7a  drew outputs/phase6d_seed_results.csv, the superseded Phase-6d
             configuration (0.940 +/- 0.005), against Table S2's released
             configuration (0.946 +/- <0.001). Now from r22_seed_values.json.
  Figure 7b  drew an illustrative reliability curve. Now the measured curve
             from outputs/r22_calibration.csv (ECE 0.058).
  Figure 8   adds the group sizes, which are small (n = 1 to 4).
  Graphical abstract  carried 0.82 and 0.77, which appear in no table; an RNA
             codon strip for a sequence model; 4.8x instead of 4.65x; "lables";
             "2.630"; and only four of the five axes.

Figures 3, 6 and S1 are unchanged and are copied through under the manuscript
numbering.

    pixi run -e ml python src/r22_revision2_figures.py
"""
import json, shutil
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
OLD = ROOT / "paper" / "Journal of Cheminformatics" / "figures"
FIG = ROOT / "paper" / "Journal of Cheminformatics" / "figures_rev2"
FIG.mkdir(parents=True, exist_ok=True)

C = dict(blue="#2C6FB3", teal="#2E9E8E", amber="#E0A33E", coral="#CB5A4C",
         slate="#7A8290", lgray="#C7CCD2", ink="#222831", purple="#7E6BB5")

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


def save(name):
    plt.savefig(FIG / ("%s.png" % name), bbox_inches="tight", pad_inches=0.10)
    plt.savefig(FIG / ("%s.pdf" % name), bbox_inches="tight", pad_inches=0.10)
    plt.close()
    print("saved", name)


# ── Figure 1 — dataset (a) + pipeline (b) ────────────────────────────────────
def fig1():
    fm = pd.read_parquet(ROOT / "features/feature_matrix.parquet")
    g = fm.groupby("target").binding_label.agg(["size", "sum"])
    g["rate"] = 100 * g["sum"] / g["size"]
    overall = 100 * fm.binding_label.mean()
    assert abs(overall - 17.8) < 0.05, overall
    g = g.sort_values("rate")
    names, rates, ns = list(g.index), list(g.rate), list(g["size"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 6.6),
                                  gridspec_kw={"width_ratios": [1, 1.4]}, layout="constrained")
    y = np.arange(len(names))
    cols = [C["blue"] if r >= 30 else "#9FC0E0" for r in rates]
    for x in (20, 40, 60, 80):
        ax.axvline(x, color="#EEF1F4", lw=0.8, zorder=0)
    ax.barh(y, rates, color=cols, height=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s  (n=%d)" % (t, n) for t, n in zip(names, ns)], fontsize=8)
    ax.set_xlim(0, 86)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Binding rate among modelled pairs (%)")
    ax.axvline(overall, color=C["coral"], lw=1.4, ls=(0, (5, 2)), zorder=3)
    ax.text(overall + 1.0, len(names) - 0.2, "all pairs %.1f%%" % overall,
            color=C["coral"], fontsize=8.5, va="center", ha="left")
    for yi, r in zip(y, rates):
        ax.text(r + 1.2, yi, "%.0f" % r, va="center", ha="left", fontsize=7.5, color="#5A626D")
    ax.tick_params(length=0, axis="y")
    panel(ax, "a")

    ax2.set_xlim(0, 14); ax2.set_ylim(0, 10); ax2.axis("off"); panel(ax2, "b", dx=-0.04)

    def box(cx, cy, w, h, t, fc, fg="white", fs=10.5):
        ax2.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                      boxstyle="round,pad=0.02,rounding_size=0.22",
                      facecolor=fc, edgecolor="none", zorder=2))
        ax2.text(cx, cy, t, ha="center", va="center", fontsize=fs, color=fg,
                 zorder=4, linespacing=1.35)
        return dict(l=cx - w / 2, r=cx + w / 2, cy=cy)

    def arr(p1, p2):
        ax2.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                      color=C["slate"], lw=1.6, zorder=3, shrinkA=0, shrinkB=0))

    seq = box(1.7, 5.0, 2.6, 1.5, "Protein\nsequence", "#3A4250")
    hc = box(5.6, 7.6, 3.0, 1.3, "Handcrafted\nfeatures", C["slate"], fs=10)
    es = box(5.6, 5.0, 3.0, 1.3, "ESM-2 650M\nembedding", C["blue"])
    pr = box(5.6, 2.4, 3.0, 1.3, "Prototype\nsimilarity", C["purple"])
    mdl = box(10.1, 5.0, 2.8, 1.7, "LightGBM\n470 features", C["teal"])
    out = box(13.0, 5.0, 1.6, 1.7, "Binding\nprediction", C["amber"], fs=9.5)
    for t in (hc, es, pr):
        arr((seq["r"], seq["cy"]), (t["l"], t["cy"]))
    for s in (hc, es, pr):
        arr((s["r"], s["cy"]), (mdl["l"], mdl["cy"]))
    arr((mdl["r"], mdl["cy"]), (out["l"], out["cy"]))
    ax2.text(13.0, 3.7, "+ interpretation\n+ design", ha="center", va="top", fontsize=8.5,
             color="#7A8290", style="italic", linespacing=1.3)
    save("fig1_dataset_pipeline")


# ── Figure 2 — model performance ─────────────────────────────────────────────
def fig2():
    iface = json.load(open(OUT / "r21_interface_row_recompute.json"))["FIXED parse (binder chain only)"]["seed42"]
    assert round(iface[0], 3) == 0.896 and round(iface[1], 3) == 0.701, iface
    M = [("Logistic regression", 0.790, 0.499, 3), ("Random forest", 0.854, 0.580, 3),
         ("XGBoost", 0.880, 0.682, 3), ("LightGBM", 0.893, 0.713, 3),
         ("XGB + ESM-2", 0.871, 0.633, 4), ("LGB + ESM-2", 0.864, 0.675, 4),
         ("ESM-2 fine-tuned", 0.854, 0.592, 5), ("Stacking ensemble", 0.883, 0.692, 5),
         ("LGB + interface", round(iface[0], 3), round(iface[1], 3), 6),
         ("XGB + Proto", 0.940, 0.748, 6), ("LGB + Proto", 0.946, 0.770, 6)]
    M = sorted(M, key=lambda m: m[1])
    names = [m[0] for m in M]; au = [m[1] for m in M]; ap = [m[2] for m in M]
    pcol = {3: C["slate"], 4: "#9FC0E0", 5: C["amber"], 6: C["blue"]}
    y = np.arange(len(M))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, layout="constrained")
    for ax, vals, metric, xlo, L, gx in zip(axes, [au, ap], ["AUROC", "AUPRC"], [0.45, 0.40],
                                            ["a", "b"], [(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                                                         (0.4, 0.5, 0.6, 0.7, 0.8)]):
        for x in gx:
            ax.axvline(x, color="#EEF1F4", lw=0.8, zorder=0)
        for yi, v, m in zip(y, vals, M):
            c = pcol[m[3]]
            ax.plot([xlo, v], [yi, yi], color=c, lw=1.4, alpha=0.5, zorder=1, solid_capstyle="round")
            ax.plot(v, yi, "o", ms=9.5, color=c, zorder=3, markeredgecolor="white", markeredgewidth=0.8)
            ax.text(v + 0.007, yi, "%.3f" % v, va="center", ha="left", fontsize=8.6, color="#5A626D")
        ax.set_xlim(xlo, 1.03); ax.set_xticks(gx); ax.set_xlabel("Test %s" % metric)
        ax.tick_params(length=0, axis="y"); panel(ax, L)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names)
    axes[0].axvline(0.895, color=C["coral"], lw=1.3, ls=(0, (5, 2)), zorder=2)
    axes[0].text(0.895, len(M) - 0.15, "nested CV 0.895", color=C["coral"],
                 fontsize=8.8, ha="center", va="bottom")
    axes[1].legend(handles=[mpatches.Patch(color=pcol[p], label="Phase %d" % p) for p in (3, 4, 5, 6)],
                   frameon=False, loc="lower right")
    save("fig2_model_comparison")


# ── Figure 4 — applicability domain ──────────────────────────────────────────
def fig4():
    g = pd.read_csv(OUT / "r13_grouped_prospective.csv")
    g = g[g.featset == "470 published"]

    def ms(split):
        v = g[g.split == split].auroc.values
        return float(v.mean()), float(v.std()), len(v)

    lao = ms("leave-author-out"); ldm = ms("leave-design_method-out"); lom = ms("leave-one-method-out")
    s11 = [0.543, 0.489, 0.515, 0.450, 0.519]          # Table S11, sequence-only per target
    ext = (float(np.mean(s11)), float(np.std(s11)), 5)
    assert round(lao[0], 3) == 0.761 and round(ldm[0], 3) == 0.702 and round(lom[0], 3) == 0.706
    assert round(ext[0], 3) == 0.503

    R = [("In-distribution (nested cross-validation)", 0.895, 0.006, "5x3 folds", C["teal"]),
         ("New campaign (leave-author-out)", lao[0], lao[1], "%d folds" % lao[2], C["amber"]),
         ("Across design methods (leave-design-method-out)", ldm[0], ldm[1], "%d folds" % ldm[2], C["amber"]),
         ("Across design methods (leave-one-method-out)", lom[0], lom[1], "%d methods" % lom[2], C["amber"]),
         ("Zero-shot, novel target (leave-one-target-out)", 0.571, None, "target-blind, 20 targets", C["coral"]),
         ("Independent dataset (de-duplicated)", ext[0], ext[1], "5 targets", C["coral"])]

    fig, ax = plt.subplots(figsize=(10.4, 5.4), layout="constrained")
    yp = np.arange(len(R))[::-1]
    for x in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        ax.axvline(x, color="#EEF1F4", lw=0.8, zorder=0)
    for i, (lab, v, sd, note, c) in enumerate(R):
        if sd is not None:
            ax.plot([v - sd, v + sd], [yp[i], yp[i]], color=c, lw=2.6, zorder=2, solid_capstyle="round")
        ax.plot(v, yp[i], "o", ms=13, color=c, zorder=3, markeredgecolor="white", markeredgewidth=1.0)
        ax.text(v, yp[i] + 0.26, "%.3f" % v, ha="center", va="bottom", fontsize=10.5, color=C["ink"])
        ax.text(1.005, yp[i], note, ha="left", va="center", fontsize=8, color=C["slate"])
    ax.axvline(0.5, ls=(0, (5, 2)), color=C["slate"], lw=1.1, zorder=1)
    ax.text(0.5, -0.72, "chance", color=C["slate"], ha="center", fontsize=9, va="center")
    ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in R], fontsize=9.5)
    ax.set_ylim(-0.9, len(R) - 0.3); ax.set_xlim(0.42, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xlabel("AUROC")
    ax.tick_params(length=0, axis="y")
    save("fig4_applicability_domain")


# ── Figure 5 — few-shot ──────────────────────────────────────────────────────
def fig5():
    d = pd.read_csv(OUT / "r11_fewshot_final.csv")
    fig, ax = plt.subplots(figsize=(7.8, 5.0), layout="constrained")
    for yv in (0.5, 0.6, 0.7):
        ax.axhline(yv, color="#EEF1F4", lw=0.8, zorder=0)
    ax.fill_between(d.k, d.auroc_ci_lo, d.auroc_ci_hi, alpha=0.16, color=C["blue"], lw=0, zorder=1)
    ax.plot(d.k, d.auroc, "-", color=C["blue"], lw=2.6, zorder=2)
    ax.plot(d.k, d.auroc, "o", color=C["blue"], ms=9, zorder=3,
            markeredgecolor="white", markeredgewidth=0.9)
    for _, r in d.iterrows():
        ax.text(r.k, r.auroc + 0.020, "%.2f" % r.auroc, ha="center", fontsize=9, color="#5A626D")
    ax.axhline(0.5, ls=(0, (5, 2)), color=C["coral"], lw=1.2, zorder=1)
    ax.text(9.7, 0.508, "chance", color=C["coral"], fontsize=9, ha="right", va="bottom")
    ax.set_xlabel("Labelled examples added per class, k\n(k binders and k non-binders; 2k support examples)")
    ax.set_ylabel("Leave-one-target-out AUROC")
    ax.set_xticks(d.k); ax.set_ylim(0.44, 0.78); ax.set_xlim(-0.5, 10.5)
    save("fig5_fewshot_recovery")


# ── Figure 7 — stability + calibration ───────────────────────────────────────
def fig7():
    s = json.load(open(OUT / "r22_seed_values.json"))
    a = np.array(s["auroc"])
    cal = pd.read_csv(OUT / "r22_calibration.csv").dropna(subset=["mean_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    ax = axes[0]
    rng = np.random.default_rng(42)
    ax.scatter(rng.uniform(-0.09, 0.09, len(a)), a, s=62, color=C["blue"], zorder=3,
               edgecolors="white", linewidths=0.9)
    ax.plot([-0.22, 0.22], [a.mean()] * 2, color=C["ink"], lw=2.2, zorder=4)
    ax.text(0.0, a.mean() + 0.0035, "mean %.3f, SD %.4f\n(seeds %s)"
            % (a.mean(), a.std(), ", ".join(str(x) for x in s["seeds"])),
            ha="center", va="bottom", fontsize=9, color=C["blue"])
    ax.set_xticks([0]); ax.set_xticklabels(["LightGBM + Proto\n(released 470-feature model)"])
    ax.set_xlim(-0.5, 0.5); ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.930, 0.960)
    ax.tick_params(length=0, axis="x"); panel(ax, "a")
    ax.set_title("Stability across 5 random seeds")

    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], ls=(0, (5, 2)), color=C["slate"], lw=1.2,
             label="Perfect calibration", zorder=1)
    ax2.plot(cal.mean_pred, cal.frac_pos, "-", color=C["blue"], lw=2.2, zorder=2,
             label="RATAN-PBind (ECE 0.058)")
    ax2.scatter(cal.mean_pred, cal.frac_pos, s=18 + 2.2 * np.sqrt(cal.n) * 6, color=C["blue"],
                zorder=3, edgecolor="white", linewidths=0.9, alpha=0.9)
    for _, r in cal.iterrows():
        ax2.text(r.mean_pred, r.frac_pos - 0.055, "n=%d" % r.n, ha="center", va="top",
                 fontsize=7.4, color="#5A626D")
    ax2.annotate("overconfident at\nhigh probability", xy=(0.975, 0.796), xytext=(0.46, 0.97),
                 fontsize=9, color=C["coral"], ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=C["coral"], lw=1.3))
    ax2.set_xlabel("Mean predicted probability in bin")
    ax2.set_ylabel("Observed binder fraction")
    ax2.set_xlim(-0.04, 1.04); ax2.set_ylim(-0.12, 1.08)
    ax2.legend(frameon=False, loc="upper left")
    panel(ax2, "b"); ax2.set_title("Calibration (test set, 10 equal-width bins)")
    save("fig7_stability_calibration")


# ── Figure 8 — structural validation ─────────────────────────────────────────
def fig8():
    bz = pd.read_csv(OUT / "r4_boltz_iptm.csv")
    order = ["binder", "designed", "nonbinder", "scrambled"]
    cmap = {"binder": C["teal"], "designed": C["blue"], "nonbinder": C["amber"], "scrambled": C["lgray"]}
    xl = {"binder": "experimental\nbinders", "designed": "oracle-designed\ncandidate",
          "nonbinder": "experimental\nnon-binders", "scrambled": "scrambled\ncontrols"}
    present = [c for c in order if (bz.category == c).any()]
    fig, ax = plt.subplots(figsize=(7.8, 5.2), layout="constrained")
    rng = np.random.default_rng(0)
    for yv in (0.2, 0.4, 0.6, 0.8):
        ax.axhline(yv, color="#EEF1F4", lw=0.8, zorder=0)
    for i, c in enumerate(present):
        v = bz[bz.category == c]["iptm"].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.07, 0.07, len(v)), v, s=78,
                   color=cmap[c], zorder=3, edgecolor="white", linewidths=0.9)
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=C["ink"], lw=2.2, zorder=4)
        ax.text(i, 1.005, "mean %.2f" % v.mean(), ha="center", va="bottom", fontsize=9.5, color=C["ink"])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(["%s\n(n = %d)" % (xl[c], (bz.category == c).sum()) for c in present])
    ax.set_ylabel("Boltz-2 interface ipTM")
    ax.set_ylim(0, 1.12); ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(length=0, axis="x")
    save("fig8_structural_validation")


# ── Graphical abstract ───────────────────────────────────────────────────────
def graphical_abstract():
    from PIL import Image
    BLUE, TEAL, AMBER, CORAL = "#2C6FB3", "#1F9E89", "#E0A33E", "#CB5A4C"
    SLATE, INK, LINE = "#5A626D", "#1E2530", "#D7DEE6"
    TINT_B, TINT_N, TINT_T = "#EEF4FB", "#F6F8FA", "#ECF7F3"

    fig = plt.figure(figsize=(9.2, 3.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def card(x, y, w, h, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                     facecolor=fill, edgecolor=LINE, linewidth=1.1, zorder=1))

    def connect(x1, x2, y=0.46):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=16,
                     color=SLATE, lw=2.4, zorder=6, shrinkA=0, shrinkB=0))

    card(0.015, 0.10, 0.205, 0.72, TINT_B)
    card(0.275, 0.10, 0.330, 0.72, TINT_N)
    card(0.660, 0.10, 0.325, 0.72, TINT_T)
    connect(0.222, 0.273); connect(0.607, 0.658)

    ax.text(0.015, 0.95, "RATAN-PBind", fontsize=13.5, fontweight="bold", color=INK, va="center")
    ax.text(0.985, 0.95, "sequence-based de novo binder pre-screening",
            fontsize=8.6, color=SLATE, va="center", ha="right", style="italic")

    # card 1 — amino-acid sequence, not codons
    ax.text(0.1175, 0.74, "De novo binder candidate", fontsize=9.2, fontweight="bold",
            ha="center", color=INK)
    seq = "MASWKELLVQRTI"
    cls = {**{a: AMBER for a in "AMLVIWFPG"}, **{a: TEAL for a in "STQNCY"},
           **{a: BLUE for a in "KRH"}, **{a: CORAL for a in "ED"}}
    n = len(seq); cw = 0.0135; x0 = 0.1175 - n * cw / 2
    for i, a in enumerate(seq):
        ax.add_patch(FancyBboxPatch((x0 + i * cw, 0.44), cw * 0.86, 0.12,
                     boxstyle="round,pad=0,rounding_size=0.015",
                     facecolor=cls.get(a, SLATE), edgecolor="white", linewidth=0.6, zorder=3))
        ax.text(x0 + i * cw + cw * 0.43, 0.50, a, fontsize=5.6, ha="center", va="center",
                color="white", zorder=4)
    ax.text(0.1175, 0.355, "amino-acid sequence", fontsize=7.4, ha="center",
            color=SLATE, style="italic")
    ax.text(0.1175, 0.20, "24 targets · 2,630 pairs", fontsize=8.4, ha="center",
            color=INK, fontweight="bold")

    # card 2
    ax.text(0.440, 0.74, "Target-conditioned prototype similarity", fontsize=9.2,
            fontweight="bold", ha="center", color=INK)
    axp = fig.add_axes([0.315, 0.295, 0.250, 0.36]); axp.set_facecolor("none")
    rng = np.random.default_rng(7)
    b = rng.normal([0.30, 0.50], [0.10, 0.135], (70, 2))
    nn = rng.normal([0.70, 0.52], [0.10, 0.135], (70, 2))
    axp.scatter(b[:, 0], b[:, 1], s=14, color=BLUE, alpha=0.45, edgecolor="none", zorder=2)
    axp.scatter(nn[:, 0], nn[:, 1], s=14, color="#AEB7C2", alpha=0.55, edgecolor="none", zorder=2)
    axp.annotate("", xy=(0.40, 0.50), xytext=(0.52, 0.45),
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8, shrinkA=2, shrinkB=4))
    axp.scatter([0.52], [0.45], s=150, marker="*", color=AMBER, edgecolor=INK,
                linewidths=0.7, zorder=5)
    axp.set_xlim(0, 1); axp.set_ylim(0.08, 0.95); axp.axis("off")
    ax.text(0.440, 0.235, "ESM-2 embedding  →  LightGBM · 470 features",
            fontsize=7.8, ha="center", color=INK)
    ly = 0.155
    for lx, col, lab, mk in [(0.345, BLUE, "binders", "o"), (0.425, "#AEB7C2", "non-binders", "o"),
                             (0.535, AMBER, "candidate", "*")]:
        ax.scatter(lx, ly, s=(46 if mk == "*" else 26), marker=mk, color=col,
                   edgecolor=(INK if mk == "*" else "none"), linewidths=0.4, zorder=4)
        ax.text(lx + 0.011, ly, lab, fontsize=6.8, va="center", color=SLATE)

    # card 3 — all five axes of Figure 4, in the same order
    ax.text(0.8225, 0.74, "Ranked binders + applicability domain", fontsize=9.2,
            fontweight="bold", ha="center", color=INK)
    axg = fig.add_axes([0.700, 0.260, 0.160, 0.345])
    vals = [0.895, 0.761, 0.702, 0.571, 0.503]
    cols = [TEAL, AMBER, AMBER, CORAL, CORAL]
    labs = ["in-\ndist.", "new\ncamp.", "x-\nmeth.", "new\ntarget", "new\ndata"]
    axg.plot(range(5), vals, color=LINE, lw=1.6, zorder=1)
    axg.scatter(range(5), vals, c=cols, s=34, zorder=3, edgecolor="white", linewidths=0.8)
    axg.axhline(0.5, ls=(0, (3, 2)), color="#C2C8D0", lw=1)
    axg.set_ylim(0.45, 0.98); axg.set_xlim(-0.45, 4.45)
    axg.set_xticks(range(5))
    axg.set_xticklabels(labs, fontsize=5.0, color=SLATE, linespacing=0.9)
    axg.set_yticks([0.5, 0.7, 0.9]); axg.tick_params(labelsize=5.8, length=2, colors=SLATE, pad=1.5)
    axg.set_ylabel("AUROC", fontsize=6.2, color=SLATE, labelpad=1)
    for sp in ["top", "right"]:
        axg.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        axg.spines[sp].set_color("#C2C8D0")
    ax.text(0.922, 0.615, "4.65×", fontsize=17, fontweight="bold", color=TEAL,
            ha="center", va="center")
    ax.text(0.922, 0.520, "enrichment", fontsize=8.0, color=INK, ha="center", va="center")
    ax.text(0.922, 0.460, "top 10%, in-distribution", fontsize=5.9, color=SLATE,
            ha="center", va="center")
    ax.text(0.922, 0.350, "few-shot", fontsize=8.4, fontweight="bold", color=BLUE,
            ha="center", va="center")
    ax.text(0.922, 0.290, "2 labels per class", fontsize=5.9, color=SLATE,
            ha="center", va="center")
    ax.text(0.922, 0.240, "on a new target", fontsize=5.9, color=SLATE,
            ha="center", va="center")

    ax.text(0.5, 0.035, "Cheap, interpretable sequence model with a measured applicability domain "
            "and few-shot extension to new targets", fontsize=7.4, color=SLATE,
            ha="center", va="center")

    p = FIG / "graphical_abstract.png"
    fig.savefig(p, dpi=100, facecolor="white", pad_inches=0)
    plt.close()
    im = Image.open(p).convert("RGB")
    if im.size != (920, 300):
        im = im.resize((920, 300), Image.LANCZOS)
    im.save(p, "PNG", optimize=True)
    import os
    print("saved graphical_abstract  %s px  %.1f KB  mode %s"
          % (im.size, os.path.getsize(p) / 1024, im.mode))


# ── carry the unchanged figures across under the manuscript numbering ────────
def carry():
    for src, dst in [("fig3_shap", "fig3_shap"),
                     ("fig4_per_target_ci", "fig6_per_target_ci"),
                     ("figS1_baselines", "figS1_baselines")]:
        for ext in ("png", "pdf"):
            s = OLD / ("%s.%s" % (src, ext))
            if s.exists():
                shutil.copy2(s, FIG / ("%s.%s" % (dst, ext)))
                print("carried", s.name, "->", "%s.%s" % (dst, ext))


if __name__ == "__main__":
    import matplotlib.font_manager as fm
    print("Arial available:", "Arial" in {f.name for f in fm.fontManager.ttflist})
    for f in [fig1, fig2, fig4, fig5, fig7, fig8, graphical_abstract, carry]:
        try:
            f()
        except Exception as e:
            print("FAIL", f.__name__, repr(e)[:300])
    print("done ->", FIG)
