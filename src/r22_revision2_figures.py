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
import io, json, os, re, shutil
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

import matplotlib.font_manager as fm

# Arial only. A substituted face changes every metric, and the graphical
# abstract has to hold its layout inside 920 x 300 px, so a missing Arial is a
# hard stop rather than a silent fallback.
_ARIAL = fm.findfont(fm.FontProperties(family="Arial"), fallback_to_default=False)
if "arial" not in os.path.basename(_ARIAL).lower():
    raise SystemExit("Arial not found; matplotlib resolved %r" % _ARIAL)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
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


# ── Graphical abstract ────────────────────────────────────────────
# Panel 1 draws a real backbone. The cartoon renderer below is a small stand-in
# for PyMOL: helical stretches come from the CA(i) to CA(i+4) distance and are
# drawn as a quad strip that twists with the helix, loops as a round-capped
# tube, everything depth sorted and Lambert shaded.
from matplotlib.collections import PolyCollection, LineCollection
from scipy.interpolate import splprep, splev

PDB = str(ROOT / "data/structures/designed_0_binder_chainA.pdb")


def ca_trace(pdb, chain="A"):
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                     for l in open(pdb)
                     if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == chain])


def helical(ca, cut=6.4):
    h = np.zeros(len(ca), bool)
    for i, ok in enumerate(np.linalg.norm(ca[4:] - ca[:-4], axis=1) < cut):
        if ok:
            h[i:i + 5] = True
    return h


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        j = i
        while j + 1 < n and mask[j + 1] == mask[i]:
            j += 1
        out.append((i, j, bool(mask[i])))
        i = j + 1
    return out


def clean(mask, gap=0, min_len=7):
    """Drop helices too short to carry a ribbon frame, and optionally close
    very short breaks. Breaks of one or two residues between helices in a
    bundle are real turns, so gap defaults to 0 and only the length filter
    runs."""
    m = mask.copy()
    for i, j, v in runs(m):
        if not v and 0 < i and j < len(m) - 1 and j - i + 1 <= gap:
            m[i:j + 1] = True
    for i, j, v in runs(m):
        if v and j - i + 1 < min_len:
            m[i:j + 1] = False
    return m


def orient(ca, tilt, roll, flip=False):
    p = ca - ca.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    p = p @ vt.T
    if flip:
        p[:, 2] *= -1
    c, s = np.cos(tilt), np.sin(tilt)
    p = p @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    c, s = np.cos(roll), np.sin(roll)
    p = p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def resample(pts, per=10):
    if len(pts) < 4:
        t = np.linspace(0, 1, len(pts))
        tt = np.linspace(0, 1, max(4, len(pts) * per))
        return np.stack([np.interp(tt, t, pts[:, k]) for k in range(3)], 1)
    tck, _ = splprep(pts.T, s=0, k=3)
    return np.asarray(splev(np.linspace(0, 1, len(pts) * per), tck)).T


def build(p, mask, width=2.2, per=6, taper=2.0):
    """Primitives in the oriented Angstrom frame: (kind, xyz..., normal).

    The ribbon rides on the CA helix, as PyMOL's does, and its plane comes from
    the vector between the helix axis and the CA. The axis is a running mean of
    four CAs, extrapolated at both ends: clamping the window there was what made
    the first and last turn flare out.
    """
    prims = []
    for i, j, is_h in runs(mask):
        if is_h:
            seg = p[i:j + 1]
            k = len(seg)
            axis = np.full((k, 3), np.nan)
            for m in range(1, k - 2):
                axis[m] = seg[m - 1:m + 3].mean(0)
            axis[0] = 2 * axis[1] - axis[2]
            axis[k - 2] = 2 * axis[k - 3] - axis[k - 4]
            axis[k - 1] = 2 * axis[k - 2] - axis[k - 3]
            sp = resample(seg, per)
            sa = resample(axis, per)
            r = unit(sp - sa)
            t = unit(np.gradient(sa, axis=0))
            r = unit(r - t * (r * t).sum(1, keepdims=True))
            s_ = unit(np.cross(t, r))
            n = len(sp)
            u = np.linspace(0, k - 1, n)                  # residue coordinate
            w = width * np.clip(np.minimum(u, (k - 1) - u) / taper, 0.30, 1.0)
            left, right = sp + s_ * (w[:, None] / 2), sp - s_ * (w[:, None] / 2)
            nrm = unit(np.cross(s_, t))
            for m in range(n - 1):
                prims.append(("quad",
                              np.array([left[m], right[m], right[m + 1], left[m + 1]]),
                              nrm[m]))
        else:
            if j - i + 1 < 2 and (i == 0 or j == len(p) - 1):
                continue                        # a lone terminal residue is a stub
            # run into the neighbouring helices so the tube and the ribbon meet
            a, b = max(0, i - 2), min(len(p) - 1, j + 2)
            if b - a < 2:
                continue
            c = resample(p[a:b + 1], per)
            for m in range(len(c) - 1):
                prims.append(("tube", c[m:m + 2], None))
    return prims


def shade(base, depth, lam):
    base = np.asarray(matplotlib.colors.to_rgb(base))
    lit = np.clip(base * (0.58 + 0.52 * lam), 0, 1)
    return np.clip(lit * (1 - 0.34 * depth) + 0.34 * depth, 0, 1)


def draw_cartoon(ax, ca, tilt=0.7, roll=0.35, flip=False, helix="#1668AD",
                 loop="#2E86C8", edge="#0C4474", width=2.2, tube=1.5, lwedge=0.5,
                 light=(-0.30, 0.50, 0.81)):
    """width and tube are Angstroms; the tube is converted to points from the
    axes geometry so it keeps its proportion to the ribbon at any figure size."""
    p = orient(ca, tilt, roll, flip)
    prims = build(p, clean(helical(ca)), width=width)

    xy = np.concatenate([q[:, :2] for _, q, _ in prims])
    lo, hi = xy.min(0), xy.max(0)
    sc = (hi - lo).max()
    zs = np.concatenate([q[:, 2] for _, q, _ in prims])
    z0, z1 = zs.min(), zs.max()
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)

    items = []
    for kind, q, nm in prims:
        d = float((q[:, 2].mean() - z0) / (z1 - z0 + 1e-9))
        g = (q[:, :2] - lo) / sc
        if kind == "quad":
            items.append((q[:, 2].mean(), kind, g, shade(helix, d, abs(float(nm @ L)))))
        elif kind == "tube":
            items.append((q[:, 2].mean(), kind, g, shade(loop, d, 0.74)))
        else:
            items.append((q[:, 2].mean() + 1e-6, kind, g, edge))
    items.sort(key=lambda t: t[0])

    # paint in depth bands so the fills and the outlines keep their order
    tubes = []
    z = 2.0
    k = 0
    while k < len(items):
        kind = items[k][1]
        m = k
        while m < len(items) and items[m][1] == kind:
            m += 1
        chunk = items[k:m]
        if kind == "quad":
            cols = [c[3] for c in chunk]
            ax.add_collection(PolyCollection([c[2] for c in chunk],
                                             facecolors=cols, edgecolors=cols,
                                             linewidths=0.35, zorder=z, snap=False))
        elif kind == "tube":
            lc = LineCollection([c[2] for c in chunk], colors=[c[3] for c in chunk],
                                capstyle="round", joinstyle="round", zorder=z)
            tubes.append(lc)
            ax.add_collection(lc)
        else:
            ax.add_collection(LineCollection([c[2] for c in chunk],
                                             colors=[c[3] for c in chunk],
                                             linewidths=lwedge, capstyle="round",
                                             zorder=z))
        z += 0.0005
        k = m
    w, h = (hi - lo) / sc
    ax.set_xlim(-0.04, w + 0.04)
    ax.set_ylim(-0.06, h + 0.06)
    ax.set_aspect("equal")
    ax.axis("off")
    lw = tube_points(ax.figure, ax, sc, tube)
    for lc in tubes:
        lc.set_linewidth(lw)
    return w, h


def tube_points(fig, ax, sc, tube_a):
    """Angstroms to points for the current axes, taking the binding dimension."""
    fw, fh = fig.get_size_inches()
    pos = ax.get_position()
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    ppu = min(fw * pos.width * 72 / xr, fh * pos.height * 72 / yr)
    return tube_a / sc * ppu


def graphical_abstract():
    """920 x 300 px, white, under 150 KB, as the journal specifies.

    Drawn at 4x and resampled, so the type is supersampled rather than
    rendered at 100 dpi. Every label is measured against its column before
    the file is written. Panel 1 is the Boltz-2 backbone of the
    oracle-designed PD-L1 binder of Figure 8 (chain A of designed_0).
    Values: 0.895 nested CV (Table 6), 0.761 and 0.702 grouped splits
    (Table S8), 0.571 target-blind LOTO, 0.503 independent dataset
    (Table S11), 4.65x top-decile enrichment (r10_budget_yield.csv).
    """
    from PIL import Image
    W, H, SS = 920, 300, 4
    INK, GREY, FAINT = "#15222D", "#79838E", "#A9B2BA"
    RULE, HAIR = "#D9DEE3", "#E8ECEF"
    BLUE, BLUE_M, BLUE_L = "#1A6AAC", "#5E9BC9", "#A8C7E0"
    AMBER, CORAL = "#D2922A", "#C14F3E"
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    C1, E1 = 0.028, 0.268
    C2, E2 = 0.318, 0.592
    C3, E3 = 0.642, 0.974
    DIV1, DIV2 = 0.293, 0.617

    TEXTS = []


    def T(x, y, s, size, color=GREY, weight="normal", ha="left", va="center",
          style="normal", ls=1.4, col=None):
        t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha,
                    va=va, style=style, linespacing=ls, zorder=5)
        TEXTS.append((t, col))
        return t


    def rule(x0, x1, y, color=RULE, lw=0.9):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=1)


    def vrule(x, y0, y1):
        ax.plot([x, x], [y0, y1], color=RULE, lw=0.9, solid_capstyle="butt", zorder=1)


    def chevron(x, y, size=0.011):
        ax.add_patch(FancyArrowPatch((x - size, y), (x + size, y), arrowstyle="-|>",
                                     mutation_scale=7, color=FAINT, lw=1.0,
                                     shrinkA=0, shrinkB=0, zorder=4))


    for i, (x, n, t) in enumerate(((C1, "1", "DE NOVO CANDIDATES"),
                                   (C2, "2", "TARGET-CONDITIONED MODEL"),
                                   (C3, "3", "APPLICABILITY DOMAIN"))):
        T(x, 0.890, n, 7.0, BLUE, "bold", col=i)
        T(x + 0.021, 0.890, t, 7.3, INK, "bold", col=i)
    rule(C1, E1, 0.848)
    rule(C2, E2, 0.848)
    rule(C3, E3, 0.848)
    vrule(DIV1, 0.13, 0.885)
    vrule(DIV2, 0.13, 0.885)
    chevron(DIV1, 0.50)
    chevron(DIV2, 0.50)

    # ── 1 — candidates ──────────────────────────────────────────────────────────
    seq = "MASWKELLV"
    fam = {**{a: BLUE for a in "KRH"}, **{a: BLUE_M for a in "STQNCY"},
           **{a: "#A7B1BA" for a in "AMLVIWFPG"}, **{a: CORAL for a in "ED"}}
    cw, ch = 0.0235, 0.088
    x0 = C1 + 0.002
    for i, a in enumerate(seq):
        ax.add_patch(FancyBboxPatch((x0 + i * cw, 0.726), cw * 0.86, ch,
                     boxstyle="round,pad=0,rounding_size=0.011",
                     facecolor=fam.get(a, FAINT), edgecolor="white", lw=0.6, zorder=3))
        ax.text(x0 + i * cw + cw * 0.43, 0.726 + ch / 2, a, fontsize=4.7, ha="center",
                va="center", color="white", zorder=4, fontweight="bold")
    axr = fig.add_axes([C1 + 0.006, 0.312, 0.228, 0.408])
    axr.set_facecolor("none")
    draw_cartoon(axr, ca_trace(PDB), tilt=0.70, roll=0.15, width=3.2, tube=1.5)

    T(C1 + 0.002, 0.286, "a de novo binder, predicted structure", 5.8, GREY,
      style="italic", col=0)
    rule(C1, E1, 0.244, HAIR)
    T(C1 + 0.002, 0.172, "2,630", 12.0, INK, "bold", col=0)
    T(C1 + 0.079, 0.166, "labelled pairs", 6.3, GREY, col=0)
    T(C1 + 0.002, 0.098, "24 targets  ·  17.8% bind", 6.3, GREY, col=0)

    # ── 2 — the model ───────────────────────────────────────────────────────────
    axp = fig.add_axes([C2 + 0.006, 0.470, 0.134, 0.340])
    axp.set_facecolor("none")
    axp.axis("off")
    axp.set_xlim(0, 1)
    axp.set_ylim(0, 1)
    rng = np.random.default_rng(11)
    pos = rng.normal([0.27, 0.50], [0.085, 0.145], (60, 2))
    neg = rng.normal([0.72, 0.46], [0.090, 0.150], (60, 2))
    axp.scatter(neg[:, 0], neg[:, 1], s=7.5, color="#C8D0D7", edgecolor="none", zorder=2)
    axp.scatter(pos[:, 0], pos[:, 1], s=7.5, color=BLUE_L, edgecolor="none", zorder=2)
    axp.scatter([0.27], [0.50], s=40, marker="P", color=BLUE, edgecolor="white",
                linewidths=0.8, zorder=5)
    axp.scatter([0.72], [0.46], s=40, marker="P", color="#8A949C", edgecolor="white",
                linewidths=0.8, zorder=5)
    axp.annotate("", xy=(0.325, 0.495), xytext=(0.495, 0.265),
                 arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.1, shrinkA=1, shrinkB=3))
    axp.scatter([0.505], [0.248], s=62, marker="*", color=AMBER, edgecolor=INK,
                linewidths=0.5, zorder=6)
    axp.text(0.27, 0.96, "binders", fontsize=5.4, color=BLUE, ha="center", va="center")
    axp.text(0.74, 0.92, "non-binders", fontsize=5.4, color="#8A949C", ha="center", va="center")
    axp.text(0.56, 0.165, "candidate", fontsize=5.4, color=AMBER, ha="left", va="center")

    T(C2 + 0.166, 0.712, "ESM-2 embedding", 6.6, INK, "bold", col=1)
    T(C2 + 0.166, 0.645, "7 prototype features:\ncosine to the target's\nown binder and\nnon-binder means",
      5.8, GREY, va="top", col=1)
    rule(C2, E2, 0.420, HAIR)
    T(C2 + 0.002, 0.355, "+ composition, physicochemical, design-method\n"
      "   and structure-derived scores", 5.8, GREY, col=1)
    rule(C2, E2, 0.276, HAIR)
    T(C2 + 0.002, 0.205, "LightGBM", 8.2, INK, "bold", col=1)
    T(C2 + 0.093, 0.200, "470 features", 6.3, GREY, col=1)
    T(C2 + 0.002, 0.120, "a new target enters through its prototype, no retraining",
      5.8, GREY, style="italic", col=1)

    # ── 3 — the measured gradient ───────────────────────────────────────────────
    axg = fig.add_axes([C3 + 0.022, 0.300, 0.170, 0.455])
    vals = [0.895, 0.761, 0.702, 0.571, 0.503]
    cols = [BLUE, BLUE, AMBER, CORAL, CORAL]
    labs = ["in\ndistribution", "new\nauthor", "new\nmethod", "new\ntarget", "new\ndataset"]
    xs = np.arange(5)
    axg.fill_between(xs, 0.5, vals, color=BLUE, alpha=0.06, lw=0, zorder=1)
    axg.plot(xs, vals, color="#C3CBD2", lw=1.2, zorder=2)
    axg.scatter(xs, vals, c=cols, s=26, zorder=4, edgecolor="white", linewidths=0.9)
    axg.axhline(0.5, ls=(0, (2.6, 2.2)), color=FAINT, lw=0.85, zorder=1)
    axg.text(0, 0.945, "0.895", fontsize=6.3, color=BLUE, ha="center", va="bottom",
             fontweight="bold")
    axg.text(4, 0.548, "0.503", fontsize=6.3, color=CORAL, ha="center", va="bottom",
             fontweight="bold")
    axg.text(2.1, 0.513, "chance", fontsize=5.1, color=FAINT, ha="center", va="bottom")
    axg.set_xlim(-0.55, 4.6)
    axg.set_ylim(0.43, 1.02)
    axg.set_xticks(xs)
    axg.set_xticklabels(labs, fontsize=4.8, color=GREY, linespacing=1.1)
    axg.set_yticks([0.5, 0.7, 0.9])
    axg.set_yticklabels(["0.5", "0.7", "0.9"], fontsize=5.1, color=GREY)
    axg.tick_params(length=1.5, width=0.7, colors=FAINT, pad=1.5)
    axg.set_ylabel("AUROC", fontsize=5.7, color=GREY, labelpad=1.2)
    for s in ("top", "right"):
        axg.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axg.spines[s].set_color(RULE)
        axg.spines[s].set_linewidth(0.8)

    SX = C3 + 0.212
    T(SX, 0.735, "4.65\u00d7", 14.5, BLUE, "bold", col=2)
    T(SX, 0.628, "top-decile enrichment,\nin distribution", 5.8, GREY, col=2)
    rule(SX, E3, 0.560, HAIR)
    T(SX, 0.495, "0.70\u20130.76", 9.5, INK, "bold", col=2)
    T(SX, 0.400, "fresh campaign,\nknown target", 5.8, GREY, col=2)
    rule(SX, E3, 0.330, HAIR)
    T(SX, 0.245, "2 + 2 labels", 9.0, INK, "bold", col=2)
    T(SX, 0.150, "move an unseen\ntarget to about 0.70", 5.8, GREY, col=2)

    # ── baseline ────────────────────────────────────────────────────────────────
    rule(C1, E3, 0.076, HAIR)
    T(C1, 0.034, "RATAN-PBind", 6.7, INK, "bold")
    T(C1 + 0.097, 0.034, "sequence-based prioritisation of de novo binders, returned "
      "with the regime each score falls in", 5.8, GREY)

    # ── geometry gate ───────────────────────────────────────────────────────────
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    BOUNDS = {0: (0.0, DIV1), 1: (DIV1, DIV2), 2: (DIV2, 1.0)}
    bad = []
    for t, col in TEXTS:
        if col is None:
            continue
        bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
        lo, hi = BOUNDS[col]
        if bb.x0 < lo - 1e-6 or bb.x1 > hi + 1e-6:
            bad.append((t.get_text().split("\n")[0][:38], bb.x0, bb.x1, lo, hi))
    for b in bad:
        print("  CROSSES COLUMN  %-40r x %.4f-%.4f  allowed %.3f-%.3f" % b)
    if bad:
        raise SystemExit("%d label(s) cross a column boundary" % len(bad))
    print("geometry gate: %d labels, all inside their columns" % len(TEXTS))

    # ── write ───────────────────────────────────────────────────────────────────
    # The journal takes jpeg, png or svg at 920 x 300 and at most 150 KB. The SVG
    # is the one to submit: it carries the same 920 x 300 canvas but stays sharp
    # at any zoom, where the PNG has only 920 x 300 pixels to show. Text goes out
    # as paths so the file does not depend on Arial being installed wherever it
    # is rendered.
    os.makedirs(FIG, exist_ok=True)
    pdf = str(FIG / "graphical_abstract.pdf")
    png = str(FIG / "graphical_abstract.png")
    svg = str(FIG / "graphical_abstract.svg")
    fig.savefig(pdf, facecolor="white", pad_inches=0)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg, facecolor="white", pad_inches=0)
    # A second SVG with the type converted to outlines. The journal copy keeps
    # text as text to stay under 150 KB, which means it needs Arial wherever it
    # is rendered; this one carries no font dependency and is what the README
    # shows, where file size does not matter.
    outlined = str(FIG / "graphical_abstract_outlined.svg")
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(outlined, facecolor="white", pad_inches=0)
    fig.savefig(png, dpi=100 * SS, facecolor="white", pad_inches=0)
    plt.close(fig)

    im = Image.open(png).convert("RGB")
    big = im.size
    im.resize((W, H), Image.LANCZOS).save(png, "PNG", optimize=True)
    # the cartoon is a few thousand polygons, so the raw SVG runs over the
    # journal's 150 KB; rounding every coordinate to 0.1 pt moves nothing on a
    # 920 px canvas and takes about a quarter off the file
    import xml.etree.ElementTree as ET
    t = io.open(svg, encoding="utf8").read()
    t = re.sub(r"<metadata>.*?</metadata>", "", t, flags=re.S)

    # Round coordinates ONLY inside path data. The text labels are literal
    # characters in the file, so a blanket number regex rewrites 0.895 as 0.8
    # and 4.65 as 4.6, which is how the first version of this figure went
    # wrong in the first place.
    def _round(m):
        return 'd="' + re.sub(r"(\d+\.\d)\d+", lambda k: k.group(1), m.group(1)) + '"'

    t = re.sub(r'd="([^"]*)"', _round, t)
    ET.fromstring(t)                      # refuse to write a malformed SVG

    # content gate: every number a reader can see must survive intact
    for want in ("0.895", "0.503", "4.65", "0.70", "0.76", "2,630", "17.8%",
                 "470 features", "2 + 2 labels", "about 0.70"):
        if want not in t:
            raise SystemExit("the SVG lost %r" % want)
    io.open(svg, "w", encoding="utf8").write(t)

    for f in (png, svg, outlined):
        if not os.path.exists(f):
            raise SystemExit("%s was not written" % f)
        kb = os.path.getsize(f) / 1024
        cap = "" if f == outlined else "   (limit 150 KB)"
        print("%-34s %6.1f KB%s" % (os.path.basename(f), kb, cap))
        if f != outlined and kb > 150:
            raise SystemExit("%s is over the 150 KB limit" % os.path.basename(f))
    print("png rendered %dx%d -> %dx%d; svg is vector on the same canvas"
          % (big[0], big[1], W, H))
    print("vector copy:", os.path.basename(pdf))


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
