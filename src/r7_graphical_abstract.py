"""Graphical abstract for J. Cheminformatics — 920x300 px, white background,
<150 KB. Designed as a cohesive infographic: three grouped stage-cards with
subtle fills, a clean sequence motif, soft embedding clusters, an applicability
gradient, and prominent number callouts."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
FIG = Path(__file__).parent.parent / "figures"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})
BLUE="#2C6FB3"; TEAL="#1F9E89"; AMBER="#E0A33E"; CORAL="#CB5A4C"
SLATE="#5A626D"; INK="#1E2530"; PURPLE="#7E6BB5"; LINE="#D7DEE6"
TINT_B="#EEF4FB"; TINT_N="#F6F8FA"; TINT_T="#ECF7F3"

fig = plt.figure(figsize=(9.2, 3.0), dpi=100); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def card(x, y, w, h, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                 facecolor=fill, edgecolor=LINE, linewidth=1.1, zorder=1))
def connect(x1, x2, y=0.46):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=16,
                 color=SLATE, lw=2.4, zorder=6, shrinkA=0, shrinkB=0))

# stage cards
card(0.015, 0.10, 0.205, 0.72, TINT_B)
card(0.275, 0.10, 0.330, 0.72, TINT_N)
card(0.660, 0.10, 0.325, 0.72, TINT_T)
connect(0.222, 0.273); connect(0.607, 0.658)

# wordmark + caption (outside cards)
ax.text(0.015, 0.95, "RATAN-PBind", fontsize=13.5, fontweight="bold", color=INK, va="center")
ax.text(0.985, 0.95, "sequence-based de novo binder pre-screening",
        fontsize=8.6, color=SLATE, va="center", ha="right", style="italic")

# ── Card 1 — de novo binder candidate (clean sequence motif) ────────────────
ax.text(0.1175, 0.74, "De novo binder candidate", fontsize=9.2, fontweight="bold",
        ha="center", color=INK)
seq = "MASWKELLVQRTI"
cls = {**{a: AMBER for a in "AMLVIWFPG"}, **{a: TEAL for a in "STQNCY"},
       **{a: BLUE for a in "KRH"}, **{a: CORAL for a in "ED"}}
n = len(seq); cw = 0.0135; x0 = 0.1175 - n*cw/2
for i, a in enumerate(seq):
    ax.add_patch(FancyBboxPatch((x0+i*cw, 0.44), cw*0.86, 0.12,
                 boxstyle="round,pad=0,rounding_size=0.015",
                 facecolor=cls.get(a, SLATE), edgecolor="white", linewidth=0.6, zorder=3))
ax.text(0.1175, 0.355, "amino-acid sequence", fontsize=7.4, ha="center", color=SLATE, style="italic")
ax.text(0.1175, 0.20, "24 targets · 2,630 pairs", fontsize=8.4, ha="center", color=INK, fontweight="bold")

# ── Card 2 — prototype similarity (soft clusters + candidate) ────────────────
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
axp.scatter([0.52], [0.45], s=150, marker="*", color=AMBER, edgecolor=INK, linewidths=0.7, zorder=5)
axp.set_xlim(0, 1); axp.set_ylim(0.08, 0.95); axp.axis("off")
# pipeline label (single clean line) + legend row, both centred under the scatter
ax.text(0.440, 0.235, "ESM-2 embedding  →  LightGBM · 470 features",
        fontsize=7.8, ha="center", color=INK)
ly = 0.155
for lx, col, lab, mk in [(0.350, BLUE, "binders", "o"), (0.430, "#AEB7C2", "non-binders", "o"),
                          (0.535, AMBER, "candidate", "*")]:
    ax.scatter(lx, ly, s=(46 if mk == "*" else 26), marker=mk, color=col,
               edgecolor=(INK if mk == "*" else "none"), linewidths=0.4, zorder=4)
    ax.text(lx+0.011, ly, lab, fontsize=6.8, va="center", color=SLATE)

# ── Card 3 — prioritise + applicability domain ──────────────────────────────
ax.text(0.8225, 0.74, "Ranked binders + applicability domain", fontsize=9.2,
        fontweight="bold", ha="center", color=INK)
axg = fig.add_axes([0.690, 0.215, 0.135, 0.40])
vals = [0.90, 0.82, 0.77, 0.55]; cols = [TEAL, TEAL, AMBER, CORAL]
labs = ["in-\ndist.", "x-\nmethod", "new\ncamp.", "new\ntarget"]
axg.plot(range(4), vals, color=LINE, lw=1.6, zorder=1)
axg.scatter(range(4), vals, c=cols, s=46, zorder=3, edgecolor="white", linewidths=0.9)
axg.axhline(0.5, ls=(0, (3, 2)), color="#C2C8D0", lw=1)
axg.set_ylim(0.45, 0.98); axg.set_xlim(-0.45, 3.45)
axg.set_xticks(range(4)); axg.set_xticklabels(labs, fontsize=6.1, color=SLATE, linespacing=0.9)
axg.set_yticks([0.5, 0.7, 0.9]); axg.tick_params(labelsize=6.1, length=2, colors=SLATE)
axg.set_ylabel("AUROC", fontsize=6.8, color=SLATE); axg.yaxis.set_label_coords(-0.22, 0.5)
for sp in ["top", "right"]: axg.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: axg.spines[sp].set_color("#C2C8D0")
# big callouts on the right of card 3
ax.text(0.905, 0.585, "4.8×", fontsize=20, fontweight="bold", color=TEAL, ha="center", va="center")
ax.text(0.905, 0.475, "enrichment", fontsize=8.2, color=INK, ha="center", va="center")
ax.text(0.905, 0.405, "(top 10% ranked)", fontsize=6.8, color=SLATE, ha="center", va="center")
ax.text(0.905, 0.285, "few-shot", fontsize=8.6, fontweight="bold", color=BLUE, ha="center", va="center")
ax.text(0.905, 0.205, "2 labels → new target", fontsize=6.9, color=SLATE, ha="center", va="center")

# bottom caption
ax.text(0.5, 0.035, "Cheap, interpretable sequence model with a quantified applicability domain "
        "and few-shot extension to new targets", fontsize=7.4, color=SLATE, ha="center", va="center")

fig.savefig(FIG / "graphical_abstract.png", dpi=100, facecolor="white", pad_inches=0)
plt.close()
import os
from PIL import Image
p = FIG / "graphical_abstract.png"
im = Image.open(p).convert("RGB")
if im.size != (920, 300): im = im.resize((920, 300), Image.LANCZOS)
im.save(p, "PNG", optimize=True)
print("size:", im.size, "px | file:", round(os.path.getsize(p)/1024, 1), "KB")
