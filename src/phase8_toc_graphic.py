"""
phase8_toc_graphic.py — JCIM Table of Contents (TOC) / Graphical Abstract

JCIM TOC specs:
  - Colorful, visually engaging summary of the paper
  - Should NOT be a data figure — artistic + scientific representation
  - Image: TIF, JPG, PNG at 300 dpi minimum
  - Size: per JCIM TOC Guidelines

This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT  = Path(__file__).parent.parent
PAPER = ROOT / "paper" / "figures"
PAPER.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.5, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")
fig.patch.set_facecolor("#0F172A")

def rounded_box(ax, x, y, w, h, facecolor, text, fontsize=8, textcolor="white",
                radius=0.25, alpha=1.0, bold=False):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=facecolor, edgecolor="none",
                         linewidth=0, zorder=3, alpha=alpha)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight=weight,
            zorder=4, multialignment="center",
            fontfamily="DejaVu Sans")

def arrow(ax, x1, y1, x2, y2, color="#4B5563"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                connectionstyle="arc3,rad=0"),
                zorder=5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(5, 3.72, "RATAN-PBind", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white",
        fontfamily="DejaVu Sans")
ax.text(5, 3.42, "Residue Attribution and Target Affinity Network for Protein Binding",
        ha="center", va="center", fontsize=7.5, color="#94A3B8",
        fontfamily="DejaVu Sans")

# ── Input: Protein sequence ───────────────────────────────────────────────────
rounded_box(ax, 0.15, 2.0, 1.7, 1.1, "#1E3A5F",
            "De novo\nProtein\nSequence", fontsize=8, bold=True)

# Amino acid letter decoration
for i, aa in enumerate(["M","A","S","W","K","E","L","L","V","Q"]):
    xpos = 0.22 + i * 0.155
    col  = "#60A5FA" if i % 3 == 0 else ("#34D399" if i % 3 == 1 else "#F472B6")
    ax.text(xpos, 1.88, aa, fontsize=5.5, color=col,
            ha="center", va="center", fontweight="bold")

arrow(ax, 1.85, 2.55, 2.35, 2.55)

# ── Feature extraction ────────────────────────────────────────────────────────
rounded_box(ax, 2.35, 1.75, 1.9, 1.6, "#1D4ED8",
            "Feature\nExtraction\n509 features\nAAC·DPC·ESM-2", fontsize=7.2)

# ESM-2 logo dots
colors_dots = ["#60A5FA","#34D399","#F472B6","#FBBF24","#A78BFA"]
for i in range(5):
    ax.plot(2.5 + i*0.3, 1.65, "o", ms=4, color=colors_dots[i], zorder=6)

arrow(ax, 4.25, 2.55, 4.75, 2.55)

# ── Prototype similarity ──────────────────────────────────────────────────────
rounded_box(ax, 4.75, 1.75, 1.9, 1.6, "#7C3AED",
            "Prototype\nSimilarity\nproto_ratio\n(#1 SHAP)", fontsize=7.2)

# Cosine similarity visualization
theta = np.linspace(0, 2*np.pi, 50)
for r, col, lw in [(0.28, "#7C3AED", 2), (0.18, "#A78BFA", 1)]:
    ax.plot(5.7 + r*np.cos(theta), 1.62 + r*np.sin(theta)*0.6,
            color=col, lw=lw, zorder=6, alpha=0.7)

arrow(ax, 6.65, 2.55, 7.15, 2.55)

# ── LightGBM ensemble ─────────────────────────────────────────────────────────
rounded_box(ax, 7.15, 1.75, 2.65, 1.6, "#0F766E",
            "LightGBM\nEnsemble\nAUROC 0.940\nAUPRC 0.765", fontsize=7.2, bold=False)

# Bar chart mini visualization
bar_heights = [0.3, 0.5, 0.7, 0.6, 0.45]
bar_colors  = ["#34D399","#34D399","#34D399","#34D399","#34D399"]
for i, (bh, bc) in enumerate(zip(bar_heights, bar_colors)):
    rect = plt.Rectangle((7.25 + i*0.22, 1.62), 0.16, bh * 0.8,
                          color=bc, alpha=0.8, zorder=6)
    ax.add_patch(rect)

# ── Bottom outputs ────────────────────────────────────────────────────────────
# Output 1: Binding prediction
rounded_box(ax, 0.15, 0.15, 2.8, 1.1, "#065F46",
            "Binding\nPrediction\n+ Confidence", fontsize=7.5)

# Output 2: LLM interpretation
rounded_box(ax, 3.6, 0.15, 2.8, 1.1, "#92400E",
            "LLM Mechanistic\nInterpretation\n(Groq · Llama-3.3-70b)", fontsize=7.2)

# Output 3: Generative design
rounded_box(ax, 7.05, 0.15, 2.8, 1.1, "#7F1D1D",
            "Generative\nDesign\n(Evo + ESM-2 MLM)", fontsize=7.2)

# Arrows from LightGBM down to outputs
arrow(ax, 1.55,  1.75, 1.55, 1.25, "#4B5563")
arrow(ax, 8.48,  1.75, 5.0,  1.25, "#4B5563")
arrow(ax, 8.48,  1.75, 8.48, 1.25, "#4B5563")

# ── AUROC badge ───────────────────────────────────────────────────────────────
badge = FancyBboxPatch((4.1, 2.9), 1.8, 0.38,
                       boxstyle="round,pad=0.08",
                       facecolor="#FBBF24", edgecolor="none", zorder=7)
ax.add_patch(badge)
ax.text(5.0, 3.09, "AUROC 0.940  ·  24 Targets",
        ha="center", va="center", fontsize=7.5, fontweight="bold",
        color="#0F172A", zorder=8)

plt.tight_layout(pad=0.1)
plt.savefig(PAPER / "toc_graphic.png", dpi=300, facecolor=fig.get_facecolor())
plt.savefig(PAPER / "toc_graphic.tif", dpi=300, facecolor=fig.get_facecolor())
plt.close()
print("TOC graphic saved: toc_graphic.png + toc_graphic.tif")
