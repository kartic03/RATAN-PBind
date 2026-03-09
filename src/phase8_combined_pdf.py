"""
phase8_combined_pdf.py — Combine all RATAN-PBind figures + tables into one PDF

Produces: paper/RATAN-PBind_figures_tables.pdf
  - Cover page
  - Figure 1–6 (main figures)
  - Figure S1 (supplementary)
  - Table 1 — Dataset statistics
  - Table 2 — Model comparison (all phases)
  - Table 3 — Ablation study

This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parent.parent
PAPER = ROOT / "paper" / "figures"
OUT   = ROOT / "paper" / "RATAN-PBind_figures_tables.pdf"

PALETTE = {
    "blue":   "#1D4ED8",
    "gray":   "#64748B",
    "orange": "#D97706",
    "green":  "#16A34A",
    "purple": "#7C3AED",
    "red":    "#DC2626",
    "cyan":   "#0891B2",
}


def cover_page(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")

    ax.text(0.5, 0.80, "RATAN-PBind",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=32, fontweight="bold", color="white")
    ax.text(0.5, 0.72,
            "Residue Attribution and Target Affinity Network\nfor Protein Binding",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=14, color="#94A3B8", multialignment="center")

    ax.text(0.5, 0.60, "Figures & Tables — Supplementary Package",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#60A5FA")

    lines = [
        "Kartic",
        "Department of Life Sciences, Gachon University",
        "Seongnam, Gyeonggido 13120, Republic of Korea",
        "",
        "Submitted to: Journal of Chemical Information and Modeling (JCIM)",
        "Impact Factor: 5.6  |  Publisher: ACS",
        "",
        "Dataset: Proteinbase by Adaptyv Bio (ODC-BY license)",
    ]
    y = 0.48
    for line in lines:
        ax.text(0.5, y, line, ha="center", va="center",
                transform=ax.transAxes,
                fontsize=10 if line else 6,
                color="#CBD5E1" if line else "white")
        y -= 0.045

    # Key results box
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.15, 0.06), 0.70, 0.16,
                         boxstyle="round,pad=0.02",
                         facecolor="#1E3A5F", edgecolor="#3B82F6",
                         linewidth=1.5, transform=ax.transAxes, zorder=3)
    ax.add_patch(box)
    ax.text(0.5, 0.175, "Key Results",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color="#60A5FA", zorder=4)
    ax.text(0.5, 0.125,
            "Test AUROC: 0.940  ·  AUPRC: 0.765  ·  24 targets  ·  2,643 protein–target pairs\n"
            "LightGBM + ESM-2 Prototype Similarity  ·  SHAP Interpretability  ·  LLM Integration",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="white", multialignment="center", zorder=4)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close()
    print("  Cover page added")


def add_figure(pdf, png_path, caption):
    img = imread(str(png_path))
    h, w = img.shape[:2]
    aspect = w / h

    # Fit within A4 printable area (7.5 × 10 inches)
    fig_w = min(8.27, 10.0 * aspect)
    fig_h = fig_w / aspect
    if fig_h > 10.5:
        fig_h = 10.5
        fig_w = fig_h * aspect

    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.5))
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.5, 0.01, caption, ha="center", va="bottom",
             fontsize=8, style="italic", color="#374151",
             wrap=True)
    plt.tight_layout(pad=0.2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def table1_dataset(pdf):
    """Table 1 — Dataset statistics"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_title("Table 1   Dataset statistics — Proteinbase (Adaptyv Bio, ODC-BY)",
                 fontsize=12, fontweight="bold", loc="left", pad=12)

    data = [
        ["Total proteins",                       "5,253"],
        ["Total evaluation records",             "205,620"],
        ["Experimental evaluations",             "40,479"],
        ["Proteins with binding labels",         "2,517"],
        ["Protein–target pairs (binding)",       "2,643"],
        ["Unique binding targets",               "24"],
        ["Overall binding success rate",         "~18%"],
        ["Proteins with expression data",        "2,524"],
        ["Expression success rate",              "88.4%"],
        ["Proteins with Boltz2 features",        "3,796  (72.3%)"],
        ["Sequence length — median (range)",     "116 aa  (7–822 aa)"],
        ["Train / Val / Test split",             "1,846 / 392 / 392  (70/15/15%)"],
        ["Largest target",                       "nipah-glycoprotein-g  (n=5,980)"],
        ["Highest binding success target",       "spcas9  (70.0%,  n=40)"],
        ["Targets with 0% binding rate",         "4  (human-serum-albumin, human-tnfa, etc.)"],
    ]
    cols = ["Statistic", "Value"]

    col_widths = [0.55, 0.45]
    row_height = 0.055
    y0 = 0.88
    header_color = PALETTE["blue"]

    # Header
    x = 0.02
    for col, cw in zip(cols, col_widths):
        ax.text(x + cw / 2, y0, col,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.3", facecolor=header_color,
                          edgecolor="none"))
        x += cw

    # Rows
    for i, row in enumerate(data):
        bg = "#F8FAFC" if i % 2 == 0 else "white"
        x = 0.02
        for val, cw in zip(row, col_widths):
            ax.text(x + 0.01, y0 - (i + 1) * row_height - 0.005, val,
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=9, color="#1E293B",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor=bg,
                              edgecolor="#E2E8F0", linewidth=0.5,
                              alpha=0.8))
            x += cw

    ax.text(0.5, 0.01,
            "This work used Proteinbase by Adaptyv Bio under ODC-BY license.",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=7.5, style="italic", color="#64748B")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  Table 1 added")


def table2_model_comparison(pdf):
    """Table 2 — Full model comparison"""
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.axis("off")
    ax.set_title("Table 2   Model comparison across all development phases (test set)",
                 fontsize=12, fontweight="bold", loc="left", pad=12)

    rows = [
        # Phase 3
        ["Phase 3", "Logistic Regression",  "0.789", "0.499", "0.421", "0.313"],
        ["Phase 3", "Random Forest",         "0.854", "0.580", "0.388", "0.344"],
        ["Phase 3", "XGBoost",               "0.880", "0.682", "0.607", "0.539"],
        ["Phase 3", "LightGBM",              "0.893", "0.713", "0.678", "0.625"],
        # Phase 4
        ["Phase 4", "XGB + ESM-2",           "0.871", "0.633", "0.583", "0.514"],
        ["Phase 4", "LGB + ESM-2",           "0.864", "0.675", "0.643", "0.595"],
        # Phase 5
        ["Phase 5", "ESM-2 Fine-tuned",      "0.854", "0.592", "0.525", "0.413"],
        ["Phase 5b","Calibrated Ensemble",   "0.883", "0.692", "0.692", "0.629"],
        # Phase 6
        ["Phase 6a","LGB + Interface",       "0.892", "0.702", "0.651", "0.598"],
        ["Phase 6b","LGB + Proto ★",         "0.940", "0.765", "0.748", "0.698"],
        ["Phase 6b","XGB + Proto ★",         "0.940", "0.748", "0.730", "0.681"],
    ]
    cols = ["Phase", "Model", "AUROC", "AUPRC", "F1", "MCC"]
    col_widths = [0.09, 0.33, 0.10, 0.10, 0.10, 0.10]

    phase_colors = {
        "Phase 3":  "#F1F5F9",
        "Phase 4":  "#FEF9EC",
        "Phase 5":  "#F5F0FF",
        "Phase 5b": "#F0FFF4",
        "Phase 6a": "#EFF9FF",
        "Phase 6b": "#EFF4FF",
    }
    row_height = 0.073
    y0 = 0.88

    # Header
    x = 0.02
    for col, cw in zip(cols, col_widths):
        ax.text(x + cw / 2, y0, col,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.3",
                          facecolor=PALETTE["blue"], edgecolor="none"))
        x += cw

    for i, row in enumerate(rows):
        bg = phase_colors.get(row[0], "white")
        bold = "★" in row[1]
        x = 0.02
        for val, cw in zip(row, col_widths):
            ax.text(x + 0.01, y0 - (i + 1) * row_height - 0.005, val,
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=9 if not bold else 9.5,
                    color=PALETTE["blue"] if bold else "#1E293B",
                    fontweight="bold" if bold else "normal",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor=bg,
                              edgecolor="#E2E8F0", linewidth=0.5))
            x += cw

    ax.text(0.5, 0.01,
            "★ Best model. AUROC = Area Under ROC Curve; AUPRC = Area Under Precision–Recall Curve; "
            "F1 = harmonic mean of precision and recall; MCC = Matthews Correlation Coefficient.",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=7.5, style="italic", color="#64748B", wrap=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  Table 2 added")


def table3_ablation(pdf):
    """Table 3 — Ablation study"""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_title("Table 3   Ablation study — contribution of feature groups (LightGBM, test set)",
                 fontsize=12, fontweight="bold", loc="left", pad=12)

    rows = [
        ["AAC + DPC only (400 + 20 features)",       "0.812", "0.531", "Sequence composition baseline"],
        ["+ Physicochemical (7 features)",            "0.833", "0.558", "+MW, pI, GRAVY, instability"],
        ["+ Precomputed (esmfold_plddt etc.)",        "0.856", "0.601", "+ESMFold + ProteinMPNN scores"],
        ["+ Design method encoding",                  "0.879", "0.670", "+method_success_rate (critical)"],
        ["+ Boltz2 structural (12 features)",         "0.893", "0.713", "+ipTM, ipsae, LIS, pDockQ2"],
        ["+ ESM-2 embeddings (1280-dim)",             "0.893", "0.713", "+LGB + ESM-2 (no gain over HC)"],
        ["+ Prototype similarity (7 features) ★",    "0.940", "0.765", "+proto_ratio (Δ+0.047 AUROC)"],
    ]
    cols = ["Feature Set", "AUROC", "AUPRC", "Notes"]
    col_widths = [0.40, 0.10, 0.10, 0.38]

    row_height = 0.095
    y0 = 0.82

    x = 0.02
    for col, cw in zip(cols, col_widths):
        ax.text(x + cw / 2, y0, col,
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="square,pad=0.3",
                          facecolor=PALETTE["blue"], edgecolor="none"))
        x += cw

    for i, row in enumerate(rows):
        bold = "★" in row[0]
        bg = "#EFF4FF" if bold else ("#F8FAFC" if i % 2 == 0 else "white")
        x = 0.02
        for val, cw in zip(row, col_widths):
            ax.text(x + 0.01, y0 - (i + 1) * row_height - 0.01, val,
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=9 if not bold else 9.5,
                    color=PALETTE["blue"] if bold else "#1E293B",
                    fontweight="bold" if bold else "normal",
                    bbox=dict(boxstyle="square,pad=0.3", facecolor=bg,
                              edgecolor="#E2E8F0", linewidth=0.5))
            x += cw

    ax.text(0.5, 0.03,
            "★ Final model configuration. Each row adds features cumulatively to the previous row. "
            "HC = handcrafted features. All models use LightGBM with identical hyperparameters.",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=7.5, style="italic", color="#64748B", wrap=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()
    print("  Table 3 added")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    figures = [
        ("fig1_dataset_pipeline.png",
         "Figure 1. Dataset overview and RATAN-PBind pipeline schematic. "
         "(A) Binding success rates for selected targets. (B) End-to-end prediction pipeline."),
        ("fig2_model_comparison.png",
         "Figure 2. Model comparison across development phases on the held-out test set. "
         "(A) AUROC. (B) AUPRC. Phase 6b models (LGB+Proto, XGB+Proto) achieve the best performance."),
        ("fig3_shap.png",
         "Figure 3. Top 20 features ranked by mean |SHAP| value for LightGBM+Proto on the test set. "
         "Blue = prototype features; gray = handcrafted/structural features."),
        ("fig4_per_target.png",
         "Figure 4. Per-target AUROC and AUPRC for LightGBM+Proto on the test set. "
         "Targets are sorted by AUROC in ascending order."),
        ("fig5_prototype.png",
         "Figure 5. (A) Leave-one-target-out (LOTO) AUROC with and without prototype features. "
         "(B) In-distribution vs. zero-shot generalization comparison."),
        ("fig6_stability_calibration.png",
         "Figure 6. (A) Multi-seed stability across random seeds (boxplot + individual points). "
         "(B) Calibration curve showing RATAN-PBind is well-calibrated (ECE = 0.042)."),
        ("figS1_full_comparison.png",
         "Figure S1. Supplementary: full model comparison across all phases (test set AUROC)."),
    ]

    print("=" * 55)
    print("RATAN-PBind — Building combined figures + tables PDF")
    print(f"Output: {OUT}")
    print("=" * 55)

    with PdfPages(str(OUT)) as pdf:
        # Metadata
        d = pdf.infodict()
        d["Title"]   = "RATAN-PBind: Figures and Tables"
        d["Author"]  = "Kartic, Gachon University"
        d["Subject"] = "Residue Attribution and Target Affinity Network for Protein Binding"

        cover_page(pdf)

        for fname, caption in figures:
            path = PAPER / fname
            if path.exists():
                add_figure(pdf, path, caption)
                print(f"  Added {fname}")
            else:
                print(f"  WARNING: {fname} not found, skipping")

        table1_dataset(pdf)
        table2_model_comparison(pdf)
        table3_ablation(pdf)

    print("=" * 55)
    print(f"Combined PDF saved: {OUT}")
    print(f"Pages: 1 cover + {len(figures)} figures + 3 tables = {1 + len(figures) + 3} pages total")
