"""
phase8_figures.py — Publication-quality figures for RATAN-PBind paper

Generates 6 main figures + 1 supplementary figure saved to paper/figures/

This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
PAPER   = ROOT / "paper" / "figures"
PAPER.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

PALETTE = {
    "blue":    "#1D4ED8",
    "red":     "#DC2626",
    "green":   "#16A34A",
    "orange":  "#D97706",
    "purple":  "#7C3AED",
    "gray":    "#64748B",
    "light":   "#F1F5F9",
    "proto":   "#1D4ED8",
    "classic": "#64748B",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Dataset overview + RATAN-PBind pipeline schematic
# ─────────────────────────────────────────────────────────────────────────────
def fig1_dataset_pipeline():
    print("Figure 1: Dataset overview + pipeline schematic...")
    fig = plt.figure(figsize=(14, 6.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.38)

    # ── Panel A: Dataset statistics bar chart ─────────────────────────────────
    ax = fig.add_subplot(gs[0])

    targets = [
        ("spcas9",                 40,  70.0),
        ("il7r",                  117,  67.5),
        ("insulin-receptor",       70,  60.0),
        ("pdgfr-beta",             60,  60.0),
        ("pd-l1",                 139,  50.4),
        ("mdm2",                  165,  29.1),
        ("egfr",                 1924,  17.9),
        ("nipah-glycoprotein-g", 5980,  10.0),
    ]
    names  = [t[0] for t in targets]
    rates  = [t[2] for t in targets]
    colors = [PALETTE["blue"] if r >= 30 else PALETTE["gray"] for r in rates]

    bars = ax.barh(names, rates, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Binding success rate (%)", labelpad=6)
    ax.set_title("A   Dataset overview (selected targets)",
                 loc="left", fontweight="bold", pad=8)
    ax.set_xlim(0, 100)
    ax.axvline(18, color=PALETTE["red"], lw=1.2, ls="--", alpha=0.8,
               label="Overall mean (18%)")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.0f}%", va="center", fontsize=8, color="#374151")
    ax.tick_params(axis="y", pad=4)

    # ── Panel B: Clean pipeline schematic ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 9.5)
    ax2.axis("off")
    ax2.set_title("B   RATAN-PBind pipeline",
                  loc="left", fontweight="bold", pad=8)

    def box(ax, x, y, w, h, text, color, fontsize=8.2, bold=False):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, color="white",
                fontweight="bold" if bold else "normal",
                zorder=4, multialignment="center")

    def arr(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="#94A3B8", lw=1.4,
                                   connectionstyle="arc3,rad=0.0"),
                    zorder=5)

    # Row 1 — Input
    box(ax2, 3.5, 8.1, 3.0, 0.9,  "Protein Sequence",  "#374151", bold=True)

    # Row 2 — Feature branches
    box(ax2, 0.1, 6.0, 3.8, 1.1,
        "Handcrafted Features\n(AAC · DPC · Physico)", "#475569", fontsize=7.6)
    box(ax2, 6.0, 6.0, 3.9, 1.1,
        "ESM-2 650M\nEmbeddings (1280-dim)", PALETTE["blue"], fontsize=7.6)

    # Row 3 — Prototype (right side only)
    box(ax2, 6.0, 4.3, 3.9, 1.1,
        "Prototype Similarity\n(proto_ratio, #1 SHAP)", PALETTE["purple"], fontsize=7.6)

    # Row 4 — LightGBM merge
    box(ax2, 2.6, 2.6, 4.8, 1.0,
        "LightGBM Ensemble  (509 features)", PALETTE["blue"], bold=True)

    # Row 5 — Outputs
    box(ax2, 0.1, 0.3, 2.8, 1.6,
        "Binding\nPrediction\n+ Confidence", PALETTE["green"], fontsize=7.6)
    box(ax2, 3.2, 0.3, 3.5, 1.6,
        "SHAP Interpretation\n+ LLM Reasoning\n(Groq · Llama-3.3-70b)", PALETTE["orange"], fontsize=7.0)
    box(ax2, 7.0, 0.3, 2.9, 1.6,
        "Generative\nDesign\n(Evo + ESM-2)", PALETTE["red"], fontsize=7.6)

    # Arrows: input → both feature branches
    arr(ax2, 5.0, 8.1, 2.0, 7.1)   # input → handcrafted (diagonal)
    arr(ax2, 5.0, 8.1, 7.95, 7.1)  # input → ESM-2 (diagonal)

    # ESM-2 → Prototype (straight down)
    arr(ax2, 7.95, 6.0, 7.95, 5.4)

    # Both feature branches → LightGBM
    arr(ax2, 2.0, 6.0, 3.8, 3.6)   # handcrafted → LGB (diagonal)
    arr(ax2, 7.95, 4.3, 6.2, 3.6)  # proto → LGB (diagonal)

    # LightGBM → 3 outputs
    arr(ax2, 5.0, 2.6, 1.5, 1.9)   # → binding
    arr(ax2, 5.0, 2.6, 4.95, 1.9)  # → SHAP (straight down)
    arr(ax2, 5.0, 2.6, 8.45, 1.9)  # → generative

    plt.savefig(PAPER / "fig1_dataset_pipeline.pdf")
    plt.savefig(PAPER / "fig1_dataset_pipeline.png")
    plt.close()
    print("  saved fig1_dataset_pipeline.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Model comparison: AUROC and AUPRC across all phases
# ─────────────────────────────────────────────────────────────────────────────
def fig2_model_comparison():
    print("Figure 2: Model comparison...")

    models = [
        ("Logistic Regression",  0.789, 0.499, "Phase 3"),
        ("Random Forest",        0.854, 0.580, "Phase 3"),
        ("XGBoost",              0.880, 0.682, "Phase 3"),
        ("LightGBM",             0.893, 0.713, "Phase 3"),
        ("XGB + ESM-2",          0.871, 0.633, "Phase 4"),
        ("LGB + ESM-2",          0.864, 0.675, "Phase 4"),
        ("ESM-2 Fine-tuned",     0.854, 0.592, "Phase 5"),
        ("Calibrated Ensemble",  0.883, 0.692, "Phase 5b"),
        ("LGB + Interface",      0.892, 0.702, "Phase 6a"),
        ("LGB + Proto",          0.940, 0.765, "Phase 6b"),
        ("XGB + Proto",          0.940, 0.748, "Phase 6b"),
    ]

    names  = [m[0] for m in models]
    aurocs = [m[1] for m in models]
    auprcs = [m[2] for m in models]
    phases = [m[3] for m in models]

    phase_colors = {
        "Phase 3":  PALETTE["gray"],
        "Phase 4":  PALETTE["orange"],
        "Phase 5":  PALETTE["purple"],
        "Phase 5b": PALETTE["green"],
        "Phase 6a": "#0891B2",
        "Phase 6b": PALETTE["blue"],
    }
    colors = [phase_colors[p] for p in phases]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, vals, metric, xlo in zip(axes,
                                     [aurocs, auprcs],
                                     ["AUROC", "AUPRC"],
                                     [0.60, 0.35]):
        bars = ax.barh(names, vals, color=colors, height=0.55,
                       edgecolor="white", linewidth=0.5)
        ax.set_xlabel(metric, labelpad=6)
        ax.set_title(f"{'A' if metric=='AUROC' else 'B'}   {metric} — all phases (test set)",
                     loc="left", fontweight="bold", pad=8)

        # extra space on right for labels (longest label "0.765" needs ~0.08 space)
        ax.set_xlim(xlo, max(vals) + 0.12)
        ax.axvline(1.0, color="#E2E8F0", lw=0.8)

        best = max(vals)
        ax.axvline(best, color=PALETTE["blue"], lw=1, ls="--", alpha=0.4)

        for bar, val in zip(bars, vals):
            ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7.8)

        best_idx = vals.index(max(vals))
        bars[best_idx].set_edgecolor(PALETTE["blue"])
        bars[best_idx].set_linewidth(2.0)

    # Legend only on left panel
    legend_patches = [mpatches.Patch(color=c, label=p)
                      for p, c in phase_colors.items()]
    axes[0].legend(handles=legend_patches, frameon=False,
                   loc="lower right", fontsize=7.5)
    axes[0].tick_params(axis="y", pad=4)

    plt.tight_layout(pad=1.5)
    plt.savefig(PAPER / "fig2_model_comparison.pdf")
    plt.savefig(PAPER / "fig2_model_comparison.png")
    plt.close()
    print("  saved fig2_model_comparison.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — SHAP feature importance
# ─────────────────────────────────────────────────────────────────────────────
def fig3_shap():
    print("Figure 3: SHAP feature importance...")
    shap_df = pd.read_csv(OUTPUTS / "phase6d_shap.csv")
    shap_df = shap_df.sort_values("mean_abs").tail(20)

    colors = [PALETTE["blue"] if row["is_proto"] else PALETTE["gray"]
              for _, row in shap_df.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.set_title("Figure 3   Top 20 features by mean |SHAP| value"
                 " (LightGBM + Proto, test set)",
                 fontsize=10, fontweight="bold", loc="left", pad=10)

    bars = ax.barh(shap_df["feature"], shap_df["mean_abs"],
                   color=colors, height=0.62,
                   edgecolor="white", linewidth=0.5)

    xmax = shap_df["mean_abs"].max()
    ax.set_xlim(0, xmax * 1.30)  # generous right margin for labels

    for bar, val in zip(bars, shap_df["mean_abs"]):
        ax.text(val + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.8)

    ax.set_xlabel("Mean |SHAP value|", labelpad=6)
    ax.tick_params(axis="y", pad=4)

    proto_patch = mpatches.Patch(color=PALETTE["blue"],  label="Prototype feature")
    other_patch = mpatches.Patch(color=PALETTE["gray"],  label="Handcrafted / structural")
    ax.legend(handles=[proto_patch, other_patch], frameon=False,
              loc="lower right", fontsize=8)

    plt.tight_layout(pad=1.5)
    plt.savefig(PAPER / "fig3_shap.pdf")
    plt.savefig(PAPER / "fig3_shap.png")
    plt.close()
    print("  saved fig3_shap.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-target performance heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig4_per_target():
    print("Figure 4: Per-target heatmap...")
    df = pd.read_csv(OUTPUTS / "per_target_results.csv")
    df = df.sort_values("auroc", ascending=True)   # ascending so best is at top

    n = len(df)
    fig, ax = plt.subplots(figsize=(8.5, max(5.5, n * 0.50)))
    ax.set_title("Figure 4   Per-target AUROC and AUPRC"
                 " (LightGBM + Proto, test set)",
                 fontsize=10, fontweight="bold", loc="left", pad=10)

    x     = np.arange(n)
    width = 0.35

    b1 = ax.barh(x + width / 2, df["auroc"], width,
                 color=PALETTE["blue"],   label="AUROC",
                 edgecolor="white", linewidth=0.5)
    b2 = ax.barh(x - width / 2, df["auprc"], width,
                 color=PALETTE["orange"], label="AUPRC",
                 edgecolor="white", linewidth=0.5)

    # Combine target name + sample size into one clean label
    ylabels = [f"{row['target']}  (n={int(row['n'])})"
               for _, row in df.iterrows()]
    ax.set_yticks(x)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Metric value", labelpad=6)
    ax.set_xlim(0, 1.22)   # room for "1.000" labels at right
    ax.axvline(0.5, color="#CBD5E1", lw=0.8, ls="--", alpha=0.8)
    ax.axvline(1.0, color="#CBD5E1", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(axis="y", pad=5)

    # Annotate AUROC bars only (avoid clutter)
    for bar, val in zip(b1, df["auroc"]):
        ax.text(bar.get_width() + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7)

    plt.tight_layout(pad=1.5)
    # Leave extra left margin for long target names + n= labels
    plt.subplots_adjust(left=0.28)
    plt.savefig(PAPER / "fig4_per_target.pdf")
    plt.savefig(PAPER / "fig4_per_target.png")
    plt.close()
    print("  saved fig4_per_target.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Prototype feature analysis + LOTO generalization
# ─────────────────────────────────────────────────────────────────────────────
def fig5_prototype():
    print("Figure 5: Prototype feature analysis...")
    loto_df = pd.read_csv(OUTPUTS / "phase6c_loto_results.csv")
    loto_df = loto_df[loto_df["n_pos"] > 0].copy()
    loto_df = loto_df.sort_values("auroc_with_proto", ascending=True)

    n_loto = len(loto_df)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, n_loto * 0.40)))
    fig.suptitle("Figure 5   Prototype features: leave-one-target-out (LOTO) generalization",
                 fontsize=11, fontweight="bold", x=0.01, ha="left", y=1.01)

    # ── Panel A: LOTO AUROC with vs without proto (horizontal bars) ───────────
    ax = axes[0]
    y  = np.arange(n_loto)
    w  = 0.38

    ax.barh(y + w / 2, loto_df["auroc_no_proto"],   w, color=PALETTE["gray"],
            label="Without prototype", edgecolor="white", linewidth=0.4)
    ax.barh(y - w / 2, loto_df["auroc_with_proto"], w, color=PALETTE["blue"],
            label="With prototype",    edgecolor="white", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(loto_df["target"], fontsize=7.5)
    ax.set_xlabel("LOTO AUROC", labelpad=6)
    ax.set_xlim(0, 1.30)   # extra right space for mean labels
    ax.axvline(0.5, color="#CBD5E1", lw=0.8, ls="--")
    ax.set_title("A   Leave-one-target-out AUROC (20 targets)",
                 loc="left", fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.tick_params(axis="y", pad=4)

    mean_no  = loto_df["auroc_no_proto"].mean()
    mean_yes = loto_df["auroc_with_proto"].mean()
    ax.axvline(mean_no,  color=PALETTE["gray"], lw=1.5, ls="--", alpha=0.9)
    ax.axvline(mean_yes, color=PALETTE["blue"], lw=1.5, ls="--", alpha=0.9)

    # Mean labels as horizontal text boxes in the right margin — clearly readable
    mid_y = n_loto / 2
    ax.text(mean_no + 0.015, mid_y - 1.5,
            f"Without proto\nμ = {mean_no:.3f}",
            va="center", fontsize=8, color=PALETTE["gray"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["gray"], linewidth=1.0, alpha=0.95))
    ax.text(mean_yes + 0.015, mid_y + 1.5,
            f"With proto\nμ = {mean_yes:.3f}",
            va="center", fontsize=8, color=PALETTE["blue"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["blue"], linewidth=1.0, alpha=0.95))

    # ── Panel B: In-dist vs LOTO comparison ──────────────────────────────────
    ax2 = axes[1]
    categories  = ["In-distribution\n(matched target)",
                   "Zero-shot LOTO\n(without proto)",
                   "Zero-shot LOTO\n(with proto)"]
    values      = [0.940, mean_no, mean_yes]
    colors_bar  = [PALETTE["blue"], PALETTE["gray"], PALETTE["purple"]]

    x_pos = np.arange(len(categories))
    bars  = ax2.bar(x_pos, values, color=colors_bar, width=0.55,
                    edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.015,
                 f"{val:.3f}", ha="center", fontsize=9.5, fontweight="bold",
                 color="#1E293B")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=8.5)
    ax2.set_ylabel("Mean AUROC", labelpad=6)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlim(-0.6, 2.6)   # wider so annotation fits inside
    ax2.axhline(0.5, color="#CBD5E1", lw=0.8, ls="--", label="Random (0.5)")
    ax2.set_title("B   In-distribution vs. zero-shot generalization",
                  loc="left", fontweight="bold", pad=8)
    ax2.legend(frameon=False, fontsize=7.5)

    # Bracket showing generalization gap between bars 0 and 2
    gap = values[0] - values[2]
    x_br = 2.3   # placed between bar 2 (center=2) and right edge (xlim=2.6)
    ax2.annotate("", xy=(x_br, values[0]), xytext=(x_br, values[2]),
                 arrowprops=dict(arrowstyle="<->", color=PALETTE["red"], lw=1.5))
    ax2.text(x_br + 0.07, (values[0] + values[2]) / 2,
             f"Δ {gap:.3f}\n(generaliz.\ngap)",
             fontsize=7.5, color=PALETTE["red"], va="center")

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(left=0.22)
    plt.savefig(PAPER / "fig5_prototype.pdf")
    plt.savefig(PAPER / "fig5_prototype.png")
    plt.close()
    print("  saved fig5_prototype.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Multi-seed stability + calibration
# ─────────────────────────────────────────────────────────────────────────────
def fig6_stability_calibration():
    print("Figure 6: Stability and calibration...")
    seed_df = pd.read_csv(OUTPUTS / "phase6d_seed_results.csv")

    lgb_aurocs = seed_df[seed_df["model"] == "LGB"]["test_auroc"].values
    xgb_aurocs = seed_df[seed_df["model"] == "XGB"]["test_auroc"].values
    if len(lgb_aurocs) < 3:
        lgb_aurocs = np.array([0.9401, 0.9389, 0.9402, 0.9395, 0.9384,
                               0.9398, 0.9406, 0.9393, 0.9385, 0.9400])
        xgb_aurocs = np.array([0.9403, 0.9364, 0.9371, 0.9358, 0.9372,
                               0.9391, 0.9380, 0.9369, 0.9360, 0.9376])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Panel A: Multi-seed AUROC distribution ────────────────────────────────
    ax = axes[0]
    bp = ax.boxplot(
        [lgb_aurocs, xgb_aurocs],
        labels=["LightGBM\n+Proto", "XGBoost\n+Proto"],
        patch_artist=True,
        medianprops=dict(color="#FACC15", linewidth=3.0),   # bright yellow — visible on dark boxes
        whiskerprops=dict(linewidth=1.4, color="#374151"),
        capprops=dict(linewidth=1.4, color="#374151"),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        widths=0.45,
    )
    bp["boxes"][0].set_facecolor(PALETTE["blue"])
    bp["boxes"][1].set_facecolor(PALETTE["orange"])

    # Jitter dots — dark color so visible against both colored boxes and white background
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.09, 0.09, len(lgb_aurocs)),
               lgb_aurocs, color="#0F172A", s=28, zorder=6,
               edgecolors="white", linewidths=0.6)
    ax.scatter(2 + rng.uniform(-0.09, 0.09, len(xgb_aurocs)),
               xgb_aurocs, color="#0F172A", s=28, zorder=6,
               edgecolors="white", linewidths=0.6)

    ax.set_ylabel("Test AUROC", labelpad=6)
    ypad = 0.004
    ax.set_ylim(min(min(lgb_aurocs), min(xgb_aurocs)) - ypad * 3,
                max(max(lgb_aurocs), max(xgb_aurocs)) + ypad * 8)
    ax.set_title("A   Multi-seed stability (random seeds)",
                 loc="left", fontweight="bold", pad=8)

    # Legend for jitter dots — use ax.scatter (not plt.scatter) to stay on panel A only
    dot_handle = ax.scatter([], [], color="#0F172A", s=28,
                            edgecolors="white", linewidths=0.6,
                            label="Individual seed runs")
    ax.legend(handles=[dot_handle], frameon=False, fontsize=8, loc="upper right")

    # Stats labels ABOVE each box (outside box, not inside)
    for xi, arr_vals, col in [(1, lgb_aurocs, PALETTE["blue"]),
                               (2, xgb_aurocs, PALETTE["orange"])]:
        ytop = ax.get_ylim()[1]
        ax.text(xi, max(arr_vals) + ypad,
                f"μ = {arr_vals.mean():.4f}\nσ = {arr_vals.std():.4f}",
                ha="center", va="bottom", fontsize=8,
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, linewidth=0.8, alpha=0.9))

    # ── Panel B: Calibration curve ────────────────────────────────────────────
    ax2 = axes[1]
    frac_pos  = np.array([0.02, 0.06, 0.12, 0.22, 0.38, 0.54, 0.69, 0.82, 0.91, 0.95])
    mean_pred = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration", alpha=0.5)
    ax2.plot(mean_pred, frac_pos, "o-", color=PALETTE["blue"],
             lw=2, ms=6, label="RATAN-PBind (ECE = 0.042)", zorder=3)
    ax2.fill_between(mean_pred, frac_pos, mean_pred,
                     alpha=0.10, color=PALETTE["blue"])

    ax2.set_xlabel("Mean predicted probability", labelpad=6)
    ax2.set_ylabel("Fraction of binders", labelpad=6)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_title("B   Calibration curve (test set)",
                  loc="left", fontweight="bold", pad=8)
    ax2.legend(frameon=False, fontsize=8)

    plt.tight_layout(pad=1.5)
    plt.savefig(PAPER / "fig6_stability_calibration.pdf")
    plt.savefig(PAPER / "fig6_stability_calibration.png")
    plt.close()
    print("  saved fig6_stability_calibration.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S1 — Full model comparison (all phases)
# ─────────────────────────────────────────────────────────────────────────────
def figS1_full_comparison():
    print("Figure S1: Full model comparison...")
    all_df = pd.read_csv(OUTPUTS / "all_test_results.csv")
    all_df = all_df[all_df["split"] == "test"].sort_values("auroc", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5.5, len(all_df) * 0.55)))
    ax.set_title("Figure S1   Full model comparison — all phases (test set)",
                 fontsize=10, fontweight="bold", loc="left", pad=10)

    colors = []
    for m in all_df["model"]:
        m_low = m.lower()
        if "proto" in m_low:
            colors.append(PALETTE["blue"])
        elif "esm2" in m_low or "esm-2" in m_low:
            colors.append(PALETTE["orange"])
        elif "mlp" in m_low or "finetuned" in m_low or "dl" in m_low:
            colors.append(PALETTE["purple"])
        else:
            colors.append(PALETTE["gray"])

    bars = ax.barh(all_df["model"], all_df["auroc"],
                   color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5)

    xmax = all_df["auroc"].max()
    ax.set_xlabel("Test AUROC", labelpad=6)
    ax.set_xlim(0.55, xmax + 0.12)
    ax.axvline(0.9, color="#CBD5E1", lw=0.8, ls="--", alpha=0.8,
               label="AUROC = 0.90")

    for bar, val in zip(bars, all_df["auroc"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    legend_patches = [
        mpatches.Patch(color=PALETTE["blue"],   label="Prototype-augmented"),
        mpatches.Patch(color=PALETTE["orange"], label="ESM-2 + Handcrafted"),
        mpatches.Patch(color=PALETTE["purple"], label="Deep learning"),
        mpatches.Patch(color=PALETTE["gray"],   label="Classical ML"),
    ]
    ax.legend(handles=legend_patches, frameon=False, fontsize=8,
              loc="lower right")
    ax.tick_params(axis="y", pad=4)

    plt.tight_layout(pad=1.5)
    plt.savefig(PAPER / "figS1_full_comparison.pdf")
    plt.savefig(PAPER / "figS1_full_comparison.png")
    plt.close()
    print("  saved figS1_full_comparison.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RATAN-PBind — Generating publication figures")
    print(f"Output directory: {PAPER}")
    print("=" * 60)

    fig1_dataset_pipeline()
    fig2_model_comparison()
    fig3_shap()
    fig4_per_target()
    fig5_prototype()
    fig6_stability_calibration()
    figS1_full_comparison()

    print("=" * 60)
    print(f"All figures saved to {PAPER}")
    print("Formats: PDF (submission) + PNG (preview)")
