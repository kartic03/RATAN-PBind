"""
RATAN-PBind — Residue Attribution and Target Affinity Network for Protein Binding
Gradio Web Application

This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os, warnings, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from protbind import ProtBind, KNOWN_TARGETS
from protbind.ai_explain import (
    ai_explain_prediction,
    ai_explain_mutations,
    ai_summarise_batch,
    ai_interpret_design,
)
from protbind.designer import ProtBindDesigner

print("Initialising RATAN-PBind...")
pb       = ProtBind()
designer = ProtBindDesigner(pb)
print("Ready.\n")

DEMO_SEQUENCES = {
    "Nipah binder (known)":
        ("MASWKELLVQNKNQFNLERSELTNGFLKPIVKVVKKLPEEVLAERIRKAFG", "nipah-glycoprotein-g"),
    "EGFR binder (known)":
        ("MLPMKKNTELKKLLEELENFKQAVPRAKLKFLADRQYKRHLKQADRQYKR", "egfr"),
    "PD-L1 candidate":
        ("MAQVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEW", "pd-l1"),
}

# ── Groq AI (primary) ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

def groq_available() -> bool:
    return bool(GROQ_API_KEY)

def ask_groq(messages: list, system: str = "") -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=full_messages, max_tokens=512, temperature=0.7)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ── Ollama (fallback) ─────────────────────────────────────────────────────────
def ollama_available():
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except:
        return False

def ask_ollama(prompt: str, system: str = "") -> str:
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "llama3.2:3b", "prompt": prompt,
                                "system": system, "stream": False}, timeout=60)
        return r.json().get("response", "No response.")
    except Exception as e:
        return f"Error: {e}"

# ── Plot helpers ──────────────────────────────────────────────────────────────
PALETTE = {
    "positive": "#2563A8",   # confident blue
    "negative": "#C0392B",   # muted red
    "neutral":  "#94A3B8",   # slate
    "bg":       "#FFFFFF",
    "grid":     "#F1F5F9",
    "text":     "#1E293B",
    "subtext":  "#64748B",
}

def _apply_base_style(ax, fig):
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")
    ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

def shap_bar_chart(top_features: list, target: str):
    names  = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    short = []
    for n in names:
        if   n.startswith("dpc_"):    n = n[4:] + "  (dipeptide)"
        elif n.startswith("aac_"):    n = n[4:] + "  (composition)"
        elif n.startswith("if_"):     n = n[3:] + "  (interface)"
        elif n.startswith("method_"): n = n[7:] + "  (method)"
        elif n.startswith("proto_"):  n = n[6:] + "  (prototype)"
        short.append(n[:40])

    colors = [PALETTE["positive"] if v > 0 else PALETTE["negative"] for v in values]
    y_pos  = np.arange(len(short))

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(short) * 0.42)))
    ax.barh(y_pos[::-1], values[::-1], color=colors[::-1],
            alpha=0.88, height=0.62, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short[::-1], fontsize=8.5, color=PALETTE["text"])
    ax.axvline(0, color=PALETTE["subtext"], lw=0.9)
    ax.set_xlabel("SHAP value — contribution to binding probability",
                  fontsize=9, color=PALETTE["subtext"])
    ax.set_title(f"Feature contributions  ·  target: {target}",
                 fontsize=10, color=PALETTE["text"], fontweight="semibold", pad=10)
    pos_patch = mpatches.Patch(color=PALETTE["positive"], alpha=0.88, label="increases binding")
    neg_patch = mpatches.Patch(color=PALETTE["negative"], alpha=0.88, label="decreases binding")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8,
              framealpha=0.0, labelcolor=PALETTE["subtext"])
    _apply_base_style(ax, fig)
    ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    plt.tight_layout(pad=1.4)
    return fig

def probability_gauge(prob: float, target: str, threshold: float):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    bar_color = PALETTE["positive"] if prob >= threshold else PALETTE["negative"]
    ax.barh([0], [prob],       color=bar_color, height=0.38, zorder=3, alpha=0.9)
    ax.barh([0], [1 - prob],   left=[prob], color=PALETTE["grid"], height=0.38, zorder=2)
    ax.axvline(threshold, color=PALETTE["subtext"], lw=1.2, ls="--", zorder=4,
               label=f"threshold {threshold:.0%}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"],
                       fontsize=9, color=PALETTE["subtext"])
    verdict = "Predicted binder" if prob >= threshold else "Predicted non-binder"
    ax.set_title(f"{verdict}  —  {prob:.1%} binding probability  ·  {target}",
                 fontsize=10, color=PALETTE["text"], fontweight="semibold", pad=9)
    ax.legend(fontsize=8, framealpha=0.0, labelcolor=PALETTE["subtext"], loc="lower right")
    _apply_base_style(ax, fig)
    ax.yaxis.grid(False)
    plt.tight_layout(pad=1.2)
    return fig

# ── Backend functions ─────────────────────────────────────────────────────────
def single_predict(sequence: str, target: str, design_method: str, load_demo: str):
    if load_demo and load_demo != "Select a demo sequence":
        sequence, target = DEMO_SEQUENCES[load_demo]

    sequence = sequence.strip().upper()
    if not sequence:
        return None, "Enter a protein sequence to begin.", None
    if not target:
        return None, "Select a binding target.", None

    try:
        result      = pb.predict(sequence, target, design_method=design_method or None)
        explanation = pb.explain(result, top_n=10)

        gauge    = probability_gauge(result["probability"], target, result["threshold"])
        shap_fig = shap_bar_chart(explanation["top_features"], target)

        prob      = result["probability"]
        conf      = result["confidence"]
        predicted = result["predicted"]
        verdict   = "Binder" if predicted else "Non-binder"
        conf_color = {"High": "#16A34A", "Medium": "#D97706", "Low": "#DC2626"}.get(conf, "#64748B")

        # ── AI interpretation (Groq) or rule-based fallback ───────────────
        if groq_available():
            ai_text = ai_explain_prediction(
                sequence       = sequence,
                target         = target,
                result         = result,
                top_features   = explanation["top_features"],
                all_feat_cols  = pb.all_feat_cols,
                feat_vec       = result["_feat_vec"],
                api_key        = GROQ_API_KEY,
                model          = GROQ_MODEL,
            )
            ai_label  = "AI Analysis"
            ai_badge  = ('<span style="font-size:10px; font-weight:600; '
                         'text-transform:uppercase; letter-spacing:0.5px; '
                         'color:#1D4ED8; background:#EFF6FF; '
                         'padding:2px 7px; border-radius:4px;">Groq · llama-3.3-70b</span>')
        else:
            ai_text  = explanation["natural_language"]
            ai_label = "Analysis"
            ai_badge = ('<span style="font-size:10px; color:#94A3B8; '
                        'background:#F1F5F9; padding:2px 7px; border-radius:4px;">'
                        'rule-based</span>')

        summary = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1E293B;padding:4px 0;">

  <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
    <span style="font-size:32px;font-weight:700;letter-spacing:-1px;color:{'#16A34A' if predicted else '#DC2626'};">{prob:.1%}</span>
    <div>
      <div style="font-size:13px;font-weight:700;color:{'#16A34A' if predicted else '#DC2626'};text-transform:uppercase;letter-spacing:0.6px;">{verdict}</div>
      <div style="font-size:12px;color:#64748B;">for target &nbsp;<strong>{target}</strong></div>
    </div>
  </div>

  <table style="width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:18px;">
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;width:55%;">Confidence</td>
      <td style="padding:5px 0;font-weight:600;color:{conf_color};">{conf}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Model uncertainty</td>
      <td style="padding:5px 0;font-weight:500;">{result['uncertainty']:.4f}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Decision threshold</td>
      <td style="padding:5px 0;font-weight:500;">{result['threshold']:.2f}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Similarity to known binders</td>
      <td style="padding:5px 0;font-weight:500;">{result['proto_cos_pos']:.4f}</td>
    </tr>
    <tr>
      <td style="padding:5px 0;color:#94A3B8;">Binder / non-binder ratio</td>
      <td style="padding:5px 0;font-weight:500;">{result['proto_ratio']:.4f}</td>
    </tr>
  </table>

  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <span style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#94A3B8;">{ai_label}</span>
    {ai_badge}
  </div>
  <div style="font-size:13px;line-height:1.8;color:#374151;">{md_to_html(ai_text)}</div>

</div>
"""
        return gauge, summary, shap_fig

    except Exception as e:
        import traceback
        return None, f"<p style='color:#DC2626;font-size:13px;'>Error: {e}</p>", None


def md_to_html(text: str) -> str:
    """Convert minimal markdown (**bold**, numbered lists) to HTML for display."""
    import re, html as _html
    # Escape any raw HTML first, then convert markdown
    text = _html.escape(text)
    # **bold** → <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # *italic* → <em>  (single asterisk, not adjacent to word chars on both sides already consumed)
    text = re.sub(r'\*([^*\n]+?)\*', r'<em>\1</em>', text)
    # Numbered list items get a small top margin for readability
    text = re.sub(r'^(\d+\. )', r'<br><span style="font-weight:600;color:#1D4ED8;">\1</span>', text, flags=re.MULTILINE)
    # Blank lines → paragraph breaks
    text = re.sub(r'\n{2,}', '<br><br>', text)
    text = text.replace('\n', '<br>')
    return text


def batch_analyze(file, target: str, seq_col: str):
    if file is None:
        return None, "Upload a CSV file to begin."
    try:
        df  = pd.read_csv(file.name)
        col = seq_col.strip() or "sequence"
        if col not in df.columns:
            return None, f"Column '{col}' not found. Available columns: {', '.join(df.columns)}"
        sequences = df[col].dropna().astype(str).tolist()
        if not sequences:
            return None, "No sequences found in the file."
        if len(sequences) > 500:
            return None, "Batch limited to 500 sequences. Please split your file."

        results   = pb.batch_predict(sequences, target)
        n_binders = int(results["predicted"].sum()) if "predicted" in results else 0
        n_total   = len(results)
        hit_rate  = n_binders / n_total if n_total else 0

        if groq_available():
            summary = ai_summarise_batch(
                results_df = results,
                target     = target,
                api_key    = GROQ_API_KEY,
                model      = GROQ_MODEL,
            )
            summary = f"**{n_binders}/{n_total} predicted binders** ({hit_rate:.1%} hit rate)\n\n" + summary
        else:
            top3 = results.head(3)
            top_lines = "\n".join(
                f"  {i+1}. {row['sequence']}  —  {row['probability']:.1%}  ({row['confidence']})"
                for i, (_, row) in enumerate(top3.iterrows())
            )
            note = ("\nNo predicted binders found." if n_binders == 0
                    else f"\nHigh hit rate ({hit_rate:.1%})." if hit_rate > 0.5 else "")
            summary = (f"**{n_binders}/{n_total}** predicted binders for **{target}** "
                       f"({hit_rate:.1%})\n\n**Top candidates:**\n{top_lines}{note}")

        return results, summary
    except Exception as e:
        return None, f"Error: {e}"


def mutation_advisor(sequence: str, target: str, top_n: int):
    sequence = sequence.strip().upper()
    if not sequence:
        return None, "Enter a protein sequence."
    if len(sequence) > 300:
        return None, "Sequence too long for mutation scan (max 300 residues). Trim to the binding region."
    try:
        baseline = pb.predict(sequence, target)
        muts = pb.suggest_mutations(sequence, target, top_n=int(top_n))

        if not muts:
            if groq_available():
                from protbind.ai_explain import call_groq, _SYSTEM
                no_mut_prompt = (
                    f"A single-point mutation scan of a {len(sequence)}-residue binder against "
                    f"{target} found no beneficial substitutions. Baseline binding probability: "
                    f"{baseline['probability']:.2%}. Proto_ratio: {baseline['proto_ratio']:.3f}. "
                    f"In 2–3 sentences, explain what this means and give 2 specific next steps "
                    f"for improving binding (e.g. multi-point mutations, redesign strategy, "
                    f"different scaffold). Be specific, not generic."
                )
                advice = call_groq(no_mut_prompt, GROQ_API_KEY, GROQ_MODEL, max_tokens=250)
            else:
                advice = (
                    f"No beneficial single mutations found for **{target}**.\n\n"
                    f"Baseline probability: {baseline['probability']:.1%}\n\n"
                    "Consider multi-point mutations, a different scaffold, "
                    "or a high-success design method (protrl, mosaic)."
                )
            return None, advice

        df = pd.DataFrame(muts)[["mutation", "original_prob", "mutant_prob", "delta"]]
        df.columns = ["Mutation", "Original", "Mutant", "Improvement"]

        if groq_available():
            advice = ai_explain_mutations(
                mutations     = muts,
                sequence      = sequence,
                target        = target,
                baseline_prob = baseline["probability"],
                api_key       = GROQ_API_KEY,
                model         = GROQ_MODEL,
            )
        else:
            advice = f"**Top {len(muts)} mutations for {target}:**\n\n"
            for m in muts:
                advice += (f"- **{m['mutation']}** — "
                           f"{m['original_prob']:.1%} → {m['mutant_prob']:.1%} "
                           f"(+{m['delta']:.1%})\n")
            advice += "\nExperimental validation is required to confirm predictions."

        return df, advice
    except Exception as e:
        return None, f"Error: {e}"


# ── Design ───────────────────────────────────────────────────────────────────
def trajectory_plot(trajectory: list, mode: str):
    """Plot binding probability over generations/rounds."""
    if not trajectory:
        return None

    key   = "generation" if "generation" in trajectory[0] else "round"
    x     = [t[key] for t in trajectory]
    best  = [t["best"] for t in trajectory]
    mean  = [t.get("mean", t["best"]) for t in trajectory]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, best, color=PALETTE["positive"], lw=2,   marker="o", ms=4,
            label="Best", zorder=3)
    ax.plot(x, mean, color=PALETTE["neutral"],  lw=1.2, ls="--",
            label="Population mean", zorder=2)
    ax.fill_between(x, mean, best, color=PALETTE["positive"], alpha=0.08)

    ax.set_xlabel("Generation" if key == "generation" else "Round",
                  fontsize=9, color=PALETTE["subtext"])
    ax.set_ylabel("Binding probability", fontsize=9, color=PALETTE["subtext"])
    ax.set_title(f"Design trajectory  ·  {mode}",
                 fontsize=10, color=PALETTE["text"], fontweight="semibold", pad=9)
    ax.set_ylim(0, min(1.05, max(best) * 1.25 + 0.05))
    ax.legend(fontsize=8, framealpha=0.0, labelcolor=PALETTE["subtext"])
    _apply_base_style(ax, fig)
    plt.tight_layout(pad=1.3)
    return fig


def run_design(
    seed_sequence: str,
    target: str,
    mode: str,
    n_generations: int,
    population_size: int,
    n_rounds: int,
    n_samples: int,
    progress: gr.Progress = gr.Progress(),
):
    seed_sequence = seed_sequence.strip().upper()
    if not seed_sequence:
        return None, None, None, "Enter a seed sequence to begin."
    if len(seed_sequence) < 10:
        return None, None, None, "Sequence too short (minimum 10 residues)."
    if len(seed_sequence) > 400:
        return None, None, None, "Sequence too long for design (max 400 residues)."

    mode_key = {"Directed Evolution": "evolution",
                "ESM-2 Redesign":     "esm2",
                "Combined (recommended)": "combined"}.get(mode, "combined")

    total_steps = n_generations if mode_key == "evolution" \
        else n_rounds if mode_key == "esm2" \
        else n_generations + n_rounds
    progress(0, desc="Initialising design run …")

    def cb(step, total, prob, seq, stage=""):
        desc = f"{stage}  ·  step {step}/{total}  ·  best {prob:.1%}"
        progress(step / max(total, 1), desc=desc.strip(" · "))

    try:
        result = designer.design(
            target         = target,
            seed_sequence  = seed_sequence,
            mode           = mode_key,
            n_generations  = int(n_generations),
            population_size= int(population_size),
            n_rounds       = int(n_rounds),
            n_samples      = int(n_samples),
            progress_cb    = cb,
        )
    except Exception as e:
        return None, None, None, f"Design error: {e}"

    progress(1.0, desc="Scoring and interpreting results …")

    best_seq  = result["best_sequence"]
    best_prob = result["best_probability"]
    seed_prob = result["seed_probability"]
    impr      = result["improvement"]

    # Score and explain best sequence with full RATAN-PBind pipeline
    best_result = pb.predict(best_seq, target)
    best_expl   = pb.explain(best_result, top_n=10)

    # Trajectory plot
    traj_fig = trajectory_plot(result["trajectory"], mode)

    # Top sequences table
    top_df = pd.DataFrame(
        [(s[:50] + ("…" if len(s) > 50 else ""), round(p, 4))
         for s, p in result["top_sequences"]],
        columns=["Sequence", "Binding Probability"],
    )

    # AI interpretation of the design run
    if groq_available():
        ai_text = ai_interpret_design(
            seed_sequence     = seed_sequence,
            best_sequence     = best_seq,
            target            = target,
            seed_prob         = seed_prob,
            best_prob         = best_prob,
            improvement       = impr,
            mode              = mode_key,
            top_sequences     = result["top_sequences"],
            best_result       = best_result,
            best_top_features = best_expl["top_features"],
            api_key           = GROQ_API_KEY,
            model             = GROQ_MODEL,
        )
        ai_badge = ('<span style="font-size:10px;font-weight:600;color:#1D4ED8;'
                    'background:#EFF6FF;padding:2px 7px;border-radius:4px;">'
                    'Groq · llama-3.3-70b</span>')
    else:
        ai_text  = (f"Design complete. Best sequence achieved {best_prob:.1%} "
                    f"binding probability ({'+' if impr >= 0 else ''}{impr:.1%} vs seed). "
                    f"Add GROQ_API_KEY for AI interpretation.")
        ai_badge = '<span style="font-size:10px;color:#94A3B8;">rule-based</span>'

    conf_color = {"High": "#16A34A", "Medium": "#D97706", "Low": "#DC2626"}.get(
        best_result.get("confidence", ""), "#64748B")

    summary_html = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1E293B;padding:4px 0;">

  <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
    <span style="font-size:32px;font-weight:700;letter-spacing:-1px;color:{'#16A34A' if impr > 0 else '#64748B'};">
      {'+' if impr >= 0 else ''}{impr:.1%}
    </span>
    <div>
      <div style="font-size:13px;font-weight:700;color:{'#16A34A' if impr > 0 else '#64748B'};text-transform:uppercase;letter-spacing:0.6px;">
        {'Improvement' if impr > 0 else 'No improvement'}
      </div>
      <div style="font-size:12px;color:#64748B;">binding probability for {target}</div>
    </div>
  </div>

  <table style="width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:18px;">
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;width:55%;">Seed probability</td>
      <td style="padding:5px 0;font-weight:500;">{seed_prob:.2%}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Best designed probability</td>
      <td style="padding:5px 0;font-weight:600;color:{'#16A34A' if best_prob > seed_prob else '#DC2626'};">{best_prob:.2%}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Confidence</td>
      <td style="padding:5px 0;font-weight:500;color:{conf_color};">{best_result.get('confidence','N/A')}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Similarity to known binders</td>
      <td style="padding:5px 0;font-weight:500;">{best_result.get('proto_cos_pos', 0):.4f}</td>
    </tr>
    <tr style="border-bottom:1px solid #F1F5F9;">
      <td style="padding:5px 0;color:#94A3B8;">Binder / non-binder ratio</td>
      <td style="padding:5px 0;font-weight:500;">{best_result.get('proto_ratio', 0):.4f}</td>
    </tr>
    <tr>
      <td style="padding:5px 0;color:#94A3B8;">Best designed sequence</td>
      <td style="padding:5px 0;font-family:monospace;font-size:11px;word-break:break-all;">{best_seq}</td>
    </tr>
  </table>

  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <span style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#94A3B8;">AI Design Interpretation</span>
    {ai_badge}
  </div>
  <div style="font-size:13px;line-height:1.8;color:#374151;">{md_to_html(ai_text)}</div>

</div>
"""
    return traj_fig, top_df, summary_html, ""


# ── AI Chat ───────────────────────────────────────────────────────────────────
CHAT_SYSTEM = """You are RATAN-PBind AI, an expert scientific assistant specialising in
protein-protein binding prediction. Help users interpret results, understand
protein biochemistry, and improve their designs. Be concise and precise.

RATAN-PBind (Residue Attribution and Target Affinity Network for Protein Binding) uses
LightGBM with ESM-2 prototype embeddings (651M parameter language model).
Trained on Proteinbase (Adaptyv Bio, ODC-BY). Supports 24 human and viral targets.
Key features: proto_ratio (cosine similarity to known binders vs non-binders in ESM-2
embedding space — the strongest single predictor), esmfold_pLDDT (predicted structural
quality), method_success_rate (historical success rate of design method used)."""

def chat_respond(message: str, history: list):
    if not message.strip():
        return history, ""

    if groq_available():
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
        messages.append({"role": "user", "content": message})
        response = ask_groq(messages, system=CHAT_SYSTEM)
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    if ollama_available():
        ctx = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in history[-8:]
        )
        response = ask_ollama(f"{ctx}\nUser: {message}\nAssistant:", system=CHAT_SYSTEM)
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    # Rule-based fallback
    msg = message.lower()
    if any(w in msg for w in ["hello", "hi", "hey", "start"]):
        response = ("Hello. I am RATAN-PBind AI, your protein binding prediction assistant. "
                    "Ask me to interpret a prediction result, explain a feature, or suggest "
                    "how to improve your binder design.")
    elif any(w in msg for w in ["auroc", "accuracy", "performance", "score"]):
        response = ("RATAN-PBind achieves AUROC 0.94 on matched-target evaluation and 0.66 on "
                    "zero-shot cross-target generalization. The model is stable: 0.9395 ± 0.0054 "
                    "across 5 random seeds. AUPRC = 0.76, approximately 4× above the random baseline.")
    elif any(w in msg for w in ["proto", "prototype", "proto_ratio"]):
        response = ("proto_ratio is the most important feature in RATAN-PBind (SHAP rank 1). "
                    "It equals cosine similarity to training binders divided by cosine similarity "
                    "to non-binders, measured in ESM-2 embedding space. A high value means your "
                    "sequence structurally resembles known binders of that target.")
    elif "target" in msg and any(w in msg for w in ["list", "which", "support"]):
        response = "RATAN-PBind supports 24 targets:\n" + ", ".join(KNOWN_TARGETS)
    elif any(t in msg for t in KNOWN_TARGETS):
        tgt = next(t for t in KNOWN_TARGETS if t in msg)
        n_pos = pb.n_pos_dict.get(tgt, 0)
        n_neg = pb.n_neg_dict.get(tgt, 0)
        rate  = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        response = (f"{tgt}: {n_pos} known binders, {n_neg} non-binders "
                    f"({rate:.1%} success rate in training data).")
    elif any(w in msg for w in ["fail", "low", "poor", "wrong", "bad"]):
        response = ("Common reasons for a low binding prediction:\n"
                    "1. Low proto_ratio — sequence does not resemble known binders in ESM-2 space\n"
                    "2. Low esmfold_pLDDT — disordered or poorly folded structure predicted\n"
                    "3. High instability index — protein may not be stable in solution\n"
                    "4. Low-success design method — consider bindcraft, protrl, or mosaic\n\n"
                    "Use the Mutation Advisor tab to identify beneficial single substitutions.")
    elif any(w in msg for w in ["improve", "better", "mutat", "design", "increase"]):
        response = ("To improve binding probability:\n"
                    "1. Run the Mutation Advisor to find beneficial single point mutations\n"
                    "2. Review negative SHAP values in the prediction chart\n"
                    "3. Aim for esmfold_pLDDT > 80 and instability index < 30\n"
                    "4. Use a design method with a higher success rate (protrl 39%, mosaic 81%)")
    elif any(w in msg for w in ["shap", "feature", "important", "contribut"]):
        response = ("Top 5 features by SHAP importance:\n"
                    "1. proto_ratio — binder/non-binder similarity ratio\n"
                    "2. method_success_rate — historical success of the design method\n"
                    "3. proto_disc_proj — discriminative projection onto binder space\n"
                    "4. proto_l2_pos — L2 distance to the binder prototype\n"
                    "5. esmfold_pLDDT — structural quality score\n\n"
                    "Prototype features occupy 5 of the top 9 positions.")
    elif any(w in msg for w in ["cite", "citation", "paper", "reference", "journal"]):
        response = ("Paper in preparation:\n"
                    "RATAN-PBind: Residue Attribution and Target Affinity Network for Protein Binding\n\n"
                    "Dataset: Proteinbase by Adaptyv Bio (ODC-BY license).")
    elif any(w in msg for w in ["use", "install", "how", "start", "begin", "python"]):
        response = ("Python API:\n"
                    "  from protbind import RatanPBind\n"
                    "  pb = RatanPBind()\n"
                    "  result = pb.predict('MASWKELLVQ...', target='egfr')\n"
                    "  explanation = pb.explain(result)\n"
                    "  print(explanation['natural_language'])\n\n"
                    "Or use this web interface directly.")
    else:
        response = ("I can help with:\n"
                    "- Interpreting a prediction result\n"
                    "- Explaining features (proto_ratio, esmfold_pLDDT, etc.)\n"
                    "- Target-specific information\n"
                    "- Design improvement strategies\n"
                    "- Citations and dataset information")

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* ═══════════════════════════════════════════════════════════
   RATAN-PBind — Force light theme in all browsers/OS modes
   ═══════════════════════════════════════════════════════════ */

/* 1. Lock color scheme so browser dark-mode never applies */
html, html.dark, html[data-theme] {
    color-scheme: light only !important;
}

/* 2. Gradient background on the app container */
body,
.gradio-container,
.gradio-container.dark,
.main.svelte-1kyws56,
.wrap.svelte-1kyws56 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: linear-gradient(150deg, #E0F7F4 0%, #EBF8FF 50%, #E8F5E9 100%) !important;
    background-attachment: fixed !important;
    color: #1E293B !important;
    min-height: 100vh;
}

/* 3. Override ALL .dark-class rules Gradio injects */
.dark body,
.dark .gradio-container,
.dark .block,
.dark .panel,
.dark .wrap,
.dark input,
.dark textarea,
.dark select,
.dark button,
.dark label,
.dark span,
.dark div {
    background-color: unset;
    color: unset;
    border-color: unset;
}

/* Header */
.pb-header {
    padding: 32px 0 20px;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 8px;
}
.pb-title {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.3px;
    margin: 0 0 4px;
}
.pb-subtitle {
    font-size: 13px;
    color: #64748B;
    margin: 0;
}

/* Tabs — force light colors regardless of browser theme */
.tab-nav, .tab-nav > * {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
.tab-nav button, .tab-nav button * {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #64748B !important;
    background: transparent !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    transition: all 0.15s ease;
}
.tab-nav button:hover {
    color: #1D4ED8 !important;
    background: #F8FAFC !important;
}
.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom-color: #1D4ED8 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* Inputs — force light background/text everywhere */
label, label span, label > span {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
input, textarea, select,
.block, .form,
[data-testid="textbox"] textarea,
[data-testid="dropdown"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border-radius: 6px !important;
    border-color: #D1D5DB !important;
    font-size: 13px !important;
}
input::placeholder, textarea::placeholder {
    color: #94A3B8 !important;
}
textarea {
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 12.5px !important;
    line-height: 1.6 !important;
}

/* Dropdown / select popup */
.dropdown-arrow, ul.options, ul.options li,
.option-string, .wrap-inner {
    background: #FFFFFF !important;
    color: #1E293B !important;
}
ul.options li:hover, ul.options li.selected {
    background: #EFF4FF !important;
    color: #1D4ED8 !important;
}

/* Block cards — white so they float on top of the gradient */
.block, .form, .panel {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}
/* Tab content area — transparent so gradient bleeds through */
.tabitem, .tabitem > .gap, .gap {
    background: transparent !important;
}

/* All plain text — inherit from container so it's always dark */
p, span, div, h1, h2, h3, h4, li, td, th {
    color: inherit;
}

/* Buttons */
button.primary {
    background: #1D4ED8 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    padding: 10px 22px !important;
    transition: background 0.15s ease !important;
}
button.primary:hover {
    background: #1E40AF !important;
}
button.secondary {
    background: transparent !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    color: #374151 !important;
    font-size: 13px !important;
}

/* Section labels */
.section-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94A3B8;
    margin-bottom: 12px;
}

/* Plots */
.gr-plot { border-radius: 8px !important; border: 1px solid #E2E8F0 !important; }

/* Dataframe */
.gr-dataframe table { font-size: 12.5px !important; }
.gr-dataframe th {
    background: #F8FAFC !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
    color: #64748B !important;
}

/* Chat */
.gr-chatbot { border-radius: 8px !important; border: 1px solid #E2E8F0 !important; }
.gr-chatbot .message { font-size: 13px !important; line-height: 1.6 !important; }
.ai-status {
    font-size: 12px;
    color: #64748B;
    padding: 8px 12px;
    background: #F1F5F9;
    border-radius: 6px;
    border-left: 3px solid #1D4ED8;
    margin-bottom: 12px;
}

/* Footer */
footer { display: none !important; }
.pb-footer {
    text-align: center;
    font-size: 11.5px;
    color: #94A3B8;
    padding: 20px 0 10px;
    border-top: 1px solid #E2E8F0;
    margin-top: 16px;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
LIGHT_THEME = gr.themes.Base().set(
    # ── backgrounds ──────────────────────────────────────────
    body_background_fill="linear-gradient(150deg,#EFF6FF 0%,#F8FAFC 50%,#F0FDF4 100%)",
    body_background_fill_dark="linear-gradient(150deg,#EFF6FF 0%,#F8FAFC 50%,#F0FDF4 100%)",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F8FAFC",
    background_fill_secondary_dark="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    panel_background_fill="#FFFFFF",
    panel_background_fill_dark="#FFFFFF",
    # ── text ─────────────────────────────────────────────────
    body_text_color="#1E293B",
    body_text_color_dark="#1E293B",
    body_text_color_subdued="#64748B",
    body_text_color_subdued_dark="#64748B",
    block_label_text_color="#374151",
    block_label_text_color_dark="#374151",
    block_title_text_color="#0F172A",
    block_title_text_color_dark="#0F172A",
    block_info_text_color="#64748B",
    block_info_text_color_dark="#64748B",
    # ── borders ──────────────────────────────────────────────
    block_border_color="#E2E8F0",
    block_border_color_dark="#E2E8F0",
    border_color_primary="#E2E8F0",
    border_color_primary_dark="#E2E8F0",
    border_color_accent="#D1D5DB",
    border_color_accent_dark="#D1D5DB",
    panel_border_color="#E2E8F0",
    panel_border_color_dark="#E2E8F0",
    # ── inputs ───────────────────────────────────────────────
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_background_fill_focus="#F8FAFC",
    input_background_fill_focus_dark="#F8FAFC",
    input_background_fill_hover="#F8FAFC",
    input_background_fill_hover_dark="#F8FAFC",
    input_border_color="#D1D5DB",
    input_border_color_dark="#D1D5DB",
    # ── accent ───────────────────────────────────────────────
    color_accent_soft="#EFF4FF",
    color_accent_soft_dark="#EFF4FF",
    # ── checkboxes / labels ───────────────────────────────────
    checkbox_background_color="#FFFFFF",
    checkbox_background_color_dark="#FFFFFF",
    checkbox_border_color="#D1D5DB",
    checkbox_border_color_dark="#D1D5DB",
    checkbox_label_background_fill="#F8FAFC",
    checkbox_label_background_fill_dark="#F8FAFC",
    checkbox_label_text_color="#374151",
    checkbox_label_text_color_dark="#374151",
    # ── table ────────────────────────────────────────────────
    table_text_color="#1E293B",
    table_text_color_dark="#1E293B",
    # ── blocks ───────────────────────────────────────────────
    block_label_background_fill="#F8FAFC",
    block_label_background_fill_dark="#F8FAFC",
    block_label_border_color="#E2E8F0",
    block_label_border_color_dark="#E2E8F0",
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    # ── code ─────────────────────────────────────────────────
    code_background_fill="#F1F5F9",
    code_background_fill_dark="#F1F5F9",
)

with gr.Blocks(title="RATAN-PBind", css=CSS, theme=LIGHT_THEME) as demo:

    gr.HTML("""
    <div class="pb-header">
      <p class="pb-title">RATAN-PBind</p>
      <p class="pb-subtitle">
        Residue Attribution and Target Affinity Network for Protein Binding &nbsp;·&nbsp;
        ESM-2 embeddings &nbsp;·&nbsp; 24 targets &nbsp;·&nbsp; 2,517 proteins
      </p>
    </div>
    """)

    with gr.Tabs():

        # ── Prediction ────────────────────────────────────────────────────────
        with gr.Tab("Prediction"):
            with gr.Row(equal_height=False):

                # Left panel — inputs
                with gr.Column(scale=2, min_width=300):
                    gr.HTML('<p class="section-label">Input</p>')
                    demo_selector = gr.Dropdown(
                        choices=["Select a demo sequence"] + list(DEMO_SEQUENCES.keys()),
                        value="Select a demo sequence",
                        label="Load example",
                        container=True)
                    seq_input = gr.Textbox(
                        label="Amino acid sequence",
                        placeholder="MASWKELLVQNKNQFNLERS...",
                        lines=5, max_lines=12)
                    tgt_input = gr.Dropdown(
                        choices=KNOWN_TARGETS,
                        value="nipah-glycoprotein-g",
                        label="Binding target")
                    method_input = gr.Textbox(
                        label="Design method  (optional)",
                        placeholder="bindcraft, boltzgen, rfdiffusion …")
                    predict_btn = gr.Button("Run prediction", variant="primary", size="lg")

                # Right panel — results
                with gr.Column(scale=3, min_width=400):
                    gr.HTML('<p class="section-label">Result</p>')
                    gauge_out   = gr.Plot(label="", show_label=False)
                    summary_out = gr.HTML()

            gr.HTML('<p class="section-label" style="margin-top:20px;">Feature contributions</p>')
            shap_out = gr.Plot(label="", show_label=False)

            predict_btn.click(
                fn=single_predict,
                inputs=[seq_input, tgt_input, method_input, demo_selector],
                outputs=[gauge_out, summary_out, shap_out],
            )
            demo_selector.change(
                fn=lambda d: (
                    DEMO_SEQUENCES[d][0] if d != "Select a demo sequence" else "",
                    DEMO_SEQUENCES[d][1] if d != "Select a demo sequence" else "nipah-glycoprotein-g"
                ),
                inputs=[demo_selector],
                outputs=[seq_input, tgt_input],
            )

        # ── Batch ─────────────────────────────────────────────────────────────
        with gr.Tab("Batch"):
            gr.HTML("""
            <p style="font-size:13px; color:#64748B; margin-bottom:16px;">
            Upload a CSV file containing protein sequences. All candidates are
            scored and ranked by predicted binding probability.
            </p>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-label">Input</p>')
                    batch_file   = gr.File(label="CSV file", file_types=[".csv"])
                    batch_target = gr.Dropdown(choices=KNOWN_TARGETS,
                                               value="egfr", label="Binding target")
                    batch_col    = gr.Textbox(label="Sequence column name",
                                             value="sequence")
                    batch_btn    = gr.Button("Run batch", variant="primary")

                with gr.Column(scale=1):
                    gr.HTML('<p class="section-label">Summary</p>')
                    batch_summary = gr.Markdown()

            gr.HTML('<p class="section-label" style="margin-top:16px;">Results</p>')
            batch_table = gr.DataFrame(interactive=False, wrap=True)

            batch_btn.click(
                fn=batch_analyze,
                inputs=[batch_file, batch_target, batch_col],
                outputs=[batch_table, batch_summary],
            )

        # ── Mutation Advisor ──────────────────────────────────────────────────
        with gr.Tab("Mutation Advisor"):
            gr.HTML("""
            <p style="font-size:13px; color:#64748B; margin-bottom:16px;">
            Scan all single amino acid substitutions and identify those predicted
            to increase binding probability. Recommended for sequences under 150 residues.
            </p>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<p class="section-label">Input</p>')
                    mut_seq    = gr.Textbox(label="Amino acid sequence",
                                           placeholder="MASWKELLVQ...", lines=5)
                    mut_target = gr.Dropdown(choices=KNOWN_TARGETS,
                                            value="egfr", label="Binding target")
                    mut_topn   = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                                          label="Mutations to return")
                    mut_btn    = gr.Button("Scan mutations", variant="primary")

                with gr.Column(scale=1):
                    gr.HTML('<p class="section-label">Recommendations</p>')
                    mut_advice = gr.Markdown()

            gr.HTML('<p class="section-label" style="margin-top:16px;">Top mutations</p>')
            mut_table = gr.DataFrame(interactive=False)

            mut_btn.click(
                fn=mutation_advisor,
                inputs=[mut_seq, mut_target, mut_topn],
                outputs=[mut_table, mut_advice],
            )

        # ── Design ───────────────────────────────────────────────────────────
        with gr.Tab("Design"):
            gr.HTML("""
            <p style="font-size:13px;color:#64748B;margin-bottom:4px;">
            Generative AI protein binder design. Two engines work together:
            <strong>Directed Evolution</strong> uses RATAN-PBind as a fitness oracle to search
            sequence space via a genetic algorithm.
            <strong>ESM-2 Redesign</strong> uses the ESM-2 protein language model (trained on
            250 million sequences) to propose chemically plausible alternatives at weak positions.
            Groq AI interprets the final design trajectory.
            </p>
            """)

            with gr.Row(equal_height=False):
                # Input panel
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<p class="section-label">Seed &amp; Target</p>')
                    design_seq = gr.Textbox(
                        label="Seed sequence",
                        placeholder="Paste a starting sequence (or use a known non-binder to optimise)…",
                        lines=5)
                    design_tgt = gr.Dropdown(
                        choices=KNOWN_TARGETS, value="egfr", label="Binding target")

                    gr.HTML('<p class="section-label" style="margin-top:14px;">Design mode</p>')
                    design_mode = gr.Radio(
                        choices=["Directed Evolution", "ESM-2 Redesign", "Combined (recommended)"],
                        value="Combined (recommended)",
                        label="")

                    with gr.Accordion("Parameters", open=False):
                        d_ngen  = gr.Slider(5, 30,  value=10, step=1,
                                            label="Generations (Evolution)")
                        d_npop  = gr.Slider(5, 30,  value=15, step=1,
                                            label="Population size")
                        d_nrnd  = gr.Slider(1, 6,   value=2,  step=1,
                                            label="Redesign rounds (ESM-2)")
                        d_nsamp = gr.Slider(3, 15,  value=6,  step=1,
                                            label="Samples per round (ESM-2)")

                    design_btn = gr.Button("Run Design", variant="primary", size="lg")
                    design_err = gr.Textbox(visible=False, show_label=False)

                # Results panel
                with gr.Column(scale=2, min_width=420):
                    gr.HTML('<p class="section-label">Result</p>')
                    design_summary = gr.HTML()

            gr.HTML('<p class="section-label" style="margin-top:18px;">Optimisation trajectory</p>')
            design_traj = gr.Plot(show_label=False)

            gr.HTML('<p class="section-label" style="margin-top:18px;">Top designed sequences</p>')
            design_table = gr.DataFrame(interactive=False, wrap=True)

            design_btn.click(
                fn=run_design,
                inputs=[design_seq, design_tgt, design_mode,
                        d_ngen, d_npop, d_nrnd, d_nsamp],
                outputs=[design_traj, design_table, design_summary, design_err],
            )

        # ── AI Assistant ──────────────────────────────────────────────────────
        with gr.Tab("AI Assistant"):
            if groq_available():
                status_html = (f'<div class="ai-status">Groq Cloud &nbsp;·&nbsp; '
                               f'{GROQ_MODEL}</div>')
            elif ollama_available():
                status_html = '<div class="ai-status">Ollama &nbsp;·&nbsp; llama3.2:3b</div>'
            else:
                status_html = ('<div class="ai-status" style="border-color:#D97706;">'
                               'Rule-based mode &nbsp;·&nbsp; add GROQ_API_KEY to .env for AI</div>')

            gr.HTML(status_html)
            chatbot  = gr.Chatbot(height=440, show_label=False)
            chat_msg = gr.Textbox(placeholder="Ask about a prediction, a target, or how to improve your design …",
                                  show_label=False, lines=1)
            with gr.Row():
                chat_send  = gr.Button("Send", variant="primary", scale=3)
                chat_clear = gr.Button("Clear", variant="secondary", scale=1)

            chat_send.click(chat_respond, [chat_msg, chatbot], [chatbot, chat_msg])
            chat_msg.submit(chat_respond, [chat_msg, chatbot], [chatbot, chat_msg])
            chat_clear.click(lambda: ([], ""), outputs=[chatbot, chat_msg])

        # ── About ─────────────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(f"""
### RATAN-PBind

**Residue Attribution and Target Affinity Network for Protein Binding**

A machine learning tool for predicting protein-protein binding across 24 human and viral targets,
trained on experimental data from the Proteinbase dataset. RATAN-PBind integrates gradient-boosted
ensemble models with large language model augmentation (Groq / Llama-3.3-70b) to bridge statistical
binding prediction and mechanistic molecular interpretation.

**Model**
LightGBM with 509 features derived from ESM-2 protein language model embeddings,
sequence composition, physicochemical properties, and Boltz2 structural predictions.
The prototype similarity features — encoding how closely a candidate resembles known binders
of each target in ESM-2 embedding space — are the most informative predictors.

**Generative Design**
RATAN-PBind includes a generative design engine combining directed evolution (RATAN-PBind as
fitness oracle) and ESM-2 masked language model redesign, with AI interpretation of design
trajectories via Groq Cloud.

**Targets ({len(KNOWN_TARGETS)})**
{", ".join(KNOWN_TARGETS)}

**Dataset**
This work used Proteinbase by Adaptyv Bio under the ODC-BY license.
5,253 proteins · 2,643 labeled pairs · 40,479 experimental evaluations.

**Citation**
Paper in preparation.
*RATAN-PBind: Residue Attribution and Target Affinity Network for Protein Binding*

**License**
Model: MIT · Dataset: Open Data Commons Attribution (ODC-By)
""")

    gr.HTML("""
    <div class="pb-footer">
      RATAN-PBind &nbsp;·&nbsp;
      This work used Proteinbase by Adaptyv Bio under ODC-BY license
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
