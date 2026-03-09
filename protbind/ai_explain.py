"""
protbind.ai_explain — RATAN-PBind LLM-augmented interpretability via Groq Cloud

Bridges ML predictions with mechanistic scientific reasoning.
The ML model answers WHAT; the LLM answers WHY.

This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

from __future__ import annotations
from typing import Optional

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """You are RATAN-PBind AI, a computational biology expert specialising in
protein-protein interactions, de novo protein binder design, and structural biology.

You receive structured quantitative output from an ML binding predictor and your job
is to provide rigorous, molecularly specific scientific interpretation — not generic
advice. Every statement must be grounded in the data provided to you.

Rules:
- Be concise and precise. No filler sentences.
- Reference specific features, values, and residue-level reasoning where possible.
- When recommending changes, be chemically specific (e.g. "introduce a salt bridge",
  "reduce hydrophobic exposure at the interface", "increase alpha-helical propensity").
- Do not hallucinate structural details not supported by the data.
- Write at the level of a Methods section in a Nature paper."""


# ── Prompt builders ───────────────────────────────────────────────────────────

def _shap_table(top_features: list) -> str:
    lines = []
    for i, (name, shap_val, feat_val) in enumerate(top_features[:10]):
        direction = "increases" if shap_val > 0 else "decreases"
        lines.append(
            f"  {i+1:2d}. {name:<42s}  SHAP={shap_val:+.4f}  ({direction} binding)  value={feat_val:.4f}"
        )
    return "\n".join(lines)


def _physico_block(top_features: list, all_feat_cols: list, feat_vec) -> str:
    """Extract physicochemical values from feature vector."""
    import numpy as np
    col_idx = {c: i for i, c in enumerate(all_feat_cols)}
    fields = [
        ("seq_length",        "Sequence length (AA)"),
        ("molecular_weight",  "Molecular weight (Da)"),
        ("isoelectric_point", "Isoelectric point (pI)"),
        ("gravy",             "GRAVY hydrophobicity"),
        ("aromaticity",       "Aromaticity"),
        ("instability_index", "Instability index"),
        ("charge_ph7",        "Net charge at pH 7"),
        ("esmfold_plddt",     "ESMFold pLDDT (structural quality, /100)"),
        ("proteinmpnn_score", "ProteinMPNN score (lower = better seq-struct fit)"),
    ]
    lines = []
    for col, label in fields:
        if col in col_idx:
            v = float(feat_vec[col_idx[col]])
            if not np.isnan(v):
                lines.append(f"  {label:<45s} {v:.4f}")
    return "\n".join(lines) if lines else "  (not available)"


def build_prediction_prompt(
    sequence: str,
    target: str,
    result: dict,
    top_features: list,
    all_feat_cols: list,
    feat_vec,
) -> str:
    prob    = result["probability"]
    verdict = "BINDER" if result["predicted"] else "NON-BINDER"
    binding_word = "bind" if result["predicted"] else "not bind"

    shap_block   = _shap_table(top_features)
    physico_block = _physico_block(top_features, all_feat_cols, feat_vec)

    prompt = f"""Protein binding prediction — requires scientific interpretation.

═══ PREDICTION ════════════════════════════════════════════════════════
Sequence        : {sequence[:80]}{'...' if len(sequence) > 80 else ''}
Sequence length : {len(sequence)} amino acids
Binding target  : {target}
Probability     : {prob:.2%}
Verdict         : {verdict}
Confidence      : {result['confidence']}  (model uncertainty = {result['uncertainty']:.4f})
Decision thr.   : {result['threshold']:.2f}

═══ PROTOTYPE SIMILARITY (ESM-2 embedding space) ════════════════════
Similarity to known {target} binders  : {result['proto_cos_pos']:.4f}
Binder / non-binder ratio (proto_ratio): {result['proto_ratio']:.4f}
  (proto_ratio > 1 = sequence resembles binders more than non-binders)

═══ TOP SHAP FEATURES ════════════════════════════════════════════════
{shap_block}

═══ PHYSICOCHEMICAL PROPERTIES ══════════════════════════════════════
{physico_block}

═══ TASK ════════════════════════════════════════════════════════════
1. Write a 3–5 sentence mechanistic interpretation of WHY this sequence
   is predicted to {binding_word} {target}. Be specific — cite feature
   names and values. Explain the molecular basis, not just the statistics.

2. Provide exactly 3 concrete, chemically specific recommendations to
   {'strengthen this prediction and prepare for experimental validation' if result['predicted'] else 'redesign or modify this sequence to improve binding probability'}.
   Each recommendation must be actionable, not generic.

Format your response exactly as:

**Mechanistic Interpretation**
[your interpretation]

**Recommendations**
1. [first recommendation]
2. [second recommendation]
3. [third recommendation]
"""
    return prompt


def build_mutation_prompt(
    mutations: list,
    sequence: str,
    target: str,
    baseline_prob: float,
) -> str:
    if not mutations:
        return ""

    mut_lines = "\n".join(
        f"  {m['mutation']:<12s}  {m['original_prob']:.2%} → {m['mutant_prob']:.2%}  "
        f"(Δ = +{m['delta']:.2%})"
        for m in mutations
    )

    prompt = f"""Single-point mutation scan results for protein binder design.

═══ CONTEXT ══════════════════════════════════════════════════════════
Sequence     : {sequence[:80]}{'...' if len(sequence) > 80 else ''}
Length       : {len(sequence)} AA
Target       : {target}
Baseline prob: {baseline_prob:.2%}

═══ TOP BENEFICIAL MUTATIONS (ranked by binding improvement) ════════
{mut_lines}

═══ TASK ════════════════════════════════════════════════════════════
For each mutation listed, provide a brief biochemical rationale (1–2 sentences)
explaining WHY this substitution is predicted to improve binding to {target}.
Consider: charge complementarity, hydrophobicity, backbone flexibility (proline/glycine),
hydrogen bonding capacity, steric effects, and known binding interface properties.

Then write 2–3 sentences on a recommended experimental validation strategy for the
top-ranked mutations.

Format:

**Mutation Analysis**
[Mutation code]: [rationale]
[Mutation code]: [rationale]
... (for each mutation)

**Experimental Validation Strategy**
[2–3 sentences]
"""
    return prompt


def build_batch_prompt(
    n_binders: int,
    n_total: int,
    target: str,
    top_candidates: list,
    hit_rate: float,
) -> str:
    cand_lines = "\n".join(
        f"  {i+1}. prob={c['probability']:.2%}  proto_ratio={c.get('proto_ratio', 0):.3f}  "
        f"conf={c.get('confidence','?')}  seq={c.get('sequence','')[:30]}..."
        for i, c in enumerate(top_candidates)
    )

    prompt = f"""Batch protein binding prediction results — requires scientific summary.

═══ BATCH OVERVIEW ════════════════════════════════════════════════════
Target         : {target}
Total sequences: {n_total}
Predicted binders: {n_binders} ({hit_rate:.1%} hit rate)

═══ TOP CANDIDATES ════════════════════════════════════════════════════
{cand_lines}

═══ TASK ══════════════════════════════════════════════════════════════
1. Interpret the overall hit rate for {target} (is {hit_rate:.1%} high/low relative to
   typical binder design campaigns, which average 10–30%?).
2. Comment on what the top candidates' proto_ratio values suggest about
   their similarity to known binders.
3. Recommend which 1–3 candidates to prioritise for experimental follow-up
   and why.
4. Suggest one improvement to the design strategy based on this batch.

Be concise. Write at the level of a Methods/Results summary in a journal paper.
"""
    return prompt


# ── Main API ─────────────────────────────────────────────────────────────────

def call_groq(prompt: str, api_key: str, model: str,
              max_tokens: int = 600) -> str:
    """Call Groq Cloud and return the response text."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,   # lower temperature = more precise/reproducible
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI analysis unavailable: {e}]"


def ai_explain_prediction(
    sequence: str,
    target: str,
    result: dict,
    top_features: list,
    all_feat_cols: list,
    feat_vec,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate an LLM-powered mechanistic interpretation of a binding prediction.

    Parameters
    ----------
    sequence     : amino acid sequence
    target       : binding target name
    result       : dict returned by RatanPBind.predict()
    top_features : list of (name, shap_val, feat_val) from RatanPBind.explain()
    all_feat_cols: full feature column list (from RatanPBind.all_feat_cols)
    feat_vec     : full feature vector (from result['_feat_vec'])
    api_key      : Groq API key
    model        : Groq model name

    Returns
    -------
    str — markdown-formatted AI analysis
    """
    prompt = build_prediction_prompt(
        sequence, target, result, top_features, all_feat_cols, feat_vec)
    return call_groq(prompt, api_key, model, max_tokens=650)


def ai_explain_mutations(
    mutations: list,
    sequence: str,
    target: str,
    baseline_prob: float,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate LLM-powered biochemical rationale for predicted beneficial mutations.
    """
    if not mutations:
        return ""
    prompt = build_mutation_prompt(mutations, sequence, target, baseline_prob)
    return call_groq(prompt, api_key, model, max_tokens=500)


def ai_summarise_batch(
    results_df,
    target: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate an LLM-powered scientific summary of a batch prediction run.
    """
    n_total   = len(results_df)
    n_binders = int(results_df["predicted"].sum()) if "predicted" in results_df else 0
    hit_rate  = n_binders / n_total if n_total else 0

    top = results_df.head(5).to_dict("records")
    prompt = build_batch_prompt(n_binders, n_total, target, top, hit_rate)
    return call_groq(prompt, api_key, model, max_tokens=400)


def build_design_prompt(
    seed_sequence: str,
    best_sequence: str,
    target: str,
    seed_prob: float,
    best_prob: float,
    improvement: float,
    mode: str,
    top_sequences: list,
    best_result: dict,
    best_top_features: list,
) -> str:
    """Build Groq prompt for AI interpretation of a design run."""

    top_lines = "\n".join(
        f"  {i+1}. prob={p:.2%}  seq={s[:50]}{'...' if len(s)>50 else ''}"
        for i, (s, p) in enumerate(top_sequences[:5])
    )

    shap_lines = "\n".join(
        f"  {name:<42s}  SHAP={sv:+.4f}  value={fv:.4f}"
        for name, sv, fv in best_top_features[:8]
    )

    mode_desc = {
        "evolution": "Directed Evolution (genetic algorithm, RATAN-PBind oracle)",
        "esm2":      "ESM-2 Masked Language Model Redesign",
        "combined":  "Combined pipeline: Directed Evolution → ESM-2 Refinement",
    }.get(mode, mode)

    prompt = f"""Generative protein binder design results — requires scientific interpretation.

═══ DESIGN RUN ════════════════════════════════════════════════════════
Method       : {mode_desc}
Target       : {target}
Seed sequence: {seed_sequence[:60]}{'...' if len(seed_sequence)>60 else ''}
Best designed: {best_sequence[:60]}{'...' if len(best_sequence)>60 else ''}

═══ PERFORMANCE ═══════════════════════════════════════════════════════
Seed binding probability   : {seed_prob:.2%}
Best designed probability  : {best_prob:.2%}
Absolute improvement       : +{improvement:.2%}
Relative improvement       : +{improvement/max(seed_prob,0.001):.0%}

═══ TOP 5 DESIGNED SEQUENCES ══════════════════════════════════════════
{top_lines}

═══ BEST SEQUENCE — SHAP FEATURE ATTRIBUTION ═════════════════════════
{shap_lines}

═══ BEST SEQUENCE — KEY PROPERTIES ══════════════════════════════════
Binding probability : {best_result.get('probability', best_prob):.2%}
Confidence          : {best_result.get('confidence', 'N/A')}
Similarity to binders (proto_cos_pos): {best_result.get('proto_cos_pos', 0):.4f}
Binder/non-binder ratio (proto_ratio): {best_result.get('proto_ratio', 0):.4f}

═══ TASK ════════════════════════════════════════════════════════════
1. Explain in 3–4 sentences what the design algorithm achieved:
   - What sequence changes occurred (seed vs best)?
   - Why did binding probability improve?
   - What does the proto_ratio and SHAP profile tell us about
     the designed sequence's relationship to known {target} binders?

2. Assess the quality of the designed sequence:
   - Is the improvement substantial or marginal?
   - What is the risk of overfitting to the ML oracle?
   - Are there any properties that could limit experimental success?

3. Recommend a concrete experimental validation strategy (2–3 steps).

Format:
**Design Outcome**
[3–4 sentences]

**Quality Assessment**
[3–4 sentences]

**Experimental Validation**
1. [step]
2. [step]
3. [step]
"""
    return prompt


def ai_interpret_design(
    seed_sequence: str,
    best_sequence: str,
    target: str,
    seed_prob: float,
    best_prob: float,
    improvement: float,
    mode: str,
    top_sequences: list,
    best_result: dict,
    best_top_features: list,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate an LLM interpretation of a generative design run.
    Explains what changed, why it worked, and how to validate experimentally.
    """
    prompt = build_design_prompt(
        seed_sequence, best_sequence, target, seed_prob, best_prob,
        improvement, mode, top_sequences, best_result, best_top_features,
    )
    return call_groq(prompt, api_key, model, max_tokens=700)
