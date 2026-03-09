"""
protbind.designer — RATAN-PBind Generative AI protein binder design

Two engines:
  1. Directed Evolution  — RATAN-PBind as fitness oracle (genetic algorithm)
  2. ESM-2 Redesign      — ESM-2 protein LM as generative model (masked prediction)

Combined pipeline: Evolution → ESM-2 refinement → Groq AI interpretation

Publication note:
  ML predicts (WHAT) · ESM-2 generates (NEW SEQUENCES) · Groq explains (WHY IT WORKS)

This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

from __future__ import annotations
import numpy as np
import random
from typing import Callable, Optional

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


class ProtBindDesigner:
    """Generative protein binder design using RATAN-PBind as fitness oracle."""

    def __init__(self, predictor):
        self.pb = predictor

    # ── Internal utilities ────────────────────────────────────────────────

    def _score(self, sequence: str, target: str) -> float:
        try:
            return float(self.pb.predict(sequence, target)["probability"])
        except Exception:
            return 0.0

    def _mutate(self, sequence: str, n: int = 1) -> str:
        """Apply n random single-point substitutions."""
        seq = list(sequence)
        positions = random.sample(range(len(seq)), min(n, len(seq)))
        for pos in positions:
            options = [aa for aa in AA20 if aa != seq[pos]]
            seq[pos] = random.choice(options)
        return "".join(seq)

    def _crossover(self, a: str, b: str) -> str:
        """Single-point crossover between two equal-length sequences."""
        if len(a) != len(b):
            return a
        pt = random.randint(1, len(a) - 1)
        return a[:pt] + b[pt:]

    def _ensure_esm2(self, seq: str):
        """Trigger ESM-2 loading in the predictor if not yet loaded."""
        if not hasattr(self.pb, "_esm_model"):
            self.pb._get_esm2_embedding(seq)

    # ── Engine 1: Directed Evolution ──────────────────────────────────────

    def directed_evolution(
        self,
        target: str,
        seed_sequence: str,
        n_generations: int = 10,
        population_size: int = 15,
        mutation_rate: float = 0.05,
        elite_frac: float = 0.30,
        seed: int = 42,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """
        (μ+λ) directed evolution with RATAN-PBind as fitness oracle.

        Each generation:
          score all → select elite fraction → reproduce via mutation/crossover

        Parameters
        ----------
        target          : binding target name
        seed_sequence   : starting amino acid sequence
        n_generations   : number of generations
        population_size : number of sequences per generation
        mutation_rate   : fraction of residues mutated per offspring
        elite_frac      : fraction of top sequences kept each generation
        seed            : random seed for reproducibility
        progress_cb     : optional callback(gen, n_gen, best_prob, best_seq)

        Returns
        -------
        dict with keys: best_sequence, best_probability, seed_probability,
                        improvement, trajectory, top_sequences
        """
        random.seed(seed)
        np.random.seed(seed)

        seq     = seed_sequence.upper().strip()
        n_elite = max(2, int(population_size * elite_frac))
        n_mut   = max(1, int(len(seq) * mutation_rate))

        # Initialise population: seed + perturbed copies
        population = [seq]
        for _ in range(population_size - 1):
            population.append(self._mutate(seq, n=random.randint(1, max(2, n_mut * 3))))

        scored_cache: dict[str, float] = {}   # avoid re-scoring duplicates
        trajectory = []

        for gen in range(n_generations):
            # Score unseen sequences
            for s in population:
                if s not in scored_cache:
                    scored_cache[s] = self._score(s, target)
            scores = [scored_cache[s] for s in population]

            best_idx  = int(np.argmax(scores))
            best_prob = scores[best_idx]
            best_seq  = population[best_idx]
            mean_prob = float(np.mean(scores))

            trajectory.append({
                "generation": gen + 1,
                "best":       round(best_prob, 4),
                "mean":       round(mean_prob, 4),
                "best_seq":   best_seq,
            })

            if progress_cb:
                progress_cb(gen + 1, n_generations, best_prob, best_seq)

            # Selection: keep elite
            ranked_idx = np.argsort(scores)[::-1]
            elite = [population[i] for i in ranked_idx[:n_elite]]

            # Reproduction
            new_pop = list(elite)
            while len(new_pop) < population_size:
                parent = elite[len(new_pop) % len(elite)]
                if len(elite) >= 2 and random.random() < 0.3:
                    p2    = random.choice(elite)
                    child = self._crossover(parent, p2)
                    child = self._mutate(child, n=max(1, n_mut // 2))
                else:
                    child = self._mutate(parent, n=n_mut)
                new_pop.append(child)

            population = new_pop

        # Final ranking across everything seen
        seen_set: dict[str, float] = {}
        for s, p in sorted(scored_cache.items(), key=lambda x: x[1], reverse=True):
            seen_set[s] = p
        ranked = sorted(seen_set.items(), key=lambda x: x[1], reverse=True)

        seed_prob = scored_cache.get(seq, self._score(seq, target))
        return {
            "best_sequence":    ranked[0][0],
            "best_probability": ranked[0][1],
            "seed_probability": seed_prob,
            "improvement":      ranked[0][1] - seed_prob,
            "trajectory":       trajectory,
            "top_sequences":    ranked[:10],
        }

    # ── Engine 2: ESM-2 Masked Language Model Redesign ────────────────────

    def _shap_weak_positions(self, sequence: str, target: str) -> list[int]:
        """
        Identify residue positions whose amino acid type has negative SHAP.
        Uses AAC SHAP values: if having more of amino acid X hurts binding,
        mask all X positions in the sequence.
        """
        result = self.pb.predict(sequence, target)
        expl   = self.pb.explain(result, top_n=40)

        bad_types = set()
        for name, shap_val, _ in expl["top_features"]:
            if shap_val < -0.015 and name.startswith("aac_"):
                aa = name[4:]
                if aa in AA20:
                    bad_types.add(aa)

        positions = [i for i, aa in enumerate(sequence) if aa in bad_types]
        return positions

    def _esm2_fill_masked(
        self,
        sequence: str,
        positions: list[int],
        temperature: float = 1.0,
    ) -> str:
        """
        Run one ESM-2 masked language model forward pass.
        Replaces masked positions by sampling from ESM-2's token distribution.
        """
        import torch
        import torch.nn.functional as F

        pb     = self.pb
        model  = pb._esm_model
        alpha  = pb._esm_alphabet
        device = pb._esm_device

        seq = sequence.upper()[: 1022]

        # Tokenise
        toks = []
        if alpha.prepend_bos:
            toks.append(alpha.cls_idx)
        for aa in seq:
            idx = alpha.get_idx(aa)
            toks.append(idx if idx is not None else alpha.unk_idx)
        if alpha.append_eos:
            toks.append(alpha.eos_idx)

        toks_t = torch.tensor([toks], dtype=torch.long, device=device)

        # Apply mask tokens
        masked_t = toks_t.clone()
        offset   = 1 if alpha.prepend_bos else 0
        for pos in positions:
            if pos < len(seq):
                masked_t[0, pos + offset] = alpha.mask_idx

        # Forward pass
        with torch.no_grad():
            out    = model(masked_t, repr_layers=[], return_contacts=False)
        logits = out["logits"]          # (1, L, vocab_size)

        # Build valid AA index map (exclude special tokens)
        valid_aa  = list("ACDEFGHIKLMNPQRSTVWY")
        valid_idx = []
        for aa in valid_aa:
            idx = alpha.get_idx(aa)
            if idx is not None:
                valid_idx.append(idx)

        # Sample at each masked position
        new_seq = list(seq)
        for pos in positions:
            if pos >= len(seq):
                continue
            pos_logits = logits[0, pos + offset, :] / max(temperature, 1e-6)
            probs = F.softmax(pos_logits, dim=-1).cpu().float().numpy()

            vp = np.array([probs[i] for i in valid_idx], dtype=np.float64)
            vp = np.clip(vp, 0.0, None)
            s  = vp.sum()
            if s < 1e-12:
                continue
            vp /= s

            chosen    = int(np.random.choice(len(valid_idx), p=vp))
            new_seq[pos] = valid_aa[chosen]

        return "".join(new_seq)

    def esm2_redesign(
        self,
        sequence: str,
        target: str,
        n_rounds: int = 3,
        mask_fraction: float = 0.20,
        n_samples: int = 8,
        temperature: float = 1.0,
        seed: int = 42,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """
        ESM-2 protein language model guided sequence redesign.

        Round 0 (SHAP-guided): masks amino acid types with negative SHAP —
          directly targets the sequence features hurting the prediction.
        Rounds 1+ (random masking): randomly masks mask_fraction of positions
          and samples from ESM-2's learned protein sequence distribution.

        Parameters
        ----------
        sequence      : seed amino acid sequence
        target        : binding target name
        n_rounds      : number of redesign rounds
        mask_fraction : fraction of residues masked per sample
        n_samples     : number of ESM-2 completions sampled per round
        temperature   : sampling temperature (higher = more diverse)
        seed          : random seed
        progress_cb   : optional callback(round, n_rounds, best_prob, best_seq)

        Returns
        -------
        dict with keys: best_sequence, best_probability, seed_probability,
                        improvement, trajectory, top_sequences
        """
        random.seed(seed)
        np.random.seed(seed)

        seq = sequence.upper().strip()
        self._ensure_esm2(seq)

        baseline = self._score(seq, target)
        best_seq  = seq
        best_prob = baseline
        n_mask    = max(1, int(len(seq) * mask_fraction))

        trajectory = [{"round": 0, "best": round(baseline, 4), "seq": seq}]
        all_candidates: list[tuple[str, float]] = [(seq, baseline)]

        for rnd in range(n_rounds):
            if rnd == 0:
                # SHAP-guided: identify and mask amino acid types with negative impact
                shap_positions = self._shap_weak_positions(best_seq, target)
                positions = shap_positions if shap_positions else list(range(len(best_seq)))
            else:
                positions = list(range(len(best_seq)))

            round_best_prob = best_prob
            round_best_seq  = best_seq

            for _ in range(n_samples):
                mask_n   = min(n_mask, len(positions))
                mask_pos = random.sample(positions, mask_n)
                candidate = self._esm2_fill_masked(best_seq, mask_pos, temperature)
                prob = self._score(candidate, target)
                all_candidates.append((candidate, prob))
                if prob > round_best_prob:
                    round_best_prob = prob
                    round_best_seq  = candidate

            if round_best_prob > best_prob:
                best_prob = round_best_prob
                best_seq  = round_best_seq

            trajectory.append({
                "round":       rnd + 1,
                "best":        round(best_prob, 4),
                "round_best":  round(round_best_prob, 4),
                "seq":         best_seq,
            })

            if progress_cb:
                progress_cb(rnd + 1, n_rounds, best_prob, best_seq)

        # Deduplicate and rank
        seen: dict[str, float] = {}
        for s, p in all_candidates:
            if s not in seen or p > seen[s]:
                seen[s] = p
        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)

        return {
            "best_sequence":    ranked[0][0],
            "best_probability": ranked[0][1],
            "seed_probability": baseline,
            "improvement":      ranked[0][1] - baseline,
            "trajectory":       trajectory,
            "top_sequences":    ranked[:10],
        }

    # ── Combined pipeline ─────────────────────────────────────────────────

    def design(
        self,
        target: str,
        seed_sequence: str,
        mode: str = "combined",
        n_generations: int = 10,
        population_size: int = 15,
        n_rounds: int = 2,
        n_samples: int = 6,
        seed: int = 42,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """
        Full design pipeline.

        Modes:
          'evolution' : directed evolution only
          'esm2'      : ESM-2 redesign only
          'combined'  : evolution → ESM-2 refinement of top candidates (recommended)
        """
        if mode == "evolution":
            return self.directed_evolution(
                target=target, seed_sequence=seed_sequence,
                n_generations=n_generations, population_size=population_size,
                seed=seed, progress_cb=progress_cb,
            )

        if mode == "esm2":
            self._ensure_esm2(seed_sequence)
            return self.esm2_redesign(
                sequence=seed_sequence, target=target,
                n_rounds=n_rounds, n_samples=n_samples,
                seed=seed, progress_cb=progress_cb,
            )

        # ── Combined: evolution then ESM-2 refinement ─────────────────────
        # Stage 1: directed evolution
        evo = self.directed_evolution(
            target=target, seed_sequence=seed_sequence,
            n_generations=n_generations, population_size=population_size,
            seed=seed,
            progress_cb=(lambda g, ng, p, s:
                progress_cb(g, ng + n_rounds, p, s, "Evolution"))
            if progress_cb else None,
        )

        # Stage 2: ESM-2 refinement on top 3 evolved candidates
        self._ensure_esm2(seed_sequence)
        all_seqs: list[tuple[str, float]] = list(evo["top_sequences"])

        for i, (evolved_seq, _) in enumerate(evo["top_sequences"][:3]):
            esm_result = self.esm2_redesign(
                sequence=evolved_seq, target=target,
                n_rounds=n_rounds, n_samples=n_samples,
                seed=seed + i,
                progress_cb=(lambda r, nr, p, s:
                    progress_cb(n_generations + r, n_generations + nr, p, s, "ESM-2 Refinement"))
                if progress_cb else None,
            )
            all_seqs.extend(esm_result["top_sequences"])

        # Final dedup + ranking
        seen: dict[str, float] = {}
        for s, p in all_seqs:
            if s not in seen or p > seen[s]:
                seen[s] = p
        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)

        seed_prob = evo["seed_probability"]
        return {
            "best_sequence":    ranked[0][0],
            "best_probability": ranked[0][1],
            "seed_probability": seed_prob,
            "improvement":      ranked[0][1] - seed_prob,
            "trajectory":       evo["trajectory"],   # evolution trajectory shown
            "top_sequences":    ranked[:10],
            "evo_best":         evo["best_probability"],
        }
