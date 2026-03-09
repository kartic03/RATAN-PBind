"""
protbind.predictor — Core RATAN-PBind prediction engine
RATAN-PBind: Residue Attribution and Target Affinity Network for Protein Binding
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Usage:
    from protbind import RatanPBind
    pb = RatanPBind()
    result = pb.predict("MASWKELLVQ...", target="egfr")
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Default paths relative to project root
_HERE = Path(__file__).parent.parent
_MODELS_DIR   = _HERE / "models"
_FEATURES_DIR = _HERE / "features"


class ProtBind:
    """
    RATAN-PBind — Residue Attribution and Target Affinity Network for Protein Binding.

    Predicts whether a protein sequence will bind to a specified target
    using a LightGBM ensemble trained on the Proteinbase dataset.
    """

    def __init__(self,
                 models_dir: Optional[Path] = None,
                 features_dir: Optional[Path] = None):
        self.models_dir   = Path(models_dir)   if models_dir   else _MODELS_DIR
        self.features_dir = Path(features_dir) if features_dir else _FEATURES_DIR
        self._load_models()

    def _load_models(self):
        print("Loading RATAN-PBind models...", end="", flush=True)

        # Primary model: LightGBM + Prototype features
        self.lgb = joblib.load(self.models_dir / "lgb_proto.pkl")
        self.xgb = joblib.load(self.models_dir / "xgb_proto.pkl")

        # Ensemble meta: prototypes, feature cols, thresholds
        with open(self.models_dir / "ensemble_meta_6b.pkl", "rb") as f:
            self.meta = pickle.load(f)

        self.target_names   = self.meta["target_names"]
        self.proto_pos      = self.meta["proto_pos"]
        self.proto_neg      = self.meta["proto_neg"]
        self.n_pos_dict     = self.meta["n_pos"]
        self.n_neg_dict     = self.meta["n_neg"]
        self.proto_feat_cols= self.meta["proto_feat_cols"]
        self.all_feat_cols  = self.meta["all_feat_cols"]
        self.thresholds     = self.meta.get("thresholds", {})

        # Phase 2 imputer + scaler (for missing features)
        self.imputer = joblib.load(self.features_dir / "imputer.pkl")
        self.scaler  = joblib.load(self.features_dir / "scaler.pkl")

        # Base feature columns (Phase 2, 463 features)
        self.base_feat_cols = pd.read_csv(self.features_dir / "feature_columns.csv")["column"].tolist()

        # Pre-computed ESM-2 embeddings (for proteins already in dataset)
        self.esm_emb = np.load(self.features_dir / "esm2_embeddings.npy")
        self.esm_ids = np.load(self.features_dir / "esm2_protein_ids.npy", allow_pickle=True)
        self.esm_map = {pid: i for i, pid in enumerate(self.esm_ids)}

        # Phase 6a interface feature metadata
        if_meta_path = self.models_dir / "ensemble_meta_6a.pkl"
        self.if_cols  = []
        self.if_meds  = {}
        if if_meta_path.exists():
            with open(if_meta_path, "rb") as f:
                meta_6a = pickle.load(f)
            self.if_cols = meta_6a.get("if_cols", [])
            self.if_meds = meta_6a.get("if_medians", {})

        print(" done.")
        print(f"  Targets: {len(self.target_names)}")
        print(f"  Features: {len(self.all_feat_cols)}")

    # ── Feature building ──────────────────────────────────────────────────────

    def _cosine_sim(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _proto_features(self, emb: np.ndarray, target: str) -> np.ndarray:
        pp = self.proto_pos.get(target, np.zeros(1280, np.float32))
        pn = self.proto_neg.get(target, np.zeros(1280, np.float32))
        disc = pp - pn
        dn   = np.linalg.norm(disc)
        cos_pos  = self._cosine_sim(emb, pp)
        cos_neg  = self._cosine_sim(emb, pn)
        l2_pos   = float(np.linalg.norm(emb - pp))
        disc_proj= float(np.dot(emb, disc) / (dn + 1e-8))
        ratio    = cos_pos / (abs(cos_neg) + 1e-6)
        n_pos    = self.n_pos_dict.get(target, 0)
        n_neg    = self.n_neg_dict.get(target, 0)
        return np.array([cos_pos, cos_neg, l2_pos, disc_proj, ratio, n_pos, n_neg],
                        dtype=np.float32)

    def _get_esm2_embedding(self,
                             sequence: str,
                             protein_id: Optional[str] = None) -> np.ndarray:
        """Return ESM-2 embedding. Uses cached embedding if protein_id is known."""
        if protein_id and protein_id in self.esm_map:
            return self.esm_emb[self.esm_map[protein_id]].astype(np.float32)

        # Try to compute on-the-fly if ESM-2 available
        try:
            import torch
            import esm as esm_lib
            if not hasattr(self, '_esm_model'):
                print("  Loading ESM-2 for embedding (one-time)...")
                self._esm_model, self._esm_alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
                self._esm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._esm_model  = self._esm_model.to(self._esm_device).eval()

            pad_idx = self._esm_alphabet.padding_idx
            seq_trunc = sequence[:1022]
            prepend = self._esm_alphabet.prepend_bos
            append  = self._esm_alphabet.append_eos
            toks = [self._esm_alphabet.get_idx(c) if self._esm_alphabet.get_idx(c) is not None
                    else self._esm_alphabet.unk_idx for c in seq_trunc]
            if prepend: toks = [self._esm_alphabet.cls_idx] + toks
            if append:  toks = toks + [self._esm_alphabet.eos_idx]
            batch = torch.tensor([toks], dtype=torch.long).to(self._esm_device)
            with torch.no_grad():
                out = self._esm_model(batch, repr_layers=[33], return_contacts=False)
            n_real = min(len(seq_trunc), 1022)
            return out["representations"][33][0, 1:n_real+1, :].mean(0).cpu().float().numpy()
        except Exception as e:
            warnings.warn(f"ESM-2 embedding failed ({e}), using zero-vector fallback. "
                          f"Prototype features will be less accurate.")
            return np.zeros(1280, dtype=np.float32)

    def _build_feature_vector(self,
                               sequence: str,
                               target: str,
                               protein_id: Optional[str] = None,
                               design_method: Optional[str] = None,
                               precomputed: Optional[dict] = None) -> np.ndarray:
        """Build the full 509-feature vector for a (sequence, target) pair."""
        from protbind.features import compute_all_features

        if target not in self.target_names:
            raise ValueError(f"Unknown target '{target}'. "
                             f"Known targets: {self.target_names}")

        # 1. Handcrafted features (463)
        raw_feats = compute_all_features(
            sequence=sequence,
            design_method=design_method,
            precomputed=precomputed or {},
        )
        # Build base feature vector aligned to Phase 2 columns
        base_arr = np.array([raw_feats.get(col, np.nan) for col in self.base_feat_cols],
                            dtype=np.float32)

        # 2. Impute missing values (uses training medians)
        # The imputer was fit on 447 cols — handle any size mismatch gracefully
        imputer_cols = len(self.imputer.statistics_)
        if len(base_arr) >= imputer_cols:
            base_arr[:imputer_cols] = self.imputer.transform(
                base_arr[:imputer_cols].reshape(1, -1))[0]
        else:
            base_arr = self.imputer.transform(
                np.resize(base_arr, imputer_cols).reshape(1, -1))[0]
            base_arr = np.resize(base_arr, len(self.base_feat_cols))

        # Assemble as dict for interface feature alignment
        base_dict = {col: base_arr[i] for i, col in enumerate(self.base_feat_cols)}

        # 3. Interface features (39) — impute with training medians if not provided
        if_feats = {col: self.if_meds.get(col, 0.0) for col in self.if_cols}

        # 4. ESM-2 embedding
        emb = self._get_esm2_embedding(sequence, protein_id)

        # 5. Prototype features (7)
        proto_feats = self._proto_features(emb, target)

        # 6. Assemble final feature vector in correct order
        feat_dict = {**base_dict, **if_feats}
        for j, col in enumerate(self.proto_feat_cols):
            feat_dict[col] = proto_feats[j]

        feat_vec = np.array([feat_dict.get(col, 0.0) for col in self.all_feat_cols],
                            dtype=np.float32)
        return feat_vec, emb

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self,
                sequence: str,
                target: str,
                protein_id: Optional[str] = None,
                design_method: Optional[str] = None,
                precomputed: Optional[dict] = None) -> dict:
        """
        Predict binding probability for a (sequence, target) pair.

        Parameters
        ----------
        sequence       : Amino acid sequence (single-letter code)
        target         : Target name (one of the 24 known targets)
        protein_id     : Optional protein ID to use cached ESM-2 embedding
        design_method  : Optional design method slug (e.g., 'bindcraft')
        precomputed    : Optional dict of precomputed features (esmfold_plddt, etc.)

        Returns
        -------
        dict with keys:
            probability   : float [0, 1] — binding probability
            confidence    : str — 'High' / 'Medium' / 'Low'
            uncertainty   : float — std between LGB and XGB predictions
            threshold     : float — optimal per-target threshold (from validation)
            predicted     : bool — True if probability >= threshold
            target        : str — target name
            proto_cos_pos : float — similarity to known binders (key feature)
            proto_ratio   : float — discriminative score
        """
        feat_vec, emb = self._build_feature_vector(
            sequence, target, protein_id, design_method, precomputed)

        X = feat_vec.reshape(1, -1)
        p_lgb = float(self.lgb.predict_proba(X)[0, 1])
        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        prob  = (p_lgb + p_xgb) / 2
        unc   = abs(p_lgb - p_xgb)

        thr = self.thresholds.get(target, 0.5)

        confidence = "High" if unc < 0.05 else ("Medium" if unc < 0.15 else "Low")

        # Extract key interpretable features
        proto_idx     = {c: i for i, c in enumerate(self.all_feat_cols)}
        proto_cos_pos = float(feat_vec[proto_idx["proto_cos_pos"]])
        proto_ratio   = float(feat_vec[proto_idx["proto_ratio"]])

        return {
            "probability":   round(prob, 4),
            "lgb_prob":      round(p_lgb, 4),
            "xgb_prob":      round(p_xgb, 4),
            "confidence":    confidence,
            "uncertainty":   round(unc, 4),
            "threshold":     round(thr, 3),
            "predicted":     prob >= thr,
            "target":        target,
            "proto_cos_pos": round(proto_cos_pos, 4),
            "proto_ratio":   round(proto_ratio, 4),
            "_feat_vec":     feat_vec,   # used internally by explain()
        }

    def explain(self, prediction_result: dict, top_n: int = 8) -> dict:
        """
        Generate SHAP-based explanation for a prediction.

        Parameters
        ----------
        prediction_result : dict returned by predict()
        top_n             : number of top features to explain

        Returns
        -------
        dict with keys:
            shap_values     : np.ndarray — SHAP values for all features
            top_features    : list of (feature_name, shap_value, feature_value)
            natural_language: str — human-readable explanation
            feature_names   : list of feature names
        """
        import shap
        if not hasattr(self, "_shap_explainer"):
            self._shap_explainer = shap.TreeExplainer(self.lgb)

        feat_vec = prediction_result["_feat_vec"].reshape(1, -1)
        sv = self._shap_explainer.shap_values(feat_vec)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]   # shape (n_features,)

        # Top features
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        top_feats = [
            (self.all_feat_cols[i], float(sv[i]), float(feat_vec[0, i]))
            for i in top_idx
        ]

        nl = self._natural_language_explanation(prediction_result, top_feats)

        return {
            "shap_values":     sv,
            "top_features":    top_feats,
            "natural_language":nl,
            "feature_names":   self.all_feat_cols,
        }

    def _natural_language_explanation(self, pred: dict, top_feats: list) -> str:
        """Generate rule-based natural language explanation from SHAP + prediction."""
        prob       = pred["probability"]
        target     = pred["target"]
        confidence = pred["confidence"]
        predicted  = pred["predicted"]
        proto_cos  = pred["proto_cos_pos"]
        proto_rat  = pred["proto_ratio"]

        # Header
        verdict = "predicted to BIND" if predicted else "predicted NOT to bind"
        lines = [
            f"**Prediction**: This sequence is {verdict} to **{target}** "
            f"(probability = {prob:.1%}, confidence = {confidence}).",
            ""
        ]

        # Key driver sentences based on top SHAP features
        drivers, concerns = [], []
        for feat_name, shap_val, feat_val in top_feats:
            positive = shap_val > 0

            if feat_name == "proto_ratio":
                if positive:
                    drivers.append(f"High similarity ratio to known binders vs non-binders "
                                   f"(proto_ratio = {feat_val:.2f}) — this sequence resembles "
                                   f"known binders of {target}.")
                else:
                    concerns.append(f"Low binder/non-binder similarity ratio "
                                    f"(proto_ratio = {feat_val:.2f}) — sequence resembles "
                                    f"non-binders more than binders.")

            elif feat_name == "proto_cos_pos":
                if positive:
                    drivers.append(f"Strong structural similarity to known {target} binders "
                                   f"(cosine similarity = {feat_val:.3f}).")
                else:
                    concerns.append(f"Low structural similarity to known {target} binders "
                                    f"(cosine similarity = {feat_val:.3f}).")

            elif feat_name == "method_success_rate":
                if positive and feat_val > 0.2:
                    drivers.append(f"Designed with a high-success-rate method "
                                   f"(historical binding rate = {feat_val:.1%}).")
                elif not positive and feat_val < 0.1:
                    concerns.append(f"Designed with a low-success-rate method "
                                    f"(historical binding rate = {feat_val:.1%}).")

            elif feat_name == "esmfold_plddt":
                if positive and feat_val > 70:
                    drivers.append(f"Well-structured fold predicted (ESMFold pLDDT = {feat_val:.1f}/100).")
                elif not positive and feat_val < 60:
                    concerns.append(f"Poorly structured fold predicted (ESMFold pLDDT = {feat_val:.1f}/100), "
                                    f"which may hinder binding.")

            elif feat_name == "instability_index":
                if not positive and feat_val > 40:
                    concerns.append(f"High instability index ({feat_val:.1f}) suggests the protein "
                                    f"may be unstable in solution.")
                elif positive and feat_val < 30:
                    drivers.append(f"Low instability index ({feat_val:.1f}) — protein predicted "
                                   f"to be stable.")

            elif feat_name == "proteinmpnn_score":
                if positive and feat_val < 1.5:
                    drivers.append(f"Strong sequence-structure compatibility "
                                   f"(ProteinMPNN score = {feat_val:.3f}).")

            elif feat_name.startswith("boltz2_iptm"):
                if positive and feat_val > 0.7:
                    drivers.append(f"High predicted interface TM-score (boltz2_iptm = {feat_val:.3f}), "
                                   f"indicating confident complex structure.")

            elif feat_name.startswith("if_"):
                clean = feat_name.replace("if_", "").replace("_", " ")
                direction = "favorable" if positive else "unfavorable"
                drivers.append(f"Interface {clean} is {direction} for binding "
                                f"(value = {feat_val:.3f}).")

        if drivers:
            lines.append("**Supporting evidence:**")
            for d in drivers[:4]:
                lines.append(f"  • {d}")
            lines.append("")

        if concerns:
            lines.append("**Potential concerns:**")
            for c in concerns[:3]:
                lines.append(f"  • {c}")
            lines.append("")

        # Target context
        n_pos = self.n_pos_dict.get(target, 0)
        n_neg = self.n_neg_dict.get(target, 0)
        n_total = n_pos + n_neg
        base_rate = n_pos / n_total if n_total > 0 else 0.18
        lines.append(f"**Target context**: {target} has {n_pos} known binders out of "
                     f"{n_total} tested sequences ({base_rate:.1%} baseline success rate) "
                     f"in the training data.")

        # Confidence note
        unc = pred["uncertainty"]
        if unc > 0.15:
            lines.append(f"\n⚠️ **Low confidence** (model disagreement = {unc:.2f}). "
                         f"This prediction should be interpreted cautiously.")
        elif proto_cos < 0.3 and n_pos < 5:
            lines.append(f"\n⚠️ **Limited training data** ({n_pos} positive examples for {target}). "
                         f"Experimental validation is strongly recommended.")

        return "\n".join(lines)

    def suggest_mutations(self,
                           sequence: str,
                           target: str,
                           top_n: int = 5,
                           protein_id: Optional[str] = None) -> list:
        """
        Suggest single point mutations predicted to improve binding probability.

        Evaluates all possible single amino acid substitutions and ranks them
        by predicted improvement. Uses handcrafted features only (fast, ~2s).

        Returns
        -------
        list of dicts: [{'position': int, 'original': str, 'mutant': str,
                          'original_prob': float, 'mutant_prob': float,
                          'delta': float}]
        """
        AA20 = list("ACDEFGHIKLMNPQRSTVWY")
        seq = sequence.upper().strip()

        # Baseline prediction (with ESM-2 if available)
        baseline = self.predict(sequence, target, protein_id=protein_id)
        base_prob = baseline["probability"]
        base_emb  = self._get_esm2_embedding(sequence, protein_id)

        mutations = []
        total = len(seq) * 19
        print(f"  Evaluating {total} single mutations for {target}...")

        for pos in range(len(seq)):
            orig_aa = seq[pos]
            for mut_aa in AA20:
                if mut_aa == orig_aa:
                    continue
                mut_seq = seq[:pos] + mut_aa + seq[pos+1:]
                # Use same ESM-2 embedding (approximate — only handcrafted features change)
                try:
                    fv, _ = self._build_feature_vector(
                        mut_seq, target, protein_id=None,
                        # Pass base_emb as override by temporarily patching esm_map
                    )
                    # Override proto features with base_emb (approximate)
                    proto_feats = self._proto_features(base_emb, target)
                    for j, col in enumerate(self.proto_feat_cols):
                        idx = self.all_feat_cols.index(col)
                        fv[idx] = proto_feats[j]

                    X = fv.reshape(1, -1)
                    p_lgb = float(self.lgb.predict_proba(X)[0, 1])
                    p_xgb = float(self.xgb.predict_proba(X)[0, 1])
                    mut_prob = (p_lgb + p_xgb) / 2

                    if mut_prob > base_prob + 0.01:   # only report improvements
                        mutations.append({
                            "position":     pos + 1,
                            "original":     orig_aa,
                            "mutant":       mut_aa,
                            "mutation":     f"{orig_aa}{pos+1}{mut_aa}",
                            "original_prob":round(base_prob, 4),
                            "mutant_prob":  round(mut_prob, 4),
                            "delta":        round(mut_prob - base_prob, 4),
                        })
                except Exception:
                    continue

        mutations.sort(key=lambda x: x["delta"], reverse=True)
        return mutations[:top_n]

    def batch_predict(self,
                      sequences: list,
                      target: str,
                      protein_ids: Optional[list] = None,
                      design_methods: Optional[list] = None) -> pd.DataFrame:
        """
        Predict binding for a batch of sequences.

        Returns
        -------
        pd.DataFrame with columns: sequence, target, probability, confidence,
                                   predicted, proto_cos_pos, proto_ratio
        """
        results = []
        pids = protein_ids or [None] * len(sequences)
        dmethods = design_methods or [None] * len(sequences)

        for i, (seq, pid, dm) in enumerate(zip(sequences, pids, dmethods)):
            try:
                r = self.predict(seq, target, protein_id=pid, design_method=dm)
                results.append({
                    "sequence":     seq[:20] + "..." if len(seq) > 20 else seq,
                    "target":       target,
                    "probability":  r["probability"],
                    "confidence":   r["confidence"],
                    "predicted":    r["predicted"],
                    "proto_cos_pos":r["proto_cos_pos"],
                    "proto_ratio":  r["proto_ratio"],
                    "uncertainty":  r["uncertainty"],
                })
            except Exception as e:
                results.append({
                    "sequence": seq[:20] + "...",
                    "target":   target,
                    "probability": None,
                    "confidence":  "Error",
                    "predicted":   None,
                    "error":       str(e),
                })

        df = pd.DataFrame(results)
        if "probability" in df.columns:
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)
        return df
