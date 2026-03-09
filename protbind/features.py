"""
protbind.features — RATAN-PBind feature extraction for new protein sequences
Computes all handcrafted features from a raw amino acid sequence.
This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import numpy as np
import re

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# Physicochemical lookup tables
MW_AA = {'A':89.09,'R':174.20,'N':132.12,'D':133.10,'C':121.16,'E':147.13,
         'Q':146.15,'G':75.03,'H':155.16,'I':131.17,'L':131.17,'K':146.19,
         'M':149.20,'F':165.19,'P':115.13,'S':105.09,'T':119.12,'W':204.23,
         'Y':181.19,'V':117.15}
PKA_COOH = 3.65
PKA_NH2  = 8.20
PKA_SIDE = {'D':3.9,'E':4.07,'H':6.04,'C':8.14,'Y':10.46,'K':10.54,'R':12.48}
HYDRO    = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,
            'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
            'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
INSTAB   = {'A':1.0,'R':0.5,'N':1.0,'D':1.0,'C':1.0,'E':1.0,'Q':1.0,
            'G':1.0,'H':1.0,'I':1.0,'L':1.0,'K':1.0,'M':1.0,'F':1.0,
            'P':1.0,'S':1.0,'T':1.0,'W':1.0,'Y':1.0,'V':1.0}
AROM     = set("FWY")
DESIGN_METHODS = ['bindcraft','boltzgen','dsm-synteract','evodiff','mosaic',
                  'pro-1','protrl','rfdiffusion','protpardelle-proteinmpnn',
                  'esm2-optimization','co-timed','bindcraft2','latentx',
                  'proteinhunter','many-steps']


def aa_composition(seq: str) -> dict:
    """20 amino acid composition features (fraction)."""
    seq = seq.upper()
    n = max(len(seq), 1)
    return {f"aac_{aa}": seq.count(aa) / n for aa in AA20}


def dipeptide_composition(seq: str) -> dict:
    """400 dipeptide composition features."""
    seq = seq.upper()
    n = max(len(seq) - 1, 1)
    counts = {}
    for i in range(len(seq) - 1):
        dp = seq[i:i+2]
        if all(c in AA20 for c in dp):
            counts[dp] = counts.get(dp, 0) + 1
    return {f"dpc_{a}{b}": counts.get(f"{a}{b}", 0) / n
            for a in AA20 for b in AA20}


def physicochemical_features(seq: str) -> dict:
    """Physicochemical features: MW, pI, GRAVY, length, aromaticity, instability."""
    seq = seq.upper()
    valid = [aa for aa in seq if aa in AA20]
    n = max(len(valid), 1)

    mw = sum(MW_AA.get(aa, 110.0) for aa in valid) - (n - 1) * 18.02
    gravy = sum(HYDRO.get(aa, 0.0) for aa in valid) / n
    arom  = sum(1 for aa in valid if aa in AROM) / n

    # Isoelectric point (binary search)
    def charge_at_ph(ph):
        c = (10**(-ph) / (10**(-ph) + 10**(-PKA_NH2)))
        c -= (10**(-PKA_COOH) / (10**(-PKA_COOH) + 10**(-ph)))
        for aa, pka in PKA_SIDE.items():
            cnt = valid.count(aa)
            if aa in ('D','E','C','Y'):
                c -= cnt * (10**(-pka) / (10**(-pka) + 10**(-ph)))
            else:
                c += cnt * (10**(-ph) / (10**(-ph) + 10**(-pka)))
        return c

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
    pi = (lo + hi) / 2

    # Simple instability index approximation (Guruprasad 1990)
    instab = sum(INSTAB.get(aa, 1.0) for aa in valid) / n * 10

    return {
        "seq_length":        len(seq),
        "molecular_weight":  mw,
        "isoelectric_point": pi,
        "gravy":             gravy,
        "aromaticity":       arom,
        "instability_index": instab,
        "charge_ph7":        charge_at_ph(7.0),
    }


def design_method_features(method: str | None,
                            method_success_rates: dict | None = None) -> dict:
    """One-hot design method + historical success rate."""
    method = (method or "other").lower().strip()
    feats = {}
    matched = "other"
    for m in DESIGN_METHODS:
        flag = 1 if m in method else 0
        feats[f"method_{m}"] = flag
        if flag:
            matched = m
    feats["method_other"] = 1 if matched == "other" else 0
    rate = (method_success_rates or {}).get(matched, 0.12)
    feats["method_success_rate"] = rate
    return feats


def compute_all_features(sequence: str,
                          design_method: str | None = None,
                          method_success_rates: dict | None = None,
                          boltz2_features: dict | None = None,
                          precomputed: dict | None = None) -> dict:
    """
    Compute the full 463-feature vector for a new sequence.
    Missing Boltz2/precomputed features are left as NaN → imputed later.
    """
    seq = sequence.upper().strip()
    feats = {}
    feats.update(dipeptide_composition(seq))
    feats.update(aa_composition(seq))
    feats.update(design_method_features(design_method, method_success_rates))
    feats.update(physicochemical_features(seq))

    # Boltz2 structural features (target-specific, usually not available for new seqs)
    boltz2_cols = ["boltz2_iptm","boltz2_ipsae","boltz2_min_ipsae",
                   "boltz2_complex_iplddt","boltz2_plddt","boltz2_complex_plddt",
                   "boltz2_ptm","boltz2_complex_pde","boltz2_lis",
                   "boltz2_pdockq","boltz2_pdockq2",
                   "shape_complimentarity_boltz2_binder_ss"]
    for col in boltz2_cols:
        feats[col] = (boltz2_features or {}).get(col, np.nan)

    # Precomputed ESMFold / ProteinMPNN features
    precomp_cols = ["esmfold_plddt","proteinmpnn_score","proteinmpnn_seq_recovery",
                    "redesigned_proteinmpnn_score"]
    for col in precomp_cols:
        feats[col] = (precomputed or {}).get(col, np.nan)

    return feats
