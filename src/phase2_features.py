"""
Phase 2: Feature Engineering
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEAT_DIR     = PROJECT_ROOT / "features"
FEAT_DIR.mkdir(exist_ok=True)

# ── Amino acid property tables ────────────────────────────────────────────────

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy index
HYDROPATHY = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# pKa values for charge at pH 7 (N-term, C-term, side chains)
PKA_SIDE = {"D": 3.9, "E": 4.1, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}
PKA_NTERM = 8.0
PKA_CTERM = 3.1

# Instability index DIWV table (Guruprasad et al. 1990) — dipeptide weights
# Full 400-entry table; only non-zero entries listed, rest default to 0
DIWV = {
    "WW": 1.0, "WC": 1.0, "WM": 24.68, "WH": 24.68, "WY": 1.0,
    "WF": 1.0, "WQ": 1.0, "WI": 1.0, "WR": 1.0, "WN": 13.34,
    "WV": -7.49, "WS": 1.0, "WP": 1.0, "WE": 1.0, "WT": -14.03,
    "WK": 1.0, "WA": -14.03, "WG": -7.49, "WL": 13.34, "WD": 1.0,
    "CW": 24.68, "CC": 1.0, "CM": 33.60, "CH": 33.60, "CY": 1.0,
    "CF": 1.0, "CQ": -6.54, "CI": 1.0, "CR": 1.0, "CN": 1.0,
    "CV": -6.54, "CS": 1.0, "CP": 20.26, "CE": 1.0, "CT": 33.60,
    "CK": 1.0, "CA": 1.0, "CG": 1.0, "CL": 20.26, "CD": 20.26,
    "MW": 1.0, "MC": 1.0, "MM": -1.88, "MH": 58.28, "MY": 24.68,
    "MF": 1.0, "MQ": -6.54, "MI": 1.0, "MR": -2.85, "MN": 1.0,
    "MV": 1.0, "MS": 44.94, "MP": 44.94, "ME": 1.0, "MT": -1.88,
    "MK": 1.0, "MA": 13.34, "MG": 1.0, "ML": 1.0, "MD": 1.0,
    "HW": -1.88, "HC": 1.0, "HM": 1.0, "HH": 1.0, "HY": 44.94,
    "HF": -9.37, "HQ": 1.0, "HI": 44.94, "HR": 1.0, "HN": 24.68,
    "HV": 1.0, "HS": 1.0, "HP": -1.88, "HE": 1.0, "HT": -6.54,
    "HK": 24.68, "HA": 1.0, "HG": -9.37, "HL": 1.0, "HD": 1.0,
    "YW": -9.37, "YC": 1.0, "YM": 44.94, "YH": 13.34, "YY": 13.34,
    "YF": 1.0, "YQ": 1.0, "YI": 1.0, "YR": -15.91, "YN": 1.0,
    "YV": 1.0, "YS": 1.0, "YP": 13.34, "YE": -6.54, "YT": -7.49,
    "YK": 1.0, "YA": 24.68, "YG": -7.49, "YL": 1.0, "YD": 1.0,
    "FW": 1.0, "FC": 1.0, "FM": 1.0, "FH": 1.0, "FY": 33.60,
    "FF": 1.0, "FQ": 1.0, "FI": 1.0, "FR": 1.0, "FN": 1.0,
    "FV": 1.0, "FS": 1.0, "FP": 20.26, "FE": 1.0, "FT": 1.0,
    "FK": -14.03, "FA": 1.0, "FG": 1.0, "FL": 1.0, "FD": 13.34,
    "QW": 1.0, "QC": -6.54, "QM": 1.0, "QH": 1.0, "QY": -6.54,
    "QF": -6.54, "QQ": 20.26, "QI": 1.0, "QR": 1.0, "QN": 1.0,
    "QV": -6.54, "QS": 44.94, "QP": 20.26, "QE": 20.26, "QT": 1.0,
    "QK": 1.0, "QA": 1.0, "QG": 1.0, "QL": 1.0, "QD": 1.0,
    "IW": 1.0, "IC": 1.0, "IM": 1.0, "IH": 13.34, "IY": 1.0,
    "IF": 1.0, "IQ": 1.0, "II": 1.0, "IR": 1.0, "IN": 1.0,
    "IV": -6.54, "IS": 1.0, "IP": -1.88, "IE": 44.94, "IT": 1.0,
    "IK": -7.49, "IA": 1.0, "IG": 1.0, "IL": 20.26, "ID": 1.0,
    "RW": 58.28, "RC": 1.0, "RM": 1.0, "RH": 20.26, "RY": -6.54,
    "RF": 1.0, "RQ": 20.26, "RI": 1.0, "RR": 58.28, "RN": 13.34,
    "RV": 1.0, "RS": 44.94, "RP": 20.26, "RE": 1.0, "RT": 1.0,
    "RK": 1.0, "RA": 1.0, "RG": -7.49, "RL": 1.0, "RD": 1.0,
    "NW": -9.37, "NC": 1.0, "NM": 1.0, "NH": 1.0, "NY": 1.0,
    "NF": -14.03, "NQ": 1.0, "NI": 44.94, "NR": 1.0, "NN": 1.0,
    "NV": 1.0, "NS": 1.0, "NP": -1.88, "NE": 1.0, "NT": -7.49,
    "NK": 24.68, "NA": 1.0, "NG": -14.03, "NL": 1.0, "ND": 1.0,
    "VW": -7.49, "VC": 1.0, "VM": 1.0, "VH": 1.0, "VY": -6.54,
    "VF": 1.0, "VQ": 1.0, "VI": 1.0, "VR": 1.0, "VN": 1.0,
    "VV": 1.0, "VS": 1.0, "VP": 20.26, "VE": 1.0, "VT": -7.49,
    "VK": -1.88, "VA": 1.0, "VG": -7.49, "VL": 1.0, "VD": -14.03,
    "SW": 1.0, "SC": 33.60, "SM": 1.0, "SH": 1.0, "SY": 1.0,
    "SF": 1.0, "SQ": 20.26, "SI": 1.0, "SR": 1.0, "SN": 1.0,
    "SV": 1.0, "SS": 20.26, "SP": 44.94, "SE": 20.26, "ST": 1.0,
    "SK": 1.0, "SA": 1.0, "SG": 1.0, "SL": 1.0, "SD": 1.0,
    "PW": -1.88, "PC": -6.54, "PM": -6.54, "PH": 1.0, "PY": 1.0,
    "PF": 20.26, "PQ": 20.26, "PI": 1.0, "PR": -6.54, "PN": 1.0,
    "PV": 20.26, "PS": 20.26, "PP": 20.26, "PE": 18.38, "PT": 1.0,
    "PK": 1.0, "PA": 20.26, "PG": 1.0, "PL": 1.0, "PD": -6.54,
    "EW": -14.03, "EC": 44.94, "EM": 1.0, "EH": -6.54, "EY": 1.0,
    "EF": 1.0, "EQ": 20.26, "EI": 1.0, "ER": 1.0, "EN": 1.0,
    "EV": 1.0, "ES": 1.0, "EP": 20.26, "EE": 33.60, "ET": -14.03,
    "EK": 1.0, "EA": 1.0, "EG": 1.0, "EL": 1.0, "ED": 1.0,
    "TW": -14.03, "TC": 1.0, "TM": 1.0, "TH": 1.0, "TY": 1.0,
    "TF": 13.34, "TQ": -6.54, "TI": 1.0, "TR": 1.0, "TN": -14.03,
    "TV": 1.0, "TS": 1.0, "TP": 1.0, "TE": 20.26, "TT": 1.0,
    "TK": 1.0, "TA": 1.0, "TG": -7.49, "TL": 1.0, "TD": 1.0,
    "KW": 1.0, "KC": 1.0, "KM": 33.60, "KH": 1.0, "KY": 1.0,
    "KF": 1.0, "KQ": 24.68, "KI": -7.49, "KR": 33.60, "KN": 1.0,
    "KV": -7.49, "KS": 1.0, "KP": -6.54, "KE": 1.0, "KT": 1.0,
    "KK": 1.0, "KA": 1.0, "KG": -7.49, "KL": -7.49, "KD": 1.0,
    "AW": 1.0, "AC": 44.94, "AM": 1.0, "AH": -7.49, "AY": 1.0,
    "AF": 1.0, "AQ": 1.0, "AI": 1.0, "AR": 1.0, "AN": 1.0,
    "AV": 1.0, "AS": 1.0, "AP": 20.26, "AE": 1.0, "AT": 1.0,
    "AK": 1.0, "AA": 1.0, "AG": 1.0, "AL": 1.0, "AD": -7.49,
    "GW": 13.34, "GC": 1.0, "GM": 1.0, "GH": 1.0, "GY": -7.49,
    "GF": 1.0, "GQ": 1.0, "GI": -7.49, "GR": 1.0, "GN": -7.49,
    "GV": 1.0, "GS": 1.0, "GP": 1.0, "GE": 1.0, "GT": -7.49,
    "GK": -7.49, "GA": 1.0, "GG": 13.34, "GL": 1.0, "GD": 1.0,
    "LW": 24.68, "LC": 1.0, "LM": 1.0, "LH": 1.0, "LY": 1.0,
    "LF": 1.0, "LQ": 33.60, "LI": 1.0, "LR": 20.26, "LN": 1.0,
    "LV": 1.0, "LS": 1.0, "LP": 20.26, "LE": 1.0, "LT": 1.0,
    "LK": -7.49, "LA": 1.0, "LG": 1.0, "LL": 1.0, "LD": 1.0,
    "DW": 1.0, "DC": 1.0, "DM": 1.0, "DH": 1.0, "DY": 1.0,
    "DF": 1.0, "DQ": 1.0, "DI": 1.0, "DR": 1.0, "DN": 1.0,
    "DV": 1.0, "DS": 1.0, "DP": 1.0, "DE": 1.0, "DT": 1.0,
    "DK": 1.0, "DA": 1.0, "DG": 1.0, "DL": 1.0, "DD": 1.0,
}

# Chou-Fasman helix/sheet/turn propensities
CF_HELIX = {
    "A": 1.45, "C": 0.77, "D": 0.98, "E": 1.53, "F": 1.12,
    "G": 0.53, "H": 1.24, "I": 1.00, "K": 1.07, "L": 1.34,
    "M": 1.20, "N": 0.73, "P": 0.59, "Q": 1.17, "R": 0.79,
    "S": 0.79, "T": 0.82, "V": 1.14, "W": 1.02, "Y": 0.61,
}
CF_SHEET = {
    "A": 0.97, "C": 1.30, "D": 0.80, "E": 0.26, "F": 1.28,
    "G": 0.81, "H": 0.71, "I": 1.60, "K": 0.74, "L": 1.22,
    "M": 1.67, "N": 0.65, "P": 0.62, "Q": 1.23, "R": 0.90,
    "S": 0.72, "T": 1.20, "V": 1.65, "W": 1.19, "Y": 1.29,
}
CF_TURN = {
    "A": 0.77, "C": 0.81, "D": 1.41, "E": 0.99, "F": 0.59,
    "G": 1.64, "H": 0.68, "I": 0.51, "K": 1.01, "L": 0.58,
    "M": 0.52, "N": 1.28, "P": 1.91, "Q": 0.98, "R": 0.88,
    "S": 1.32, "T": 1.04, "V": 0.47, "W": 0.76, "Y": 1.05,
}

# ── Feature computation functions ─────────────────────────────────────────────

def aa_composition(seq):
    """20 features: fraction of each standard AA."""
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return {f"aac_{aa}": 0.0 for aa in STANDARD_AAS}
    return {f"aac_{aa}": seq.count(aa) / n for aa in STANDARD_AAS}


def dipeptide_composition(seq):
    """400 features: fraction of each AA pair."""
    seq = seq.upper()
    n = len(seq) - 1
    pairs = {f"dpc_{a}{b}": 0.0 for a, b in product(STANDARD_AAS, repeat=2)}
    if n <= 0:
        return pairs
    for i in range(n):
        dp = seq[i:i+2]
        key = f"dpc_{dp}"
        if key in pairs:
            pairs[key] += 1.0 / n
    return pairs


def physicochemical(seq):
    """Scalar physicochemical descriptors."""
    seq = seq.upper()
    n = len(seq)
    feats = {}

    # GRAVY
    feats["gravy"] = sum(HYDROPATHY.get(aa, 0) for aa in seq) / n if n else 0.0

    # Instability index
    inst = sum(DIWV.get(seq[i:i+2], 1.0) for i in range(n - 1))
    feats["instability_index"] = (10.0 / n * inst) if n > 1 else 0.0

    # Aromaticity (F + Y + W fraction)
    feats["aromaticity"] = sum(seq.count(aa) for aa in "FYW") / n if n else 0.0

    # Charge at pH 7
    ph = 7.0
    charge = 0.0
    charge += 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))   # N-terminus (+)
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))   # C-terminus (-)
    for aa, pka in PKA_SIDE.items():
        count = seq.count(aa)
        if count == 0:
            continue
        if aa in "KRH":  # positive
            charge += count * (1.0 / (1.0 + 10 ** (ph - pka)))
        else:            # negative
            charge -= count * (1.0 / (1.0 + 10 ** (pka - ph)))
    feats["charge_ph7"] = charge

    # Chou-Fasman propensities (mean over sequence)
    feats["cf_helix"]  = np.mean([CF_HELIX.get(aa, 1.0)  for aa in seq]) if n else 1.0
    feats["cf_sheet"]  = np.mean([CF_SHEET.get(aa, 1.0)  for aa in seq]) if n else 1.0
    feats["cf_turn"]   = np.mean([CF_TURN.get(aa, 1.0)   for aa in seq]) if n else 1.0

    return feats


def compute_all_seq_features(seq):
    feats = {}
    feats.update(aa_composition(seq))
    feats.update(dipeptide_composition(seq))
    feats.update(physicochemical(seq))
    return feats


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading processed data...")
pairs_df   = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
proteins_df = pd.read_parquet(DATA_DIR / "proteins.parquet")
print(f"  Pairs : {len(pairs_df):,}")
print(f"  Proteins: {len(proteins_df):,}")

# ── 2.1 Sequence-based features ───────────────────────────────────────────────

print("\nComputing sequence-based features (AA comp + dipeptide + physicochemical)...")
seq_feat_records = []
for _, row in proteins_df.iterrows():
    seq = str(row["sequence"]).upper()
    feats = {"protein_id": row["protein_id"]}
    feats.update(compute_all_seq_features(seq))
    seq_feat_records.append(feats)

seq_feat_df = pd.DataFrame(seq_feat_records)
print(f"  Shape: {seq_feat_df.shape}  ({seq_feat_df.shape[1]-1} features per protein)")

# ── 2.2 Precomputed features ─────────────────────────────────────────────────

print("\nExtracting precomputed sequence-level features...")
PRECOMPUTED_COLS = [
    "protein_id", "seq_length",
    "esmfold_plddt", "proteinmpnn_score", "proteinmpnn_seq_recovery",
    "redesigned_proteinmpnn_score", "molecular_weight", "isoelectric_point",
]
available_cols = [c for c in PRECOMPUTED_COLS if c in proteins_df.columns]
precomp_df = proteins_df[available_cols].copy()
print(f"  Extracted columns: {available_cols}")

# ── 2.3 Boltz2 structural features (already in pairs_df, target-specific) ────

print("\nExtracting Boltz2 structural features...")
BOLTZ2_COLS = [
    "boltz2_iptm", "boltz2_ipsae", "boltz2_min_ipsae", "boltz2_complex_iplddt",
    "boltz2_plddt", "boltz2_complex_plddt", "boltz2_ptm", "boltz2_complex_pde",
    "boltz2_lis", "boltz2_pdockq", "boltz2_pdockq2",
    "shape_complimentarity_boltz2_binder_ss",
]
boltz2_available = [c for c in BOLTZ2_COLS if c in pairs_df.columns]
boltz2_df = pairs_df[["protein_id", "target"] + boltz2_available].copy()
coverage = boltz2_df[boltz2_available].notna().any(axis=1).sum()
print(f"  Boltz2 coverage: {coverage:,} / {len(boltz2_df):,} pairs ({coverage/len(boltz2_df):.1%})")

# ── 2.4 Design method encoding ────────────────────────────────────────────────

print("\nEncoding design methods...")
TOP_N_METHODS = 15
method_counts = proteins_df["design_method"].value_counts()
top_methods   = method_counts.head(TOP_N_METHODS).index.tolist()

proteins_df["design_method_clean"] = proteins_df["design_method"].apply(
    lambda m: m if m in top_methods else "other"
)
method_dummies = pd.get_dummies(
    proteins_df[["protein_id", "design_method_clean"]].set_index("protein_id")["design_method_clean"],
    prefix="method"
).reset_index()
print(f"  One-hot encoded {method_dummies.shape[1]-1} method categories")

# Also add historical success rate per design method
method_success = (
    pairs_df.groupby("design_method")["binding_label"]
    .mean()
    .reset_index()
    .rename(columns={"binding_label": "method_success_rate"})
)
proteins_df = proteins_df.merge(method_success, on="design_method", how="left")
proteins_df["method_success_rate"] = proteins_df["method_success_rate"].fillna(
    pairs_df["binding_label"].mean()
)

# ── 2.5 Assemble final feature matrix ─────────────────────────────────────────

print("\nAssembling final feature matrix...")

# Start from pairs (protein-target pairs with labels and splits)
label_cols = ["protein_id", "target", "binding_label", "binding_strength", "expressed", "split"]
feature_matrix = pairs_df[label_cols].copy()

# Merge sequence features
feature_matrix = feature_matrix.merge(seq_feat_df, on="protein_id", how="left")

# Merge precomputed features
feature_matrix = feature_matrix.merge(precomp_df, on="protein_id", how="left")

# Merge method encoding
feature_matrix = feature_matrix.merge(method_dummies, on="protein_id", how="left")

# Merge method success rate
success_rate_df = proteins_df[["protein_id", "method_success_rate"]]
feature_matrix = feature_matrix.merge(success_rate_df, on="protein_id", how="left")

# Merge Boltz2 (target-specific)
feature_matrix = feature_matrix.merge(boltz2_df, on=["protein_id", "target"], how="left")

print(f"  Full matrix shape: {feature_matrix.shape}")

# Identify feature columns (exclude metadata and labels)
META_COLS  = ["protein_id", "target", "split"]
LABEL_COLS = ["binding_label", "binding_strength", "expressed"]
feat_cols  = [c for c in feature_matrix.columns if c not in META_COLS + LABEL_COLS]

print(f"  Feature columns: {len(feat_cols)}")
missing_pct = feature_matrix[feat_cols].isna().mean()
print(f"  Features with >50% missing: {(missing_pct > 0.5).sum()}")
print(f"  Features with any missing : {(missing_pct > 0).sum()}")

# ── Impute missing values ─────────────────────────────────────────────────────

print("\nImputing missing values (median strategy)...")

# Separate boolean/int columns (method dummies) from float columns
bool_cols  = [c for c in feat_cols if c.startswith("method_") and c != "method_success_rate"]
float_cols = [c for c in feat_cols if c not in bool_cols]

imputer = SimpleImputer(strategy="median")
feature_matrix[float_cols] = imputer.fit_transform(feature_matrix[float_cols])
feature_matrix[bool_cols]  = feature_matrix[bool_cols].fillna(0).astype(int)

print(f"  Missing after imputation: {feature_matrix[feat_cols].isna().sum().sum()}")

# ── Normalize float features ──────────────────────────────────────────────────

print("\nNormalizing features (StandardScaler)...")

# Don't scale binary method dummies or label columns
cols_to_scale = float_cols
scaler = StandardScaler()
feature_matrix[cols_to_scale] = scaler.fit_transform(feature_matrix[cols_to_scale])

# ── Save ──────────────────────────────────────────────────────────────────────

print("\nSaving feature matrix and artifacts...")

feature_matrix.to_parquet(FEAT_DIR / "feature_matrix.parquet", index=False)

# Save column metadata
col_meta = pd.DataFrame({
    "column": feat_cols,
    "group": [
        "aa_composition"       if c.startswith("aac_") else
        "dipeptide"            if c.startswith("dpc_") else
        "physicochemical"      if c in ["gravy","instability_index","aromaticity","charge_ph7","cf_helix","cf_sheet","cf_turn"] else
        "precomputed_seq"      if c in ["seq_length","esmfold_plddt","proteinmpnn_score","proteinmpnn_seq_recovery","redesigned_proteinmpnn_score","molecular_weight","isoelectric_point"] else
        "design_method"        if c.startswith("method_") else
        "boltz2"               if c.startswith("boltz2_") or c == "shape_complimentarity_boltz2_binder_ss" else
        "other"
        for c in feat_cols
    ]
})
col_meta.to_csv(FEAT_DIR / "feature_columns.csv", index=False)

# Save scaler and imputer for inference
joblib.dump(scaler,   FEAT_DIR / "scaler.pkl")
joblib.dump(imputer,  FEAT_DIR / "imputer.pkl")

print(f"  Saved: features/feature_matrix.parquet  ({feature_matrix.shape})")
print(f"  Saved: features/feature_columns.csv")
print(f"  Saved: features/scaler.pkl, imputer.pkl")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n── Feature Summary ──────────────────────────────────────────────────────")
grp = col_meta["group"].value_counts()
for g, cnt in grp.items():
    print(f"  {g:<30} {cnt:>5} features")
print(f"  {'TOTAL':<30} {len(feat_cols):>5} features")

train = feature_matrix[feature_matrix["split"] == "train"]
val   = feature_matrix[feature_matrix["split"] == "val"]
test  = feature_matrix[feature_matrix["split"] == "test"]
print(f"\n  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
print(f"  Train binding rate: {train['binding_label'].mean():.1%}")

print("\nPhase 2 complete.")
