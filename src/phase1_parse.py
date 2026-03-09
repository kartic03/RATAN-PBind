"""
Phase 1: Data Parsing & Preparation
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = PROJECT_ROOT / "proteinbase_all_data_28_01_2026.csv"

# ── 1. Load raw CSV ──────────────────────────────────────────────────────────

print("Loading CSV...")
raw = pd.read_csv(CSV_PATH)
print(f"  Proteins: {len(raw):,}")
print(f"  Columns : {list(raw.columns)}")
print(f"  Missing name: {raw['name'].isna().sum()}, author: {raw['author'].isna().sum()}, designMethod: {raw['designMethod'].isna().sum()}")

# ── 2. Parse evaluations JSON ────────────────────────────────────────────────

print("\nParsing evaluations JSON...")

records = []
for _, row in raw.iterrows():
    protein_id = row["id"]
    sequence   = row["sequence"]
    name       = row["name"]
    author     = row["author"]
    design_method = row["designMethod"]

    try:
        evals = json.loads(row["evaluations"])
    except (json.JSONDecodeError, TypeError):
        evals = []

    if not evals:
        # Keep protein with no evaluations so we don't lose it
        records.append({
            "protein_id": protein_id,
            "name": name,
            "sequence": sequence,
            "author": author,
            "design_method": design_method,
            "eval_type": None,
            "metric": None,
            "target": None,
            "value_type": None,
            "unit": None,
            "value": None,
        })
    else:
        for ev in evals:
            records.append({
                "protein_id": protein_id,
                "name": name,
                "sequence": sequence,
                "author": author,
                "design_method": design_method,
                "eval_type": ev.get("type"),
                "metric": ev.get("metric"),
                "target": ev.get("target"),
                "value_type": ev.get("valueType"),
                "unit": ev.get("unit"),
                "value": ev.get("value"),
            })

evals_df = pd.DataFrame(records)
print(f"  Total evaluation records: {len(evals_df):,}")
print(f"  Unique proteins: {evals_df['protein_id'].nunique():,}")
print(f"  Unique metrics : {evals_df['metric'].nunique()}")
print(f"  Unique targets : {evals_df['target'].nunique()}")
print(f"  Eval types     : {evals_df['eval_type'].value_counts().to_dict()}")

# ── 3. Build protein-level feature table (one row per protein) ───────────────

print("\nBuilding protein-level feature table...")

NUMERIC_METRICS = [
    "esmfold_plddt", "proteinmpnn_score", "proteinmpnn_seq_recovery",
    "redesigned_proteinmpnn_score", "molecular_weight", "isoelectric_point",
    "ted_confidence",
]

BOLTZ2_METRICS = [
    "boltz2_iptm", "boltz2_ipsae", "boltz2_min_ipsae", "boltz2_complex_iplddt",
    "boltz2_plddt", "boltz2_complex_plddt", "boltz2_ptm", "boltz2_complex_pde",
    "boltz2_lis", "boltz2_pdockq", "boltz2_pdockq2",
    "shape_complimentarity_boltz2_binder_ss",
]

def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

# Sequence-level numeric features (not target-specific)
seq_features = evals_df[evals_df["metric"].isin(NUMERIC_METRICS)].copy()
seq_features["value_float"] = seq_features["value"].apply(safe_float)
seq_pivot = (
    seq_features.groupby(["protein_id", "metric"])["value_float"]
    .first()
    .unstack("metric")
    .reset_index()
)

# Protein metadata (one row per protein)
meta = (
    evals_df[["protein_id", "name", "sequence", "author", "design_method"]]
    .drop_duplicates("protein_id")
    .reset_index(drop=True)
)
meta["seq_length"] = meta["sequence"].str.len()

protein_df = meta.merge(seq_pivot, on="protein_id", how="left")
print(f"  Protein table shape: {protein_df.shape}")

# ── 4. Build binding label table (protein-target pairs) ─────────────────────

print("\nBuilding binding label table...")

binding_rows = evals_df[evals_df["metric"] == "binding"].copy()

def parse_binding(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        if v.lower() == "true":  return 1
        if v.lower() == "false": return 0
    return np.nan

binding_rows["binding_label"] = binding_rows["value"].apply(parse_binding)
binding_rows = binding_rows.dropna(subset=["binding_label", "target"])
binding_rows["binding_label"] = binding_rows["binding_label"].astype(int)

# Binding strength (ordinal)
strength_map = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
strength_rows = evals_df[evals_df["metric"] == "binding_strength"].copy()
strength_rows["binding_strength"] = strength_rows["value"].str.lower().map(strength_map)

# Expression label
expr_rows = evals_df[evals_df["metric"] == "expressed"].copy()
expr_rows["expressed"] = expr_rows["value"].apply(parse_binding)

# Boltz2 features (target-specific)
boltz2_rows = evals_df[evals_df["metric"].isin(BOLTZ2_METRICS)].copy()
boltz2_rows["value_float"] = boltz2_rows["value"].apply(safe_float)
boltz2_pivot = (
    boltz2_rows.groupby(["protein_id", "target", "metric"])["value_float"]
    .first()
    .unstack("metric")
    .reset_index()
)

# Assemble protein-target pair table
pair_df = (
    binding_rows[["protein_id", "target", "binding_label", "eval_type"]]
    .drop_duplicates(["protein_id", "target"])
    .reset_index(drop=True)
)

# Merge binding strength
strength_agg = (
    strength_rows.groupby(["protein_id", "target"])["binding_strength"]
    .first()
    .reset_index()
)
pair_df = pair_df.merge(strength_agg, on=["protein_id", "target"], how="left")

# Merge expression
expr_agg = (
    expr_rows.groupby(["protein_id"])["expressed"]
    .first()
    .reset_index()
)
pair_df = pair_df.merge(expr_agg, on="protein_id", how="left")

# Merge protein-level features
pair_df = pair_df.merge(protein_df, on="protein_id", how="left")

# Merge Boltz2 features
pair_df = pair_df.merge(boltz2_pivot, on=["protein_id", "target"], how="left")

print(f"  Pair table shape: {pair_df.shape}")
print(f"  Unique proteins with binding: {pair_df['protein_id'].nunique():,}")
print(f"  Unique targets: {pair_df['target'].nunique()}")
print(f"  Binding rate: {pair_df['binding_label'].mean():.1%}")

# Per-target stats
print("\n  Per-target binding rates:")
target_stats = (
    pair_df.groupby("target")["binding_label"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "binders", "count": "total", "mean": "rate"})
    .sort_values("total", ascending=False)
)
print(target_stats.to_string())

# ── 5. Sequence quality checks ───────────────────────────────────────────────

print("\nSequence quality checks...")
STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
NON_STANDARD = set("BJOUXXZ")

def has_nonstandard(seq):
    return bool(set(str(seq).upper()) & NON_STANDARD)

protein_df["has_nonstandard_aa"] = protein_df["sequence"].apply(has_nonstandard)
protein_df["is_duplicate_seq"]   = protein_df["sequence"].duplicated(keep=False)

ns_count  = protein_df["has_nonstandard_aa"].sum()
dup_count = protein_df["is_duplicate_seq"].sum()
short     = (protein_df["seq_length"] < 10).sum()
long_     = (protein_df["seq_length"] > 800).sum()

print(f"  Non-standard AAs: {ns_count}")
print(f"  Duplicate seqs  : {dup_count}")
print(f"  Too short (<10) : {short}")
print(f"  Too long (>800) : {long_}")

# ── 6. Train / Val / Test split (70/15/15, stratified by target+label) ───────

print("\nCreating train/val/test split...")
from sklearn.model_selection import StratifiedShuffleSplit

pair_df = pair_df.reset_index(drop=True)

# Stratify by target + binding label combined
pair_df["strat_key"] = pair_df["target"] + "__" + pair_df["binding_label"].astype(str)

# Remove strat keys with fewer than 2 samples (can't split)
key_counts = pair_df["strat_key"].value_counts()
valid_keys = key_counts[key_counts >= 2].index
pair_df_split = pair_df[pair_df["strat_key"].isin(valid_keys)].copy()
pair_df_rare  = pair_df[~pair_df["strat_key"].isin(valid_keys)].copy()

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
train_idx, temp_idx = next(sss1.split(pair_df_split, pair_df_split["strat_key"]))

temp_df = pair_df_split.iloc[temp_idx].reset_index(drop=True)
train_df = pair_df_split.iloc[train_idx].reset_index(drop=True)

# Split temp into val/test (50/50 of the 30%)
key_counts2 = temp_df["strat_key"].value_counts()
valid_keys2 = key_counts2[key_counts2 >= 2].index
temp_valid = temp_df[temp_df["strat_key"].isin(valid_keys2)].copy()
temp_rare  = temp_df[~temp_df["strat_key"].isin(valid_keys2)].copy()

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
val_idx, test_idx = next(sss2.split(temp_valid, temp_valid["strat_key"]))

val_df  = temp_valid.iloc[val_idx].reset_index(drop=True)
test_df = temp_valid.iloc[test_idx].reset_index(drop=True)

# Assign rare samples to train
train_df = pd.concat([train_df, pair_df_rare, temp_rare], ignore_index=True)

train_df["split"] = "train"
val_df["split"]   = "val"
test_df["split"]  = "test"

full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
full_split_df = full_split_df.drop(columns=["strat_key"])

print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"  Train binding rate: {train_df['binding_label'].mean():.1%}")
print(f"  Val   binding rate: {val_df['binding_label'].mean():.1%}")
print(f"  Test  binding rate: {test_df['binding_label'].mean():.1%}")

# ── 7. Save outputs ──────────────────────────────────────────────────────────

print("\nSaving outputs...")
DATA_DIR.mkdir(exist_ok=True)

# Convert mixed-type value column to string for parquet compatibility
evals_df["value"] = evals_df["value"].apply(
    lambda v: json.dumps(v) if isinstance(v, (dict, list)) else (str(v) if v is not None else None)
)
evals_df.to_parquet(DATA_DIR / "evaluations_flat.parquet", index=False)
protein_df.to_parquet(DATA_DIR / "proteins.parquet", index=False)
full_split_df.to_parquet(DATA_DIR / "pairs_with_splits.parquet", index=False)

# Also save split indices separately for reproducibility
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    df[["protein_id", "target"]].to_csv(DATA_DIR / f"{split_name}_ids.csv", index=False)

print(f"  Saved: data/evaluations_flat.parquet  ({len(evals_df):,} rows)")
print(f"  Saved: data/proteins.parquet          ({len(protein_df):,} rows)")
print(f"  Saved: data/pairs_with_splits.parquet ({len(full_split_df):,} rows)")
print(f"  Saved: data/train_ids.csv, val_ids.csv, test_ids.csv")

print("\nPhase 1 complete.")
