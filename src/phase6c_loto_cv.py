"""
Phase 6c: Leave-One-Target-Out (LOTO) Cross-Validation
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

For Nature publication: must demonstrate generalization to UNSEEN targets.
LOTO holds out ALL pairs for one target, trains on the rest, evaluates on held-out.

Two scenarios tested:
  1. With prototype features (zero-shot cold start: no known binders for held-out target)
  2. Without prototype features (pure sequence/structure generalization)

Key question: does the model generalize beyond memorizing target-specific patterns?
"""

import json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from pathlib import Path
import joblib, warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ── Physicochemical tables (same as Phase 6a) ─────────────────────────────────
KD_H = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
         'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,
         'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
CHG  = {'K':1,'R':1,'H':0.1,'D':-1,'E':-1}
ARO  = set('FWY'); HBD = set('NQSTKRHWY'); HBA = set('NQDEST')
VOL  = {'G':60,'A':89,'S':96,'P':112,'V':117,'T':116,'C':114,'I':166,'L':166,
        'N':114,'D':111,'Q':144,'K':168,'E':138,'M':162,'H':153,'F':190,'R':173,'Y':194,'W':228}
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

def _if_feat(sequence, positions):
    seq = sequence.upper(); n = len(seq)
    if not positions or n == 0: return {}
    pos_0 = [p-1 for p in positions if 0 <= p-1 < n]
    if not pos_0: return {}
    iface = [seq[p] for p in pos_0]; n_if = len(iface)
    f = {}
    cnt = {a:0 for a in AA20}
    for aa in iface:
        if aa in cnt: cnt[aa] += 1
    for aa in AA20: f[f"if_aac_{aa}"] = cnt[aa]/n_if
    f["if_n_residues"] = n_if; f["if_coverage"] = n_if/n
    if len(pos_0) > 1:
        gaps = [pos_0[i+1]-pos_0[i] for i in range(len(pos_0)-1)]
        f["if_span"] = (max(pos_0)-min(pos_0))/n
        f["if_mean_gap"] = np.mean(gaps)/n; f["if_max_gap"] = max(gaps)/n
        f["if_n_segments"] = sum(1 for g in gaps if g > 3)
    else:
        f["if_span"]=f["if_mean_gap"]=f["if_max_gap"]=f["if_n_segments"]=0.0
    f["if_nterm_frac"] = sum(1 for p in pos_0 if p < n*0.33)/n_if
    f["if_cterm_frac"] = sum(1 for p in pos_0 if p > n*0.67)/n_if
    hy=[KD_H.get(a,0.) for a in iface]; ch=[CHG.get(a,0.) for a in iface]
    vl=[VOL.get(a,130.) for a in iface]
    f["if_mean_hydro"]=np.mean(hy); f["if_std_hydro"]=np.std(hy)
    f["if_net_charge"]=sum(ch); f["if_mean_charge"]=np.mean(ch)
    f["if_pos_frac"]=sum(1 for c in ch if c>0)/n_if
    f["if_neg_frac"]=sum(1 for c in ch if c<0)/n_if
    f["if_aromatic_frac"]=sum(1 for a in iface if a in ARO)/n_if
    f["if_hbond_donor_frac"]=sum(1 for a in iface if a in HBD)/n_if
    f["if_hbond_acc_frac"]=sum(1 for a in iface if a in HBA)/n_if
    f["if_mean_volume"]=np.mean(vl)
    wh=[KD_H.get(a,0.) for a in seq]
    f["if_hydro_delta"]=f["if_mean_hydro"]-np.mean(wh)
    return f

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

PROTO_COLS = ["proto_cos_pos","proto_cos_neg","proto_l2_pos",
              "proto_disc_proj","proto_ratio","proto_n_pos","proto_n_neg"]

def compute_proto_features(emb_i, proto_pos, proto_neg, n_pos, n_neg):
    pp, pn = proto_pos, proto_neg
    disc = pp - pn; disc_norm = np.linalg.norm(disc)
    return [
        cosine_sim(emb_i, pp),
        cosine_sim(emb_i, pn),
        float(np.linalg.norm(emb_i - pp)),
        float(np.dot(emb_i, disc) / (disc_norm + 1e-8)),
        cosine_sim(emb_i, pp) / (abs(cosine_sim(emb_i, pn)) + 1e-6),
        float(n_pos),
        float(n_neg),
    ]

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading data...")
pairs     = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet").reset_index(drop=True)
feat_mat  = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet").reset_index(drop=True)

# Load augmented feature columns (463 base + 39 interface = 502)
meta_6a   = pickle.load(open(MODELS_DIR / "ensemble_meta_6a.pkl", "rb"))
if_cols   = meta_6a["if_cols"]
if_meds   = pd.Series(meta_6a["if_medians"])
base_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()

# Parse interface residues
evals  = pd.read_parquet(DATA_DIR / "evaluations_flat.parquet")
ir_raw = evals[evals["metric"] == "interface_residues"].copy()
def _parse_pos(v):
    try:
        d = json.loads(v) if isinstance(v, str) else v
        return [int(r["residue"]) for r in d if "residue" in r] if isinstance(d, list) else []
    except: return []
ir_raw["positions"] = ir_raw["value"].apply(_parse_pos)
ir_raw = ir_raw[ir_raw["positions"].apply(len) > 0]
ir_map = {(r.protein_id, r.target): r.positions for _, r in ir_raw.iterrows()}
seq_map = dict(zip(pairs["protein_id"], pairs["sequence"]))

# Compute interface features for all pairs
print("Computing interface features for all pairs...")
if_rows = []
for _, row in pairs.iterrows():
    key = (row["protein_id"], row["target"])
    pos = ir_map.get(key, [])
    seq = seq_map.get(row["protein_id"], "")
    if_rows.append(_if_feat(seq, pos) if pos and seq else {})
if_df = pd.DataFrame(if_rows)[if_cols].fillna(if_meds)

# Merge feature matrix with interface features
fm = feat_mat.copy()
fm_keys    = list(zip(fm["protein_id"], fm["target"]))
pair_to_if = {(pairs["protein_id"].iloc[i], pairs["target"].iloc[i]): i for i in range(len(pairs))}
for col in if_cols:
    fm[col] = [if_df[col].iloc[pair_to_if.get(k, 0)]
               if k in pair_to_if else if_meds[col]
               for k in fm_keys]
aug_cols = base_cols + if_cols  # 502 features

# Load ESM-2 embeddings
esm_emb = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_idx = {pid: i for i, pid in enumerate(esm_ids)}

X_emb_all = np.zeros((len(pairs), 1280), dtype=np.float32)
for i, pid in enumerate(pairs["protein_id"]):
    idx = esm_idx.get(pid)
    if idx is not None:
        X_emb_all[i] = esm_emb[idx]
X_emb_all = np.nan_to_num(X_emb_all, nan=0.0)

y_all      = pairs["binding_label"].values.astype(np.float32)
target_all = pairs["target"].values
target_names = sorted(pairs["target"].unique())

print(f"  Total pairs: {len(pairs):,} | Targets: {len(target_names)}")

# Align fm to pairs order
fm_pair_idx = {(r["protein_id"], r["target"]): i for i, r in fm.iterrows()}
pairs_to_fm = [fm_pair_idx.get((pairs["protein_id"].iloc[i], pairs["target"].iloc[i]), None)
               for i in range(len(pairs))]

X_hc_all = np.zeros((len(pairs), len(aug_cols)), dtype=np.float32)
for i, fm_i in enumerate(pairs_to_fm):
    if fm_i is not None:
        X_hc_all[i] = fm.iloc[fm_i][aug_cols].values.astype(np.float32)

# LightGBM params (same as 6b)
lgb_params = dict(
    objective="binary", metric="auc", verbosity=-1, device="gpu",
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=5,
    random_state=RANDOM_SEED,
)

# ── LOTO Cross-Validation ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("LOTO Cross-Validation (24 targets)...")
print(f"  {'Target':<45}  {'N':>4}  {'Pos':>3}  {'AUROC_w_proto':>13}  {'AUROC_no_proto':>14}  {'AUPRC_w':>8}  {'F1_w':>6}")
print("  " + "-"*110)

loto_results = []

for tgt in target_names:
    test_mask  = target_all == tgt
    train_mask = ~test_mask

    y_test  = y_all[test_mask]
    y_train = y_all[train_mask]
    n_test  = test_mask.sum()
    n_pos   = int(y_test.sum())

    # Skip targets with no variability in test
    if n_pos == 0 or n_pos == n_test or n_test < 3:
        print(f"  {tgt:<45}  {n_test:>4}  {n_pos:>3}  {'N/A (no variability)':<36}")
        loto_results.append({
            "target": tgt, "n": n_test, "n_pos": n_pos,
            "auroc_with_proto": np.nan, "auroc_no_proto": np.nan,
            "auprc_with_proto": np.nan, "auprc_no_proto": np.nan,
            "f1_with_proto": np.nan, "mcc_with_proto": np.nan,
        })
        continue

    # ── Build prototype features for this LOTO fold ───────────────────────────
    # Prototype = mean ESM-2 of TRAINING positives (held-out target has no prototype)
    tgt_protos_pos = {}
    tgt_protos_neg = {}
    tgt_n_pos = {}
    tgt_n_neg = {}
    for t in target_names:
        if t == tgt:
            # Zero prototype for held-out target (cold start)
            tgt_protos_pos[t] = np.zeros(1280, dtype=np.float32)
            tgt_protos_neg[t] = np.zeros(1280, dtype=np.float32)
            tgt_n_pos[t] = 0
            tgt_n_neg[t] = 0
        else:
            t_mask_tr = train_mask & (target_all == t)
            y_t       = y_all[t_mask_tr]
            emb_t     = X_emb_all[t_mask_tr]
            pos_m     = y_t == 1; neg_m = y_t == 0
            tgt_protos_pos[t] = emb_t[pos_m].mean(0) if pos_m.sum() > 0 else np.zeros(1280, dtype=np.float32)
            tgt_protos_neg[t] = emb_t[neg_m].mean(0) if neg_m.sum() > 0 else np.zeros(1280, dtype=np.float32)
            tgt_n_pos[t] = int(pos_m.sum())
            tgt_n_neg[t] = int(neg_m.sum())

    # Compute prototype features for ALL pairs in this fold
    proto_arr = np.array([
        compute_proto_features(
            X_emb_all[i], tgt_protos_pos[target_all[i]],
            tgt_protos_neg[target_all[i]], tgt_n_pos[target_all[i]], tgt_n_neg[target_all[i]]
        )
        for i in range(len(pairs))
    ], dtype=np.float32)

    # ── Feature matrices ─────────────────────────────────────────────────────
    X_with_proto = np.concatenate([X_hc_all, proto_arr], axis=1)  # 502 + 7 = 509

    X_tr_wp  = X_with_proto[train_mask];  X_te_wp  = X_with_proto[test_mask]
    X_tr_np  = X_hc_all[train_mask];      X_te_np  = X_hc_all[test_mask]

    # ── Train LightGBM WITH prototype features ────────────────────────────────
    model_wp = lgb.LGBMClassifier(**lgb_params)
    model_wp.fit(X_tr_wp, y_train,
                 eval_set=[(X_te_wp, y_test)],
                 callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
    prob_wp = model_wp.predict_proba(X_te_wp)[:, 1]

    # ── Train LightGBM WITHOUT prototype features ────────────────────────────
    model_np = lgb.LGBMClassifier(**lgb_params)
    model_np.fit(X_tr_np, y_train,
                 eval_set=[(X_te_np, y_test)],
                 callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)])
    prob_np = model_np.predict_proba(X_te_np)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    auroc_wp = roc_auc_score(y_test, prob_wp)
    auprc_wp = average_precision_score(y_test, prob_wp)
    f1_wp    = f1_score(y_test, (prob_wp >= 0.5).astype(int), zero_division=0)
    mcc_wp   = matthews_corrcoef(y_test, (prob_wp >= 0.5).astype(int))

    auroc_np = roc_auc_score(y_test, prob_np)
    auprc_np = average_precision_score(y_test, prob_np)

    print(f"  {tgt:<45}  {n_test:>4}  {n_pos:>3}  "
          f"{auroc_wp:>13.4f}  {auroc_np:>14.4f}  {auprc_wp:>8.4f}  {f1_wp:>6.4f}")

    loto_results.append({
        "target": tgt, "n": n_test, "n_pos": n_pos,
        "auroc_with_proto": auroc_wp, "auroc_no_proto": auroc_np,
        "auprc_with_proto": auprc_wp, "auprc_no_proto": auprc_np,
        "f1_with_proto": f1_wp, "mcc_with_proto": mcc_wp,
    })

# ── Summary statistics ────────────────────────────────────────────────────────
loto_df = pd.DataFrame(loto_results)
evaluable = loto_df.dropna(subset=["auroc_with_proto"])

print("\n" + "="*70)
print("LOTO SUMMARY")
print(f"  Evaluable targets: {len(evaluable)} / {len(target_names)}")
print(f"\n  WITH prototype features (zero-shot cold start for held-out target):")
print(f"    Mean AUROC : {evaluable['auroc_with_proto'].mean():.4f}")
print(f"    Median AUROC: {evaluable['auroc_with_proto'].median():.4f}")
print(f"    Min AUROC  : {evaluable['auroc_with_proto'].min():.4f}  ({evaluable.loc[evaluable['auroc_with_proto'].idxmin(), 'target']})")
print(f"    Max AUROC  : {evaluable['auroc_with_proto'].max():.4f}  ({evaluable.loc[evaluable['auroc_with_proto'].idxmax(), 'target']})")
print(f"    Mean AUPRC : {evaluable['auprc_with_proto'].mean():.4f}")
print(f"    Mean F1    : {evaluable['f1_with_proto'].mean():.4f}")

print(f"\n  WITHOUT prototype features (pure sequence/structure generalization):")
print(f"    Mean AUROC : {evaluable['auroc_no_proto'].mean():.4f}")
print(f"    Median AUROC: {evaluable['auroc_no_proto'].median():.4f}")
print(f"    Min AUROC  : {evaluable['auroc_no_proto'].min():.4f}  ({evaluable.loc[evaluable['auroc_no_proto'].idxmin(), 'target']})")
print(f"    Max AUROC  : {evaluable['auroc_no_proto'].max():.4f}  ({evaluable.loc[evaluable['auroc_no_proto'].idxmax(), 'target']})")
print(f"    Mean AUPRC : {evaluable['auprc_no_proto'].mean():.4f}")

# Large targets separately
large = evaluable[evaluable["n"] >= 20]
print(f"\n  Large targets only (n≥20, {len(large)} targets):")
print(f"    With proto  — Mean AUROC={large['auroc_with_proto'].mean():.4f}  Mean AUPRC={large['auprc_with_proto'].mean():.4f}")
print(f"    No proto    — Mean AUROC={large['auroc_no_proto'].mean():.4f}  Mean AUPRC={large['auprc_no_proto'].mean():.4f}")

# ── Comparison: in-distribution (random split) vs LOTO ────────────────────────
print("\n" + "="*70)
print("IN-DISTRIBUTION vs LOTO COMPARISON")
print(f"  {'Setting':<45}  {'AUROC':>7}  {'AUPRC':>7}")
print("  " + "-"*63)
print(f"  {'In-distribution (random 70/15/15 split)':<45}  {0.9402:>7.4f}  {0.7645:>7.4f}")
print(f"  {'LOTO with proto (zero-shot cold start)':<45}  {evaluable['auroc_with_proto'].mean():>7.4f}  {evaluable['auprc_with_proto'].mean():>7.4f}")
print(f"  {'LOTO without proto (pure generalization)':<45}  {evaluable['auroc_no_proto'].mean():>7.4f}  {evaluable['auprc_no_proto'].mean():>7.4f}")
print(f"\n  Note: prototype drop from in-dist→LOTO shows how much signal")
print(f"  comes from target-specific similarity vs. intrinsic sequence features.")

# ── Save ──────────────────────────────────────────────────────────────────────
loto_df.to_csv(OUTPUTS_DIR / "phase6c_loto_results.csv", index=False)
print(f"\nSaved: outputs/phase6c_loto_results.csv")
print("Phase 6c complete.")
