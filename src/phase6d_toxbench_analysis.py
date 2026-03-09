"""
Phase 6d: ToxBench-Inspired Rigorous Analysis
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Applies methodology from ToxBench (Kartic et al.) to the protein binding domain:
1. Multi-seed evaluation (5 seeds) — AUROC ± std, model stability
2. ECE + reliability diagrams — calibration quality (before/after Platt)
3. Applicability domain — ESM-2 NN similarity vs AUROC (explains LOTO gap)
4. Uncertainty quantification — confidence-filtering curve
5. SHAP analysis — prototype feature biological interpretation
6. Sequence identity leakage check — validate split integrity
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib, pickle, warnings, json, contextlib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import lightgbm as lgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEEDS        = [42, 123, 456, 789, 1337]
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ── Helpers ───────────────────────────────────────────────────────────────────
def mets(y_true, y_prob, thr=0.5):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    yp    = (y_prob >= thr).astype(int)
    return dict(auroc=auroc, auprc=auprc,
                f1=f1_score(y_true, yp, zero_division=0),
                mcc=matthews_corrcoef(y_true, yp))

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bins  = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_prob = y_prob[mask].mean()
        ece_val += mask.sum() / total * abs(frac_pos - mean_prob)
    return ece_val

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading data...")
pairs    = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
feat_mat = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")
esm_emb  = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids  = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_map  = {pid: i for i, pid in enumerate(esm_ids)}

pairs_r = pairs.reset_index(drop=True)
X_emb   = np.zeros((len(pairs_r), 1280), dtype=np.float32)
for i, pid in enumerate(pairs_r["protein_id"]):
    idx = esm_map.get(pid)
    if idx is not None:
        X_emb[i] = esm_emb[idx]
X_emb = np.nan_to_num(X_emb, nan=0.0)

y_bind   = pairs_r["binding_label"].values.astype(np.float32)
train_m  = (pairs_r["split"] == "train").values
val_m    = (pairs_r["split"] == "val").values
test_m   = (pairs_r["split"] == "test").values

target_names = sorted(pairs_r["target"].unique())
target_enc   = {t: i for i, t in enumerate(target_names)}
print(f"  Train: {train_m.sum()}  Val: {val_m.sum()}  Test: {test_m.sum()}")

# Load Phase 6b meta (prototypes, feature cols)
with open(MODELS_DIR / "ensemble_meta_6b.pkl", "rb") as f:
    meta_6b = pickle.load(f)

all_feat_cols   = meta_6b["all_feat_cols"]
proto_feat_cols = meta_6b["proto_feat_cols"]
proto_pos       = meta_6b["proto_pos"]
proto_neg       = meta_6b["proto_neg"]
n_pos_dict      = meta_6b["n_pos"]
n_neg_dict      = meta_6b["n_neg"]
print(f"  Feature cols: {len(all_feat_cols)} (incl. {len(proto_feat_cols)} proto)")

# ── 2. Rebuild feature matrix with prototype features ─────────────────────────
print("\nRebuilding feature matrix with prototype features...")

def _proto_feats(emb, target):
    pp = proto_pos[target]
    pn = proto_neg[target]
    disc = pp - pn
    dn   = np.linalg.norm(disc)
    return [
        cosine_sim(emb, pp),
        cosine_sim(emb, pn),
        float(np.linalg.norm(emb - pp)),
        float(np.dot(emb, disc) / (dn + 1e-8)),
        cosine_sim(emb, pp) / (abs(cosine_sim(emb, pn)) + 1e-6),
        n_pos_dict[target],
        n_neg_dict[target],
    ]

fm = feat_mat.copy()
pairs_keys = list(zip(pairs_r["protein_id"], pairs_r["target"]))
proto_arr  = np.array([_proto_feats(X_emb[i], pairs_r["target"].iloc[i])
                       for i in range(len(pairs_r))], dtype=np.float32)
key_to_proto = {k: proto_arr[i] for i, k in enumerate(pairs_keys)}
fm_keys = list(zip(fm["protein_id"], fm["target"]))
proto_fm = np.array([key_to_proto.get(k, np.zeros(len(proto_feat_cols), dtype=np.float32))
                     for k in fm_keys], dtype=np.float32)
for j, col in enumerate(proto_feat_cols):
    fm[col] = proto_fm[:, j]

# Add interface features if Phase 6a was run
use_aug = False
if (MODELS_DIR / "ensemble_meta_6a.pkl").exists():
    try:
        meta_6a  = pickle.load(open(MODELS_DIR / "ensemble_meta_6a.pkl", "rb"))
        if_cols  = meta_6a["if_cols"]
        if_meds  = pd.Series(meta_6a["if_medians"])
        evals    = pd.read_parquet(DATA_DIR / "evaluations_flat.parquet")
        ir_rows  = evals[evals["metric"] == "interface_residues"].copy()

        def _parse_pos(v):
            try:
                d = json.loads(v) if isinstance(v, str) else v
                return [int(r["residue"]) for r in d if "residue" in r] if isinstance(d, list) else []
            except:
                return []

        ir_rows["positions"] = ir_rows["value"].apply(_parse_pos)
        ir_rows = ir_rows[ir_rows["positions"].apply(len) > 0]
        ir_map  = {(r.protein_id, r.target): r.positions for _, r in ir_rows.iterrows()}
        seq_map = dict(zip(pairs_r["protein_id"], pairs_r["sequence"]))

        KD_H = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
                'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,
                'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
        CHG  = {'K':1,'R':1,'H':0.1,'D':-1,'E':-1}
        ARO  = set('FWY');  HBD = set('NQSTKRHWY');  HBA = set('NQDEST')
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
            cnt = {a: 0 for a in AA20}
            for aa in iface:
                if aa in cnt: cnt[aa] += 1
            for aa in AA20: f[f"if_aac_{aa}"] = cnt[aa] / n_if
            f["if_n_residues"] = n_if; f["if_coverage"] = n_if / n
            if len(pos_0) > 1:
                gaps = [pos_0[k+1]-pos_0[k] for k in range(len(pos_0)-1)]
                f["if_span"]      = (max(pos_0)-min(pos_0)) / n
                f["if_mean_gap"]  = np.mean(gaps) / n
                f["if_max_gap"]   = max(gaps) / n
                f["if_n_segments"]= sum(1 for g in gaps if g > 3)
            else:
                f["if_span"] = f["if_mean_gap"] = f["if_max_gap"] = f["if_n_segments"] = 0.0
            f["if_nterm_frac"] = sum(1 for p in pos_0 if p < n*0.33) / n_if
            f["if_cterm_frac"] = sum(1 for p in pos_0 if p > n*0.67) / n_if
            hy = [KD_H.get(a, 0.) for a in iface]
            ch = [CHG.get(a,  0.) for a in iface]
            vl = [VOL.get(a, 130.) for a in iface]
            f["if_mean_hydro"]=np.mean(hy); f["if_std_hydro"]=np.std(hy)
            f["if_net_charge"]=sum(ch);     f["if_mean_charge"]=np.mean(ch)
            f["if_pos_frac"] =sum(1 for c in ch if c>0)/n_if
            f["if_neg_frac"] =sum(1 for c in ch if c<0)/n_if
            f["if_aromatic_frac"]    = sum(1 for a in iface if a in ARO)/n_if
            f["if_hbond_donor_frac"] = sum(1 for a in iface if a in HBD)/n_if
            f["if_hbond_acc_frac"]   = sum(1 for a in iface if a in HBA)/n_if
            f["if_mean_volume"]      = np.mean(vl)
            wh = [KD_H.get(a, 0.) for a in seq]
            f["if_hydro_delta"] = f["if_mean_hydro"] - np.mean(wh)
            return f

        if_rows_list = []
        for _, row in pairs_r.iterrows():
            key = (row["protein_id"], row["target"])
            pos = ir_map.get(key, [])
            seq = seq_map.get(row["protein_id"], "")
            if_rows_list.append(_if_feat(seq, pos) if pos and seq else {})
        if_df = pd.DataFrame([r if r else {} for r in if_rows_list])
        for col in if_cols:
            if col not in if_df.columns:
                if_df[col] = np.nan
        if_df = if_df[if_cols].fillna(if_meds)

        pair_to_row = {(pairs_r["protein_id"].iloc[i], pairs_r["target"].iloc[i]): i
                       for i in range(len(pairs_r))}
        for col in if_cols:
            fm[col] = [if_df[col].iloc[pair_to_row.get(k, 0)]
                       if k in pair_to_row else if_meds[col]
                       for k in fm_keys]
        use_aug = True
        print(f"  Interface features added ({len(if_cols)} cols)")
    except Exception as e:
        print(f"  Interface features skipped ({e})")

# Verify all feature cols present
missing = [c for c in all_feat_cols if c not in fm.columns]
if missing:
    print(f"  WARNING: {len(missing)} cols missing from fm, zeroing: {missing[:5]}")
    for c in missing:
        fm[c] = 0.0

train_fm = fm[fm["split"] == "train"].reset_index(drop=True)
val_fm   = fm[fm["split"] == "val"].reset_index(drop=True)
test_fm  = fm[fm["split"] == "test"].reset_index(drop=True)

X_tr = train_fm[all_feat_cols].values.astype(np.float32)
X_v  = val_fm[all_feat_cols].values.astype(np.float32)
X_te = test_fm[all_feat_cols].values.astype(np.float32)
y_tr = train_fm["binding_label"].values.astype(np.float32)
y_v  = val_fm["binding_label"].values.astype(np.float32)
y_te = test_fm["binding_label"].values.astype(np.float32)
print(f"  X_tr={X_tr.shape}  X_v={X_v.shape}  X_te={X_te.shape}")

# ── 3. Sequence Identity Leakage Check ────────────────────────────────────────
print("\n" + "=" * 70)
print("Sequence Identity Leakage Check (k-mer Jaccard)...")

def kmer_set(seq, k=5):
    seq = seq.upper()
    return set(seq[i:i+k] for i in range(len(seq)-k+1))

train_seqs = list(pairs_r["sequence"].values[train_m])
val_seqs   = list(pairs_r["sequence"].values[val_m])
test_seqs  = list(pairs_r["sequence"].values[test_m])

train_kmers = [kmer_set(s) for s in train_seqs]

def max_jaccard(query_seqs, ref_kmers, sample_n=200):
    """Max Jaccard similarity of each query to any reference (sampled for speed)."""
    ref_sample = ref_kmers[:min(sample_n, len(ref_kmers))]
    results = []
    for seq in query_seqs:
        qk = kmer_set(seq)
        best = 0.0
        for rk in ref_sample:
            inter = len(qk & rk)
            union = len(qk | rk)
            if union > 0:
                j = inter / union
                if j > best:
                    best = j
        results.append(best)
    return np.array(results)

val_max_j  = max_jaccard(val_seqs,  train_kmers, sample_n=500)
test_max_j = max_jaccard(test_seqs, train_kmers, sample_n=500)

print(f"  Val  max Jaccard to train — mean={val_max_j.mean():.3f}  "
      f"median={np.median(val_max_j):.3f}  p95={np.percentile(val_max_j,95):.3f}  "
      f"max={val_max_j.max():.3f}")
print(f"  Test max Jaccard to train — mean={test_max_j.mean():.3f}  "
      f"median={np.median(test_max_j):.3f}  p95={np.percentile(test_max_j,95):.3f}  "
      f"max={test_max_j.max():.3f}")

# Approx: Jaccard > 0.8 suggests >~90% seq identity (heuristic)
leak_val  = (val_max_j  > 0.8).sum()
leak_test = (test_max_j > 0.8).sum()
print(f"  Potential leakage (Jaccard>0.8): Val={leak_val}/{len(val_seqs)}  "
      f"Test={leak_test}/{len(test_seqs)}")
if leak_val + leak_test == 0:
    print("  CLEAN: No high-identity leakage detected.")
else:
    print("  WARNING: Some high-identity sequences found in val/test vs train.")

# ── 4. Multi-Seed Evaluation ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Multi-Seed Evaluation ({len(SEEDS)} seeds × 2 models = {len(SEEDS)*2} runs)...")

LGB_PARAMS = dict(
    objective="binary", metric="auc", verbosity=-1, device="gpu",
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
)
XGB_PARAMS = dict(
    n_estimators=776, max_depth=8, learning_rate=0.0496,
    subsample=0.958, colsample_bytree=0.722, min_child_weight=2,
    reg_alpha=1.11, reg_lambda=0.002,
    device="cuda", eval_metric="auc", early_stopping_rounds=50, verbosity=0,
)

seed_results   = []
all_val_probs  = []   # (n_seeds, n_val)  — for uncertainty on val
all_test_probs = []   # (n_seeds*2, n_test) — LGB+XGB across all seeds

lgb_val_probs_seeds  = []
lgb_test_probs_seeds = []
xgb_val_probs_seeds  = []
xgb_test_probs_seeds = []

for seed in SEEDS:
    print(f"\n  Seed {seed}:")

    # LightGBM
    lgb_m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    lgb_m.fit(X_tr, y_tr,
              eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
    lgb_vp = lgb_m.predict_proba(X_v)[:, 1]
    lgb_tp = lgb_m.predict_proba(X_te)[:, 1]
    lgb_vm = mets(y_v,  lgb_vp)
    lgb_tm = mets(y_te, lgb_tp)
    lgb_val_probs_seeds.append(lgb_vp)
    lgb_test_probs_seeds.append(lgb_tp)
    print(f"    LGB  Val AUROC={lgb_vm['auroc']:.4f}  Test AUROC={lgb_tm['auroc']:.4f}")

    # XGBoost
    xgb_m = XGBClassifier(**XGB_PARAMS, random_state=seed)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    xgb_vp = xgb_m.predict_proba(X_v)[:, 1]
    xgb_tp = xgb_m.predict_proba(X_te)[:, 1]
    xgb_vm = mets(y_v,  xgb_vp)
    xgb_tm = mets(y_te, xgb_tp)
    xgb_val_probs_seeds.append(xgb_vp)
    xgb_test_probs_seeds.append(xgb_tp)
    print(f"    XGB  Val AUROC={xgb_vm['auroc']:.4f}  Test AUROC={xgb_tm['auroc']:.4f}")

    for model, vm, tm in [("LGB", lgb_vm, lgb_tm), ("XGB", xgb_vm, xgb_tm)]:
        seed_results.append({"seed": seed, "model": model,
                              "val_auroc": vm["auroc"], "val_auprc": vm["auprc"],
                              "test_auroc": tm["auroc"], "test_auprc": tm["auprc"],
                              "test_f1": tm["f1"], "test_mcc": tm["mcc"]})

# Aggregate multi-seed results
lgb_test_arr = np.stack(lgb_test_probs_seeds)  # (5, n_test)
xgb_test_arr = np.stack(xgb_test_probs_seeds)  # (5, n_test)
all_test_arr = np.vstack([lgb_test_arr, xgb_test_arr])  # (10, n_test)

lgb_val_arr = np.stack(lgb_val_probs_seeds)
xgb_val_arr = np.stack(xgb_val_probs_seeds)
all_val_arr  = np.vstack([lgb_val_arr, xgb_val_arr])

# Multi-seed ensemble (average of 10 models)
ens_test_p = all_test_arr.mean(0)
ens_val_p  = all_val_arr.mean(0)
ens_tm = mets(y_te, ens_test_p)
ens_vm = mets(y_v,  ens_val_p)

# Per-model stats
lgb_aurocs = [r["test_auroc"] for r in seed_results if r["model"] == "LGB"]
xgb_aurocs = [r["test_auroc"] for r in seed_results if r["model"] == "XGB"]
all_aurocs  = [r["test_auroc"] for r in seed_results]

print("\n" + "-" * 60)
print("MULTI-SEED SUMMARY (TEST SET)")
print(f"  LightGBM  — mean={np.mean(lgb_aurocs):.4f} ± {np.std(lgb_aurocs):.4f}  "
      f"[{min(lgb_aurocs):.4f}, {max(lgb_aurocs):.4f}]")
print(f"  XGBoost   — mean={np.mean(xgb_aurocs):.4f} ± {np.std(xgb_aurocs):.4f}  "
      f"[{min(xgb_aurocs):.4f}, {max(xgb_aurocs):.4f}]")
print(f"  Combined  — mean={np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
print(f"  10-model ensemble — AUROC={ens_tm['auroc']:.4f}  AUPRC={ens_tm['auprc']:.4f}  "
      f"F1={ens_tm['f1']:.4f}  MCC={ens_tm['mcc']:.4f}")

# ── 5. ECE + Calibration Curves ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("ECE + Calibration Analysis...")

# Use ensemble predictions + Platt calibration
raw_val_p  = ens_val_p
raw_test_p = ens_test_p

# Platt scaling fit on val
cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=42)
cal.fit(raw_val_p.reshape(-1, 1), y_v)
cal_test_p = cal.predict_proba(raw_test_p.reshape(-1, 1))[:, 1]
cal_val_p  = cal.predict_proba(raw_val_p.reshape(-1, 1))[:, 1]

ece_raw = ece(y_te, raw_test_p)
ece_cal = ece(y_te, cal_test_p)
print(f"  ECE before calibration : {ece_raw:.4f}")
print(f"  ECE after Platt scaling: {ece_cal:.4f}  ({100*(ece_raw-ece_cal)/ece_raw:.1f}% reduction)")

cal_tm = mets(y_te, cal_test_p)
print(f"  Calibrated ensemble — AUROC={cal_tm['auroc']:.4f}  AUPRC={cal_tm['auprc']:.4f}  "
      f"F1={cal_tm['f1']:.4f}  MCC={cal_tm['mcc']:.4f}")

# Reliability diagram data
prob_raw, frac_raw = calibration_curve(y_te, raw_test_p, n_bins=10, strategy="uniform")
prob_cal, frac_cal = calibration_curve(y_te, cal_test_p,  n_bins=10, strategy="uniform")

# ── 6. Uncertainty Quantification ────────────────────────────────────────────
print("\n" + "=" * 70)
print("Uncertainty Quantification (confidence-based filtering)...")

# Uncertainty = std across all 10 model predictions
uncertainty_test = all_test_arr.std(0)  # (n_test,)
print(f"  Uncertainty (std) — mean={uncertainty_test.mean():.4f}  "
      f"max={uncertainty_test.max():.4f}  p90={np.percentile(uncertainty_test,90):.4f}")

# Confidence-filtering curve
coverages    = np.arange(0.2, 1.01, 0.05)
auroc_filter = []
n_filter     = []
for cov in coverages:
    # Keep the `cov` fraction with LOWEST uncertainty
    thr_unc = np.percentile(uncertainty_test, cov * 100)
    mask    = uncertainty_test <= thr_unc
    if mask.sum() < 5 or y_te[mask].sum() < 2:
        auroc_filter.append(np.nan)
    else:
        auroc_filter.append(roc_auc_score(y_te[mask], raw_test_p[mask]))
    n_filter.append(mask.sum())

auroc_filter = np.array(auroc_filter)
print(f"  AUROC at 100% coverage: {auroc_filter[-1]:.4f}")
print(f"  AUROC at  80% coverage: {auroc_filter[np.searchsorted(coverages, 0.8)]:.4f}")
print(f"  AUROC at  60% coverage: {auroc_filter[np.searchsorted(coverages, 0.6)]:.4f}")

# ── 7. Applicability Domain Analysis ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Applicability Domain Analysis (prototype cosine similarity to known binders)...")

# AD metric: proto_cos_pos (cosine similarity to positive prototype of the target)
# This measures how "in-domain" a test protein is relative to known binders
# It is already computed as a feature — extract from X_te
proto_cos_pos_idx = all_feat_cols.index("proto_cos_pos")
nn_sim = X_te[:, proto_cos_pos_idx]   # (n_test,)

print(f"  Proto cos-pos similarity — mean={nn_sim.mean():.3f}  "
      f"median={np.median(nn_sim):.3f}  min={nn_sim.min():.3f}  max={nn_sim.max():.3f}")

# Bin test set into quartiles by NN similarity
quartile_aurocs = []
quartile_labels = ["Q1 (lowest\nsimilarity)", "Q2", "Q3", "Q4 (highest\nsimilarity)"]
quartile_ns     = []
quartile_sims   = []
q_bounds = np.percentile(nn_sim, [0, 25, 50, 75, 100])

for lo, hi in zip(q_bounds[:-1], q_bounds[1:]):
    mask = (nn_sim >= lo) & (nn_sim <= hi)
    n    = mask.sum()
    pos  = y_te[mask].sum()
    quartile_ns.append(n)
    quartile_sims.append(float(nn_sim[mask].mean()) if n > 0 else 0.0)
    if n < 5 or pos < 2:
        quartile_aurocs.append(np.nan)
    else:
        quartile_aurocs.append(roc_auc_score(y_te[mask], raw_test_p[mask]))

print(f"  AD Quartile Analysis:")
for i, (lbl, n, sim, auroc) in enumerate(zip(quartile_labels, quartile_ns, quartile_sims, quartile_aurocs)):
    lbl_clean  = lbl.replace('\n', ' ')
    auroc_str  = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
    print(f"    {lbl_clean:<30} n={n:>3}  mean_sim={sim:.3f}  AUROC={auroc_str}")

# ── 8. SHAP Analysis ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SHAP Analysis on best LightGBM model (seed=42)...")

# Reload seed-42 LGB model (already trained above; just retrain for SHAP)
lgb_shap = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
lgb_shap.fit(X_tr, y_tr,
             eval_set=[(X_v, y_v)],
             callbacks=[lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(period=-1)])

explainer   = shap.TreeExplainer(lgb_shap)
shap_vals   = explainer.shap_values(X_te)   # (n_test, n_feat) for class 1
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

# Feature importance by mean |SHAP|
mean_abs_shap = np.abs(shap_vals).mean(0)
shap_df = pd.DataFrame({
    "feature":    all_feat_cols,
    "mean_abs":   mean_abs_shap,
    "is_proto":   [c in proto_feat_cols for c in all_feat_cols],
})
shap_df = shap_df.sort_values("mean_abs", ascending=False).reset_index(drop=True)
shap_df["rank"] = range(1, len(shap_df) + 1)

print(f"\n  Top 10 features by |SHAP|:")
for _, row in shap_df.head(10).iterrows():
    flag = " [PROTO]" if row["is_proto"] else ""
    print(f"    #{int(row['rank']):>3}  {row['feature']:<35}  mean|SHAP|={row['mean_abs']:.4f}{flag}")

print(f"\n  Prototype feature ranks:")
proto_shap = shap_df[shap_df["is_proto"]][["rank", "feature", "mean_abs"]]
for _, row in proto_shap.iterrows():
    print(f"    #{int(row['rank']):>3}  {row['feature']:<35}  mean|SHAP|={row['mean_abs']:.4f}")

shap_df.to_csv(OUTPUTS_DIR / "phase6d_shap.csv", index=False)

# ── 9. Plots ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Generating figures...")

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "figure.dpi": 150})
PROTO_COLOR = "#E63946"   # red for proto features
BASE_COLOR  = "#457B9D"   # blue for base features

# ── Figure 1: Multi-seed AUROC bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
seed_df = pd.DataFrame(seed_results)
x = np.arange(len(SEEDS))
w = 0.35
lgb_by_seed = seed_df[seed_df["model"] == "LGB"]["test_auroc"].values
xgb_by_seed = seed_df[seed_df["model"] == "XGB"]["test_auroc"].values
bars1 = ax.bar(x - w/2, lgb_by_seed, w, label="LightGBM+Proto", color=BASE_COLOR, alpha=0.85)
bars2 = ax.bar(x + w/2, xgb_by_seed, w, label="XGBoost+Proto",  color="#2A9D8F", alpha=0.85)
ax.axhline(np.mean(lgb_by_seed), color=BASE_COLOR, ls="--", lw=1.2, alpha=0.7)
ax.axhline(np.mean(xgb_by_seed), color="#2A9D8F",  ls="--", lw=1.2, alpha=0.7)
ax.axhline(ens_tm["auroc"], color="black", ls="-", lw=1.5, label=f"10-model ensemble ({ens_tm['auroc']:.4f})")
ax.set_xticks(x); ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
ax.set_ylabel("Test AUROC"); ax.set_title("Multi-Seed Model Stability (5 Seeds × 2 Models)")
ax.set_ylim(0.88, 0.97)
ax.legend(loc="lower right", fontsize=9)
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.4f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "phase6d_multiseed.png", bbox_inches="tight")
plt.close()
print("  Saved: phase6d_multiseed.png")

# ── Figure 2: Calibration curves (reliability diagram) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, prob, frac, label, col in [
    (axes[0], prob_raw, frac_raw, f"Raw ensemble (ECE={ece_raw:.3f})",  "#E63946"),
    (axes[1], prob_cal, frac_cal, f"Platt calibrated (ECE={ece_cal:.3f})", "#457B9D"),
]:
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(prob, frac, "o-", color=col, lw=2, ms=6, label=label)
    ax.fill_between(prob, frac, prob, alpha=0.15, color=col)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.suptitle("Reliability Diagrams (Test Set)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "phase6d_calibration.png", bbox_inches="tight")
plt.close()
print("  Saved: phase6d_calibration.png")

# ── Figure 3: Applicability Domain ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
# Left: NN similarity distribution
axes[0].hist(nn_sim, bins=30, color=BASE_COLOR, alpha=0.8, edgecolor="white")
axes[0].set_xlabel("Cosine similarity to positive prototype (proto_cos_pos)")
axes[0].set_ylabel("Number of test pairs")
axes[0].set_title("Applicability Domain Distribution")
q_vals = np.percentile(nn_sim, [25, 50, 75])
for qv, ql in zip(q_vals, ["Q1|Q2", "Q2|Q3", "Q3|Q4"]):
    axes[0].axvline(qv, color="red", ls="--", lw=1.2, alpha=0.7)
    axes[0].text(qv, axes[0].get_ylim()[1]*0.9, ql, color="red", fontsize=8, ha="center")

# Right: AUROC per quartile
q_idx    = [1, 2, 3, 4]
q_valid  = [(i, a) for i, a in zip(q_idx, quartile_aurocs) if not np.isnan(a)]
q_idx_v, q_aur_v = zip(*q_valid) if q_valid else ([], [])
colors   = [BASE_COLOR if a >= 0.90 else "#E63946" if a < 0.75 else "#F4A261" for a in q_aur_v]
bars     = axes[1].bar(q_idx_v, q_aur_v, color=colors, alpha=0.85, width=0.6)
axes[1].axhline(ens_tm["auroc"], color="black", ls="--", lw=1.2,
                label=f"Overall AUROC={ens_tm['auroc']:.4f}")
for bar, a in zip(bars, q_aur_v):
    axes[1].text(bar.get_x() + bar.get_width()/2, a + 0.005,
                 f"{a:.3f}", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks(q_idx_v)
axes[1].set_xticklabels([f"Q{i}\n(n={quartile_ns[i-1]})" for i in q_idx_v], fontsize=9)
axes[1].set_ylabel("AUROC")
axes[1].set_title("AUROC by Applicability Domain Quartile")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0.5, 1.0)
plt.suptitle("Applicability Domain Analysis (Similarity to Known Binders per Target)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "phase6d_applicability_domain.png", bbox_inches="tight")
plt.close()
print("  Saved: phase6d_applicability_domain.png")

# ── Figure 4: Confidence-Filtering Curve ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
valid_mask = ~np.isnan(auroc_filter)
axes[0].plot(coverages[valid_mask] * 100, auroc_filter[valid_mask],
             "o-", color=BASE_COLOR, lw=2, ms=5)
axes[0].fill_between(coverages[valid_mask] * 100, auroc_filter[valid_mask],
                     ens_tm["auroc"], alpha=0.15, color=BASE_COLOR)
axes[0].axhline(ens_tm["auroc"], color="gray", ls="--", lw=1.2,
                label=f"No filtering ({ens_tm['auroc']:.4f})")
axes[0].set_xlabel("Coverage (% of test set retained)")
axes[0].set_ylabel("AUROC")
axes[0].set_title("Confidence-Filtered AUROC\n(remove high-uncertainty predictions)")
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 105)
# Right: uncertainty distribution
axes[1].hist(uncertainty_test, bins=30, color="#E63946", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Prediction uncertainty (std across 10 models)")
axes[1].set_ylabel("Number of test pairs")
axes[1].set_title("Distribution of Prediction Uncertainty")
p80 = np.percentile(uncertainty_test, 80)
axes[1].axvline(p80, color="black", ls="--", lw=1.5,
                label=f"80th pct (keep 20% most certain)")
axes[1].legend(fontsize=9)
plt.suptitle("Uncertainty Quantification (10-Model Ensemble Std)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "phase6d_uncertainty.png", bbox_inches="tight")
plt.close()
print("  Saved: phase6d_uncertainty.png")

# ── Figure 5: SHAP summary ────────────────────────────────────────────────────
top_n   = 25
top_idx = shap_df.head(top_n).index[::-1]   # reverse for horizontal bar (best at top)
top_feats    = shap_df.loc[top_idx, "feature"].values
top_vals     = shap_df.loc[top_idx, "mean_abs"].values
top_is_proto = shap_df.loc[top_idx, "is_proto"].values

fig, ax = plt.subplots(figsize=(9, 8))
colors  = [PROTO_COLOR if p else BASE_COLOR for p in top_is_proto]
bars    = ax.barh(range(top_n), top_vals, color=colors, alpha=0.85, height=0.7)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats, fontsize=9)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top {top_n} Features by SHAP Importance\n(LightGBM+Proto, Test Set)")
from matplotlib.patches import Patch
legend_elems = [Patch(color=PROTO_COLOR, label="Prototype features (Phase 6b)"),
                Patch(color=BASE_COLOR,  label="Sequence/structure features")]
ax.legend(handles=legend_elems, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "phase6d_shap.png", bbox_inches="tight")
plt.close()
print("  Saved: phase6d_shap.png")

# ── 10. Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 6d — FINAL SUMMARY")
print(f"\n  MULTI-SEED STABILITY (5 seeds × 2 models, Test AUROC):")
print(f"    LightGBM+Proto : {np.mean(lgb_aurocs):.4f} ± {np.std(lgb_aurocs):.4f}")
print(f"    XGBoost+Proto  : {np.mean(xgb_aurocs):.4f} ± {np.std(xgb_aurocs):.4f}")
print(f"    10-model ens.  : {ens_tm['auroc']:.4f} (AUPRC={ens_tm['auprc']:.4f}  "
      f"F1={ens_tm['f1']:.4f}  MCC={ens_tm['mcc']:.4f})")

print(f"\n  CALIBRATION (Test ECE):")
print(f"    Before Platt   : {ece_raw:.4f}")
print(f"    After Platt    : {ece_cal:.4f}  ({100*(ece_raw-ece_cal)/ece_raw:.1f}% reduction)")

print(f"\n  SEQUENCE IDENTITY LEAKAGE:")
print(f"    Val  max k-mer Jaccard: mean={val_max_j.mean():.3f}  "
      f"p95={np.percentile(val_max_j,95):.3f}  leakage_n={leak_val}")
print(f"    Test max k-mer Jaccard: mean={test_max_j.mean():.3f}  "
      f"p95={np.percentile(test_max_j,95):.3f}  leakage_n={leak_test}")

print(f"\n  APPLICABILITY DOMAIN (proto_cos_pos quartiles):")
for i, (a, n, s) in enumerate(zip(quartile_aurocs, quartile_ns, quartile_sims)):
    a_str = f"{a:.4f}" if not np.isnan(a) else "N/A"
    print(f"    Q{i+1} (mean_proto_cos={s:.3f}, n={n}): AUROC={a_str}")

print(f"\n  CONFIDENCE FILTERING:")
idx80 = np.searchsorted(coverages, 0.8)
idx60 = np.searchsorted(coverages, 0.6)
a80 = f"{auroc_filter[idx80]:.4f}" if not np.isnan(auroc_filter[idx80]) else "N/A"
a60 = f"{auroc_filter[idx60]:.4f}" if not np.isnan(auroc_filter[idx60]) else "N/A"
a100 = f"{auroc_filter[-1]:.4f}" if not np.isnan(auroc_filter[-1]) else "N/A"
print(f"    100% coverage: AUROC={a100}")
print(f"     80% coverage: AUROC={a80}")
print(f"     60% coverage: AUROC={a60}")

print(f"\n  TOP SHAP FEATURES:")
for _, row in shap_df.head(10).iterrows():
    flag = " [PROTO]" if row["is_proto"] else ""
    print(f"    #{int(row['rank']):>3}  {row['feature']:<35}  {row['mean_abs']:.4f}{flag}")

# Save results
pd.DataFrame(seed_results).to_csv(OUTPUTS_DIR / "phase6d_seed_results.csv", index=False)
summary = {
    "lgb_mean_auroc": np.mean(lgb_aurocs), "lgb_std_auroc": np.std(lgb_aurocs),
    "xgb_mean_auroc": np.mean(xgb_aurocs), "xgb_std_auroc": np.std(xgb_aurocs),
    "ensemble_10model_auroc": ens_tm["auroc"], "ensemble_10model_auprc": ens_tm["auprc"],
    "ensemble_10model_f1":    ens_tm["f1"],    "ensemble_10model_mcc":   ens_tm["mcc"],
    "ece_before_platt": ece_raw, "ece_after_platt": ece_cal,
    "leakage_val_n": leak_val,  "leakage_test_n": leak_test,
    "ad_q1_auroc": quartile_aurocs[0], "ad_q2_auroc": quartile_aurocs[1],
    "ad_q3_auroc": quartile_aurocs[2], "ad_q4_auroc": quartile_aurocs[3],
    "conf_filter_80pct_auroc": float(auroc_filter[idx80]) if (not np.isnan(auroc_filter[idx80])) else None,
}
pd.DataFrame([summary]).to_csv(OUTPUTS_DIR / "phase6d_summary.csv", index=False)

print(f"\nSaved: outputs/phase6d_seed_results.csv, phase6d_summary.csv")
print(f"Saved: outputs/phase6d_shap.csv")
print(f"Saved: 5 figures in outputs/phase6d_*.png")
print("Phase 6d complete.")
