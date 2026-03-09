"""
Phase 5b: Ensemble + Calibration + Per-Target Threshold Optimization
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Steps:
  1. Weighted ensemble: LightGBM + XGBoost + TargetAwareMLP (weights optimized on val)
  2. Platt scaling calibration (fit on val, evaluate on test)
  3. Per-target threshold optimization (tuned on val, applied to test)
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
# Load feature matrix as Phase 3 did — already contains imputed values + split column
feat_mat  = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")
feat_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()

train_fm = feat_mat[feat_mat["split"] == "train"].reset_index(drop=True)
val_fm   = feat_mat[feat_mat["split"] == "val"].reset_index(drop=True)
test_fm  = feat_mat[feat_mat["split"] == "test"].reset_index(drop=True)

X_hc_train = train_fm[feat_cols].values.astype(np.float32)
X_hc_val   = val_fm[feat_cols].values.astype(np.float32)
X_hc_test  = test_fm[feat_cols].values.astype(np.float32)
y_val      = train_fm["binding_label"].values  # placeholder — overwritten below
y_val      = val_fm["binding_label"].values.astype(np.float32)
y_test     = test_fm["binding_label"].values.astype(np.float32)

# Also load pairs for sequence / target info (needed for TargetAwareMLP)
pairs = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
pairs = pairs.merge(feat_mat[["protein_id", "target"] + feat_cols],
                    on=["protein_id", "target"], how="left", suffixes=("", "_feat"))

esm_emb = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_map = {pid: i for i, pid in enumerate(esm_ids)}

emb_dim = esm_emb.shape[1]
X_emb   = np.zeros((len(pairs), emb_dim), dtype=np.float32)
for i, pid in enumerate(pairs["protein_id"]):
    idx = esm_map.get(pid)
    if idx is not None:
        X_emb[i] = esm_emb[idx]

X_hc_all = np.nan_to_num(pairs[feat_cols].values.astype(np.float32), nan=0.0)
X_emb    = np.nan_to_num(X_emb, nan=0.0)
X_comb   = np.concatenate([X_emb, X_hc_all], axis=1)

y_bind = pairs["binding_label"].values.astype(np.float32)

target_names = sorted(pairs["target"].unique())
target_enc   = {t: i for i, t in enumerate(target_names)}
t_idx_arr    = pairs["target"].map(target_enc).values.astype(np.int64)
n_targets    = len(target_names)

train_m = (pairs["split"] == "train").values
val_m   = (pairs["split"] == "val").values
test_m  = (pairs["split"] == "test").values

print(f"  Train: {train_m.sum()}  Val: {val_m.sum()}  Test: {test_m.sum()}")

def metrics(y_true, y_prob, threshold=0.5):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    yp    = (y_prob >= threshold).astype(int)
    return auroc, auprc, f1_score(y_true, yp, zero_division=0), matthews_corrcoef(y_true, yp)

# ── 1. Load Phase 3 models and get predictions ────────────────────────────────
print("\n" + "="*70)
print("Step 1: Loading Phase 3 models (LightGBM, XGBoost)...")

lgb_model = joblib.load(MODELS_DIR / "lgb_best.pkl")
xgb_model = joblib.load(MODELS_DIR / "xgb_best.pkl")

lgb_val_prob  = lgb_model.predict_proba(X_hc_val)[:, 1]
lgb_test_prob = lgb_model.predict_proba(X_hc_test)[:, 1]
xgb_val_prob  = xgb_model.predict_proba(X_hc_val)[:, 1]
xgb_test_prob = xgb_model.predict_proba(X_hc_test)[:, 1]

print(f"  LightGBM  — Val AUROC={roc_auc_score(y_val,  lgb_val_prob):.4f}  "
      f"Test AUROC={roc_auc_score(y_test, lgb_test_prob):.4f}")
print(f"  XGBoost   — Val AUROC={roc_auc_score(y_val,  xgb_val_prob):.4f}  "
      f"Test AUROC={roc_auc_score(y_test, xgb_test_prob):.4f}")

# ── 2. Load Phase 5 TargetAwareMLP and get predictions ───────────────────────
print("\nLoading Phase 5 TargetAwareMLP...")

class TargetAwareMLP(nn.Module):
    def __init__(self, in_dim, n_targets, t_emb=32, dropout=0.3):
        super().__init__()
        self.t_emb = nn.Embedding(n_targets, t_emb)
        d = in_dim + t_emb
        self.net = nn.Sequential(
            nn.Linear(d,   512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(128, 1),
        )
    def forward(self, x, t_idx):
        return self.net(torch.cat([x, self.t_emb(t_idx)], dim=-1))

modelC = TargetAwareMLP(X_comb.shape[1], n_targets)
modelC.load_state_dict(torch.load(MODELS_DIR / "target_aware_mlp.pt", map_location="cpu"))
modelC.to(device)
modelC.eval()

def get_mlp_probs(mask):
    ds  = TensorDataset(torch.tensor(X_comb[mask], dtype=torch.float32),
                        torch.tensor(t_idx_arr[mask], dtype=torch.long))
    ldr = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for xb, tb in ldr:
            logits = modelC(xb.to(device), tb.to(device)).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)

mlp_val_prob  = get_mlp_probs(val_m)
mlp_test_prob = get_mlp_probs(test_m)
print(f"  TargetAwareMLP — Val AUROC={roc_auc_score(y_val,  mlp_val_prob):.4f}  "
      f"Test AUROC={roc_auc_score(y_test, mlp_test_prob):.4f}")

# ── 3. Optimize ensemble weights on validation set ────────────────────────────
print("\n" + "="*70)
print("Step 2: Optimizing ensemble weights on validation set...")

val_preds  = np.stack([lgb_val_prob,  xgb_val_prob,  mlp_val_prob],  axis=1)
test_preds = np.stack([lgb_test_prob, xgb_test_prob, mlp_test_prob], axis=1)

def neg_auprc(w):
    w = np.abs(w)
    w = w / w.sum()
    return -average_precision_score(y_val, (val_preds * w).sum(axis=1))

# Grid search seed for Nelder-Mead
best_ap, best_w0 = -1, np.array([0.4, 0.4, 0.2])
for w0 in np.arange(0.2, 0.7, 0.1):
    for w1 in np.arange(0.1, 0.5, 0.1):
        w2 = 1.0 - w0 - w1
        if w2 <= 0:
            continue
        w = np.array([w0, w1, w2])
        ap = average_precision_score(y_val, (val_preds * w).sum(axis=1))
        if ap > best_ap:
            best_ap, best_w0 = ap, w.copy()

res   = minimize(neg_auprc, best_w0, method='Nelder-Mead',
                 options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
w_opt = np.abs(res.x) / np.abs(res.x).sum()
print(f"  Weights — LightGBM: {w_opt[0]:.3f}  XGBoost: {w_opt[1]:.3f}  "
      f"TargetAwareMLP: {w_opt[2]:.3f}")

ens_val_prob  = (val_preds  * w_opt).sum(axis=1)
ens_test_prob = (test_preds * w_opt).sum(axis=1)
simple_val    = val_preds.mean(axis=1)
simple_test   = test_preds.mean(axis=1)

m_wt  = metrics(y_val,  ens_val_prob)
m_wt_t = metrics(y_test, ens_test_prob)
m_si  = metrics(y_val,  simple_val)
m_si_t = metrics(y_test, simple_test)

print(f"  Weighted avg Val  — AUROC={m_wt[0]:.4f}  AUPRC={m_wt[1]:.4f}  F1={m_wt[2]:.4f}  MCC={m_wt[3]:.4f}")
print(f"  Weighted avg Test — AUROC={m_wt_t[0]:.4f}  AUPRC={m_wt_t[1]:.4f}  F1={m_wt_t[2]:.4f}  MCC={m_wt_t[3]:.4f}")
print(f"  Simple avg   Val  — AUROC={m_si[0]:.4f}  AUPRC={m_si[1]:.4f}  F1={m_si[2]:.4f}  MCC={m_si[3]:.4f}")
print(f"  Simple avg   Test — AUROC={m_si_t[0]:.4f}  AUPRC={m_si_t[1]:.4f}  F1={m_si_t[2]:.4f}  MCC={m_si_t[3]:.4f}")

# Pick best ensemble by val AUPRC
if m_wt[1] >= m_si[1]:
    ens_val_best, ens_test_best = ens_val_prob, ens_test_prob
    ens_name = "WeightedEnsemble"
else:
    ens_val_best, ens_test_best = simple_val, simple_test
    w_opt = np.array([1/3, 1/3, 1/3])
    ens_name = "SimpleAvgEnsemble"

print(f"  Selected: {ens_name}")

# ── 4. Platt scaling calibration ─────────────────────────────────────────────
print("\n" + "="*70)
print("Step 3: Platt scaling calibration (fit on val, apply to test)...")

cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=RANDOM_SEED)
cal.fit(ens_val_best.reshape(-1, 1), y_val)

cal_val_prob  = cal.predict_proba(ens_val_best.reshape(-1, 1))[:, 1]
cal_test_prob = cal.predict_proba(ens_test_best.reshape(-1, 1))[:, 1]

m_cal_v = metrics(y_val,  cal_val_prob)
m_cal_t = metrics(y_test, cal_test_prob)
print(f"  Calibrated Val  — AUROC={m_cal_v[0]:.4f}  AUPRC={m_cal_v[1]:.4f}  F1={m_cal_v[2]:.4f}  MCC={m_cal_v[3]:.4f}")
print(f"  Calibrated Test — AUROC={m_cal_t[0]:.4f}  AUPRC={m_cal_t[1]:.4f}  F1={m_cal_t[2]:.4f}  MCC={m_cal_t[3]:.4f}")

# ── 5. Per-target threshold optimization ─────────────────────────────────────
print("\n" + "="*70)
print("Step 4: Per-target threshold optimization (val → test)...")

val_targets  = pairs["target"].values[val_m]
test_targets = pairs["target"].values[test_m]

thresholds = {}
for t in target_names:
    t_mask = val_targets == t
    if t_mask.sum() < 5:
        thresholds[t] = 0.5
        continue
    yt = y_val[t_mask]
    yp = cal_val_prob[t_mask]
    if yt.sum() == 0 or yt.sum() == t_mask.sum():
        thresholds[t] = 0.5
        continue
    best_f1, best_thr = -1, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(yt, (yp >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    thresholds[t] = best_thr

print(f"  Per-target thresholds (val-optimized):")
for t in sorted(thresholds):
    t_mask = val_targets == t
    if t_mask.sum() >= 5:
        n_pos = int(y_val[val_targets == t].sum())
        n_tot = t_mask.sum()
        print(f"    {t:<45} thr={thresholds[t]:.2f}  (val: {n_pos}/{n_tot} pos)")

# Apply per-target thresholds to test
y_pred_pt = np.zeros(len(y_test), dtype=int)
for t in target_names:
    t_mask = test_targets == t
    if t_mask.sum() == 0:
        continue
    y_pred_pt[t_mask] = (cal_test_prob[t_mask] >= thresholds[t]).astype(int)

auroc_pt = roc_auc_score(y_test, cal_test_prob)
auprc_pt = average_precision_score(y_test, cal_test_prob)
f1_pt    = f1_score(y_test, y_pred_pt, zero_division=0)
mcc_pt   = matthews_corrcoef(y_test, y_pred_pt)
print(f"\n  Calibrated + Per-Target Thr Test — AUROC={auroc_pt:.4f}  AUPRC={auprc_pt:.4f}  "
      f"F1={f1_pt:.4f}  MCC={mcc_pt:.4f}")

# ── 6. Per-target breakdown on final model ────────────────────────────────────
print("\n" + "="*70)
print("Per-target results (final calibrated ensemble):")
print(f"  {'Target':<45}  {'N':>4}  {'Pos':>3}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>6}")
print("  " + "-"*80)

per_target_rows = []
for t in target_names:
    t_mask = test_targets == t
    if t_mask.sum() == 0:
        continue
    yt = y_test[t_mask]
    yp = cal_test_prob[t_mask]
    yp_bin = (yp >= thresholds[t]).astype(int)
    n_pos  = int(yt.sum())
    if n_pos == 0 or n_pos == len(yt):
        au, ap, f1v = float('nan'), float('nan'), float('nan')
    else:
        au = roc_auc_score(yt, yp)
        ap = average_precision_score(yt, yp)
        f1v = f1_score(yt, yp_bin, zero_division=0)
    print(f"  {t:<45}  {t_mask.sum():>4}  {n_pos:>3}  "
          f"{au:>7.4f}  {ap:>7.4f}  {f1v:>6.4f}")
    per_target_rows.append({"target": t, "n": t_mask.sum(), "n_pos": n_pos,
                             "auroc": au, "auprc": ap, "f1": f1v,
                             "threshold": thresholds[t]})

# ── 7. Final summary ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 5b — FINAL SUMMARY (TEST SET)")
print(f"  {'Model':<45}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>7}  {'MCC':>7}")
print("  " + "-"*75)
print(f"  {'--- Baselines ---':<45}")
print(f"  {'LightGBM (Phase 3 best)':<45}  {0.8925:>7.4f}  {0.7132:>7.4f}  {0.6777:>7.4f}  {0.6247:>7.4f}")
print(f"  {'XGBoost (Phase 3)':<45}  {0.8796:>7.4f}  {0.6820:>7.4f}  {0.6066:>7.4f}  {0.5385:>7.4f}")
print(f"  {'TargetAwareMLP (Phase 5)':<45}  {0.8427:>7.4f}  {0.5613:>7.4f}  {0.5280:>7.4f}  {0.4406:>7.4f}")
print(f"  {'--- Ensemble ---':<45}")
print(f"  {ens_name:<45}  {m_wt_t[0]:>7.4f}  {m_wt_t[1]:>7.4f}  {m_wt_t[2]:>7.4f}  {m_wt_t[3]:>7.4f}")
print(f"  {'Calibrated Ensemble':<45}  {m_cal_t[0]:>7.4f}  {m_cal_t[1]:>7.4f}  {m_cal_t[2]:>7.4f}  {m_cal_t[3]:>7.4f}")
print(f"  {'Calibrated + Per-Target Threshold':<45}  {auroc_pt:>7.4f}  {auprc_pt:>7.4f}  {f1_pt:>7.4f}  {mcc_pt:>7.4f}")

# ── 8. Save ───────────────────────────────────────────────────────────────────
summary_rows = [
    {"model": "LightGBM_Phase3",              "auroc": 0.8925, "auprc": 0.7132, "f1": 0.6777, "mcc": 0.6247},
    {"model": "XGBoost_Phase3",               "auroc": 0.8796, "auprc": 0.6820, "f1": 0.6066, "mcc": 0.5385},
    {"model": "TargetAwareMLP_Phase5",         "auroc": 0.8427, "auprc": 0.5613, "f1": 0.5280, "mcc": 0.4406},
    {"model": ens_name,                        "auroc": m_wt_t[0], "auprc": m_wt_t[1], "f1": m_wt_t[2], "mcc": m_wt_t[3]},
    {"model": "CalibratedEnsemble",            "auroc": m_cal_t[0], "auprc": m_cal_t[1], "f1": m_cal_t[2], "mcc": m_cal_t[3]},
    {"model": "CalibratedEnsemble_PerTgtThr",  "auroc": auroc_pt, "auprc": auprc_pt, "f1": f1_pt, "mcc": mcc_pt},
]
pd.DataFrame(summary_rows).to_csv(OUTPUTS_DIR / "phase5b_summary.csv", index=False)
pd.DataFrame(per_target_rows).to_csv(OUTPUTS_DIR / "phase5b_per_target.csv", index=False)

ensemble_meta = {
    "model_names":  ["LightGBM", "XGBoost", "TargetAwareMLP"],
    "weights":       w_opt,
    "thresholds":    thresholds,
    "target_names":  target_names,
    "target_enc":    target_enc,
}
with open(MODELS_DIR / "ensemble_meta.pkl", "wb") as f:
    pickle.dump(ensemble_meta, f)
joblib.dump(cal, MODELS_DIR / "calibrator.pkl")

print(f"\nSaved: outputs/phase5b_summary.csv")
print(f"Saved: outputs/phase5b_per_target.csv")
print(f"Saved: models/ensemble_meta.pkl")
print(f"Saved: models/calibrator.pkl")
print("Phase 5b complete.")
