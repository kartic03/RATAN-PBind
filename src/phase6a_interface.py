"""
Phase 6a: Interface Residue Feature Engineering
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Key insight: interface_residues field (Boltz2-predicted binding interface)
  - Coverage: 1,072 / 2,630 binding pairs (40.8%), almost all nipah + some fcrn/il7r/pd-l1
  - Mean ~66 interface residues vs ~114 total sequence length
  - Extracting physicochemical features SPECIFICALLY at interface vs. whole protein

Part 1 of 2:
  - Handcrafted interface features (~39 features)
  - Retrain LightGBM + XGBoost with augmented features
  - Ensemble with calibration + per-target thresholds

Part 2 (phase6b_interface_esm2.py):
  - Re-run ESM-2, pool over interface positions only → interface embedding
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from pathlib import Path
import joblib, pickle, warnings
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

# ── Physicochemical lookup tables ─────────────────────────────────────────────
KD_HYDRO = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}
CHARGE   = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}
AROMATIC = set('FWY')
HBOND_D  = set('NQSTKRHWY')
HBOND_A  = set('NQDEST')
VOLUME   = {
    'G': 60, 'A': 89, 'S': 96, 'P': 112, 'V': 117, 'T': 116, 'C': 114,
    'I': 166, 'L': 166, 'N': 114, 'D': 111, 'Q': 144, 'K': 168, 'E': 138,
    'M': 162, 'H': 153, 'F': 190, 'R': 173, 'Y': 194, 'W': 228,
}
AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def interface_features(sequence: str, positions: list) -> dict:
    """~39 features derived from interface residues (1-indexed positions)."""
    seq = sequence.upper()
    n   = len(seq)
    if not positions or n == 0:
        return {}
    pos_0 = [p - 1 for p in positions if 0 <= p - 1 < n]
    if not pos_0:
        return {}
    iface = [seq[p] for p in pos_0]
    n_if  = len(iface)
    feat  = {}

    # AA composition at interface (20 features)
    cnt = {aa: 0 for aa in AA20}
    for aa in iface:
        if aa in cnt:
            cnt[aa] += 1
    for aa in AA20:
        feat[f"if_aac_{aa}"] = cnt[aa] / n_if

    # Size & coverage
    feat["if_n_residues"] = n_if
    feat["if_coverage"]   = n_if / n

    # Geometry
    if len(pos_0) > 1:
        gaps = [pos_0[i+1] - pos_0[i] for i in range(len(pos_0) - 1)]
        feat["if_span"]      = (max(pos_0) - min(pos_0)) / n
        feat["if_mean_gap"]  = np.mean(gaps) / n
        feat["if_max_gap"]   = max(gaps) / n
        feat["if_n_segments"]= sum(1 for g in gaps if g > 3)
    else:
        feat["if_span"] = feat["if_mean_gap"] = feat["if_max_gap"] = 0.0
        feat["if_n_segments"] = 0.0

    feat["if_nterm_frac"] = sum(1 for p in pos_0 if p < n * 0.33) / n_if
    feat["if_cterm_frac"] = sum(1 for p in pos_0 if p > n * 0.67) / n_if

    # Physicochemical at interface
    hydro  = [KD_HYDRO.get(aa, 0.0)   for aa in iface]
    charge = [CHARGE.get(aa, 0.0)     for aa in iface]
    vol    = [VOLUME.get(aa, 130.0)   for aa in iface]

    feat["if_mean_hydro"]      = np.mean(hydro)
    feat["if_std_hydro"]       = np.std(hydro)
    feat["if_net_charge"]      = sum(charge)
    feat["if_mean_charge"]     = np.mean(charge)
    feat["if_pos_frac"]        = sum(1 for c in charge if c > 0) / n_if
    feat["if_neg_frac"]        = sum(1 for c in charge if c < 0) / n_if
    feat["if_aromatic_frac"]   = sum(1 for aa in iface if aa in AROMATIC) / n_if
    feat["if_hbond_donor_frac"]= sum(1 for aa in iface if aa in HBOND_D)  / n_if
    feat["if_hbond_acc_frac"]  = sum(1 for aa in iface if aa in HBOND_A)  / n_if
    feat["if_mean_volume"]     = np.mean(vol)

    # Delta vs whole-sequence average
    whole_hydro = [KD_HYDRO.get(aa, 0.0) for aa in seq]
    feat["if_hydro_delta"] = feat["if_mean_hydro"] - np.mean(whole_hydro)

    return feat


def metrics(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    yp    = (y_prob >= 0.5).astype(int)
    return auroc, auprc, f1_score(y_true, yp, zero_division=0), matthews_corrcoef(y_true, yp)


# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
pairs     = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
feat_mat  = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")
feat_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()
evals     = pd.read_parquet(DATA_DIR / "evaluations_flat.parquet")

# ── Parse interface_residues ──────────────────────────────────────────────────
print("Parsing interface_residues from evaluations...")
ir_rows = evals[evals["metric"] == "interface_residues"].copy()

def parse_positions(v):
    try:
        data = json.loads(v) if isinstance(v, str) else v
        if isinstance(data, list):
            return [int(r["residue"]) for r in data if "residue" in r]
    except Exception:
        pass
    return []

ir_rows["positions"] = ir_rows["value"].apply(parse_positions)
ir_rows = ir_rows[ir_rows["positions"].apply(len) > 0]
ir_map  = {(row.protein_id, row.target): row.positions
           for _, row in ir_rows.iterrows()}

# Check overlap with binding pairs
pairs_keys = set(zip(pairs["protein_id"], pairs["target"]))
overlap    = pairs_keys & set(ir_map.keys())
print(f"  IR entries total: {len(ir_map):,}")
print(f"  Overlap with binding pairs: {len(overlap):,} / {len(pairs_keys):,} ({len(overlap)/len(pairs_keys):.1%})")

seq_map = dict(zip(pairs["protein_id"], pairs["sequence"]))

# ── Extract handcrafted interface features ────────────────────────────────────
print("\nExtracting handcrafted interface features...")
if_feat_rows = []
for _, row in pairs.iterrows():
    key = (row["protein_id"], row["target"])
    pos = ir_map.get(key, [])
    seq = seq_map.get(row["protein_id"], "")
    if pos and seq:
        if_feat_rows.append(interface_features(seq, pos))
    else:
        if_feat_rows.append({})   # empty dict → NaN after DataFrame construction

# Build DataFrame by expanding list of dicts (empty dicts → NaN rows)
if_df   = pd.DataFrame(if_feat_rows)
if_cols = [c for c in if_df.columns if c.startswith("if_")]
if_df   = if_df[if_cols]

n_nonzero = if_df.notna().all(axis=1).sum()
print(f"  Interface feature columns: {len(if_cols)}")
print(f"  Pairs with interface data: {n_nonzero:,} / {len(pairs):,} ({n_nonzero/len(pairs):.1%})")
print(f"  Sample columns: {if_cols[:6]}")

# Impute missing with median
if_medians = if_df.median()
if_df_imp  = if_df.fillna(if_medians)

# ── Augment feature matrix ────────────────────────────────────────────────────
print("\nBuilding augmented feature matrix...")
feat_mat_aug = feat_mat.copy().reset_index(drop=True)
pairs_reset  = pairs.reset_index(drop=True)

# Align if_df_imp to feat_mat row order via protein_id + target
pairs_order = list(zip(pairs_reset["protein_id"], pairs_reset["target"]))
fm_order    = list(zip(feat_mat_aug["protein_id"], feat_mat_aug["target"]))

# Build lookup from (pid, tgt) → if_df row index
pair_to_if  = {k: i for i, k in enumerate(pairs_order)}
if_aligned  = pd.DataFrame(
    [if_df_imp.iloc[pair_to_if[k]].to_dict() if k in pair_to_if else {}
     for k in fm_order],
    columns=if_cols
).fillna(if_medians)

for col in if_cols:
    feat_mat_aug[col] = if_aligned[col].values

all_feat_cols = feat_cols + if_cols
print(f"  Original features : {len(feat_cols)}")
print(f"  Interface features: {len(if_cols)}")
print(f"  Total             : {len(all_feat_cols)}")

# ── Split ─────────────────────────────────────────────────────────────────────
train_fm = feat_mat_aug[feat_mat_aug["split"] == "train"].reset_index(drop=True)
val_fm   = feat_mat_aug[feat_mat_aug["split"] == "val"].reset_index(drop=True)
test_fm  = feat_mat_aug[feat_mat_aug["split"] == "test"].reset_index(drop=True)

X_train = train_fm[all_feat_cols].values.astype(np.float32)
X_val   = val_fm[all_feat_cols].values.astype(np.float32)
X_test  = test_fm[all_feat_cols].values.astype(np.float32)
y_train = train_fm["binding_label"].values.astype(np.float32)
y_val   = val_fm["binding_label"].values.astype(np.float32)
y_test  = test_fm["binding_label"].values.astype(np.float32)

print(f"  Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

# ── Retrain LightGBM ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Retraining LightGBM with augmented features (+interface HC)...")
import lightgbm as lgb

lgb_params = dict(
    objective="binary", metric="auc", verbosity=-1, device="gpu",
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
    random_state=RANDOM_SEED,
)
lgb_aug = lgb.LGBMClassifier(**lgb_params)
lgb_aug.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)
lgb_val_p  = lgb_aug.predict_proba(X_val)[:, 1]
lgb_test_p = lgb_aug.predict_proba(X_test)[:, 1]
m_lgb_v    = metrics(y_val,  lgb_val_p)
m_lgb_t    = metrics(y_test, lgb_test_p)
print(f"  LightGBM+IfaceHC Val  — AUROC={m_lgb_v[0]:.4f}  AUPRC={m_lgb_v[1]:.4f}  F1={m_lgb_v[2]:.4f}  MCC={m_lgb_v[3]:.4f}")
print(f"  LightGBM+IfaceHC Test — AUROC={m_lgb_t[0]:.4f}  AUPRC={m_lgb_t[1]:.4f}  F1={m_lgb_t[2]:.4f}  MCC={m_lgb_t[3]:.4f}")
joblib.dump(lgb_aug, MODELS_DIR / "lgb_interface_hc.pkl")

# ── Retrain XGBoost ───────────────────────────────────────────────────────────
print("\nRetraining XGBoost with augmented features (+interface HC)...")
from xgboost import XGBClassifier

xgb_aug = XGBClassifier(
    n_estimators=776, max_depth=8, learning_rate=0.0496,
    subsample=0.958, colsample_bytree=0.722, min_child_weight=2,
    reg_alpha=1.11, reg_lambda=0.002,
    device="cuda", eval_metric="auc", early_stopping_rounds=50,
    random_state=RANDOM_SEED, verbosity=0,
)
xgb_aug.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_val_p  = xgb_aug.predict_proba(X_val)[:, 1]
xgb_test_p = xgb_aug.predict_proba(X_test)[:, 1]
m_xgb_v    = metrics(y_val,  xgb_val_p)
m_xgb_t    = metrics(y_test, xgb_test_p)
print(f"  XGBoost+IfaceHC Val  — AUROC={m_xgb_v[0]:.4f}  AUPRC={m_xgb_v[1]:.4f}  F1={m_xgb_v[2]:.4f}  MCC={m_xgb_v[3]:.4f}")
print(f"  XGBoost+IfaceHC Test — AUROC={m_xgb_t[0]:.4f}  AUPRC={m_xgb_t[1]:.4f}  F1={m_xgb_t[2]:.4f}  MCC={m_xgb_t[3]:.4f}")
joblib.dump(xgb_aug, MODELS_DIR / "xgb_interface_hc.pkl")

# ── Load Phase 5 TargetAwareMLP ───────────────────────────────────────────────
print("\nLoading Phase 5 TargetAwareMLP for ensemble...")

esm_emb = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_map_orig = {pid: i for i, pid in enumerate(esm_ids)}

# Build X_comb (ESM-2 + handcrafted, same as Phase 5) using augmented handcrafted
pairs_aug = pairs_reset.merge(
    feat_mat_aug[["protein_id", "target"] + all_feat_cols],
    on=["protein_id", "target"], how="left", suffixes=("", "_feat")
)
X_hc_full = np.nan_to_num(pairs_aug[all_feat_cols].values.astype(np.float32), nan=0.0)
X_emb_all = np.zeros((len(pairs_aug), 1280), dtype=np.float32)
for i, pid in enumerate(pairs_aug["protein_id"]):
    idx = esm_map_orig.get(pid)
    if idx is not None:
        X_emb_all[i] = esm_emb[idx]
X_emb_all = np.nan_to_num(X_emb_all, nan=0.0)
X_comb    = np.concatenate([X_emb_all, X_hc_full], axis=1)

train_m = (pairs_aug["split"] == "train").values
val_m   = (pairs_aug["split"] == "val").values
test_m  = (pairs_aug["split"] == "test").values
y_bind  = pairs_aug["binding_label"].values.astype(np.float32)

target_names = sorted(pairs_aug["target"].unique())
target_enc   = {t: i for i, t in enumerate(target_names)}
t_idx_arr    = pairs_aug["target"].map(target_enc).values.astype(np.int64)
n_targets    = len(target_names)

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

# Train new TargetAwareMLP with augmented combined features
print("Training TargetAwareMLP with augmented features (ESM-2 + interface HC)...")
print(f"  Input dim: {X_comb.shape[1]} (ESM-2 1280 + HC {X_hc_full.shape[1]})")

import contextlib
amp_ctx = lambda: torch.amp.autocast('cuda') if device.type == 'cuda' else contextlib.nullcontext()

def train_mlp(X, n_epochs=120, patience=15, lr=1e-3, batch=64):
    model  = TargetAwareMLP(X.shape[1], n_targets).to(device)
    bce    = nn.BCEWithLogitsLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    def ldr(mask, shuf):
        ds = TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
                           torch.tensor(y_bind[mask], dtype=torch.float32),
                           torch.tensor(t_idx_arr[mask], dtype=torch.long))
        return DataLoader(ds, batch_size=batch if shuf else 256, shuffle=shuf, num_workers=0)

    tr_ldr, val_ldr = ldr(train_m, True), ldr(val_m, False)
    best_ap, best_state, no_imp = -1.0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb, tb in tr_ldr:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            opt.zero_grad()
            with amp_ctx():
                loss = bce(model(xb, tb).squeeze(-1), yb)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval()
        probs, yt = [], []
        with torch.no_grad(), amp_ctx():
            for xb, yb, tb in val_ldr:
                probs.append(torch.sigmoid(model(xb.to(device), tb.to(device)).squeeze(-1)).cpu().numpy())
                yt.append(yb.numpy())
        probs = np.concatenate(probs); yt = np.concatenate(yt)
        auroc = roc_auc_score(yt, probs)
        auprc = average_precision_score(yt, probs)
        sched.step(auprc)
        imp = auprc > best_ap
        if epoch <= 5 or imp or epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}{'*' if imp else ' '}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if imp:
            best_ap = auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model

modelC_aug = train_mlp(X_comb)
torch.save(modelC_aug.state_dict(), MODELS_DIR / "target_aware_mlp_aug.pt")

def mlp_probs(model, X, mask):
    model.eval()
    ds  = TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
                        torch.tensor(t_idx_arr[mask], dtype=torch.long))
    ldr = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad(), amp_ctx():
        for xb, tb in ldr:
            out.append(torch.sigmoid(modelC_aug(xb.to(device), tb.to(device)).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)

mlp_val_p  = mlp_probs(modelC_aug, X_comb, val_m)
mlp_test_p = mlp_probs(modelC_aug, X_comb, test_m)
m_mlp_v    = metrics(y_bind[val_m],  mlp_val_p)
m_mlp_t    = metrics(y_bind[test_m], mlp_test_p)
print(f"  MLP+Aug Val  — AUROC={m_mlp_v[0]:.4f}  AUPRC={m_mlp_v[1]:.4f}  F1={m_mlp_v[2]:.4f}  MCC={m_mlp_v[3]:.4f}")
print(f"  MLP+Aug Test — AUROC={m_mlp_t[0]:.4f}  AUPRC={m_mlp_t[1]:.4f}  F1={m_mlp_t[2]:.4f}  MCC={m_mlp_t[3]:.4f}")

# ── Weighted ensemble ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Building ensemble (LGB_aug + XGB_aug + MLP_aug)...")

y_v = y_bind[val_m]; y_t = y_bind[test_m]
val_stack  = np.stack([lgb_val_p,  xgb_val_p,  mlp_val_p],  axis=1)
test_stack = np.stack([lgb_test_p, xgb_test_p, mlp_test_p], axis=1)

def neg_ap(w):
    w = np.abs(w) / np.abs(w).sum()
    return -average_precision_score(y_v, (val_stack * w).sum(1))

best_ap0, best_w0 = -1, np.array([0.4, 0.3, 0.3])
for w0 in np.arange(0.2, 0.7, 0.1):
    for w1 in np.arange(0.1, 0.5, 0.1):
        w2 = 1.0 - w0 - w1
        if w2 <= 0: continue
        ap = average_precision_score(y_v, (val_stack * np.array([w0, w1, w2])).sum(1))
        if ap > best_ap0: best_ap0, best_w0 = ap, np.array([w0, w1, w2])

res   = minimize(neg_ap, best_w0, method='Nelder-Mead',
                 options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000})
w_opt = np.abs(res.x) / np.abs(res.x).sum()
print(f"  Weights — LGB: {w_opt[0]:.3f}  XGB: {w_opt[1]:.3f}  MLP: {w_opt[2]:.3f}")

ens_val_p  = (val_stack  * w_opt).sum(1)
ens_test_p = (test_stack * w_opt).sum(1)
m_ens_v    = metrics(y_v, ens_val_p)
m_ens_t    = metrics(y_t, ens_test_p)
print(f"  Ensemble Val  — AUROC={m_ens_v[0]:.4f}  AUPRC={m_ens_v[1]:.4f}  F1={m_ens_v[2]:.4f}  MCC={m_ens_v[3]:.4f}")
print(f"  Ensemble Test — AUROC={m_ens_t[0]:.4f}  AUPRC={m_ens_t[1]:.4f}  F1={m_ens_t[2]:.4f}  MCC={m_ens_t[3]:.4f}")

# Platt calibration
cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=RANDOM_SEED)
cal.fit(ens_val_p.reshape(-1, 1), y_v)
cal_val_p  = cal.predict_proba(ens_val_p.reshape(-1, 1))[:, 1]
cal_test_p = cal.predict_proba(ens_test_p.reshape(-1, 1))[:, 1]
m_cal_v    = metrics(y_v, cal_val_p)
m_cal_t    = metrics(y_t, cal_test_p)
print(f"  Calibrated Val  — AUROC={m_cal_v[0]:.4f}  AUPRC={m_cal_v[1]:.4f}  F1={m_cal_v[2]:.4f}  MCC={m_cal_v[3]:.4f}")
print(f"  Calibrated Test — AUROC={m_cal_t[0]:.4f}  AUPRC={m_cal_t[1]:.4f}  F1={m_cal_t[2]:.4f}  MCC={m_cal_t[3]:.4f}")

# Per-target thresholds
val_tgts  = pairs_aug["target"].values[val_m]
test_tgts = pairs_aug["target"].values[test_m]
thresholds = {}
for t in target_names:
    tm = val_tgts == t
    if tm.sum() < 5: thresholds[t] = 0.5; continue
    yt_, yp_ = y_v[tm], cal_val_p[tm]
    if yt_.sum() == 0 or yt_.sum() == tm.sum(): thresholds[t] = 0.5; continue
    bf, bt = -1, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(yt_, (yp_ >= thr).astype(int), zero_division=0)
        if f1 > bf: bf, bt = f1, thr
    thresholds[t] = bt

y_pred_pt = np.zeros(len(y_t), dtype=int)
for t in target_names:
    tm = test_tgts == t
    if tm.sum() == 0: continue
    y_pred_pt[tm] = (cal_test_p[tm] >= thresholds[t]).astype(int)

auroc_pt = roc_auc_score(y_t, cal_test_p)
auprc_pt = average_precision_score(y_t, cal_test_p)
f1_pt    = f1_score(y_t, y_pred_pt, zero_division=0)
mcc_pt   = matthews_corrcoef(y_t, y_pred_pt)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 6a — FINAL SUMMARY (TEST SET)")
print(f"  {'Model':<50}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>7}  {'MCC':>7}")
print("  " + "-"*82)
print(f"  {'--- Previous Best ---':<50}")
print(f"  {'LightGBM Phase3':<50}  {0.8925:>7.4f}  {0.7132:>7.4f}  {0.6777:>7.4f}  {0.6247:>7.4f}")
print(f"  {'Ensemble+CalibPerTgt Phase5b':<50}  {0.8825:>7.4f}  {0.6790:>7.4f}  {0.6917:>7.4f}  {0.6288:>7.4f}")
print(f"  {'--- Phase 6a (Interface Features) ---':<50}")
print(f"  {'LightGBM+IfaceHC':<50}  {m_lgb_t[0]:>7.4f}  {m_lgb_t[1]:>7.4f}  {m_lgb_t[2]:>7.4f}  {m_lgb_t[3]:>7.4f}")
print(f"  {'XGBoost+IfaceHC':<50}  {m_xgb_t[0]:>7.4f}  {m_xgb_t[1]:>7.4f}  {m_xgb_t[2]:>7.4f}  {m_xgb_t[3]:>7.4f}")
print(f"  {'TargetAwareMLP+Aug(ESM2+IfaceHC)':<50}  {m_mlp_t[0]:>7.4f}  {m_mlp_t[1]:>7.4f}  {m_mlp_t[2]:>7.4f}  {m_mlp_t[3]:>7.4f}")
print(f"  {'WeightedEnsemble_6a':<50}  {m_ens_t[0]:>7.4f}  {m_ens_t[1]:>7.4f}  {m_ens_t[2]:>7.4f}  {m_ens_t[3]:>7.4f}")
print(f"  {'CalibratedEnsemble_6a':<50}  {m_cal_t[0]:>7.4f}  {m_cal_t[1]:>7.4f}  {m_cal_t[2]:>7.4f}  {m_cal_t[3]:>7.4f}")
print(f"  {'CalibratedEnsemble_6a+PerTgtThr':<50}  {auroc_pt:>7.4f}  {auprc_pt:>7.4f}  {f1_pt:>7.4f}  {mcc_pt:>7.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
rows = [
    {"model": "LightGBM_Phase3",            "auroc": 0.8925, "auprc": 0.7132, "f1": 0.6777, "mcc": 0.6247},
    {"model": "Ensemble_Phase5b",            "auroc": 0.8825, "auprc": 0.6790, "f1": 0.6917, "mcc": 0.6288},
    {"model": "LightGBM+IfaceHC",            "auroc": m_lgb_t[0], "auprc": m_lgb_t[1], "f1": m_lgb_t[2], "mcc": m_lgb_t[3]},
    {"model": "XGBoost+IfaceHC",             "auroc": m_xgb_t[0], "auprc": m_xgb_t[1], "f1": m_xgb_t[2], "mcc": m_xgb_t[3]},
    {"model": "TargetAwareMLP+Aug",          "auroc": m_mlp_t[0], "auprc": m_mlp_t[1], "f1": m_mlp_t[2], "mcc": m_mlp_t[3]},
    {"model": "WeightedEnsemble_6a",         "auroc": m_ens_t[0], "auprc": m_ens_t[1], "f1": m_ens_t[2], "mcc": m_ens_t[3]},
    {"model": "CalibratedEnsemble_6a",       "auroc": m_cal_t[0], "auprc": m_cal_t[1], "f1": m_cal_t[2], "mcc": m_cal_t[3]},
    {"model": "CalibratedEnsemble_6a+PerTgt","auroc": auroc_pt,   "auprc": auprc_pt,   "f1": f1_pt,     "mcc": mcc_pt},
]
pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "phase6a_results.csv", index=False)

meta = {
    "model_names": ["LGB_IfaceHC", "XGB_IfaceHC", "MLP_Aug"],
    "weights": w_opt, "thresholds": thresholds,
    "target_names": target_names, "target_enc": target_enc,
    "if_cols": if_cols, "if_medians": if_medians.to_dict(),
    "all_feat_cols": all_feat_cols,
}
with open(MODELS_DIR / "ensemble_meta_6a.pkl", "wb") as f:
    pickle.dump(meta, f)
joblib.dump(cal, MODELS_DIR / "calibrator_6a.pkl")
# Save augmented feature columns for downstream use
pd.DataFrame({"column": all_feat_cols}).to_csv(FEATURES_DIR / "feature_columns_aug.csv", index=False)

print(f"\nSaved: outputs/phase6a_results.csv")
print(f"Saved: models/lgb_interface_hc.pkl, xgb_interface_hc.pkl, target_aware_mlp_aug.pt")
print(f"Saved: models/ensemble_meta_6a.pkl, calibrator_6a.pkl")
print(f"Saved: features/feature_columns_aug.csv")
print("Phase 6a complete.")
