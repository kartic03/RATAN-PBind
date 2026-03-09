"""
Phase 6b: Target Prototype Embeddings (data-driven target representation)
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Key insight: instead of a 32-dim LEARNED target embedding (Phase 5),
use the MEAN ESM-2 embedding of known training binders as the target prototype.

Why this works:
  - Known binders of the same target share structural features (common epitope)
  - Their mean ESM-2 embedding encodes "what this target likes to bind"
  - Cosine similarity between query binder and target prototype is extremely informative
  - This is essentially prototype networks (Snell et al. 2017) applied to protein binding

Features added (7 scalar):
  1. proto_cos_pos   — cosine sim to positive prototype (known binders)
  2. proto_cos_neg   — cosine sim to negative prototype (known non-binders)
  3. proto_l2_pos    — L2 distance to positive prototype
  4. proto_disc_proj — projection onto discriminative axis (pos_proto - neg_proto)
  5. proto_ratio     — proto_cos_pos / (proto_cos_neg + ε)
  6. proto_n_pos     — number of training positives for this target (prototype reliability)
  7. proto_n_neg     — number of training negatives for this target

TargetAwareMLP variant: uses 1280-dim prototype as target context
(much richer than 32-dim learned embedding)

No external data needed — completely self-contained.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from pathlib import Path
import joblib, pickle, warnings, contextlib
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_ctx = lambda: torch.amp.autocast('cuda') if device.type == 'cuda' else contextlib.nullcontext()
print(f"Device: {device}")

def metrics(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    yp    = (y_prob >= 0.5).astype(int)
    return auroc, auprc, f1_score(y_true, yp, zero_division=0), matthews_corrcoef(y_true, yp)

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
pairs     = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
feat_mat  = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")

# Use augmented feature columns from Phase 6a (463 + 39 interface features)
aug_col_path = FEATURES_DIR / "feature_columns_aug.csv"
if aug_col_path.exists():
    feat_cols = pd.read_csv(aug_col_path)["column"].tolist()
    print(f"  Using augmented features from Phase 6a: {len(feat_cols)} columns")
    # Load augmented feature matrix with interface features
    feat_mat_aug_path = FEATURES_DIR / "feature_matrix_aug.parquet"
    # Reconstruct augmented feat_mat by merging with 6a models metadata
    meta_6a = pickle.load(open(MODELS_DIR / "ensemble_meta_6a.pkl", "rb"))
    all_feat_cols_6a = meta_6a["all_feat_cols"]
    # Try loading from 6a saved data; fallback to original if aug not saved separately
    import json
    try:
        lgb_6a = joblib.load(MODELS_DIR / "lgb_interface_hc.pkl")
        use_aug = True
    except Exception:
        use_aug = False
else:
    feat_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()
    all_feat_cols_6a = feat_cols
    use_aug = False

# Load base feature matrix (with splits) for classical ML
feat_mat_base = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")

# Load ESM-2 embeddings
esm_emb  = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids  = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_map  = {pid: i for i, pid in enumerate(esm_ids)}

# Align ESM-2 embeddings to pairs
pairs_r  = pairs.reset_index(drop=True)
X_emb    = np.zeros((len(pairs_r), 1280), dtype=np.float32)
for i, pid in enumerate(pairs_r["protein_id"]):
    idx = esm_map.get(pid)
    if idx is not None:
        X_emb[i] = esm_emb[idx]
X_emb = np.nan_to_num(X_emb, nan=0.0)

y_bind  = pairs_r["binding_label"].values.astype(np.float32)
train_m = (pairs_r["split"] == "train").values
val_m   = (pairs_r["split"] == "val").values
test_m  = (pairs_r["split"] == "test").values

target_names = sorted(pairs_r["target"].unique())
target_enc   = {t: i for i, t in enumerate(target_names)}
t_idx_arr    = pairs_r["target"].map(target_enc).values.astype(np.int64)
n_targets    = len(target_names)

print(f"  Train: {train_m.sum()}  Val: {val_m.sum()}  Test: {test_m.sum()}")
print(f"  Targets: {n_targets}")

# ── Compute per-target prototypes from TRAINING SET only ──────────────────────
print("\nComputing per-target prototypes from training set...")

X_emb_train = X_emb[train_m]
y_train_bind = y_bind[train_m]
targets_train = pairs_r["target"].values[train_m]

proto_pos  = {}  # target → mean ESM-2 of train positives
proto_neg  = {}  # target → mean ESM-2 of train negatives
n_pos_dict = {}
n_neg_dict = {}

for t in target_names:
    t_mask    = targets_train == t
    y_t       = y_train_bind[t_mask]
    emb_t     = X_emb_train[t_mask]
    pos_mask  = y_t == 1
    neg_mask  = y_t == 0
    n_pos_dict[t] = int(pos_mask.sum())
    n_neg_dict[t] = int(neg_mask.sum())
    proto_pos[t]  = emb_t[pos_mask].mean(0) if pos_mask.sum() > 0 else np.zeros(1280, dtype=np.float32)
    proto_neg[t]  = emb_t[neg_mask].mean(0) if neg_mask.sum() > 0 else np.zeros(1280, dtype=np.float32)

print(f"  {'Target':<45}  {'N_pos':>5}  {'N_neg':>5}")
for t in target_names:
    print(f"  {t:<45}  {n_pos_dict[t]:>5}  {n_neg_dict[t]:>5}")

# ── Extract prototype similarity features for all pairs ───────────────────────
print("\nExtracting prototype similarity features...")

def proto_features(emb, target):
    """7 features from prototype similarity for one protein-target pair."""
    pp = proto_pos[target]
    pn = proto_neg[target]
    disc = pp - pn
    disc_norm = np.linalg.norm(disc)

    cos_pos  = cosine_sim(emb, pp)
    cos_neg  = cosine_sim(emb, pn)
    l2_pos   = float(np.linalg.norm(emb - pp))
    disc_proj = float(np.dot(emb, disc) / (disc_norm + 1e-8))
    ratio    = cos_pos / (abs(cos_neg) + 1e-6)
    n_pos    = n_pos_dict[target]
    n_neg    = n_neg_dict[target]

    return [cos_pos, cos_neg, l2_pos, disc_proj, ratio, n_pos, n_neg]

proto_feat_cols = ["proto_cos_pos", "proto_cos_neg", "proto_l2_pos",
                   "proto_disc_proj", "proto_ratio", "proto_n_pos", "proto_n_neg"]

proto_feats = np.array([
    proto_features(X_emb[i], pairs_r["target"].iloc[i])
    for i in range(len(pairs_r))
], dtype=np.float32)

print(f"  Proto features shape: {proto_feats.shape}")
print(f"  Proto features (sample val pair):")
for j, col in enumerate(proto_feat_cols):
    print(f"    {col:<20}: train_mean={proto_feats[train_m, j].mean():.4f}  "
          f"val_range=[{proto_feats[val_m, j].min():.3f}, {proto_feats[val_m, j].max():.3f}]")

# ── Build augmented handcrafted feature matrices ──────────────────────────────
print("\nBuilding augmented feature matrices...")

# Rebuild Phase 6a interface features inline (re-run the feature extraction)
# Load base feature cols (463 original)
base_feat_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()

# Load augmented feature columns from 6a if available
if use_aug:
    all_feat_cols = meta_6a["all_feat_cols"]  # 463 + 39 interface = 502
else:
    all_feat_cols = base_feat_cols

# Rebuild augmented feat_mat for classical ML (Phase 6a + prototype features)
# Load feature matrix (Phase 3 style — already imputed)
fm = feat_mat_base.copy()

# Add prototype features to fm aligned by protein_id + target
fm_keys = list(zip(fm["protein_id"], fm["target"]))
pairs_keys = list(zip(pairs_r["protein_id"], pairs_r["target"]))
key_to_proto = {k: proto_feats[i] for i, k in enumerate(pairs_keys)}

proto_arr = np.array([
    key_to_proto.get(k, np.zeros(len(proto_feat_cols), dtype=np.float32))
    for k in fm_keys
], dtype=np.float32)

for j, col in enumerate(proto_feat_cols):
    fm[col] = proto_arr[:, j]

# Add interface features from 6a if available
if use_aug:
    # Load the 6a interface feature data from saved model meta
    if_cols    = meta_6a["if_cols"]
    if_medians = pd.Series(meta_6a["if_medians"])
    # Re-extract interface features (same logic as phase6a)
    import json as _json
    evals = pd.read_parquet(DATA_DIR / "evaluations_flat.parquet")
    ir_rows = evals[evals["metric"] == "interface_residues"].copy()
    def _parse_pos(v):
        try:
            data = _json.loads(v) if isinstance(v, str) else v
            return [int(r["residue"]) for r in data if "residue" in r] if isinstance(data, list) else []
        except: return []
    ir_rows["positions"] = ir_rows["value"].apply(_parse_pos)
    ir_rows = ir_rows[ir_rows["positions"].apply(len) > 0]
    ir_map_local = {(r.protein_id, r.target): r.positions for _, r in ir_rows.iterrows()}
    seq_map_local = dict(zip(pairs_r["protein_id"], pairs_r["sequence"]))

    # Import interface_features function logic inline
    KD_H = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
    CHG  = {'K':1,'R':1,'H':0.1,'D':-1,'E':-1}
    ARO  = set('FWY')
    HBD  = set('NQSTKRHWY')
    HBA  = set('NQDEST')
    VOL  = {'G':60,'A':89,'S':96,'P':112,'V':117,'T':116,'C':114,'I':166,'L':166,'N':114,'D':111,'Q':144,'K':168,'E':138,'M':162,'H':153,'F':190,'R':173,'Y':194,'W':228}
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
        hy=[KD_H.get(a,0.) for a in iface]; ch=[CHG.get(a,0.) for a in iface]; vl=[VOL.get(a,130.) for a in iface]
        f["if_mean_hydro"]=np.mean(hy); f["if_std_hydro"]=np.std(hy)
        f["if_net_charge"]=sum(ch); f["if_mean_charge"]=np.mean(ch)
        f["if_pos_frac"]=sum(1 for c in ch if c>0)/n_if; f["if_neg_frac"]=sum(1 for c in ch if c<0)/n_if
        f["if_aromatic_frac"]=sum(1 for a in iface if a in ARO)/n_if
        f["if_hbond_donor_frac"]=sum(1 for a in iface if a in HBD)/n_if
        f["if_hbond_acc_frac"]=sum(1 for a in iface if a in HBA)/n_if
        f["if_mean_volume"]=np.mean(vl)
        wh=[KD_H.get(a,0.) for a in seq]; f["if_hydro_delta"]=f["if_mean_hydro"]-np.mean(wh)
        return f

    # Build per-pairs interface features
    if_rows_pairs = []
    for _, row in pairs_r.iterrows():
        key = (row["protein_id"], row["target"])
        pos = ir_map_local.get(key, [])
        seq = seq_map_local.get(row["protein_id"], "")
        if_rows_pairs.append(_if_feat(seq, pos) if pos and seq else {})
    if_df_pairs = pd.DataFrame(if_rows_pairs)[if_cols].fillna(if_medians)

    # Align to fm row order
    fm_order   = list(zip(fm["protein_id"], fm["target"]))
    pair_to_if = {(pairs_r["protein_id"].iloc[i], pairs_r["target"].iloc[i]): i
                  for i in range(len(pairs_r))}
    for col in if_cols:
        fm[col] = [if_df_pairs[col].iloc[pair_to_if.get(k, 0)]
                   if k in pair_to_if else if_medians[col]
                   for k in fm_order]

    all_feat_cols = base_feat_cols + if_cols + proto_feat_cols
else:
    all_feat_cols = base_feat_cols + proto_feat_cols

print(f"  Total feature columns: {len(all_feat_cols)}")

# ── Split ─────────────────────────────────────────────────────────────────────
train_fm = fm[fm["split"] == "train"].reset_index(drop=True)
val_fm   = fm[fm["split"] == "val"].reset_index(drop=True)
test_fm  = fm[fm["split"] == "test"].reset_index(drop=True)

X_tr = train_fm[all_feat_cols].values.astype(np.float32)
X_v  = val_fm[all_feat_cols].values.astype(np.float32)
X_te = test_fm[all_feat_cols].values.astype(np.float32)
y_tr = train_fm["binding_label"].values.astype(np.float32)
y_v  = val_fm["binding_label"].values.astype(np.float32)
y_te = test_fm["binding_label"].values.astype(np.float32)
print(f"  Train {X_tr.shape} | Val {X_v.shape} | Test {X_te.shape}")

# ── Retrain LightGBM ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LightGBM with prototype features...")
import lightgbm as lgb

lgb_params = dict(
    objective="binary", metric="auc", verbosity=-1, device="gpu",
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
    random_state=RANDOM_SEED,
)
lgb_6b = lgb.LGBMClassifier(**lgb_params)
lgb_6b.fit(X_tr, y_tr,
           eval_set=[(X_v, y_v)],
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
lgb_val_p  = lgb_6b.predict_proba(X_v)[:, 1]
lgb_test_p = lgb_6b.predict_proba(X_te)[:, 1]
m_lgb_v = metrics(y_v,  lgb_val_p)
m_lgb_t = metrics(y_te, lgb_test_p)
print(f"  LightGBM+Proto Val  — AUROC={m_lgb_v[0]:.4f}  AUPRC={m_lgb_v[1]:.4f}  F1={m_lgb_v[2]:.4f}  MCC={m_lgb_v[3]:.4f}")
print(f"  LightGBM+Proto Test — AUROC={m_lgb_t[0]:.4f}  AUPRC={m_lgb_t[1]:.4f}  F1={m_lgb_t[2]:.4f}  MCC={m_lgb_t[3]:.4f}")
joblib.dump(lgb_6b, MODELS_DIR / "lgb_proto.pkl")

# Feature importance — check proto features
fi = lgb_6b.feature_importances_
fi_df = pd.DataFrame({"feature": all_feat_cols, "importance": fi}).sort_values("importance", ascending=False)
proto_ranks = fi_df[fi_df["feature"].isin(proto_feat_cols)][["feature", "importance"]]
print(f"\n  Proto feature importance (rank among {len(all_feat_cols)} total):")
for _, row in proto_ranks.iterrows():
    rank = fi_df[fi_df["feature"] == row["feature"]].index[0]
    print(f"    {row['feature']:<25}: importance={row['importance']:>6.0f}  (rank #{list(fi_df['feature']).index(row['feature'])+1})")

# ── Retrain XGBoost ───────────────────────────────────────────────────────────
print("\nXGBoost with prototype features...")
from xgboost import XGBClassifier

xgb_6b = XGBClassifier(
    n_estimators=776, max_depth=8, learning_rate=0.0496,
    subsample=0.958, colsample_bytree=0.722, min_child_weight=2,
    reg_alpha=1.11, reg_lambda=0.002,
    device="cuda", eval_metric="auc", early_stopping_rounds=50,
    random_state=RANDOM_SEED, verbosity=0,
)
xgb_6b.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
xgb_val_p  = xgb_6b.predict_proba(X_v)[:, 1]
xgb_test_p = xgb_6b.predict_proba(X_te)[:, 1]
m_xgb_v = metrics(y_v,  xgb_val_p)
m_xgb_t = metrics(y_te, xgb_test_p)
print(f"  XGBoost+Proto Val  — AUROC={m_xgb_v[0]:.4f}  AUPRC={m_xgb_v[1]:.4f}  F1={m_xgb_v[2]:.4f}  MCC={m_xgb_v[3]:.4f}")
print(f"  XGBoost+Proto Test — AUROC={m_xgb_t[0]:.4f}  AUPRC={m_xgb_t[1]:.4f}  F1={m_xgb_t[2]:.4f}  MCC={m_xgb_t[3]:.4f}")
joblib.dump(xgb_6b, MODELS_DIR / "xgb_proto.pkl")

# ── TargetAwareMLP with 1280-dim prototype target context ─────────────────────
print("\n" + "="*70)
print("TargetAwareMLP with 1280-dim prototype target context...")

# Build per-target prototype tensor (fixed, not learned)
proto_tensor = torch.zeros(n_targets, 1280, dtype=torch.float32)
for t, idx in target_enc.items():
    proto_tensor[idx] = torch.from_numpy(proto_pos[t].astype(np.float32))

class ProtoTargetMLP(nn.Module):
    """
    TargetAwareMLP where target context = 1280-dim prototype embedding (fixed).
    Projection layer reduces prototype to 64-dim before concatenation.
    """
    def __init__(self, in_dim, proto_dim=1280, proj_dim=64, dropout=0.3):
        super().__init__()
        # Learnable projection of the prototype (makes it trainable context)
        self.proto_proj = nn.Sequential(
            nn.Linear(proto_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        d = in_dim + proj_dim
        self.net = nn.Sequential(
            nn.Linear(d,   512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(128, 1),
        )
    def forward(self, x, proto):
        ctx = self.proto_proj(proto)
        return self.net(torch.cat([x, ctx], dim=-1))

# Build combined input: ESM-2 + augmented handcrafted
pairs_aug_6b = pairs_r.merge(
    fm[["protein_id", "target"] + all_feat_cols],
    on=["protein_id", "target"], how="left", suffixes=("", "_feat")
)
X_hc_6b  = np.nan_to_num(pairs_aug_6b[all_feat_cols].values.astype(np.float32), nan=0.0)
X_comb_6b = np.concatenate([X_emb, X_hc_6b], axis=1)
y_bind_6b = pairs_aug_6b["binding_label"].values.astype(np.float32)
t_idx_6b  = pairs_aug_6b["target"].map(target_enc).values.astype(np.int64)

print(f"  Input dim: {X_comb_6b.shape[1]}")

def train_proto_mlp(X, t_idx, y, train_m, val_m,
                    n_epochs=120, patience=15, lr=1e-3, batch=64):
    model  = ProtoTargetMLP(X.shape[1]).to(device)
    pt_dev = proto_tensor.to(device)  # (n_targets, 1280)
    bce    = nn.BCEWithLogitsLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    def make_ldr(mask, shuf):
        ds = TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
                           torch.tensor(y[mask], dtype=torch.float32),
                           torch.tensor(t_idx[mask], dtype=torch.long))
        return DataLoader(ds, batch_size=batch if shuf else 256, shuffle=shuf, num_workers=0)

    tr_ldr, val_ldr = make_ldr(train_m, True), make_ldr(val_m, False)
    best_ap, best_state, no_imp = -1.0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb, tb in tr_ldr:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            proto_b = pt_dev[tb]  # (batch, 1280)
            opt.zero_grad()
            with amp_ctx():
                loss = bce(model(xb, proto_b).squeeze(-1), yb)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        model.eval()
        probs, yt_list = [], []
        with torch.no_grad(), amp_ctx():
            for xb, yb, tb in val_ldr:
                xb, tb = xb.to(device), tb.to(device)
                probs.append(torch.sigmoid(model(xb, pt_dev[tb]).squeeze(-1)).cpu().numpy())
                yt_list.append(yb.numpy())
        probs = np.concatenate(probs); yt = np.concatenate(yt_list)
        auroc = roc_auc_score(yt, probs)
        auprc = average_precision_score(yt, probs)
        sched.step(auprc)
        imp = auprc > best_ap
        if epoch <= 5 or imp or epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}{'*' if imp else ' '}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if imp:
            best_ap = auprc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: print(f"  Early stopping at epoch {epoch}"); break

    model.load_state_dict(best_state)
    return model

model_proto = train_proto_mlp(X_comb_6b, t_idx_6b, y_bind_6b, train_m, val_m)
torch.save(model_proto.state_dict(), MODELS_DIR / "proto_target_mlp.pt")

def eval_proto_mlp(model, X, t_idx, y, mask):
    model.eval()
    pt_dev = proto_tensor.to(device)
    ds  = TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
                        torch.tensor(y[mask], dtype=torch.float32),
                        torch.tensor(t_idx[mask], dtype=torch.long))
    ldr = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probs, yt_list = [], []
    with torch.no_grad(), amp_ctx():
        for xb, yb, tb in ldr:
            xb, tb = xb.to(device), tb.to(device)
            probs.append(torch.sigmoid(model(xb, pt_dev[tb]).squeeze(-1)).cpu().numpy())
            yt_list.append(yb.numpy())
    return metrics(np.concatenate(yt_list), np.concatenate(probs))

mlp_val  = eval_proto_mlp(model_proto, X_comb_6b, t_idx_6b, y_bind_6b, val_m)
mlp_test = eval_proto_mlp(model_proto, X_comb_6b, t_idx_6b, y_bind_6b, test_m)
print(f"  ProtoTargetMLP Val  — AUROC={mlp_val[0]:.4f}  AUPRC={mlp_val[1]:.4f}  F1={mlp_val[2]:.4f}  MCC={mlp_val[3]:.4f}")
print(f"  ProtoTargetMLP Test — AUROC={mlp_test[0]:.4f}  AUPRC={mlp_test[1]:.4f}  F1={mlp_test[2]:.4f}  MCC={mlp_test[3]:.4f}")

# ── Ensemble: all 6b models ───────────────────────────────────────────────────
print("\n" + "="*70)
print("Building final ensemble (LGB+Proto, XGB+Proto, ProtoMLP, XGB+IfaceHC_6a)...")

# Load Phase 6a best model (XGBoost with interface HC, our previous best)
xgb_6a = joblib.load(MODELS_DIR / "xgb_interface_hc.pkl")
# Rebuild 6a features for val/test (base + interface only, no proto)
feat_cols_6a = meta_6a["all_feat_cols"] if use_aug else base_feat_cols

# Get 6a feature matrices
fm_6a = feat_mat_base.copy()
if use_aug:
    if_cols_6a = meta_6a["if_cols"]
    if_meds_6a = pd.Series(meta_6a["if_medians"])
    # Use the already-computed interface features (same as above)
    fm_order_6a = list(zip(fm_6a["protein_id"], fm_6a["target"]))
    pair_to_if_6a = {(pairs_r["protein_id"].iloc[i], pairs_r["target"].iloc[i]): i for i in range(len(pairs_r))}
    for col in if_cols_6a:
        fm_6a[col] = [if_df_pairs[col].iloc[pair_to_if_6a.get(k, 0)]
                      if k in pair_to_if_6a else if_meds_6a[col]
                      for k in fm_order_6a]

X_6a_v  = fm_6a[fm_6a["split"] == "val"].reset_index(drop=True)[feat_cols_6a].values.astype(np.float32)
X_6a_te = fm_6a[fm_6a["split"] == "test"].reset_index(drop=True)[feat_cols_6a].values.astype(np.float32)

xgb_6a_val_p  = xgb_6a.predict_proba(X_6a_v)[:, 1]
xgb_6a_test_p = xgb_6a.predict_proba(X_6a_te)[:, 1]

# Get ProtoMLP probs
def get_mlp_probs_arr(model, X, t_idx, y, mask):
    model.eval()
    pt_dev = proto_tensor.to(device)
    ds  = TensorDataset(torch.tensor(X[mask], dtype=torch.float32),
                        torch.tensor(t_idx[mask], dtype=torch.long))
    ldr = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad(), amp_ctx():
        for xb, tb in ldr:
            out.append(torch.sigmoid(model(xb.to(device), pt_dev[tb.to(device)]).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)

mlp_val_p  = get_mlp_probs_arr(model_proto, X_comb_6b, t_idx_6b, y_bind_6b, val_m)
mlp_test_p = get_mlp_probs_arr(model_proto, X_comb_6b, t_idx_6b, y_bind_6b, test_m)

val_stack  = np.stack([lgb_val_p, xgb_val_p, xgb_6a_val_p,  mlp_val_p],  axis=1)
test_stack = np.stack([lgb_test_p, xgb_test_p, xgb_6a_test_p, mlp_test_p], axis=1)

def neg_ap(w):
    w = np.abs(w) / np.abs(w).sum()
    return -average_precision_score(y_v, (val_stack * w).sum(1))

best_ap0, best_w0 = -1, np.array([0.25, 0.35, 0.25, 0.15])
for w0 in np.arange(0.1, 0.6, 0.1):
    for w1 in np.arange(0.1, 0.5, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = 1.0 - w0 - w1 - w2
            if w3 <= 0: continue
            ap = average_precision_score(y_v, (val_stack * np.array([w0,w1,w2,w3])).sum(1))
            if ap > best_ap0: best_ap0, best_w0 = ap, np.array([w0,w1,w2,w3])

res   = minimize(neg_ap, best_w0, method='Nelder-Mead',
                 options={'xatol':1e-5, 'fatol':1e-5, 'maxiter':2000})
w_opt = np.abs(res.x) / np.abs(res.x).sum()
print(f"  Weights — LGB_Proto:{w_opt[0]:.3f}  XGB_Proto:{w_opt[1]:.3f}  XGB_6a:{w_opt[2]:.3f}  MLP_Proto:{w_opt[3]:.3f}")

ens_val_p  = (val_stack  * w_opt).sum(1)
ens_test_p = (test_stack * w_opt).sum(1)
m_ens_v    = metrics(y_v,  ens_val_p)
m_ens_t    = metrics(y_te, ens_test_p)
print(f"  Ensemble Val  — AUROC={m_ens_v[0]:.4f}  AUPRC={m_ens_v[1]:.4f}  F1={m_ens_v[2]:.4f}  MCC={m_ens_v[3]:.4f}")
print(f"  Ensemble Test — AUROC={m_ens_t[0]:.4f}  AUPRC={m_ens_t[1]:.4f}  F1={m_ens_t[2]:.4f}  MCC={m_ens_t[3]:.4f}")

# Platt calibration + per-target threshold
cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=RANDOM_SEED)
cal.fit(ens_val_p.reshape(-1, 1), y_v)
cal_val_p  = cal.predict_proba(ens_val_p.reshape(-1, 1))[:, 1]
cal_test_p = cal.predict_proba(ens_test_p.reshape(-1, 1))[:, 1]
m_cal_v    = metrics(y_v,  cal_val_p)
m_cal_t    = metrics(y_te, cal_test_p)
print(f"  Calibrated Val  — AUROC={m_cal_v[0]:.4f}  AUPRC={m_cal_v[1]:.4f}  F1={m_cal_v[2]:.4f}  MCC={m_cal_v[3]:.4f}")
print(f"  Calibrated Test — AUROC={m_cal_t[0]:.4f}  AUPRC={m_cal_t[1]:.4f}  F1={m_cal_t[2]:.4f}  MCC={m_cal_t[3]:.4f}")

val_tgts  = pairs_aug_6b["target"].values[val_m]
test_tgts = pairs_aug_6b["target"].values[test_m]
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

y_pred_pt = np.zeros(len(y_te), dtype=int)
for t in target_names:
    tm = test_tgts == t
    if tm.sum() == 0: continue
    y_pred_pt[tm] = (cal_test_p[tm] >= thresholds[t]).astype(int)

auroc_pt = roc_auc_score(y_te, cal_test_p)
auprc_pt = average_precision_score(y_te, cal_test_p)
f1_pt    = f1_score(y_te, y_pred_pt, zero_division=0)
mcc_pt   = matthews_corrcoef(y_te, y_pred_pt)

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 6b — FINAL SUMMARY (TEST SET)")
print(f"  {'Model':<50}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>7}  {'MCC':>7}")
print("  " + "-"*82)
print(f"  {'--- Historical bests ---':<50}")
print(f"  {'LightGBM Phase3':<50}  {0.8925:>7.4f}  {0.7132:>7.4f}  {0.6777:>7.4f}  {0.6247:>7.4f}")
print(f"  {'XGBoost+IfaceHC Phase6a (best single)':<50}  {0.8942:>7.4f}  {0.7022:>7.4f}  {0.6239:>7.4f}  {0.5877:>7.4f}")
print(f"  {'Ensemble Phase5b+CalibPerTgt':<50}  {0.8825:>7.4f}  {0.6790:>7.4f}  {0.6917:>7.4f}  {0.6288:>7.4f}")
print(f"  {'--- Phase 6b (Prototype embeddings) ---':<50}")
print(f"  {'LightGBM+Proto':<50}  {m_lgb_t[0]:>7.4f}  {m_lgb_t[1]:>7.4f}  {m_lgb_t[2]:>7.4f}  {m_lgb_t[3]:>7.4f}")
print(f"  {'XGBoost+Proto':<50}  {m_xgb_t[0]:>7.4f}  {m_xgb_t[1]:>7.4f}  {m_xgb_t[2]:>7.4f}  {m_xgb_t[3]:>7.4f}")
print(f"  {'ProtoTargetMLP':<50}  {mlp_test[0]:>7.4f}  {mlp_test[1]:>7.4f}  {mlp_test[2]:>7.4f}  {mlp_test[3]:>7.4f}")
print(f"  {'WeightedEnsemble_6b':<50}  {m_ens_t[0]:>7.4f}  {m_ens_t[1]:>7.4f}  {m_ens_t[2]:>7.4f}  {m_ens_t[3]:>7.4f}")
print(f"  {'CalibratedEnsemble_6b':<50}  {m_cal_t[0]:>7.4f}  {m_cal_t[1]:>7.4f}  {m_cal_t[2]:>7.4f}  {m_cal_t[3]:>7.4f}")
print(f"  {'CalibratedEnsemble_6b+PerTgtThr':<50}  {auroc_pt:>7.4f}  {auprc_pt:>7.4f}  {f1_pt:>7.4f}  {mcc_pt:>7.4f}")

# Save
rows = [
    {"model": "LightGBM_Phase3",         "auroc": 0.8925, "auprc": 0.7132, "f1": 0.6777, "mcc": 0.6247},
    {"model": "XGBoost+IfaceHC_Phase6a", "auroc": 0.8942, "auprc": 0.7022, "f1": 0.6239, "mcc": 0.5877},
    {"model": "LightGBM+Proto",          "auroc": m_lgb_t[0], "auprc": m_lgb_t[1], "f1": m_lgb_t[2], "mcc": m_lgb_t[3]},
    {"model": "XGBoost+Proto",           "auroc": m_xgb_t[0], "auprc": m_xgb_t[1], "f1": m_xgb_t[2], "mcc": m_xgb_t[3]},
    {"model": "ProtoTargetMLP",          "auroc": mlp_test[0], "auprc": mlp_test[1], "f1": mlp_test[2], "mcc": mlp_test[3]},
    {"model": "WeightedEnsemble_6b",     "auroc": m_ens_t[0], "auprc": m_ens_t[1], "f1": m_ens_t[2], "mcc": m_ens_t[3]},
    {"model": "CalibratedEnsemble_6b",   "auroc": m_cal_t[0], "auprc": m_cal_t[1], "f1": m_cal_t[2], "mcc": m_cal_t[3]},
    {"model": "CalibratedEnsemble_6b+PerTgt","auroc": auroc_pt, "auprc": auprc_pt, "f1": f1_pt, "mcc": mcc_pt},
]
pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "phase6b_results.csv", index=False)

meta_6b = {
    "model_names": ["LGB_Proto", "XGB_Proto", "XGB_IfaceHC_6a", "ProtoTargetMLP"],
    "weights": w_opt, "thresholds": thresholds,
    "target_names": target_names, "target_enc": target_enc,
    "proto_pos": proto_pos, "proto_neg": proto_neg,
    "n_pos": n_pos_dict, "n_neg": n_neg_dict,
    "proto_feat_cols": proto_feat_cols, "all_feat_cols": all_feat_cols,
}
with open(MODELS_DIR / "ensemble_meta_6b.pkl", "wb") as f:
    pickle.dump(meta_6b, f)
joblib.dump(cal, MODELS_DIR / "calibrator_6b.pkl")
np.save(MODELS_DIR / "proto_tensor.npy", proto_tensor.numpy())

print(f"\nSaved: outputs/phase6b_results.csv")
print(f"Saved: models/lgb_proto.pkl, xgb_proto.pkl, proto_target_mlp.pt")
print(f"Saved: models/ensemble_meta_6b.pkl, calibrator_6b.pkl, proto_tensor.npy")
print("Phase 6b complete.")
