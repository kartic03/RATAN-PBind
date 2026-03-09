"""
Phase 4: ESM-2 Embeddings + Embedding-based Models
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import esm as fair_esm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEAT_DIR     = PROJECT_ROOT / "features"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg): print(msg, flush=True)

log(f"Device: {DEVICE}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load data ─────────────────────────────────────────────────────────────────

log("\nLoading data...")
proteins_df = pd.read_parquet(DATA_DIR / "proteins.parquet")
pairs_df    = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
col_meta    = pd.read_csv(FEAT_DIR / "feature_columns.csv")
feat_cols   = col_meta["column"].tolist()
fm          = pd.read_parquet(FEAT_DIR / "feature_matrix.parquet")

train_df = fm[fm["split"] == "train"].reset_index(drop=True)
val_df   = fm[fm["split"] == "val"].reset_index(drop=True)
test_df  = fm[fm["split"] == "test"].reset_index(drop=True)

log(f"  Proteins: {len(proteins_df):,}")
log(f"  Pairs — Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ── Step 1: Generate ESM-2 embeddings ─────────────────────────────────────────

EMB_PATH = FEAT_DIR / "esm2_embeddings.npy"
ID_PATH  = FEAT_DIR / "esm2_protein_ids.npy"

if EMB_PATH.exists() and ID_PATH.exists():
    log("\nFound cached ESM-2 embeddings — loading...")
    embeddings   = np.load(EMB_PATH)
    protein_ids  = np.load(ID_PATH, allow_pickle=True)
    log(f"  Loaded: {embeddings.shape}")
else:
    log("\n[Step 1] Generating ESM-2 embeddings (esm2_t33_650M_UR50D)...")
    log("  Loading model via fair-esm...")

    esm_model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(DEVICE)
    esm_model.eval()

    param_count = sum(p.numel() for p in esm_model.parameters()) / 1e6
    log(f"  Model loaded: {param_count:.0f}M parameters")

    sequences   = proteins_df["sequence"].tolist()
    protein_ids = proteins_df["protein_id"].values

    # Clean sequences: keep only standard ESM-2 amino acids
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")  # ESM-2 supported tokens
    def clean_seq(s):
        return "".join(c for c in str(s).upper() if c in VALID_AA) or "A"

    sequences = [clean_seq(s) for s in sequences]
    n_cleaned = sum(1 for s in sequences if len(s) == 0)
    log(f"  Sequences cleaned ({n_cleaned} were empty, replaced with 'A')")

    BATCH_SIZE  = 32
    MAX_SEQ_LEN = 1022  # ESM-2 max tokens
    embeddings_list = []

    log(f"  Embedding {len(sequences):,} proteins in batches of {BATCH_SIZE}...")

    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i : i + BATCH_SIZE]
            # Truncate long sequences
            batch_data = [(str(j), s[:MAX_SEQ_LEN]) for j, s in enumerate(batch_seqs)]

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(DEVICE)

            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]  # (B, L+2, 1280)

            # Mean pool over sequence (exclude BOS token 0 and EOS)
            for k, (_, seq) in enumerate(batch_data):
                seq_len = min(len(seq), MAX_SEQ_LEN)
                emb = token_reps[k, 1:seq_len+1].mean(0)  # exclude BOS
                embeddings_list.append(emb.cpu().numpy())

            if (i // BATCH_SIZE + 1) % 10 == 0 or i + BATCH_SIZE >= len(sequences):
                done = min(i + BATCH_SIZE, len(sequences))
                log(f"    {done:>5}/{len(sequences)}  ({done/len(sequences)*100:.1f}%)")

    embeddings = np.vstack(embeddings_list)
    log(f"  Embeddings shape: {embeddings.shape}  (dtype: {embeddings.dtype})")

    np.save(EMB_PATH, embeddings)
    np.save(ID_PATH,  protein_ids)
    log(f"  Saved: features/esm2_embeddings.npy  ({embeddings.nbytes/1e6:.1f} MB)")

    del esm_model
    torch.cuda.empty_cache()

# ── Build embedding lookup ────────────────────────────────────────────────────

id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

def get_emb(protein_id):
    idx = id_to_idx.get(protein_id)
    return embeddings[idx] if idx is not None else np.zeros(1280)

log("\nBuilding embedding matrices for splits...")
X_emb_train = np.vstack([get_emb(pid) for pid in train_df["protein_id"]])
X_emb_val   = np.vstack([get_emb(pid) for pid in val_df["protein_id"]])
X_emb_test  = np.vstack([get_emb(pid) for pid in test_df["protein_id"]])

y_train = train_df["binding_label"].values
y_val   = val_df["binding_label"].values
y_test  = test_df["binding_label"].values

# Handcrafted features
X_hc_train = train_df[feat_cols].values
X_hc_val   = val_df[feat_cols].values
X_hc_test  = test_df[feat_cols].values

# Combined
X_comb_train = np.hstack([X_emb_train, X_hc_train])
X_comb_val   = np.hstack([X_emb_val,   X_hc_val])
X_comb_test  = np.hstack([X_emb_test,  X_hc_test])

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
log(f"  Emb dim: {X_emb_train.shape[1]}  |  Combined dim: {X_comb_train.shape[1]}")

# ── Evaluation helper ─────────────────────────────────────────────────────────

results = []

def evaluate(name, proba, y, split="val"):
    preds = (proba >= 0.5).astype(int)
    r = {
        "model": name, "split": split,
        "auroc": roc_auc_score(y, proba),
        "auprc": average_precision_score(y, proba),
        "f1":    f1_score(y, preds, zero_division=0),
        "mcc":   matthews_corrcoef(y, preds),
    }
    log(f"  ✓ {name:<35} AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  F1={r['f1']:.4f}  MCC={r['mcc']:.4f}")
    return r

# ── Step 2: XGBoost on ESM-2 embeddings only ──────────────────────────────────

log("\n[Step 2] XGBoost on ESM-2 embeddings only (Optuna 60 trials)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def xgb_obj(trial, X, y):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 800),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("lr", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("col", 0.3, 1.0),
        min_child_weight = trial.suggest_int("mcw", 1, 10),
        reg_alpha        = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        scale_pos_weight = pos_weight,
        tree_method      = "hist",
        device           = "cuda",
        eval_metric      = "aucpr",
        random_state     = RANDOM_SEED,
        n_jobs           = 1,
    )
    scores = []
    for tr, vl in skf.split(X, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr], y[tr], verbose=False)
        scores.append(average_precision_score(y[vl], m.predict_proba(X[vl])[:,1]))
    return np.mean(scores)

study_emb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_emb.optimize(lambda t: xgb_obj(t, X_emb_train, y_train),
                   n_trials=60, show_progress_bar=True)
log(f"  Best CV AUPRC: {study_emb.best_value:.4f}")

xgb_emb = xgb.XGBClassifier(**{**study_emb.best_params,
                                 "scale_pos_weight": pos_weight, "tree_method": "hist",
                                 "device": "cuda", "eval_metric": "aucpr",
                                 "random_state": RANDOM_SEED, "n_jobs": -1})
xgb_emb.fit(X_emb_train, y_train)
results.append(evaluate("XGB_ESM2only", xgb_emb.predict_proba(X_emb_val)[:,1], y_val))
joblib.dump(xgb_emb, MODEL_DIR / "xgb_esm2only.pkl")

# ── Step 3: XGBoost on ESM-2 + handcrafted (expected best) ───────────────────

log("\n[Step 3] XGBoost on ESM-2 + Handcrafted features (Optuna 60 trials)...")
study_comb = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_comb.optimize(lambda t: xgb_obj(t, X_comb_train, y_train),
                    n_trials=60, show_progress_bar=True)
log(f"  Best CV AUPRC: {study_comb.best_value:.4f}")

xgb_comb = xgb.XGBClassifier(**{**study_comb.best_params,
                                  "scale_pos_weight": pos_weight, "tree_method": "hist",
                                  "device": "cuda", "eval_metric": "aucpr",
                                  "random_state": RANDOM_SEED, "n_jobs": -1})
xgb_comb.fit(X_comb_train, y_train)
results.append(evaluate("XGB_ESM2+Handcrafted", xgb_comb.predict_proba(X_comb_val)[:,1], y_val))
joblib.dump(xgb_comb, MODEL_DIR / "xgb_esm2_combined.pkl")

# ── Step 4: LightGBM on ESM-2 + handcrafted ───────────────────────────────────

log("\n[Step 4] LightGBM on ESM-2 + Handcrafted (Optuna 60 trials)...")

def lgb_obj(trial, X, y):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 800),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("lr", 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 100),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("col", 0.3, 1.0),
        min_child_samples = trial.suggest_int("mcs", 5, 50),
        reg_alpha         = trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        is_unbalance      = True,
        random_state      = RANDOM_SEED,
        n_jobs            = 1,
        verbose           = -1,
    )
    scores = []
    for tr, vl in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr])
        scores.append(average_precision_score(y[vl], m.predict_proba(X[vl])[:,1]))
    return np.mean(scores)

study_lgb_comb = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_lgb_comb.optimize(lambda t: lgb_obj(t, X_comb_train, y_train),
                         n_trials=60, show_progress_bar=True)
log(f"  Best CV AUPRC: {study_lgb_comb.best_value:.4f}")

lgb_comb = lgb.LGBMClassifier(**{**study_lgb_comb.best_params,
                                   "is_unbalance": True, "random_state": RANDOM_SEED,
                                   "n_jobs": -1, "verbose": -1})
lgb_comb.fit(X_comb_train, y_train)
results.append(evaluate("LGB_ESM2+Handcrafted", lgb_comb.predict_proba(X_comb_val)[:,1], y_val))
joblib.dump(lgb_comb, MODEL_DIR / "lgb_esm2_combined.pkl")

# ── Step 5: MLP on ESM-2 embeddings ──────────────────────────────────────────

log("\n[Step 5] MLP on ESM-2 embeddings (PyTorch, GPU)...")

class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Normalize embeddings for MLP
from sklearn.preprocessing import StandardScaler as SS
emb_scaler = SS()
X_emb_tr_n = emb_scaler.fit_transform(X_emb_train)
X_emb_vl_n = emb_scaler.transform(X_emb_val)
X_emb_te_n = emb_scaler.transform(X_emb_test)
joblib.dump(emb_scaler, FEAT_DIR / "emb_scaler.pkl")

train_ds = ProteinDataset(X_emb_tr_n, y_train)
val_ds   = ProteinDataset(X_emb_vl_n, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128)

mlp = MLP(input_dim=1280).to(DEVICE)
pos_w = torch.tensor([pos_weight], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

best_auprc, best_state = 0.0, None
patience, patience_ctr = 10, 0

log("  Training MLP (up to 100 epochs, early stopping patience=10)...")
for epoch in range(1, 101):
    mlp.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(mlp(Xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    mlp.eval()
    probs = []
    with torch.no_grad():
        for Xb, _ in val_loader:
            probs.append(torch.sigmoid(mlp(Xb.to(DEVICE))).cpu().numpy())
    proba = np.concatenate(probs)
    auprc = average_precision_score(y_val, proba)

    if auprc > best_auprc:
        best_auprc = auprc
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1

    if epoch % 10 == 0 or patience_ctr == 0:
        auroc = roc_auc_score(y_val, proba)
        log(f"  Epoch {epoch:>3}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}  (best={best_auprc:.4f})")

    if patience_ctr >= patience:
        log(f"  Early stopping at epoch {epoch}")
        break

mlp.load_state_dict(best_state)
mlp.eval()
with torch.no_grad():
    val_proba = torch.sigmoid(mlp(torch.FloatTensor(X_emb_vl_n).to(DEVICE))).cpu().numpy()
results.append(evaluate("MLP_ESM2", val_proba, y_val))
torch.save(mlp.state_dict(), MODEL_DIR / "mlp_esm2.pt")

# ── Step 6: MLP on ESM-2 + Handcrafted features ───────────────────────────────

log("\n[Step 6] MLP on ESM-2 + Handcrafted features...")

# Combine normalized embeddings with already-scaled handcrafted features
X_comb_tr_n = np.hstack([X_emb_tr_n, X_hc_train])
X_comb_vl_n = np.hstack([X_emb_vl_n, X_hc_val])
X_comb_te_n = np.hstack([X_emb_te_n, X_hc_test])

train_ds2 = ProteinDataset(X_comb_tr_n, y_train)
val_ds2   = ProteinDataset(X_comb_vl_n, y_val)
train_loader2 = DataLoader(train_ds2, batch_size=64, shuffle=True)
val_loader2   = DataLoader(val_ds2,   batch_size=128)

mlp2 = MLP(input_dim=X_comb_tr_n.shape[1], hidden=[512, 256, 128]).to(DEVICE)
opt2 = torch.optim.AdamW(mlp2.parameters(), lr=1e-3, weight_decay=0.01)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=80)

best_auprc2, best_state2, patience_ctr2 = 0.0, None, 0
log("  Training MLP (up to 100 epochs, early stopping patience=10)...")

for epoch in range(1, 101):
    mlp2.train()
    for Xb, yb in train_loader2:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt2.zero_grad()
        criterion(mlp2(Xb), yb).backward()
        opt2.step()
    sch2.step()

    mlp2.eval()
    probs2 = []
    with torch.no_grad():
        for Xb, _ in val_loader2:
            probs2.append(torch.sigmoid(mlp2(Xb.to(DEVICE))).cpu().numpy())
    proba2 = np.concatenate(probs2)
    auprc2 = average_precision_score(y_val, proba2)

    if auprc2 > best_auprc2:
        best_auprc2 = auprc2
        best_state2 = {k: v.clone() for k, v in mlp2.state_dict().items()}
        patience_ctr2 = 0
    else:
        patience_ctr2 += 1

    if epoch % 10 == 0 or patience_ctr2 == 0:
        auroc2 = roc_auc_score(y_val, proba2)
        log(f"  Epoch {epoch:>3}: AUPRC={auprc2:.4f}  AUROC={auroc2:.4f}  (best={best_auprc2:.4f})")

    if patience_ctr2 >= patience:
        log(f"  Early stopping at epoch {epoch}")
        break

mlp2.load_state_dict(best_state2)
mlp2.eval()
with torch.no_grad():
    val_proba2 = torch.sigmoid(mlp2(torch.FloatTensor(X_comb_vl_n).to(DEVICE))).cpu().numpy()
results.append(evaluate("MLP_ESM2+Handcrafted", val_proba2, y_val))
torch.save(mlp2.state_dict(), MODEL_DIR / "mlp_esm2_combined.pt")

# ── Final results ─────────────────────────────────────────────────────────────

log("\n══ PHASE 4 VALIDATION RESULTS ══════════════════════════════════════════")
rdf = pd.DataFrame(results).sort_values("auprc", ascending=False)
log(f"  {'Model':<35} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'MCC':>7}")
log(f"  {'-'*65}")
for _, r in rdf.iterrows():
    log(f"  {r['model']:<35} {r['auroc']:>7.4f} {r['auprc']:>7.4f} {r['f1']:>7.4f} {r['mcc']:>7.4f}")

# ── Test set evaluation ───────────────────────────────────────────────────────

log("\n══ TEST SET RESULTS ════════════════════════════════════════════════════")
test_results = []

models_test = [
    ("XGB_ESM2only",         xgb_emb,  X_emb_test),
    ("XGB_ESM2+Handcrafted", xgb_comb, X_comb_test),
    ("LGB_ESM2+Handcrafted", lgb_comb, X_comb_test),
]
for name, model, X in models_test:
    proba = model.predict_proba(X)[:,1]
    r = evaluate(name, proba, y_test, split="test")
    test_results.append(r)

# MLP test
mlp.eval()
with torch.no_grad():
    te_proba = torch.sigmoid(mlp(torch.FloatTensor(X_emb_te_n).to(DEVICE))).cpu().numpy()
test_results.append(evaluate("MLP_ESM2", te_proba, y_test, split="test"))

mlp2.eval()
with torch.no_grad():
    te_proba2 = torch.sigmoid(mlp2(torch.FloatTensor(X_comb_te_n).to(DEVICE))).cpu().numpy()
test_results.append(evaluate("MLP_ESM2+Handcrafted", te_proba2, y_test, split="test"))

# ── Compare with Phase 3 best ─────────────────────────────────────────────────

log("\n══ COMPARISON: Phase 3 vs Phase 4 (Test Set) ═══════════════════════════")
ph3_test = pd.read_csv(OUTPUT_DIR / "test_results.csv")
ph4_test = pd.DataFrame(test_results)
comparison = pd.concat([ph3_test, ph4_test]).sort_values("auprc", ascending=False)
log(f"  {'Model':<35} {'Split':>5} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'MCC':>7}")
log(f"  {'-'*75}")
for _, r in comparison.iterrows():
    log(f"  {r['model']:<35} {r['split']:>5} {r['auroc']:>7.4f} {r['auprc']:>7.4f} {r['f1']:>7.4f} {r['mcc']:>7.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────

rdf.to_csv(OUTPUT_DIR / "phase4_val_results.csv", index=False)
ph4_test.to_csv(OUTPUT_DIR / "phase4_test_results.csv", index=False)
comparison.to_csv(OUTPUT_DIR / "all_test_results.csv", index=False)

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
comp_plot = comparison.copy()
comp_plot["label"] = comp_plot["model"].str.replace("_", "\n")
palette = ["#DD8452" if "ESM2" in m else "#4C72B0" for m in comp_plot["model"]]
for ax, metric in zip(axes, ["auroc", "auprc"]):
    bars = ax.bar(comp_plot["label"], comp_plot[metric], color=palette)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"All Models — Test {metric.upper()}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, comp_plot[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#DD8452", label="ESM-2 models"),
                   Patch(color="#4C72B0", label="Phase 3 baseline")],
          loc="lower right")
plt.suptitle("Phase 3 vs Phase 4 — Test Set Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "phase4_comparison.png", dpi=150)
plt.close()

log("\nSaved: outputs/phase4_val_results.csv  phase4_test_results.csv  all_test_results.csv")
log("Saved: outputs/phase4_comparison.png")
log("\n✓ Phase 4 complete.")
