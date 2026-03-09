"""
Phase 5: Deep Learning Models
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Models:
  A: Advanced MLP (ESM-2 + Handcrafted, BatchNorm, GELU, plain BCE)
  B: Multi-Task MLP (binding + binding_strength + expression)
  C: Target-Aware MLP (learnable target embeddings - best DL model)
  D: Fine-Tuned ESM-2 (last 4 transformer layers; manual tokenizer for Python 3.13)
  E: Fine-Tuned ESM-2 + Handcrafted features (expected best DL)

GPU: All neural networks
"""

import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from pathlib import Path
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
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

amp_ctx = lambda: torch.amp.autocast('cuda') if device.type == 'cuda' else contextlib.nullcontext()

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
pairs    = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
feat_mat = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")
feat_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()

esm_emb = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_map = {pid: i for i, pid in enumerate(esm_ids)}

# Merge handcrafted features
pairs = pairs.merge(feat_mat[["protein_id", "target"] + feat_cols],
                    on=["protein_id", "target"], how="left", suffixes=("", "_feat"))

# ESM-2 embeddings aligned to pairs
emb_dim = esm_emb.shape[1]
X_emb = np.zeros((len(pairs), emb_dim), dtype=np.float32)
for i, pid in enumerate(pairs["protein_id"]):
    idx = esm_map.get(pid)
    if idx is not None:
        X_emb[i] = esm_emb[idx]

X_hc   = np.nan_to_num(pairs[feat_cols].values.astype(np.float32), nan=0.0)
X_emb  = np.nan_to_num(X_emb, nan=0.0)
X_comb = np.concatenate([X_emb, X_hc], axis=1)

# Labels
y_bind    = pairs["binding_label"].values.astype(np.float32)
y_str_raw = pairs["binding_strength"].values.astype(float)
y_expr_raw = pd.to_numeric(pairs["expressed"], errors="coerce").values.astype(float)
y_str  = np.where(np.isnan(y_str_raw),  -1.0, y_str_raw).astype(np.float32)
y_expr = np.where(np.isnan(y_expr_raw), -1.0, y_expr_raw).astype(np.float32)

# Target encoding
target_names = sorted(pairs["target"].unique())
target_enc   = {t: i for i, t in enumerate(target_names)}
t_idx_arr    = pairs["target"].map(target_enc).values.astype(np.int64)
n_targets    = len(target_names)

train_m = (pairs["split"] == "train").values
val_m   = (pairs["split"] == "val").values
test_m  = (pairs["split"] == "test").values

pos_rate = y_bind[train_m].mean()
print(f"  Train: {train_m.sum()}  Val: {val_m.sum()}  Test: {test_m.sum()}")
print(f"  Binding rate (train): {pos_rate:.1%}")
print(f"  Combined feature dim: {X_comb.shape[1]}")

# ── Metrics & helpers ─────────────────────────────────────────────────────────
def metrics(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    yp    = (y_prob >= 0.5).astype(int)
    return auroc, auprc, f1_score(y_true, yp, zero_division=0), matthews_corrcoef(y_true, yp)

# ── Dataset ───────────────────────────────────────────────────────────────────
class BindingDataset(Dataset):
    def __init__(self, X, y_bind, y_str, y_expr, t_idx):
        self.X      = torch.tensor(X,      dtype=torch.float32)
        self.y_bind = torch.tensor(y_bind, dtype=torch.float32)
        self.y_str  = torch.tensor(y_str,  dtype=torch.float32)
        self.y_expr = torch.tensor(y_expr, dtype=torch.float32)
        self.t_idx  = torch.tensor(t_idx,  dtype=torch.long)
    def __len__(self): return len(self.y_bind)
    def __getitem__(self, i):
        return self.X[i], self.y_bind[i], self.y_str[i], self.y_expr[i], self.t_idx[i]

def make_ds(X, mask):
    return BindingDataset(X[mask], y_bind[mask], y_str[mask], y_expr[mask], t_idx_arr[mask])

# ── Training loop
#    KEY FIXES vs. first attempt:
#    1. Use standard BCE (no pos_weight) - Phase 4 MLP worked this way
#    2. Use BatchNorm instead of LayerNorm (better for tabular features)
#    3. lr=1e-3 to match Phase 4 MLP convergence speed
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(model, tr_mask, val_mask, X,
               task="single", name="model",
               n_epochs=100, patience=12, lr=1e-3, wd=0.01, batch=64):

    bce = nn.BCEWithLogitsLoss()  # NO pos_weight (Phase 4 approach)

    tr_ds  = make_ds(X, tr_mask)
    val_ds = make_ds(X, val_mask)
    tr_ldr  = DataLoader(tr_ds,  batch_size=batch, shuffle=True,  num_workers=0)
    val_ldr = DataLoader(val_ds, batch_size=256,   shuffle=False, num_workers=0)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_auprc, best_state, no_imp = -1.0, None, 0
    model.to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch_data in tr_ldr:
            X_b, yb, ys, ye, ti = [t.to(device) for t in batch_data]
            opt.zero_grad()
            with amp_ctx():
                if task == "single":
                    loss = bce(model(X_b).squeeze(-1), yb)
                elif task == "multitask":
                    bl, sl, el = model(X_b)
                    loss = bce(bl.squeeze(-1), yb)
                    sm = ys >= 0
                    if sm.sum() > 0:
                        loss = loss + 0.3 * F.cross_entropy(sl[sm], ys[sm].long())
                    em = ye >= 0
                    if em.sum() > 0:
                        loss = loss + 0.2 * bce(el.squeeze(-1)[em], ye[em])
                elif task == "target_aware":
                    loss = bce(model(X_b, ti).squeeze(-1), yb)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        probs_list, y_true_list = [], []
        with torch.no_grad(), amp_ctx():
            for batch_data in val_ldr:
                X_b, yb, ys, ye, ti = [t.to(device) for t in batch_data]
                if task == "single":
                    logits = model(X_b).squeeze(-1)
                elif task == "multitask":
                    logits, _, _ = model(X_b)
                    logits = logits.squeeze(-1)
                elif task == "target_aware":
                    logits = model(X_b, ti).squeeze(-1)
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
                y_true_list.append(yb.cpu().numpy())

        probs  = np.concatenate(probs_list)
        y_true = np.concatenate(y_true_list)
        auroc, auprc, f1, mcc = metrics(y_true, probs)
        sched.step(auprc)

        improved = auprc > best_auprc
        if epoch <= 10 or improved or epoch % 10 == 0:
            marker = "*" if improved else " "
            print(f"  Epoch {epoch:3d}{marker}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  (best={best_auprc:.4f})")
        if improved:
            best_auprc = auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model

def evaluate(model, X, mask, task="single"):
    model.eval()
    ds  = make_ds(X, mask)
    ldr = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    probs_list, y_true_list = [], []
    with torch.no_grad(), amp_ctx():
        for batch_data in ldr:
            X_b, yb, ys, ye, ti = [t.to(device) for t in batch_data]
            if task == "single":
                logits = model(X_b).squeeze(-1)
            elif task == "multitask":
                logits, _, _ = model(X_b)
                logits = logits.squeeze(-1)
            elif task == "target_aware":
                logits = model(X_b, ti).squeeze(-1)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
            y_true_list.append(yb.cpu().numpy())
    return metrics(np.concatenate(y_true_list), np.concatenate(probs_list))

# ══════════════════════════════════════════════════════════════════════════════
# MODEL A: Advanced MLP (ESM-2 + Handcrafted)
#          BatchNorm + GELU + standard BCE (no pos_weight)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[Model A] Advanced MLP (ESM-2 + Handcrafted, BatchNorm, BCE)")

class AdvancedMLP(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

modelA = AdvancedMLP(X_comb.shape[1])
modelA = train_loop(modelA, train_m, val_m, X_comb,
                    task="single", name="AdvancedMLP")
torch.save(modelA.state_dict(), MODELS_DIR / "adv_mlp.pt")
val_A  = evaluate(modelA, X_comb, val_m)
test_A = evaluate(modelA, X_comb, test_m)
print(f"  Val  — AUROC={val_A[0]:.4f}  AUPRC={val_A[1]:.4f}  F1={val_A[2]:.4f}  MCC={val_A[3]:.4f}")
print(f"  Test — AUROC={test_A[0]:.4f}  AUPRC={test_A[1]:.4f}  F1={test_A[2]:.4f}  MCC={test_A[3]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL B: Multi-Task MLP (binding + binding_strength + expression)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[Model B] Multi-Task MLP (binding + strength + expression)")

class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, n_str=4, dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.bind_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        self.str_head  = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, n_str))
        self.expr_head = nn.Sequential(nn.Linear(256, 64),  nn.GELU(), nn.Linear(64,  1))
    def forward(self, x):
        h = self.shared(x)
        return self.bind_head(h), self.str_head(h), self.expr_head(h)

modelB = MultiTaskMLP(X_comb.shape[1])
modelB = train_loop(modelB, train_m, val_m, X_comb,
                    task="multitask", name="MultiTaskMLP")
torch.save(modelB.state_dict(), MODELS_DIR / "multitask_mlp.pt")
val_B  = evaluate(modelB, X_comb, val_m,  task="multitask")
test_B = evaluate(modelB, X_comb, test_m, task="multitask")
print(f"  Val  — AUROC={val_B[0]:.4f}  AUPRC={val_B[1]:.4f}  F1={val_B[2]:.4f}  MCC={val_B[3]:.4f}")
print(f"  Test — AUROC={test_B[0]:.4f}  AUPRC={test_B[1]:.4f}  F1={test_B[2]:.4f}  MCC={test_B[3]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL C: Target-Aware MLP (learnable target embeddings)
#          Best Phase 5 model — target identity is crucial information
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[Model C] Target-Aware MLP (learnable 32-dim target embeddings)")

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
modelC = train_loop(modelC, train_m, val_m, X_comb,
                    task="target_aware", name="TargetAwareMLP",
                    n_epochs=150, patience=15)
torch.save(modelC.state_dict(), MODELS_DIR / "target_aware_mlp.pt")
val_C  = evaluate(modelC, X_comb, val_m,  task="target_aware")
test_C = evaluate(modelC, X_comb, test_m, task="target_aware")
print(f"  Val  — AUROC={val_C[0]:.4f}  AUPRC={val_C[1]:.4f}  F1={val_C[2]:.4f}  MCC={val_C[3]:.4f}")
print(f"  Test — AUROC={test_C[0]:.4f}  AUPRC={test_C[1]:.4f}  F1={test_C[2]:.4f}  MCC={test_C[3]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL D: Fine-Tuned ESM-2 (last 4 transformer layers + classification head)
#
# WORKAROUND for Python 3.13 + fair-esm tokenizer segfault:
#   fair-esm's batch_converter uses regex-based tokenization that crashes in
#   Python 3.13 (changed re.split cache behavior in C extension).
#   FIX: manual_tokenize() replicates ESM-2's character-level tokenization
#   directly using alphabet.tok_to_idx without any regex, avoiding the crash.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[Model D] Fine-Tuned ESM-2 (last 4 layers, manual tokenizer workaround)")

import esm as fair_esm

print("  Loading ESM-2 (esm2_t33_650M_UR50D)...")
esm_model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
esm_model.eval()

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUOGJ")
MAX_LEN  = 1022

def clean_seq(seq):
    return "".join(c if c in VALID_AA else "X" for c in str(seq).upper())

def manual_tokenize(seqs, max_len=MAX_LEN):
    """
    ESM-2 is character-level (one amino acid = one token, no BPE).
    This manually replicates batch_converter without calling the buggy
    Python 3.13-incompatible regex tokenizer in fair-esm.
    """
    tok2idx = alphabet.tok_to_idx
    unk_idx = tok2idx.get('<unk>', 3)
    pad_idx = alphabet.padding_idx
    cls_idx = alphabet.cls_idx  # BOS
    eos_idx = alphabet.eos_idx  # EOS

    all_toks = []
    for seq in seqs:
        toks = [cls_idx]
        for c in clean_seq(seq)[:max_len]:
            toks.append(tok2idx.get(c, unk_idx))
        toks.append(eos_idx)
        all_toks.append(toks)

    max_tok_len = max(len(t) for t in all_toks)
    result = torch.full((len(seqs), max_tok_len), pad_idx, dtype=torch.long)
    for i, toks in enumerate(all_toks):
        result[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return result

# Tokenize all sequences upfront (avoids tokenizer in DataLoader)
print("  Tokenizing sequences (manual, Python-3.13-safe)...")
pairs_train = pairs[train_m].reset_index(drop=True)
pairs_val   = pairs[val_m].reset_index(drop=True)
pairs_test  = pairs[test_m].reset_index(drop=True)

tok_train = manual_tokenize(pairs_train["sequence"].tolist())
tok_val   = manual_tokenize(pairs_val["sequence"].tolist())
tok_test  = manual_tokenize(pairs_test["sequence"].tolist())
print(f"  Token shapes — train:{tuple(tok_train.shape)}  val:{tuple(tok_val.shape)}")

y_tr = torch.tensor(pairs_train["binding_label"].values, dtype=torch.float32)
y_v  = torch.tensor(pairs_val["binding_label"].values,   dtype=torch.float32)
y_te = torch.tensor(pairs_test["binding_label"].values,  dtype=torch.float32)

tr_ds_D   = TensorDataset(tok_train, y_tr)
val_ds_D  = TensorDataset(tok_val,   y_v)
test_ds_D = TensorDataset(tok_test,  y_te)
tr_ldr_D  = DataLoader(tr_ds_D,  batch_size=16, shuffle=True,  num_workers=0)
val_ldr_D = DataLoader(val_ds_D, batch_size=32, shuffle=False, num_workers=0)

# Freeze all, then unfreeze last 4 transformer blocks + final layer norm
for param in esm_model.parameters():
    param.requires_grad = False
n_layers = len(esm_model.layers)  # 33
for i in range(n_layers - 4, n_layers):
    for param in esm_model.layers[i].parameters():
        param.requires_grad = True
for param in esm_model.emb_layer_norm_after.parameters():
    param.requires_grad = True

class ESMClassifier(nn.Module):
    def __init__(self, esm_model, repr_layer=33, dropout=0.2):
        super().__init__()
        self.esm        = esm_model
        self.repr_layer = repr_layer
        self.head = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256,  64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64,   1),
        )
    def forward(self, tokens):
        out  = self.esm(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]
        mask = (tokens != alphabet.padding_idx) & (tokens != alphabet.cls_idx)
        mask = mask.unsqueeze(-1).float()
        emb  = (reps * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(emb)

modelD = ESMClassifier(esm_model)
trainable = sum(p.numel() for p in modelD.parameters() if p.requires_grad)
total     = sum(p.numel() for p in modelD.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

modelD.to(device)
opt_D    = torch.optim.AdamW(
               filter(lambda p: p.requires_grad, modelD.parameters()),
               lr=5e-6, weight_decay=0.01)
sched_D  = torch.optim.lr_scheduler.ReduceLROnPlateau(
               opt_D, mode='max', factor=0.5, patience=4, min_lr=1e-7)
scaler_D = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
bce_D    = nn.BCEWithLogitsLoss()

best_auprc_D, best_state_D, no_imp_D = -1.0, None, 0
print("  Fine-tuning ESM-2 (up to 50 epochs, patience=10)...")

for epoch in range(1, 51):
    modelD.train()
    for tokens, yb in tr_ldr_D:
        tokens, yb = tokens.to(device), yb.to(device)
        opt_D.zero_grad()
        with amp_ctx():
            loss = bce_D(modelD(tokens).squeeze(-1), yb)
        if scaler_D:
            scaler_D.scale(loss).backward()
            scaler_D.unscale_(opt_D)
            nn.utils.clip_grad_norm_(modelD.parameters(), 1.0)
            scaler_D.step(opt_D)
            scaler_D.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(modelD.parameters(), 1.0)
            opt_D.step()

    modelD.eval()
    probs_D, yt_D = [], []
    with torch.no_grad(), amp_ctx():
        for tokens, yb in val_ldr_D:
            probs_D.append(torch.sigmoid(modelD(tokens.to(device)).squeeze(-1)).cpu().numpy())
            yt_D.append(yb.numpy())
    probs_D  = np.concatenate(probs_D)
    yt_D     = np.concatenate(yt_D)
    auroc_D, auprc_D, f1_D, mcc_D = metrics(yt_D, probs_D)
    sched_D.step(auprc_D)

    improved = auprc_D > best_auprc_D
    if epoch <= 10 or improved or epoch % 10 == 0:
        marker = "*" if improved else " "
        lr_now = opt_D.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}{marker}: AUPRC={auprc_D:.4f}  AUROC={auroc_D:.4f}  "
              f"lr={lr_now:.2e}  (best={best_auprc_D:.4f})")
    if improved:
        best_auprc_D = auprc_D
        best_state_D = {k: v.cpu().clone() for k, v in modelD.state_dict().items()}
        no_imp_D = 0
    else:
        no_imp_D += 1
        if no_imp_D >= 10:
            print(f"  Early stopping at epoch {epoch}")
            break

modelD.load_state_dict(best_state_D)
torch.save(modelD.state_dict(), MODELS_DIR / "esm2_finetuned.pt")

def eval_esm(model, ds):
    ldr = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    probs, yt = [], []
    model.eval()
    with torch.no_grad(), amp_ctx():
        for tokens, yb in ldr:
            probs.append(torch.sigmoid(model(tokens.to(device)).squeeze(-1)).cpu().numpy())
            yt.append(yb.numpy())
    return metrics(np.concatenate(yt), np.concatenate(probs))

val_D  = eval_esm(modelD, val_ds_D)
test_D = eval_esm(modelD, test_ds_D)
print(f"  Val  — AUROC={val_D[0]:.4f}  AUPRC={val_D[1]:.4f}  F1={val_D[2]:.4f}  MCC={val_D[3]:.4f}")
print(f"  Test — AUROC={test_D[0]:.4f}  AUPRC={test_D[1]:.4f}  F1={test_D[2]:.4f}  MCC={test_D[3]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL E: Fine-Tuned ESM-2 embeddings + Handcrafted features
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("[Model E] Fine-Tuned ESM-2 embeddings + Handcrafted features")

def extract_ft_embeddings(model, tok_tensor):
    all_emb = []
    model.eval()
    BS = 32
    for i in range(0, len(tok_tensor), BS):
        batch_tok = tok_tensor[i:i+BS].to(device)
        with torch.no_grad(), amp_ctx():
            out  = model.esm(batch_tok, repr_layers=[model.repr_layer], return_contacts=False)
            reps = out["representations"][model.repr_layer]
            mask = (batch_tok != alphabet.padding_idx) & (batch_tok != alphabet.cls_idx)
            mask = mask.unsqueeze(-1).float()
            emb  = (reps * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)

print("  Extracting fine-tuned ESM-2 embeddings...")
ft_tr  = extract_ft_embeddings(modelD, tok_train)
ft_val = extract_ft_embeddings(modelD, tok_val)
ft_te  = extract_ft_embeddings(modelD, tok_test)

XE_tr  = np.concatenate([ft_tr,  X_hc[train_m]], axis=1)
XE_val = np.concatenate([ft_val, X_hc[val_m]],   axis=1)
XE_te  = np.concatenate([ft_te,  X_hc[test_m]],  axis=1)

XE_all = np.zeros((len(pairs), XE_tr.shape[1]), dtype=np.float32)
XE_all[train_m] = XE_tr
XE_all[val_m]   = XE_val
XE_all[test_m]  = XE_te

modelE = AdvancedMLP(XE_all.shape[1])
modelE = train_loop(modelE, train_m, val_m, XE_all,
                    task="single", name="ESM2_FT_Combined")
torch.save(modelE.state_dict(), MODELS_DIR / "esm2_ft_combined.pt")
val_E  = evaluate(modelE, XE_all, val_m)
test_E = evaluate(modelE, XE_all, test_m)
print(f"  Val  — AUROC={val_E[0]:.4f}  AUPRC={val_E[1]:.4f}  F1={val_E[2]:.4f}  MCC={val_E[3]:.4f}")
print(f"  Test — AUROC={test_E[0]:.4f}  AUPRC={test_E[1]:.4f}  F1={test_E[2]:.4f}  MCC={test_E[3]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
results_val = {
    "AdvancedMLP_ESM2+HC":    val_A,
    "MultiTaskMLP_ESM2+HC":   val_B,
    "TargetAwareMLP_ESM2+HC": val_C,
    "ESM2_Finetuned":         val_D,
    "ESM2_Finetuned+HC":      val_E,
}
results_test = {
    "AdvancedMLP_ESM2+HC":    test_A,
    "MultiTaskMLP_ESM2+HC":   test_B,
    "TargetAwareMLP_ESM2+HC": test_C,
    "ESM2_Finetuned":         test_D,
    "ESM2_Finetuned+HC":      test_E,
}

for split_name, results in [("VALIDATION", results_val), ("TEST", results_test)]:
    print(f"\n{'='*70}")
    print(f"PHASE 5 — {split_name} RESULTS")
    print(f"  {'Model':<35}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>7}  {'MCC':>7}")
    print("  " + "-"*63)
    for name, (au, ap, f1, mcc) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"  {name:<35}  {au:>7.4f}  {ap:>7.4f}  {f1:>7.4f}  {mcc:>7.4f}")

rows = [{"model": n, "split": "val",  "auroc": r[0], "auprc": r[1], "f1": r[2], "mcc": r[3]}
        for n, r in results_val.items()] + \
       [{"model": n, "split": "test", "auroc": r[0], "auprc": r[1], "f1": r[2], "mcc": r[3]}
        for n, r in results_test.items()]
pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "phase5_results.csv", index=False)
print("\nSaved: outputs/phase5_results.csv")
print("Phase 5 complete.")
