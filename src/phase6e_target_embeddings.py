"""
Phase 6e: Target Protein ESM-2 Embeddings + Binder-Target Interaction Features
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license

Key insight: the model knew the BINDER sequence via ESM-2 (1280-dim) but only
knew the TARGET by a 1-hot label or a proxy prototype. True molecular recognition
requires knowing BOTH binder and target in the same embedding space.

New features per (binder, target) pair:
  1. tgt_cos_sim       — cosine similarity between binder and target ESM-2
  2. tgt_l2_dist       — L2 distance in ESM-2 space
  3. tgt_dot_prod      — unnormalized dot product
  4. tgt_norm          — L2 norm of target embedding (proxy for size/complexity)
  5. binder_norm       — L2 norm of binder embedding
  6-21. tgt_pca_01..16 — top 16 PCA components of (binder × target) element-wise
                          product (fitted on training pairs only)

LOTO improvement:
  Target ESM-2 embeddings are available even for UNSEEN targets → real interaction
  features instead of zeros → expect significant LOTO AUROC improvement.
"""

import numpy as np
import pandas as pd
import torch
import urllib.request
import json, pickle, warnings, contextlib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

RANDOM_SEED  = 42
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def mets(y_true, y_prob, thr=0.5):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    yp    = (y_prob >= thr).astype(int)
    return dict(auroc=auroc, auprc=auprc,
                f1=f1_score(y_true, yp, zero_division=0),
                mcc=matthews_corrcoef(y_true, yp))

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── 1. Target protein sequences (UniProt) ─────────────────────────────────────
# Manual mapping from Proteinbase slug → UniProt accession
# Verified against UniProt canonical human sequences + viral proteomes
UNIPROT_MAP = {
    "egfr":                    "P00533",   # Human EGFR
    "nipah-glycoprotein-g":    "Q9IH63",   # Nipah virus attachment glycoprotein G
    "pd-l1":                   "Q9NZQ7",   # Human CD274/PD-L1
    "mdm2":                    "Q00987",   # Human MDM2
    "il7r":                    "P16871",   # Human IL-7 receptor alpha chain
    "fcrn":                    "P55899",   # Human FcRn (FCGRT heavy chain)
    "human-insulin-receptor":  "P06213",   # Human insulin receptor
    "human-pdgfr-beta":        "P09619",   # Human PDGFR-beta
    "fgf-r1":                  "P11362",   # Human FGFR1
    "human-phyh":              "O14832",   # Human phytanoyl-CoA dioxygenase
    "human-pmvk":              "Q15126",   # Human phosphomevalonate kinase
    "human-rfk":               "Q969G2",   # Human riboflavin kinase
    "ifnar2":                  "P48551",   # Human IFNAR2
    "hnmt":                    "P50135",   # Human histamine N-methyltransferase
    "human-ambp":              "P02760",   # Human AMBP (alpha-1-microglobulin)
    "human-idi2":              "Q9GZU1",   # Human isopentenyl-diphosphate isomerase 2
    "spcas9":                  "Q99ZW2",   # SpCas9 (S. pyogenes)
    "human-serum-albumin":     "P02768",   # Human serum albumin
    "human-tnfa":              "P01375",   # Human TNF-alpha
    "human-orm2":              "P19652",   # Human orosomucoid-2
    "human-gm2a":              "P17900",   # Human GM2 ganglioside activator
    "human-mzb1-perp1":       "Q8WU39",   # Human MZB1 (primary subunit)
    # der21 and der7: custom Adaptyv Bio designed targets, not in UniProt
    # Will be handled with zero-vector fallback
}

print("\n" + "=" * 70)
print("Fetching target protein sequences from UniProt...")

def fetch_uniprot_seq(acc, timeout=15):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            lines = r.read().decode().strip().split("\n")
            return "".join(lines[1:])
    except Exception as e:
        print(f"    Failed {acc}: {e}")
        return None

target_sequences = {}
pairs = pd.read_parquet(DATA_DIR / "pairs_with_splits.parquet")
all_targets = sorted(pairs["target"].unique())

for tgt in all_targets:
    acc = UNIPROT_MAP.get(tgt)
    if acc:
        seq = fetch_uniprot_seq(acc)
        if seq:
            target_sequences[tgt] = seq
            print(f"  {tgt:<45} {acc}  len={len(seq)}")
        else:
            target_sequences[tgt] = None
            print(f"  {tgt:<45} {acc}  FETCH FAILED — zero-vector")
    else:
        target_sequences[tgt] = None
        print(f"  {tgt:<45} no UniProt ID — zero-vector fallback")

n_found = sum(1 for v in target_sequences.values() if v)
print(f"\n  Found sequences: {n_found}/{len(all_targets)}")

# ── 2. Embed target sequences with ESM-2 ─────────────────────────────────────
print("\n" + "=" * 70)
print("Embedding target sequences with ESM-2 (fair-esm, cached weights)...")

import esm as esm_lib

esm_model_loaded, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
esm_model_loaded = esm_model_loaded.to(device).eval()
print(f"  ESM-2 loaded from local cache: esm2_t33_650M_UR50D")

# Manual tokenize workaround for Python 3.13 compatibility (avoids segfault)
def manual_tokenize(seqs, max_len=1022):
    """Safe tokenizer for Python 3.13 — avoids fair-esm batch converter segfault."""
    prepend = alphabet.prepend_bos
    append  = alphabet.append_eos
    pad_idx = alphabet.padding_idx
    results = []
    for seq in seqs:
        seq = seq[:max_len]
        toks = [alphabet.get_idx(c) if alphabet.get_idx(c) is not None
                else alphabet.unk_idx for c in seq]
        if prepend: toks = [alphabet.cls_idx] + toks
        if append:  toks = toks + [alphabet.eos_idx]
        results.append(toks)
    max_l = max(len(t) for t in results)
    batch  = torch.ones(len(results), max_l, dtype=torch.long) * pad_idx
    for i, toks in enumerate(results):
        batch[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return batch

def embed_sequence_esm2(seq, max_len=1022):
    """Mean-pool ESM-2 representation for a single sequence."""
    tokens = manual_tokenize([seq], max_len=max_len).to(device)
    with torch.no_grad():
        out = esm_model_loaded(tokens, repr_layers=[33], return_contacts=False)
    reps = out["representations"][33]  # (1, L+2, 1280)
    # Exclude CLS and EOS tokens
    n_real = min(len(seq), max_len)
    emb = reps[0, 1:n_real+1, :].mean(0).cpu().float().numpy()
    return emb

target_emb = {}   # target_name → 1280-dim numpy array
for tgt in all_targets:
    seq = target_sequences.get(tgt)
    if seq:
        emb = embed_sequence_esm2(seq)
        target_emb[tgt] = emb
        print(f"  {tgt:<45} norm={np.linalg.norm(emb):.3f}")
    else:
        target_emb[tgt] = np.zeros(1280, dtype=np.float32)
        print(f"  {tgt:<45} ZERO VECTOR (no sequence)")

np.save(MODELS_DIR / "target_esm2_embeddings.npy",
        np.stack([target_emb[t] for t in all_targets]))
print(f"\n  Saved target ESM-2 embeddings: {len(all_targets)} × 1280")

# Free GPU memory — we'll use binder embeddings next
del esm_model_loaded
torch.cuda.empty_cache()

# ── 3. Load binder ESM-2 embeddings ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Loading binder ESM-2 embeddings + feature matrix...")

esm_emb  = np.load(FEATURES_DIR / "esm2_embeddings.npy")
esm_ids  = np.load(FEATURES_DIR / "esm2_protein_ids.npy", allow_pickle=True)
esm_map  = {pid: i for i, pid in enumerate(esm_ids)}

pairs_r  = pairs.reset_index(drop=True)
X_emb    = np.zeros((len(pairs_r), 1280), dtype=np.float32)
for i, pid in enumerate(pairs_r["protein_id"]):
    idx = esm_map.get(pid)
    if idx is not None:
        X_emb[i] = esm_emb[idx]
X_emb = np.nan_to_num(X_emb, nan=0.0)

y_bind   = pairs_r["binding_label"].values.astype(np.float32)
train_m  = (pairs_r["split"] == "train").values
val_m    = (pairs_r["split"] == "val").values
test_m   = (pairs_r["split"] == "test").values
print(f"  Train: {train_m.sum()}  Val: {val_m.sum()}  Test: {test_m.sum()}")

# ── 4. Compute binder-target interaction features ────────────────────────────
print("\nComputing binder-target interaction features...")

INTER_FEAT_COLS_SCALAR = [
    "tgt_cos_sim", "tgt_l2_dist", "tgt_dot_prod",
    "tgt_norm",    "binder_norm"
]
N_PCA = 16
INTER_FEAT_COLS_PCA = [f"tgt_pca_{i:02d}" for i in range(N_PCA)]
INTER_FEAT_COLS = INTER_FEAT_COLS_SCALAR + INTER_FEAT_COLS_PCA

def interaction_scalar(binder_emb, tgt_emb):
    cos  = cosine_sim(binder_emb, tgt_emb)
    l2   = float(np.linalg.norm(binder_emb - tgt_emb))
    dot  = float(np.dot(binder_emb, tgt_emb))
    tn   = float(np.linalg.norm(tgt_emb))
    bn   = float(np.linalg.norm(binder_emb))
    return [cos, l2, dot, tn, bn]

# Compute element-wise products for all pairs (for PCA)
print("  Computing element-wise products for PCA...")
elem_prods = np.zeros((len(pairs_r), 1280), dtype=np.float32)
scalars    = np.zeros((len(pairs_r), len(INTER_FEAT_COLS_SCALAR)), dtype=np.float32)

for i, row in pairs_r.iterrows():
    tgt_e = target_emb[row["target"]]
    bin_e = X_emb[i]
    scalars[i]    = interaction_scalar(bin_e, tgt_e)
    elem_prods[i] = bin_e * tgt_e   # element-wise product

# Fit PCA on TRAINING pairs only (no leakage)
print(f"  Fitting PCA (n={N_PCA}) on training element-wise products...")
pca = PCA(n_components=N_PCA, random_state=RANDOM_SEED)
pca.fit(elem_prods[train_m])
pca_feats = pca.transform(elem_prods).astype(np.float32)
print(f"  PCA explained variance ratio (cumulative): "
      f"{pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

inter_feats = np.concatenate([scalars, pca_feats], axis=1)
print(f"  Interaction features shape: {inter_feats.shape}  "
      f"({len(INTER_FEAT_COLS)} features)")
print(f"  Sample val tgt_cos_sim — mean={scalars[val_m, 0].mean():.4f}  "
      f"std={scalars[val_m, 0].std():.4f}")

# ── 5. Build augmented feature matrix ─────────────────────────────────────────
print("\nBuilding augmented feature matrix (Phase 6b features + interaction)...")

# Load Phase 6b meta
with open(MODELS_DIR / "ensemble_meta_6b.pkl", "rb") as f:
    meta_6b = pickle.load(f)
all_feat_cols_6b = meta_6b["all_feat_cols"]   # 509 features
proto_feat_cols  = meta_6b["proto_feat_cols"]
proto_pos        = meta_6b["proto_pos"]
proto_neg        = meta_6b["proto_neg"]
n_pos_dict       = meta_6b["n_pos"]
n_neg_dict       = meta_6b["n_neg"]

def _proto_feats(emb, target):
    pp, pn = proto_pos[target], proto_neg[target]
    disc   = pp - pn
    dn     = np.linalg.norm(disc)
    return [cosine_sim(emb, pp), cosine_sim(emb, pn),
            float(np.linalg.norm(emb - pp)),
            float(np.dot(emb, disc) / (dn + 1e-8)),
            cosine_sim(emb, pp) / (abs(cosine_sim(emb, pn)) + 1e-6),
            n_pos_dict[target], n_neg_dict[target]]

# Rebuild 6b feature matrix
feat_mat = pd.read_parquet(FEATURES_DIR / "feature_matrix.parquet")
fm = feat_mat.copy()
pairs_keys  = list(zip(pairs_r["protein_id"], pairs_r["target"]))
proto_arr   = np.array([_proto_feats(X_emb[i], pairs_r["target"].iloc[i])
                        for i in range(len(pairs_r))], dtype=np.float32)
key_to_proto = {k: proto_arr[i] for i, k in enumerate(pairs_keys)}
fm_keys     = list(zip(fm["protein_id"], fm["target"]))
proto_fm    = np.array([key_to_proto.get(k, np.zeros(7, dtype=np.float32))
                        for k in fm_keys], dtype=np.float32)
for j, col in enumerate(proto_feat_cols):
    fm[col] = proto_fm[:, j]

# Add Phase 6a interface features if available
import json as _json
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
                d = _json.loads(v) if isinstance(v, str) else v
                return [int(r["residue"]) for r in d if "residue" in r] if isinstance(d, list) else []
            except: return []

        ir_rows["positions"] = ir_rows["value"].apply(_parse_pos)
        ir_rows = ir_rows[ir_rows["positions"].apply(len) > 0]
        ir_map  = {(r.protein_id, r.target): r.positions for _, r in ir_rows.iterrows()}
        seq_map = dict(zip(pairs_r["protein_id"], pairs_r["sequence"]))

        KD_H={'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
        CHG={'K':1,'R':1,'H':0.1,'D':-1,'E':-1}
        ARO=set('FWY'); HBD=set('NQSTKRHWY'); HBA=set('NQDEST')
        VOL={'G':60,'A':89,'S':96,'P':112,'V':117,'T':116,'C':114,'I':166,'L':166,'N':114,'D':111,'Q':144,'K':168,'E':138,'M':162,'H':153,'F':190,'R':173,'Y':194,'W':228}
        AA20=list("ACDEFGHIKLMNPQRSTVWY")

        def _if_feat(sequence, positions):
            seq=sequence.upper(); n=len(seq)
            if not positions or n==0: return {}
            pos_0=[p-1 for p in positions if 0<=p-1<n]
            if not pos_0: return {}
            iface=[seq[p] for p in pos_0]; n_if=len(iface)
            f={}
            cnt={a:0 for a in AA20}
            for aa in iface:
                if aa in cnt: cnt[aa]+=1
            for aa in AA20: f[f"if_aac_{aa}"]=cnt[aa]/n_if
            f["if_n_residues"]=n_if; f["if_coverage"]=n_if/n
            if len(pos_0)>1:
                gaps=[pos_0[k+1]-pos_0[k] for k in range(len(pos_0)-1)]
                f["if_span"]=(max(pos_0)-min(pos_0))/n; f["if_mean_gap"]=np.mean(gaps)/n
                f["if_max_gap"]=max(gaps)/n; f["if_n_segments"]=sum(1 for g in gaps if g>3)
            else:
                f["if_span"]=f["if_mean_gap"]=f["if_max_gap"]=f["if_n_segments"]=0.0
            f["if_nterm_frac"]=sum(1 for p in pos_0 if p<n*0.33)/n_if
            f["if_cterm_frac"]=sum(1 for p in pos_0 if p>n*0.67)/n_if
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

        if_rows_list=[]
        for _,row in pairs_r.iterrows():
            key=(row["protein_id"],row["target"])
            pos=ir_map.get(key,[])
            seq=seq_map.get(row["protein_id"],"")
            if_rows_list.append(_if_feat(seq,pos) if pos and seq else {})
        if_df=pd.DataFrame([r if r else {} for r in if_rows_list])
        for col in if_cols:
            if col not in if_df.columns: if_df[col]=np.nan
        if_df=if_df[if_cols].fillna(if_meds)
        pair_to_row={(pairs_r["protein_id"].iloc[i],pairs_r["target"].iloc[i]):i for i in range(len(pairs_r))}
        for col in if_cols:
            fm[col]=[if_df[col].iloc[pair_to_row.get(k,0)] if k in pair_to_row else if_meds[col] for k in fm_keys]
        use_aug=True
        print(f"  Interface features added ({len(if_cols)} cols)")
    except Exception as e:
        print(f"  Interface features skipped ({e})")

# Add interaction features to fm
pair_to_inter = {k: inter_feats[i] for i, k in enumerate(pairs_keys)}
inter_fm = np.array([pair_to_inter.get(k, np.zeros(len(INTER_FEAT_COLS), dtype=np.float32))
                     for k in fm_keys], dtype=np.float32)
for j, col in enumerate(INTER_FEAT_COLS):
    fm[col] = inter_fm[:, j]

# Final feature cols: 6b features + interaction features
all_feat_cols = all_feat_cols_6b + INTER_FEAT_COLS
missing = [c for c in all_feat_cols if c not in fm.columns]
if missing:
    for c in missing: fm[c] = 0.0
    print(f"  WARNING: zeroed {len(missing)} missing cols")

print(f"  Total features: {len(all_feat_cols)}  "
      f"(was 509, added {len(INTER_FEAT_COLS)} interaction)")

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

# ── 6. Retrain LightGBM + XGBoost ────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training LightGBM+Proto+Interaction...")

lgb_6e = lgb.LGBMClassifier(
    objective="binary", metric="auc", verbosity=-1, device="gpu",
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
    random_state=RANDOM_SEED,
)
lgb_6e.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
           callbacks=[lgb.early_stopping(50, verbose=False),
                      lgb.log_evaluation(period=-1)])
lgb_vp = lgb_6e.predict_proba(X_v)[:, 1]
lgb_tp = lgb_6e.predict_proba(X_te)[:, 1]
m_lgb_v = mets(y_v, lgb_vp); m_lgb_t = mets(y_te, lgb_tp)
print(f"  LightGBM+Interaction Val  — AUROC={m_lgb_v['auroc']:.4f}  AUPRC={m_lgb_v['auprc']:.4f}  F1={m_lgb_v['f1']:.4f}  MCC={m_lgb_v['mcc']:.4f}")
print(f"  LightGBM+Interaction Test — AUROC={m_lgb_t['auroc']:.4f}  AUPRC={m_lgb_t['auprc']:.4f}  F1={m_lgb_t['f1']:.4f}  MCC={m_lgb_t['mcc']:.4f}")
joblib.dump(lgb_6e, MODELS_DIR / "lgb_interaction.pkl")

# Feature importance: where do interaction features rank?
fi = lgb_6e.feature_importances_
fi_df = pd.DataFrame({"feature": all_feat_cols, "importance": fi}).sort_values("importance", ascending=False).reset_index(drop=True)
print(f"\n  Interaction feature ranks (out of {len(all_feat_cols)}):")
for col in INTER_FEAT_COLS:
    rank = fi_df[fi_df["feature"] == col].index[0] + 1
    imp  = fi_df[fi_df["feature"] == col]["importance"].values[0]
    print(f"    #{rank:>3}  {col:<25}  importance={imp:.0f}")

print("\nTraining XGBoost+Proto+Interaction...")
xgb_6e = XGBClassifier(
    n_estimators=776, max_depth=8, learning_rate=0.0496,
    subsample=0.958, colsample_bytree=0.722, min_child_weight=2,
    reg_alpha=1.11, reg_lambda=0.002,
    device="cuda", eval_metric="auc", early_stopping_rounds=50,
    random_state=RANDOM_SEED, verbosity=0,
)
xgb_6e.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
xgb_vp = xgb_6e.predict_proba(X_v)[:, 1]
xgb_tp = xgb_6e.predict_proba(X_te)[:, 1]
m_xgb_v = mets(y_v, xgb_vp); m_xgb_t = mets(y_te, xgb_tp)
print(f"  XGBoost+Interaction Val  — AUROC={m_xgb_v['auroc']:.4f}  AUPRC={m_xgb_v['auprc']:.4f}  F1={m_xgb_v['f1']:.4f}  MCC={m_xgb_v['mcc']:.4f}")
print(f"  XGBoost+Interaction Test — AUROC={m_xgb_t['auroc']:.4f}  AUPRC={m_xgb_t['auprc']:.4f}  F1={m_xgb_t['f1']:.4f}  MCC={m_xgb_t['mcc']:.4f}")
joblib.dump(xgb_6e, MODELS_DIR / "xgb_interaction.pkl")

# ── 7. LOTO CV with real target embeddings ───────────────────────────────────
print("\n" + "=" * 70)
print("LOTO CV with real target ESM-2 embeddings (zero-shot improvement)...")

# Build full feature array aligned to pairs_r
all_feat_arr = np.array([pair_to_inter.get(k, np.zeros(len(INTER_FEAT_COLS), dtype=np.float32))
                          for k in pairs_keys], dtype=np.float32)

# Load base handcrafted feature matrix aligned to pairs_r
base_feat_cols = pd.read_csv(FEATURES_DIR / "feature_columns.csv")["column"].tolist()
fm_aligned = fm.set_index(["protein_id", "target"]).reindex(
    pd.MultiIndex.from_tuples(pairs_keys, names=["protein_id", "target"])
).reset_index()

# Full feature matrix for LOTO: Phase6b (509 features without proto) + interaction (21)
# We recompute proto from each train fold inside the LOTO loop
base_no_proto_cols = [c for c in all_feat_cols_6b if c not in proto_feat_cols]

target_all   = pairs_r["target"].values
targets_loto = sorted(set(t for t in all_targets
                          if (target_all == t).sum() >= 3
                          and y_bind[target_all == t].sum() >= 1
                          and y_bind[target_all == t].sum() < (target_all == t).sum()))

print(f"  Evaluable targets: {len(targets_loto)}")

LGB_LOTO_PARAMS = dict(
    objective="binary", metric="auc", verbosity=-1, device="gpu",
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=5,
    random_state=RANDOM_SEED,
)

loto_rows = []
print(f"\n  {'Target':<45}  {'N':>5}  {'Pos':>3}  {'AUROC_with_tgt':>14}  {'AUROC_no_tgt':>12}")
print("  " + "-"*88)

for tgt in targets_loto:
    test_mask  = target_all == tgt
    train_mask = ~test_mask
    n_te = test_mask.sum()
    n_pos = int(y_bind[test_mask].sum())
    if n_pos == 0 or n_pos == n_te:
        print(f"  {tgt:<45}  {n_te:>5}  {n_pos:>3}  N/A (no variability)")
        continue

    # ── Recompute prototype from this fold's train set ──
    fold_proto_pos, fold_proto_neg = {}, {}
    fold_n_pos, fold_n_neg = {}, {}
    for t2 in all_targets:
        t2_mask = (target_all == t2) & train_mask
        y_t2    = y_bind[t2_mask]
        e_t2    = X_emb[t2_mask]
        pm, nm  = y_t2 == 1, y_t2 == 0
        fold_n_pos[t2] = int(pm.sum())
        fold_n_neg[t2] = int(nm.sum())
        fold_proto_pos[t2] = e_t2[pm].mean(0) if pm.sum() > 0 else np.zeros(1280, np.float32)
        fold_proto_neg[t2] = e_t2[nm].mean(0) if nm.sum() > 0 else np.zeros(1280, np.float32)
    # Held-out target: zero prototype (cold start for prototype)
    fold_proto_pos[tgt] = np.zeros(1280, np.float32)
    fold_proto_neg[tgt] = np.zeros(1280, np.float32)

    # ── Recompute interaction features with/without target embedding ──
    def _fold_inter(i, use_real_tgt=True):
        b = X_emb[i]
        t = target_emb[pairs_r["target"].iloc[i]] if use_real_tgt else np.zeros(1280, np.float32)
        sc  = interaction_scalar(b, t)
        ep  = (b * t).reshape(1, -1)
        pc  = pca.transform(ep)[0].tolist()
        return np.array(sc + pc, dtype=np.float32)

    # Build feature arrays for this fold
    def build_fold_feats(mask, use_real_tgt=True):
        rows = []
        for i in np.where(mask)[0]:
            pp   = fold_proto_pos[pairs_r["target"].iloc[i]]
            pn   = fold_proto_neg[pairs_r["target"].iloc[i]]
            disc = pp - pn
            dn   = np.linalg.norm(disc)
            b    = X_emb[i]
            pf   = [cosine_sim(b,pp), cosine_sim(b,pn),
                    float(np.linalg.norm(b-pp)),
                    float(np.dot(b,disc)/(dn+1e-8)),
                    cosine_sim(b,pp)/(abs(cosine_sim(b,pn))+1e-6),
                    fold_n_pos[pairs_r["target"].iloc[i]],
                    fold_n_neg[pairs_r["target"].iloc[i]]]
            inter = _fold_inter(i, use_real_tgt)
            # base hc features (no proto, no inter)
            fm_row = fm.iloc[i][base_no_proto_cols].values.astype(np.float32)
            rows.append(np.concatenate([fm_row, pf, inter]))
        return np.array(rows, dtype=np.float32)

    X_tr_wp = build_fold_feats(train_mask, use_real_tgt=True)
    X_te_wp = build_fold_feats(test_mask,  use_real_tgt=True)
    X_tr_np = build_fold_feats(train_mask, use_real_tgt=False)
    X_te_np = build_fold_feats(test_mask,  use_real_tgt=False)
    y_train  = y_bind[train_mask]
    y_test   = y_bind[test_mask]

    # Eval set for early stopping: use 10% of train as pseudo-val
    n_trn = len(y_train)
    val_n = max(5, n_trn // 10)
    rng   = np.random.RandomState(RANDOM_SEED)
    val_idx = rng.choice(n_trn, val_n, replace=False)
    tr_idx  = np.setdiff1d(np.arange(n_trn), val_idx)

    def fit_lgb(Xtr, Xte, ytr, yte, val_i, tr_i):
        m = lgb.LGBMClassifier(**LGB_LOTO_PARAMS)
        m.fit(Xtr[tr_i], ytr[tr_i],
              eval_set=[(Xtr[val_i], ytr[val_i])],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])
        if yte.sum() == 0 or yte.sum() == len(yte): return np.nan
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    auroc_wp = fit_lgb(X_tr_wp, X_te_wp, y_train, y_test, val_idx, tr_idx)
    auroc_np = fit_lgb(X_tr_np, X_te_np, y_train, y_test, val_idx, tr_idx)

    wp_str = f"{auroc_wp:.4f}" if not np.isnan(auroc_wp) else "N/A"
    np_str = f"{auroc_np:.4f}" if not np.isnan(auroc_np) else "N/A"
    print(f"  {tgt:<45}  {n_te:>5}  {n_pos:>3}  {wp_str:>14}  {np_str:>12}")
    loto_rows.append({"target": tgt, "n": n_te, "n_pos": n_pos,
                      "auroc_with_tgt": auroc_wp, "auroc_no_tgt": auroc_np})

loto_df = pd.DataFrame(loto_rows)
valid   = loto_df.dropna(subset=["auroc_with_tgt", "auroc_no_tgt"])
print("\n" + "=" * 70)
print("LOTO SUMMARY")
print(f"  Phase 6c (no real target, proto=zeros)  mean AUROC: 0.6584")
print(f"  Phase 6e WITH real target ESM-2         mean AUROC: {valid['auroc_with_tgt'].mean():.4f}")
print(f"  Phase 6e WITHOUT target ESM-2 (base)    mean AUROC: {valid['auroc_no_tgt'].mean():.4f}")
print(f"  Delta (6e vs 6c): {valid['auroc_with_tgt'].mean() - 0.6584:+.4f}")

# ── 8. Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 6e — FINAL SUMMARY (TEST SET)")
print(f"  {'Model':<55}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>7}  {'MCC':>7}")
print("  " + "-" * 85)
print(f"  {'LightGBM+Proto (Phase 6b best)':<55}  {0.9402:>7.4f}  {0.7645:>7.4f}  {0.7481:>7.4f}  {0.6979:>7.4f}")
print(f"  {'LightGBM+Proto+TargetInteraction (Phase 6e)':<55}  {m_lgb_t['auroc']:>7.4f}  {m_lgb_t['auprc']:>7.4f}  {m_lgb_t['f1']:>7.4f}  {m_lgb_t['mcc']:>7.4f}")
print(f"  {'XGBoost+Proto+TargetInteraction (Phase 6e)':<55}  {m_xgb_t['auroc']:>7.4f}  {m_xgb_t['auprc']:>7.4f}  {m_xgb_t['f1']:>7.4f}  {m_xgb_t['mcc']:>7.4f}")

# Save
rows = [
    {"model": "LightGBM_Phase6b", "auroc": 0.9402, "auprc": 0.7645, "f1": 0.7481, "mcc": 0.6979},
    {"model": "LightGBM+Interaction", "auroc": m_lgb_t["auroc"], "auprc": m_lgb_t["auprc"],
     "f1": m_lgb_t["f1"], "mcc": m_lgb_t["mcc"]},
    {"model": "XGBoost+Interaction",  "auroc": m_xgb_t["auroc"], "auprc": m_xgb_t["auprc"],
     "f1": m_xgb_t["f1"], "mcc": m_xgb_t["mcc"]},
]
pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "phase6e_results.csv", index=False)
loto_df.to_csv(OUTPUTS_DIR / "phase6e_loto_results.csv", index=False)

# Save meta for Phase 7 inference
meta_6e = {
    "all_feat_cols": all_feat_cols,
    "inter_feat_cols": INTER_FEAT_COLS,
    "proto_feat_cols": proto_feat_cols,
    "target_emb": target_emb,
    "pca": pca,
    "n_pos": n_pos_dict,
    "n_neg": n_neg_dict,
    "proto_pos": proto_pos,
    "proto_neg": proto_neg,
    "uniprot_map": UNIPROT_MAP,
    "target_sequences": {k: v for k, v in target_sequences.items() if v},
}
with open(MODELS_DIR / "ensemble_meta_6e.pkl", "wb") as f:
    pickle.dump(meta_6e, f)

print(f"\nSaved: outputs/phase6e_results.csv, phase6e_loto_results.csv")
print(f"Saved: models/lgb_interaction.pkl, xgb_interaction.pkl")
print(f"Saved: models/target_esm2_embeddings.npy, ensemble_meta_6e.pkl")
print("Phase 6e complete.")
