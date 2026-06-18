#!/usr/bin/env python3
"""R9b - regenerate per-target reliability (Table S1) and shared-vs-per-target
(Table S6) from the DETERMINISTIC 470-feature headline model, so the committed
artifacts reproduce exactly (the older r3 versions were multithreaded and drifted).
CPU, deterministic. Overwrites outputs/r3_per_target_ci.csv and r3_single_vs_shared.csv."""
import os, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.RandomState(42)
ZERO = {"human-serum-albumin", "human-tnfa", "human-orm2", "human-gm2a"}
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
           colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0, is_unbalance=True,
           verbosity=-1, n_jobs=1, num_threads=1, deterministic=True, force_row_wise=True,
           random_state=42, bagging_seed=42, feature_fraction_seed=42)
fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)
y = fm["binding_label"].values.astype(int); tgt = fm["target"].values; split = fm["split"].values
tr = (split == "train") | (split == "val"); te = split == "test"; eps = 1e-8

def proto_feats(E, tgt, y, trmask):
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t; pos = E[m & trmask & (y == 1)]; neg = E[m & trmask & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1]); pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn; dn = np.linalg.norm(diff) + eps; ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + eps)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + eps)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn,
                           cp / (cn + eps), np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out

P = proto_feats(E, tgt, y, tr); X = np.hstack([fm[base_cols].values.astype(np.float32), P])
clf = lgb.LGBMClassifier(**LGB); clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]
yte = y[te]; Xte = X[te]; tgt_te = tgt[te]

def boot_ci(yy, pp, n=500):
    vals = []; idx = np.arange(len(yy))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yy[b])) < 2: continue
        try: vals.append(roc_auc_score(yy[b], pp[b]))
        except Exception: pass
    return (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))

rows = []
for t in sorted(set(tgt) - ZERO):
    mm = tgt_te == t
    if mm.sum() < 2 or len(np.unique(yte[mm])) < 2: continue
    au = roc_auc_score(yte[mm], p[mm]); lo, hi = boot_ci(yte[mm], p[mm])
    rel = "OK" if mm.sum() >= 20 else ("low-n" if mm.sum() >= 8 else "UNRELIABLE")
    rows.append((t, int(mm.sum()), int(yte[mm].sum()), round(au, 3), round(lo, 3), round(hi, 3), rel))
rdf = pd.DataFrame(rows, columns=["target", "n", "binders", "auroc", "ci_lo", "ci_hi", "reliability"]).sort_values("n", ascending=False)
rdf.to_csv(os.path.join(ROOT, "outputs/r3_per_target_ci.csv"), index=False)
print("PER-TARGET (deterministic):"); print(rdf.to_string(index=False))

cmp = []
for t in sorted(set(tgt) - ZERO):
    mm = tgt_te == t
    if mm.sum() < 15 or len(np.unique(yte[mm])) < 2: continue
    sh = roc_auc_score(yte[mm], p[mm]); mt = tgt == t
    if (tr & mt & (y == 1)).sum() < 3: continue
    ms = lgb.LGBMClassifier(**LGB); ms.fit(X[tr & mt], y[tr & mt]); ps = ms.predict_proba(Xte[mm])[:, 1]
    si = roc_auc_score(yte[mm], ps); cmp.append((t, int(mm.sum()), round(sh, 3), round(si, 3), round(sh - si, 3)))
cdf = pd.DataFrame(cmp, columns=["target", "n_te", "shared_auroc", "single_auroc", "delta"])
cdf.to_csv(os.path.join(ROOT, "outputs/r3_single_vs_shared.csv"), index=False)
print("\nSHARED vs PER-TARGET (deterministic):"); print(cdf.to_string(index=False))
print(f"\nmean delta(shared-single) = {cdf.delta.mean():+.3f}")
print("\noverwrote outputs/r3_per_target_ci.csv, outputs/r3_single_vs_shared.csv")
