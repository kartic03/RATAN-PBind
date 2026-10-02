#!/usr/bin/env python3
"""Noise floor for the LEAVE-AUTHOR-OUT feature-block deltas (r15_feature_block_bias.py).

r15b measured run-to-run variation (seed, column order) on the random split only
(SD 0.0017, range 0.0062). The leave-author-out deltas in r15 have not had their own
floor, and grouped folds are smaller and noisier. This measures it on the full model:
3 seeds x 3 column permutations (9 runs), each a full 5-fold GroupKFold by author with
prototypes recomputed per training fold.

    pixi run -e ml python src/r15c_loao_noise_floor.py
"""
import os, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-8
fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
pw = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))[["protein_id", "target", "author", "design_method"]]
fm = fm.merge(pw, on=["protein_id", "target"], how="left")
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y = fm.binding_label.values.astype(int); tgt = fm.target.values
g = fm.author.fillna("NA").astype(str).values
Xb = fm[base_cols].values.astype(np.float32)

def proto(trm):
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t
        pos, neg = E[m & trm & (y == 1)], E[m & trm & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1]); pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn; dn = np.linalg.norm(diff) + EPS; ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + EPS)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + EPS)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn, cp / (cn + EPS),
                           np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out

folds = list(GroupKFold(5).split(np.zeros((len(y), 1)), y, g))
PK = {k: proto(np.isin(np.arange(len(y)), a)) for k, (a, b) in enumerate(folds)}

def run(seed, perm):
    prm = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
               colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0, is_unbalance=True, verbosity=-1,
               n_jobs=1, num_threads=1, deterministic=True, force_row_wise=True,
               random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)
    aucs = []
    for k, (a, b) in enumerate(folds):
        if len(np.unique(y[b])) < 2: continue
        X = np.hstack([Xb, PK[k]])
        if perm is not None: X = X[:, perm]
        m = lgb.LGBMClassifier(**prm).fit(X[a], y[a])
        aucs.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
    return float(np.mean(aucs))

rng = np.random.RandomState(0)
perms = [None] + [rng.permutation(Xb.shape[1] + 7) for _ in range(2)]
vals = [run(s, p) for s in (42, 7, 2024) for p in perms]
v = np.array(vals)
print("leave-author-out, full model, 9 nuisance runs (3 seeds x 3 column orders):")
print("  values:", " ".join("%.4f" % x for x in vals))
print("  mean %.4f   SD %.4f   range %.4f" % (v.mean(), v.std(), v.max() - v.min()))
print("  -> leave-author-out deltas below about %.3f AUROC are not interpretable" % (2 * v.std()))
pd.DataFrame({"auroc": vals}).to_csv(os.path.join(ROOT, "outputs/r15c_loao_noise_floor.csv"), index=False)
