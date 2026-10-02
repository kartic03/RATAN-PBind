#!/usr/bin/env python3
"""Noise floor for the feature-block ablations.

r12 and r15 disagree by 0.006 on the identical 458-feature configuration. The only
difference between them is the ORDER of the feature columns, which interacts with
LightGBM's colsample_bytree / feature_fraction_seed. Before any ablation delta is
interpreted, the size of that nuisance variation has to be known.

Three sources are measured on the FULL 470-feature model, random split:
    seed      vary random_state / bagging_seed / feature_fraction_seed
    order     permute the column order, seed fixed
    both
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y = fm.binding_label.values.astype(int)
tgt = fm.target.values
split = fm.split.values
tr, te = (split == "train") | (split == "val"), split == "test"
EPS = 1e-8


def proto_feats(trm):
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t
        pos, neg = E[m & trm & (y == 1)], E[m & trm & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1])
        pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn
        dn = np.linalg.norm(diff) + EPS
        ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + EPS)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + EPS)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn,
                           cp / (cn + EPS), np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out


X0 = np.hstack([fm[base_cols].values.astype(np.float32), proto_feats(tr)])


def run(seed, perm):
    p = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
             subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
             is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
             force_row_wise=True, random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)
    X = X0[:, perm] if perm is not None else X0
    m = lgb.LGBMClassifier(**p).fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])


base = run(42, None)
print("full 470, canonical (seed 42, original order): %.4f\n" % base)

seeds = [run(s, None) for s in [42, 1, 7, 2024, 99]]
print("vary SEED only        : %s" % " ".join("%.4f" % v for v in seeds))
print("   mean %.4f  SD %.4f  range %.4f" % (np.mean(seeds), np.std(seeds), max(seeds) - min(seeds)))

rng = np.random.RandomState(0)
orders = [run(42, rng.permutation(X0.shape[1])) for _ in range(5)]
print("\nvary COLUMN ORDER only: %s" % " ".join("%.4f" % v for v in orders))
print("   mean %.4f  SD %.4f  range %.4f" % (np.mean(orders), np.std(orders), max(orders) - min(orders)))

both = [run(s, rng.permutation(X0.shape[1])) for s in [1, 7, 2024, 99, 5]]
print("\nvary BOTH             : %s" % " ".join("%.4f" % v for v in both))
print("   mean %.4f  SD %.4f  range %.4f" % (np.mean(both), np.std(both), max(both) - min(both)))

allv = seeds + orders + both
print("\nNOISE FLOOR across all %d nuisance runs: SD %.4f, range %.4f"
      % (len(allv), np.std(allv), max(allv) - min(allv)))
print("Any ablation delta smaller than about %.3f AUROC is not interpretable."
      % (2 * np.std(allv)))
pd.DataFrame({"kind": ["seed"] * len(seeds) + ["order"] * len(orders) + ["both"] * len(both),
              "auroc": allv}).to_csv(os.path.join(ROOT, "outputs/r15b_noise_floor.csv"), index=False)
print("saved outputs/r15b_noise_floor.csv")
