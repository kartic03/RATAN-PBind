#!/usr/bin/env python3
"""Corrections to r17: three rows there measured the wrong thing.

  1. "Sequence only (427)" wrongly kept the 7 prototype features (434 columns).
  2. ESM-2 nearest-centroid used ONE global centroid pair; the baseline is
     per-target, matching how prototypes are built everywhere else.
  3. The expression controls never ran because `expressed` is not a column of
     pairs_with_splits; it lives in evaluations_flat.

Also re-checks Table S2 under alternative readings, since the published SD
(0.005) is eight times the SD that varying the seed actually produces.
"""
import os, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, EPS = 42, 1e-8
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
base_cols = list(cols.column); grp = dict(zip(cols.column, cols.group))
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float64)
y = fm.binding_label.values.astype(int); tgt = fm.target.values; split = fm.split.values
tr, te = (split == "train") | (split == "val"), split == "test"
yte = y[te]
SEQ = [c for c in base_cols if grp[c] in ("aa_composition", "dipeptide", "physicochemical")]

print("=" * 72)
print("FIX 1 -- sequence-only, NO prototype features (published: AUROC 0.821, AUPRC 0.581)")
X = fm[SEQ].values.astype(np.float32)
print("   columns: %d" % X.shape[1])
m = lgb.LGBMClassifier(**LGB).fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
print("   recomputed AUROC %.4f   AUPRC %.4f" % (roc_auc_score(yte, p), average_precision_score(yte, p)))

print("\nFIX 2 -- ESM-2 nearest-centroid (published 0.750)")
def nc_scores(per_target):
    s = np.zeros(len(E))
    if per_target:
        for t in np.unique(tgt):
            mt = tgt == t
            pos, neg = E[mt & tr & (y == 1)], E[mt & tr & (y == 0)]
            if not len(pos) or not len(neg):
                continue
            pp, pn = pos.mean(0), neg.mean(0)
            ei = E[mt]; nei = np.linalg.norm(ei, axis=1)
            s[mt] = ((ei @ pp) / (nei * np.linalg.norm(pp) + EPS)
                     - (ei @ pn) / (nei * np.linalg.norm(pn) + EPS))
    else:
        pp, pn = E[tr & (y == 1)].mean(0), E[tr & (y == 0)].mean(0)
        ei = E; nei = np.linalg.norm(ei, axis=1)
        s = ((ei @ pp) / (nei * np.linalg.norm(pp) + EPS)
             - (ei @ pn) / (nei * np.linalg.norm(pn) + EPS))
    return s
for lbl, pt in [("global centroid pair", False), ("per-target centroids", True)]:
    print("   %-24s AUROC %.4f" % (lbl, roc_auc_score(yte, nc_scores(pt)[te])))
# also: L2 nearest-centroid rather than cosine
pp, pn = E[tr & (y == 1)].mean(0), E[tr & (y == 0)].mean(0)
l2 = np.linalg.norm(E - pn, axis=1) - np.linalg.norm(E - pp, axis=1)
print("   %-24s AUROC %.4f" % ("global, L2 distance", roc_auc_score(yte, l2[te])))

print("\nFIX 3 -- expression controls (published: expressed-only 0.945, predict expression 0.778)")
ev = pd.read_parquet(os.path.join(ROOT, "data/evaluations_flat.parquet"))
ex = ev[ev.metric == "expressed"][["protein_id", "target", "value"]].copy()
def tobool(v):
    s = str(v).strip().lower()
    return 1 if s in ("true", "1", "yes") else (0 if s in ("false", "0", "no") else np.nan)
ex["e"] = ex.value.map(tobool)
key = pd.MultiIndex.from_arrays([fm.protein_id, fm.target])
exd = ex.dropna(subset=["e"])
print("   'expressed' rows: %d   distinct targets on those rows: %s"
      % (len(exd), exd.target.dropna().unique()[:4].tolist()))
# expression is a property of the PROTEIN, not the protein-target pair
exmap = exd.groupby("protein_id").e.max()
evec = fm.protein_id.map(exmap).values.astype(float)
print("   expression labels matched: %d of %d  (expressed %d)"
      % (np.sum(~pd.isna(evec)), len(evec), np.nansum(evec)))

P = np.zeros((len(E), 7))
for t in np.unique(tgt):
    mt = tgt == t
    pos, neg = E[mt & tr & (y == 1)], E[mt & tr & (y == 0)]
    pp_ = pos.mean(0) if len(pos) else np.zeros(E.shape[1])
    pn_ = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
    dd = pp_ - pn_; dn = np.linalg.norm(dd) + EPS
    ei = E[mt]; nei = np.linalg.norm(ei, axis=1)
    cp = (ei @ pp_) / (nei * np.linalg.norm(pp_) + EPS)
    cn = (ei @ pn_) / (nei * np.linalg.norm(pn_) + EPS)
    P[mt] = np.stack([cp, cn, np.linalg.norm(ei - pp_, axis=1), (ei @ dd) / dn,
                      cp / (cn + EPS), np.full(mt.sum(), len(pos)), np.full(mt.sum(), len(neg))], 1)
X470 = np.hstack([fm[base_cols].values.astype(np.float32), P]).astype(np.float32)
mb = lgb.LGBMClassifier(**LGB).fit(X470[tr], y[tr])
pb = mb.predict_proba(X470)[:, 1]

ok = ~pd.isna(evec)
sub = te & ok & (evec == 1)
if sub.sum() > 20:
    print("   expressed-only subset n=%d  AUROC %.4f" % (sub.sum(), roc_auc_score(y[sub], pb[sub])))
trm2, tem2 = tr & ok, te & ok
if len(np.unique(evec[tem2])) > 1:
    me = lgb.LGBMClassifier(**LGB).fit(X470[trm2], evec[trm2].astype(int))
    print("   predicting EXPRESSION  n=%d  AUROC %.4f"
          % (tem2.sum(), roc_auc_score(evec[tem2].astype(int), me.predict_proba(X470[tem2])[:, 1])))

print("\nTable S2 -- alternative readings (published AUROC 0.940 +/- 0.005)")
for lbl, Xv in [("470 full", X470), ("463 base, no proto", fm[base_cols].values.astype(np.float32))]:
    vals = []
    for s in [42, 1, 7, 2024, 99]:
        pr = dict(LGB); pr.update(random_state=s, bagging_seed=s, feature_fraction_seed=s)
        q = lgb.LGBMClassifier(**pr).fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
        vals.append(roc_auc_score(yte, q))
    print("   %-20s mean %.4f  sd %.4f" % (lbl, np.mean(vals), np.std(vals)))
