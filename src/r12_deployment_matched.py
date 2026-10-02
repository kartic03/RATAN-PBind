#!/usr/bin/env python3
"""Deployment-matched ablation.

The headline 470-feature model mixes inputs that differ sharply in what they cost
to obtain at inference time. This script separates them and reports a ladder.

  TIER A  sequence + prototype (437)
          amino-acid composition (20), dipeptide composition (400),
          physicochemical (7), seq_length / molecular_weight / isoelectric_point (3),
          prototype-similarity (7, from the candidate's ESM-2 embedding).
          Needs: the candidate sequence, one ESM-2 forward pass, and target prototypes
          already in the database. Sub-second on GPU. THIS IS THE DEPLOYMENT MODEL.

  TIER B  + design-method metadata (17) = 454
          Needs: which method generated the candidate, plus that method's historical
          success rate from the training corpus. Free in a real campaign, but it is
          campaign metadata, not sequence.

  TIER C  + ESMFold pLDDT and ProteinMPNN scores (4) = 458
          Needs: folding the binder. Seconds to minutes per candidate, not sub-second.
          This tier is the existing "Without Boltz2 structural features" row.

  TIER D  + Boltz-2 interface metrics (12) = 470  -- the published headline.
          Needs: folding the binder-target COMPLEX. Minutes to hours per candidate.
          Available for only 40.8% of the training pairs; median-imputed otherwise.

Training protocol matches src/r9_release_artifacts.py exactly: LightGBM with the
released hyperparameters, train+val for fitting and for prototype construction,
evaluation on the held-out test split, F1/MCC at a FIXED 0.5 threshold.

    pixi run -e ml python src/r12_deployment_matched.py
"""
import os, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, N_BOOT, THRESH = 42, 1000, 0.5
rng = np.random.RandomState(SEED)

LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
base_cols = list(cols.column)
grp = dict(zip(cols.column, cols.group))

emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y = fm.binding_label.values.astype(int)
tgt = fm.target.values
split = fm.split.values
tr, te = (split == "train") | (split == "val"), split == "test"


def proto_feats(E, tgt, y, trmask):
    eps = 1e-8
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t
        pos, neg = E[m & trmask & (y == 1)], E[m & trmask & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1])
        pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn
        dn = np.linalg.norm(diff) + eps
        ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + eps)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + eps)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn,
                           cp / (cn + eps), np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out


P = proto_feats(E, tgt, y, tr)

SEQ_COMP = [c for c in base_cols if grp[c] in ("aa_composition", "dipeptide", "physicochemical")]
SEQ_SCALAR = ["seq_length", "molecular_weight", "isoelectric_point"]
FOLD_BINDER = ["esmfold_plddt", "proteinmpnn_score", "proteinmpnn_seq_recovery",
               "redesigned_proteinmpnn_score"]
METHOD = [c for c in base_cols if grp[c] == "design_method"]
BOLTZ = [c for c in base_cols if grp[c] == "boltz2"]

TIERS = [
    ("A  sequence + prototype",        SEQ_COMP + SEQ_SCALAR, True,  "sequence only; ESM-2 pass", "sub-second (GPU)"),
    ("B  + design-method metadata",    SEQ_COMP + SEQ_SCALAR + METHOD, True, "+ campaign metadata", "sub-second (GPU)"),
    ("C  + ESMFold / ProteinMPNN",     SEQ_COMP + SEQ_SCALAR + METHOD + FOLD_BINDER, True, "+ fold the binder", "seconds-minutes"),
    ("D  + Boltz-2 (published 470)",   base_cols, True, "+ fold the complex", "minutes-hours"),
]


def boot_ci(yy, pp, fn):
    vals, idx = [], np.arange(len(yy))
    r = np.random.RandomState(SEED)
    for _ in range(N_BOOT):
        b = r.choice(idx, len(idx), replace=True)
        if len(np.unique(yy[b])) > 1:
            vals.append(fn(yy[b], pp[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


yte = y[te]
base_rate = yte.mean()
print("test n=%d  binders=%d  base rate=%.3f  threshold=%.1f (fixed)\n" % (len(yte), yte.sum(), base_rate, THRESH))
rows = []
for name, feats, use_proto, needs, cost in TIERS:
    # Keep the CANONICAL column order (base_cols) for every tier. Order interacts
    # with LightGBM column subsampling (run-to-run range ~0.006 AUROC), and a
    # tier built in a different order disagrees with Table S3 for the same feature set.
    feats = [c for c in base_cols if c in set(feats)]
    X = fm[feats].values.astype(np.float32)
    if use_proto:
        X = np.hstack([X, P])
    clf = lgb.LGBMClassifier(**LGB).fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    au, ap = roc_auc_score(yte, p), average_precision_score(yte, p)
    lo, hi = boot_ci(yte, p, roc_auc_score)
    order = np.argsort(-p)
    k10 = max(1, int(0.1 * len(yte)))
    enrich = yte[order[:k10]].mean() / base_rate
    pred = (p >= THRESH).astype(int)
    rows.append({"tier": name, "n_features": X.shape[1], "needs": needs, "cost": cost,
                 "auroc": round(au, 3), "auroc_ci": [round(lo, 3), round(hi, 3)],
                 "auprc": round(ap, 3), "top10_precision": round(float(yte[order[:k10]].mean()), 3),
                 "top10_enrichment": round(float(enrich), 2),
                 "f1": round(f1_score(yte, pred), 3), "mcc": round(matthews_corrcoef(yte, pred), 3)})
    print("%-32s %3d feats  AUROC %.3f [%.3f-%.3f]  AUPRC %.3f  top10%% %.3f (%.2fx)  F1 %.3f  MCC %.3f"
          % (name, X.shape[1], au, lo, hi, ap, rows[-1]["top10_precision"], enrich, rows[-1]["f1"], rows[-1]["mcc"]))

d = pd.DataFrame(rows)
print("\ncost of the structural tiers, relative to the deployment model (tier A):")
a = d.iloc[0]
for r in rows[1:]:
    print("   %-32s dAUROC %+.3f   dAUPRC %+.3f" % (r["tier"], r["auroc"] - a.auroc, r["auprc"] - a.auprc))

print("\nreference: manuscript reports 470 = 0.946, and 458 'without Boltz2' = 0.931")
json.dump(rows, open(os.path.join(ROOT, "outputs/r12_deployment_matched.json"), "w"), indent=2)
d.to_csv(os.path.join(ROOT, "outputs/r12_deployment_matched.csv"), index=False)
print("saved outputs/r12_deployment_matched.{json,csv}")
