#!/usr/bin/env python3
"""Per-feature-block contribution, in-distribution vs across campaigns (Reviewer 3, comment 1).

R3 asks for a rationale for the feature groups and warns that design-method metadata,
historical success rates, target prototypes and structural scores "could also introduce
campaign- or target-specific biases if not handled carefully".

That is an empirical question, and the dataset can answer it. For each block, drop it
and measure the loss twice:

    random split       how much the block contributes in-distribution
    leave-author-out   how much survives when the evaluation campaign is unseen

A block that contributes in-distribution but not across authors is, by construction,
encoding campaign-specific structure. The ratio of the two deltas is the diagnostic.

    pixi run -e ml python src/r15_feature_block_bias.py
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
base_cols = list(cols.column)
grp = dict(zip(cols.column, cols.group))
pw = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))[
    ["protein_id", "target", "author", "design_method"]]
fm = fm.merge(pw, on=["protein_id", "target"], how="left")

emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y = fm.binding_label.values.astype(int)
tgt = fm.target.values
split = fm.split.values
tr, te = (split == "train") | (split == "val"), split == "test"


def proto_feats(E, tgt, y, trm):
    eps = 1e-8
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t
        pos, neg = E[m & trm & (y == 1)], E[m & trm & (y == 0)]
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


BLOCKS = {
    "prototype (7)":            ("proto", None),
    "design-method (17)":       ("cols", [c for c in base_cols if grp[c] == "design_method"]),
    "Boltz-2 (12)":             ("cols", [c for c in base_cols if grp[c] == "boltz2"]),
    "ESMFold/ProteinMPNN (4)":  ("cols", ["esmfold_plddt", "proteinmpnn_score",
                                          "proteinmpnn_seq_recovery", "redesigned_proteinmpnn_score"]),
    "dipeptide (400)":          ("cols", [c for c in base_cols if grp[c] == "dipeptide"]),
    "aa-composition (20)":      ("cols", [c for c in base_cols if grp[c] == "aa_composition"]),
    "physicochemical (7)":      ("cols", [c for c in base_cols if grp[c] == "physicochemical"]),
}


def build(drop_cols, drop_proto, trmask):
    keep = [c for c in base_cols if c not in set(drop_cols or [])]
    X = fm[keep].values.astype(np.float32)
    if not drop_proto:
        X = np.hstack([X, proto_feats(E, tgt, y, trmask)])
    return X


def eval_random(drop_cols, drop_proto):
    X = build(drop_cols, drop_proto, tr)
    m = lgb.LGBMClassifier(**LGB).fit(X[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])


def eval_loao(drop_cols, drop_proto):
    g = fm.author.fillna("NA").astype(str).values
    aucs = []
    for a, b in GroupKFold(5).split(np.zeros((len(y), 1)), y, g):
        if len(np.unique(y[b])) < 2:
            continue
        trmask = np.isin(np.arange(len(y)), a)
        X = build(drop_cols, drop_proto, trmask)
        m = lgb.LGBMClassifier(**LGB).fit(X[a], y[a])
        aucs.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
    return float(np.mean(aucs))


full_rand = eval_random([], False)
full_loao = eval_loao([], False)
print("FULL MODEL   random split %.3f   leave-author-out %.3f\n" % (full_rand, full_loao))
print("%-26s %9s %9s   %9s %9s   %s" % ("block removed", "rand", "d_rand", "LOAO", "d_LOAO", "survives?"))

rows = []
for name, (kind, c) in BLOCKS.items():
    dp = kind == "proto"
    dc = c if kind == "cols" else []
    r, l = eval_random(dc, dp), eval_loao(dc, dp)
    dr, dl = full_rand - r, full_loao - l
    ratio = (dl / dr) if abs(dr) > 1e-9 else float("nan")
    verdict = "campaign-specific" if (dr > 0.005 and dl < 0.3 * dr) else ("transfers" if dl > 0.005 else "-")
    rows.append({"block": name, "random": round(r, 3), "d_random": round(dr, 3),
                 "loao": round(l, 3), "d_loao": round(dl, 3),
                 "retained_frac": round(ratio, 2) if ratio == ratio else None, "verdict": verdict})
    print("%-26s %9.3f %9.3f   %9.3f %9.3f   %s" % (name, r, dr, l, dl, verdict))

d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "outputs/r15_feature_block_bias.csv"), index=False)
print("\nd_rand = AUROC lost by removing the block on the random split")
print("d_LOAO = AUROC lost by removing it when the evaluation campaign is unseen")
print("A block with large d_rand and near-zero d_LOAO encodes campaign-specific structure.")
print("\nsaved outputs/r15_feature_block_bias.csv")
