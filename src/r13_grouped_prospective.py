#!/usr/bin/env python3
"""AUPRC and top-k enrichment under grouped prospective splits (Reviewer 1, comment 2).

outputs/r9_grouped_leakage.csv reports AUROC only. The manuscript states that at a
17.8% binding rate, AUPRC and top-k enrichment are the decision-relevant metrics --
but the 4.7x enrichment it reports comes from the RANDOM held-out test split, not
from any grouped split. This script closes that gap.

Splits (protocol identical to src/r9_release_artifacts.py):
  leave-author-out        GroupKFold(5) on author
  leave-design_method-out GroupKFold(5) on design_method
  leave-one-method-out    each design method with >= 40 pairs, held out whole

Prototypes are recomputed from the training fold every time.

Feature sets:
  470  published headline
  437  deployment-matched (sequence + prototype only; see src/r12_deployment_matched.py)

Enrichment is computed against EACH FOLD'S OWN base rate, since grouped folds have
very different binder proportions.

    pixi run -e ml python src/r13_grouped_prospective.py
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
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

DEPLOY = ([c for c in base_cols if grp[c] in ("aa_composition", "dipeptide", "physicochemical")]
          + ["seq_length", "molecular_weight", "isoelectric_point"])
FEATSETS = {"470 published": base_cols, "437 deployment": DEPLOY}


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


def topk(yy, pp, frac):
    k = max(1, int(round(frac * len(yy))))
    sel = yy[np.argsort(-pp)][:k]
    prec = float(sel.mean())
    base = float(yy.mean())
    return prec, (prec / base if base > 0 else np.nan), k


def evaluate(train_idx, test_idx, feats):
    P = proto_feats(E, tgt, y, np.isin(np.arange(len(y)), train_idx))
    X = np.hstack([fm[feats].values.astype(np.float32), P])
    m = lgb.LGBMClassifier(**LGB).fit(X[train_idx], y[train_idx])
    p = m.predict_proba(X[test_idx])[:, 1]
    yy = y[test_idx]
    r = {"n": len(yy), "binders": int(yy.sum()), "base_rate": round(float(yy.mean()), 3),
         "auroc": round(roc_auc_score(yy, p), 3), "auprc": round(average_precision_score(yy, p), 3)}
    for f in (0.05, 0.10, 0.20):
        prec, enr, k = topk(yy, p, f)
        r["top%d_prec" % int(f * 100)] = round(prec, 3)
        r["top%d_enrich" % int(f * 100)] = round(enr, 2)
    return r


rows = []
for fsname, feats in FEATSETS.items():
    for gcol in ["author", "design_method"]:
        g = fm[gcol].fillna("NA").astype(str).values
        Xd = np.zeros((len(y), 1))
        for k, (a, b) in enumerate(GroupKFold(5).split(Xd, y, g)):
            if len(np.unique(y[b])) < 2:
                continue
            r = evaluate(a, b, feats)
            r.update({"featset": fsname, "split": "leave-%s-out" % gcol, "fold": str(k)})
            rows.append(r)
    dm = fm.design_method.fillna("unknown").values
    vc = pd.Series(dm).value_counts()
    for meth in vc[vc >= 40].index.tolist()[:8]:
        hm = dm == meth
        if len(np.unique(y[hm])) < 2 or hm.sum() < 20:
            continue
        r = evaluate(np.where(~hm)[0], np.where(hm)[0], feats)
        r.update({"featset": fsname, "split": "leave-one-method-out", "fold": str(meth)[:24]})
        rows.append(r)

d = pd.DataFrame(rows)
order = ["featset", "split", "fold", "n", "binders", "base_rate", "auroc", "auprc",
         "top5_prec", "top5_enrich", "top10_prec", "top10_enrich", "top20_prec", "top20_enrich"]
d = d[order]
print(d.to_string(index=False))

print("\n=== fold means by split and feature set ===")
agg = d.groupby(["featset", "split"]).agg(
    folds=("fold", "size"), base_rate=("base_rate", "mean"), auroc=("auroc", "mean"),
    auprc=("auprc", "mean"), top10_enrich=("top10_enrich", "mean"),
    top10_prec=("top10_prec", "mean")).round(3)
print(agg.to_string())

print("\nreference, RANDOM held-out test split (src/r12_deployment_matched.py):")
print("   470 published : AUROC 0.946  AUPRC 0.770  top-10%% 0.795  enrichment 4.65x  base rate 0.171")
print("   437 deployment: AUROC 0.933  AUPRC 0.757  top-10%% 0.821  enrichment 4.80x  base rate 0.171")

d.to_csv(os.path.join(ROOT, "outputs/r13_grouped_prospective.csv"), index=False)
agg.to_csv(os.path.join(ROOT, "outputs/r13_grouped_prospective_summary.csv"))
print("\nsaved outputs/r13_grouped_prospective{,_summary}.csv")
