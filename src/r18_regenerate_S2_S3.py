#!/usr/bin/env python3
"""Regenerate Supplementary Tables S2 and S3 under one canonical protocol.

Canonical path, identical to src/r9_release_artifacts.py:
    embeddings cast to float32 before prototype construction
    prototypes from train+val
    LightGBM with the released hyperparameters
    evaluation on the held-out test split
    F1 / MCC at a fixed 0.5 threshold

Emits publication-ready markdown for both tables, the recomputed prototype gain
with its paired-bootstrap interval, and the measured noise floor.

    pixi run -e ml python src/r18_regenerate_S2_S3.py
"""
import os, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, EPS, N_BOOT = 42, 1e-8, 2000
SEEDS = [42, 123, 456, 789, 1337]   # the five seeds named in the Table S2 caption
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
E = np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)   # canonical dtype
y = fm.binding_label.values.astype(int)
tgt = fm.target.values
split = fm.split.values
tr, te = (split == "train") | (split == "val"), split == "test"
yte = y[te]


def proto_feats(E, tgt, y, trmask):
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t
        pos, neg = E[m & trmask & (y == 1)], E[m & trmask & (y == 0)]
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


P = proto_feats(E, tgt, y, tr)
SEQ = [c for c in base_cols if grp[c] in ("aa_composition", "dipeptide", "physicochemical")]


def run(X, seed=SEED, model="lgb"):
    if model == "xgb":
        from xgboost import XGBClassifier
        npos, nneg = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
        m = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                          subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                          scale_pos_weight=nneg / max(npos, 1), eval_metric="auc",
                          random_state=seed, n_jobs=1, verbosity=0, tree_method="exact")
        m.fit(X[tr], y[tr])
        return m.predict_proba(X[te])[:, 1]
    prm = dict(LGB)
    prm.update(random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)
    m = lgb.LGBMClassifier(**prm).fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def metrics(p):
    b = (p >= 0.5).astype(int)
    return (roc_auc_score(yte, p), average_precision_score(yte, p),
            f1_score(yte, b), matthews_corrcoef(yte, b))


# ---------- Table S3 ----------
X470 = np.hstack([fm[base_cols].values.astype(np.float32), P])
ROWS = [
    ("Full model", 470, X470),
    ("Without prototype features", 463, fm[base_cols].values.astype(np.float32)),
    ("Without Boltz2 structural features", 458,
     np.hstack([fm[[c for c in base_cols if grp[c] != "boltz2"]].values.astype(np.float32), P])),
    ("Without design-method encoding", 453,
     np.hstack([fm[[c for c in base_cols if grp[c] != "design_method"]].values.astype(np.float32), P])),
    ("Sequence features only (AAC + DPC + physico)", 427, fm[SEQ].values.astype(np.float32)),
    ("ESM-2 embeddings only", 1280, E.astype(np.float32)),
]
s3, preds = [], {}
for name, n, X in ROWS:
    assert X.shape[1] == n, "%s expected %d cols, got %d" % (name, n, X.shape[1])
    # the published ESM-2-only row was fitted with XGBoost, not LightGBM
    p = run(X, model="xgb" if n == 1280 else "lgb")
    preds[name] = p
    au, ap, _, _ = metrics(p)
    s3.append({"Feature set": name, "N": n, "AUROC": round(au, 3), "AUPRC": round(ap, 3)})
full = s3[0]["AUROC"]
for r in s3:
    r["dAUROC"] = "-" if r["N"] == 470 else round(r["AUROC"] - full, 3)

# ---------- Table S2 ----------
s2 = []
for label, X, mdl in [("LightGBM + Proto", X470, "lgb"),
                      ("XGBoost + Proto", X470, "xgb")]:
    vals = np.array([metrics(run(X, s, mdl)) for s in SEEDS])
    s2.append({"Model": label,
               "AUROC": "%.3f +/- %.3f" % (vals[:, 0].mean(), vals[:, 0].std()),
               "AUPRC": "%.3f +/- %.3f" % (vals[:, 1].mean(), vals[:, 1].std()),
               "F1": "%.3f +/- %.3f" % (vals[:, 2].mean(), vals[:, 2].std()),
               "MCC": "%.3f +/- %.3f" % (vals[:, 3].mean(), vals[:, 3].std())})

# ---------- prototype gain, paired bootstrap ----------
pa, pb = preds["Full model"], preds["Without prototype features"]
rng = np.random.RandomState(SEED)
d = []
for _ in range(N_BOOT):
    i = rng.randint(0, len(yte), len(yte))
    if len(np.unique(yte[i])) > 1:
        d.append(roc_auc_score(yte[i], pa[i]) - roc_auc_score(yte[i], pb[i]))
d = np.array(d)
gain = roc_auc_score(yte, pa) - roc_auc_score(yte, pb)
lo, hi = np.percentile(d, [2.5, 97.5])
gain_ap = average_precision_score(yte, pa) - average_precision_score(yte, pb)

# ---------- noise floor ----------
nf = []
r2 = np.random.RandomState(0)
for s in SEEDS:
    nf.append(roc_auc_score(yte, run(X470, s)))
for _ in range(5):
    perm = r2.permutation(X470.shape[1])
    nf.append(roc_auc_score(yte, run(X470[:, perm])))
nf = np.array(nf)

print("=" * 78)
print("TABLE S3 (regenerated, canonical protocol)\n")
print("| Feature set | N | AUROC | AUPRC | dAUROC |")
print("|---|---|---|---|---|")
for r in s3:
    print("| %s | %s | %.3f | %.3f | %s |" % (r["Feature set"], "{:,}".format(r["N"]),
                                              r["AUROC"], r["AUPRC"], r["dAUROC"]))
print("\nTABLE S2 (regenerated, %d seeds: %s)\n" % (len(SEEDS), SEEDS))
print("| Model | AUROC | AUPRC | F1 | MCC |")
print("|---|---|---|---|---|")
for r in s2:
    print("| %s | %s | %s | %s | %s |" % (r["Model"], r["AUROC"], r["AUPRC"], r["F1"], r["MCC"]))

print("\nPROTOTYPE GAIN (470 vs 463, paired bootstrap %d resamples)" % N_BOOT)
print("   AUROC  +%.3f  95%% CI [%.3f, %.3f]   P(gain>0) = %.4f" % (gain, lo, hi, (d > 0).mean()))
print("   AUPRC  +%.3f" % gain_ap)
print("   published: +0.065 [0.031, 0.105], p < 0.0001")

print("\nNOISE FLOOR (5 seeds + 5 column permutations on the full model)")
print("   mean %.4f   SD %.4f   range %.4f   -> deltas below %.3f not interpretable"
      % (nf.mean(), nf.std(), nf.max() - nf.min(), 2 * nf.std()))

out = {"table_s3": s3, "table_s2": s2,
       "prototype_gain": {"auroc": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)],
                          "p_gt_0": round(float((d > 0).mean()), 4), "auprc": round(gain_ap, 4)},
       "noise_floor": {"mean": round(float(nf.mean()), 4), "sd": round(float(nf.std()), 4),
                       "range": round(float(nf.max() - nf.min()), 4)}}
json.dump(out, open(os.path.join(ROOT, "outputs/r18_tables_S2_S3.json"), "w"), indent=2)
pd.DataFrame(s3).to_csv(os.path.join(ROOT, "outputs/r18_table_S3.csv"), index=False)
pd.DataFrame(s2).to_csv(os.path.join(ROOT, "outputs/r18_table_S2.csv"), index=False)
print("\nsaved outputs/r18_tables_S2_S3.json, r18_table_S2.csv, r18_table_S3.csv")
