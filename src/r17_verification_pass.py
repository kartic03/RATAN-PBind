#!/usr/bin/env python3
"""Verification pass: recompute every reported headline number and compare.

Prints PUBLISHED vs RECOMPUTED vs delta for Tables 2, 3, 4, 7, S1, S2, S3,
nested CV and ECE. Marks a row REPRODUCES when within the measured noise floor
(0.003 AUROC, from src/r15b_noise_floor.py), else DRIFT.

Canonical protocol from src/r9_release_artifacts.py: LightGBM with the released
hyperparameters, train+val for fitting and for prototype construction, held-out
test split, F1/MCC at a fixed 0.5 threshold.

    pixi run -e ml python src/r17_verification_pass.py
"""
import os, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, EPS, NOISE = 42, 1e-8, 0.003
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
base_cols = list(cols.column)
grp = dict(zip(cols.column, cols.group))
pw = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y = fm.binding_label.values.astype(int)
tgt = fm.target.values
split = fm.split.values
tr, te = (split == "train") | (split == "val"), split == "test"
yte = y[te]

rows = []


def rec(section, metric, published, got, kind="auroc"):
    d = None if published is None or got is None else got - published
    if d is None:
        verdict = "n/a"
    elif kind == "auroc":
        verdict = "reproduces" if abs(d) <= NOISE else ("close" if abs(d) <= 0.01 else "DRIFT")
    else:
        verdict = "reproduces" if abs(d) <= 0.01 else "DRIFT"
    rows.append({"section": section, "metric": metric, "published": published,
                 "recomputed": None if got is None else round(got, 4),
                 "delta": None if d is None else round(d, 4), "verdict": verdict})


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


P = proto_feats(tr)
X470 = np.hstack([fm[base_cols].values.astype(np.float32), P])


def fit_eval(X, trmask=tr, temask=te, yy=None):
    yy = y if yy is None else yy
    m = lgb.LGBMClassifier(**LGB).fit(X[trmask], yy[trmask])
    return m.predict_proba(X[temask])[:, 1]


print("=" * 78)
print("1. HEADLINE 470-FEATURE MODEL (Table 2)")
p = fit_eval(X470)
rec("headline", "AUROC", 0.946, roc_auc_score(yte, p))
rec("headline", "AUPRC", 0.770, average_precision_score(yte, p))
rec("headline", "F1 @0.5", 0.752, f1_score(yte, (p >= 0.5).astype(int)), "other")
rec("headline", "MCC @0.5", 0.701, matthews_corrcoef(yte, (p >= 0.5).astype(int)), "other")
rec("headline", "ECE (10 bins)", 0.058,
    float(sum(abs(p[(p >= lo) & (p < hi)].mean() - yte[(p >= lo) & (p < hi)].mean())
              * ((p >= lo) & (p < hi)).sum() / len(p)
              for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])
              if ((p >= lo) & (p < hi)).sum() > 0)), "other")

print("2. NESTED CROSS-VALIDATION (the 'rigorous headline')")
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
nested = []
for a, b in outer.split(X470, y):
    trm = np.zeros(len(y), bool); trm[a] = True
    Pk = proto_feats(trm)
    Xk = np.hstack([fm[base_cols].values.astype(np.float32), Pk])
    mk = lgb.LGBMClassifier(**LGB).fit(Xk[a], y[a])
    nested.append(roc_auc_score(y[b], mk.predict_proba(Xk[b])[:, 1]))
rec("nested CV", "AUROC mean", 0.895, float(np.mean(nested)))
rec("nested CV", "AUROC sd", 0.006, float(np.std(nested)), "other")
print("   folds: %s" % " ".join("%.3f" % v for v in nested))

print("3. FEATURE-GROUP ABLATION (Table S3)")
SEQ = [c for c in base_cols if grp[c] in ("aa_composition", "dipeptide", "physicochemical")]
ABL = {
    ("Full model", 470, 0.946, 0.770): base_cols,
    ("Without prototype", 463, 0.880, 0.713): None,
    ("Without Boltz2", 458, 0.931, 0.751): [c for c in base_cols if grp[c] != "boltz2"],
    ("Without design-method", 453, 0.912, 0.728): [c for c in base_cols if grp[c] != "design_method"],
    ("Sequence only", 427, 0.821, 0.581): SEQ,
}
for (name, n, pa, pp_), keep in ABL.items():
    if keep is None:
        X = fm[base_cols].values.astype(np.float32)
    else:
        X = np.hstack([fm[keep].values.astype(np.float32), P])
    pr = fit_eval(X)
    rec("Table S3", "%s (n=%d) AUROC" % (name, n), pa, roc_auc_score(yte, pr))
    rec("Table S3", "%s (n=%d) AUPRC" % (name, n), pp_, average_precision_score(yte, pr), "other")

print("4. MULTI-SEED STABILITY (Table S2)")
seeds = []
for s in [42, 1, 7, 2024, 99]:
    prm = dict(LGB); prm.update(random_state=s, bagging_seed=s, feature_fraction_seed=s)
    mm = lgb.LGBMClassifier(**prm).fit(X470[tr], y[tr])
    q = mm.predict_proba(X470[te])[:, 1]
    seeds.append((roc_auc_score(yte, q), average_precision_score(yte, q),
                  f1_score(yte, (q >= .5).astype(int)), matthews_corrcoef(yte, (q >= .5).astype(int))))
a = np.array(seeds)
for j, (nm, pm, ps) in enumerate([("AUROC", 0.940, 0.005), ("AUPRC", 0.759, 0.015),
                                  ("F1", 0.718, 0.019), ("MCC", 0.664, 0.020)]):
    rec("Table S2", "%s mean" % nm, pm, float(a[:, j].mean()), "other")
    rec("Table S2", "%s sd" % nm, ps, float(a[:, j].std()), "other")

print("5. NEGATIVE CONTROLS (Table 3)")
rs = np.random.RandomState(SEED)
yp = y.copy(); yp[tr] = rs.permutation(y[tr])
rec("Table 3", "label permutation", 0.467, roc_auc_score(yte, fit_eval(X470, yy=yp)))
if "expressed" in pw.columns:
    ex = pw.set_index(["protein_id", "target"]).expressed.reindex(
        pd.MultiIndex.from_arrays([fm.protein_id, fm.target])).values
    exb = pd.Series(ex).map({True: 1, False: 0, 1: 1, 0: 0}).values
    m_ex = te & (exb == 1)
    if m_ex.sum() > 20:
        rec("Table 3", "expressed-only AUROC", 0.945, roc_auc_score(y[m_ex], p[np.where(te)[0].searchsorted(np.where(m_ex)[0])]))
    ok = ~pd.isna(exb)
    if ok.sum() > 100 and len(np.unique(exb[ok & te])) > 1:
        rec("Table 3", "predicting expression", 0.778,
            roc_auc_score(exb[te & ok].astype(int), fit_eval(X470, yy=np.nan_to_num(exb).astype(int))[ok[te]]))
else:
    rec("Table 3", "expression controls", None, None)

print("6. SINGLE-FEATURE BASELINES (Table 4)")
pr_ratio = P[:, 4]
rec("Table 4", "proto_ratio alone", 0.772, roc_auc_score(yte, pr_ratio[te]))
pos = E[tr & (y == 1)].mean(0); neg = E[tr & (y == 0)].mean(0)
ei = E[te].astype(np.float64); nei = np.linalg.norm(ei, axis=1)
nc = (ei @ pos) / (nei * np.linalg.norm(pos) + EPS) - (ei @ neg) / (nei * np.linalg.norm(neg) + EPS)
rec("Table 4", "ESM-2 nearest-centroid", 0.750, roc_auc_score(yte, nc))
for col, pub in [("esmfold_plddt", None), ("isoelectric_point", 0.585), ("seq_length", 0.54)]:
    if col in fm.columns:
        v = fm[col].values[te]
        rec("Table 4", col, pub, max(roc_auc_score(yte, v), roc_auc_score(yte, -v)))
ms = [c for c in base_cols if "success" in c.lower()]
if ms:
    rec("Table 4", "design-method success rate", 0.719, roc_auc_score(yte, fm[ms[0]].values[te]))
bz = [c for c in base_cols if c.startswith("boltz2_iptm")]
if bz:
    nip = te & (tgt == "nipah-glycoprotein-g")
    rec("Table 4", "Boltz2 ipTM (nipah)", 0.682, roc_auc_score(y[nip], fm[bz[0]].values[nip]))

print("7. BUDGET vs YIELD (Table 7)")
order = np.argsort(-p)
base_rate = yte.mean()
rec("Table 7", "test base rate", 0.171, float(base_rate), "other")
for frac, pp_, pe in [(0.01, 0.75, 4.4), (0.05, 0.90, 5.3), (0.10, 0.795, 4.7), (0.20, 0.667, 3.9)]:
    k = max(1, int(round(frac * len(yte))))
    prec = float(yte[order[:k]].mean())
    rec("Table 7", "top-%d%% precision" % int(frac * 100), pp_, prec, "other")
    rec("Table 7", "top-%d%% enrichment" % int(frac * 100), pe, prec / base_rate, "other")

d = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(d.to_string(index=False))
n_dr = (d.verdict == "DRIFT").sum()
print("\nreproduces %d   close %d   DRIFT %d   n/a %d"
      % ((d.verdict == "reproduces").sum(), (d.verdict == "close").sum(), n_dr, (d.verdict == "n/a").sum()))
if n_dr:
    print("\nDRIFTED:")
    print(d[d.verdict == "DRIFT"].to_string(index=False))
d.to_csv(os.path.join(ROOT, "outputs/r17_verification_pass.csv"), index=False)
print("\nsaved outputs/r17_verification_pass.csv")
