#!/usr/bin/env python3
"""Per-seed metrics and the measured reliability curve for the released model.

Figure 7A was drawn from outputs/phase6d_seed_results.csv, which belongs to the
superseded Phase-6d configuration (0.940 +/- 0.005), while Table S2 reports the
released deterministic configuration (0.946 +/- <0.001). Figure 7B was drawn
from an illustrative curve rather than from the test predictions. This script
writes both quantities from the released protocol so the figure can be rebuilt
from data.

Protocol is identical to src/r18_regenerate_S2_S3.py:
    embeddings cast to float32 before prototype construction
    prototypes from train+val
    LightGBM with the released hyperparameters, deterministic, single thread
    evaluation on the held-out test split, F1/MCC at a fixed 0.5 threshold

    pixi run -e ml python src/r22_seed_and_calibration.py

Writes outputs/r22_seed_values.json and outputs/r22_calibration.csv.
"""
import os, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, EPS = 42, 1e-8
SEEDS = [42, 123, 456, 789, 1337]
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
base_cols = list(cols.column)
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)
y = fm.binding_label.values.astype(int)
tgt, split = fm.target.values, fm.split.values
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


X = np.hstack([fm[base_cols].values.astype(np.float32), proto_feats(E, tgt, y, tr)])


def predict(seed):
    prm = dict(LGB)
    prm.update(random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)
    return lgb.LGBMClassifier(**prm).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


# ---------- per-seed metrics (Figure 7A, Table S2) ----------
vals = []
for s in SEEDS:
    p = predict(s)
    b = (p >= 0.5).astype(int)
    vals.append([roc_auc_score(yte, p), average_precision_score(yte, p),
                 f1_score(yte, b), matthews_corrcoef(yte, b)])
vals = np.array(vals)
print("LightGBM + Proto over %d seeds %s" % (len(SEEDS), SEEDS))
print("   AUROC %.3f +/- %.3f   AUPRC %.3f +/- %.3f   F1 %.3f +/- %.3f   MCC %.3f +/- %.3f"
      % (vals[:, 0].mean(), vals[:, 0].std(), vals[:, 1].mean(), vals[:, 1].std(),
         vals[:, 2].mean(), vals[:, 2].std(), vals[:, 3].mean(), vals[:, 3].std()))
for s, v in zip(SEEDS, vals):
    print("   seed %-5d AUROC %.5f  AUPRC %.4f  F1 %.4f  MCC %.4f" % (s, v[0], v[1], v[2], v[3]))

# ---------- reliability curve (Figure 7B) ----------
p42 = predict(SEED)
edges = np.linspace(0, 1, 11)
rows, ece = [], 0.0
for i in range(10):
    lo, hi = edges[i], edges[i + 1]
    sel = (p42 >= lo) & (p42 < hi) if i < 9 else (p42 >= lo) & (p42 <= hi)
    n = int(sel.sum())
    if n == 0:
        rows.append(dict(bin_lo=round(lo, 2), bin_hi=round(hi, 2), n=0,
                         mean_pred=np.nan, frac_pos=np.nan))
        continue
    mp, fp = float(p42[sel].mean()), float(yte[sel].mean())
    ece += n / len(p42) * abs(fp - mp)
    rows.append(dict(bin_lo=round(lo, 2), bin_hi=round(hi, 2), n=n,
                     mean_pred=round(mp, 4), frac_pos=round(fp, 4)))
cal = pd.DataFrame(rows)
print("\nreliability curve, 10 equal-width bins, seed 42 (the released model)")
print(cal.to_string(index=False))
print("ECE = %.4f   (published 0.058)" % ece)

json.dump({"seeds": SEEDS, "auroc": vals[:, 0].tolist(), "auprc": vals[:, 1].tolist(),
           "f1": vals[:, 2].tolist(), "mcc": vals[:, 3].tolist(),
           "ece_seed42": round(ece, 4), "n_test": int(len(p42))},
          open(os.path.join(ROOT, "outputs/r22_seed_values.json"), "w"), indent=2)
cal.to_csv(os.path.join(ROOT, "outputs/r22_calibration.csv"), index=False)
print("\nsaved outputs/r22_seed_values.json, outputs/r22_calibration.csv")
