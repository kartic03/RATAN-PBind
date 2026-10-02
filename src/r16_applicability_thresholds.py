#!/usr/bin/env python3
"""Empirical basis for the applicability-domain thresholds.

protbind/predictor.py hardcodes three regimes on n_pos, the target's known binders
in the prototype database:

    n_pos >= 20  "in-domain"      claims "in-distribution reliability (~0.90-0.95 AUROC)"
    n_pos >=  2  "few-shot"       claims "expect ~0.70 AUROC"
    else         "extrapolation"  "near chance"

Neither cut point is derived anywhere in the repository. This script supplies the
evidence: a fine sweep of prototype support size using the leave-one-target-out
few-shot machinery, so the curve can be inspected where the thresholds sit.

    pixi run -e ml python src/r16_applicability_thresholds.py
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-8
KS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]
N_DRAWS = 30
LGB = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=42, bagging_seed=42, feature_fraction_seed=42)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float32)
y = fm.binding_label.values.astype(int)
tgt = fm.target.values
Xb = fm[base_cols].values.astype(np.float32)


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


def row(e, pp, pn, a, b):
    ne = np.linalg.norm(e)
    cp = float(e @ pp) / (ne * np.linalg.norm(pp) + EPS)
    cn = float(e @ pn) / (ne * np.linalg.norm(pn) + EPS)
    d = pp - pn
    return [cp, cn, np.linalg.norm(e - pp), float(e @ d) / (np.linalg.norm(d) + EPS), cp / (cn + EPS), a, b]


cand = [t for t in np.unique(tgt)
        if ((tgt == t) & (y == 1)).sum() >= 35 and ((tgt == t) & (y == 0)).sum() >= 35]
print("targets with >=35 binders and >=35 non-binders (needed to reach k=30): %s\n" % cand)

rows = []
for held in cand:
    hm = tgt == held
    P = proto_feats(~hm)
    model = lgb.LGBMClassifier(**LGB).fit(np.hstack([Xb, P])[~hm], y[~hm])
    pidx, nidx = np.where(hm & (y == 1))[0], np.where(hm & (y == 0))[0]
    for k in KS:
        if k > min(len(pidx), len(nidx)) - 5:
            continue
        for r in range(N_DRAWS if k > 0 else 1):
            rs = np.random.RandomState(r)
            if k == 0:
                pp = pn = np.zeros(E.shape[1]); shots = set()
            else:
                sp = rs.choice(pidx, k, False); sn = rs.choice(nidx, k, False)
                shots = set(sp) | set(sn)
                pp, pn = E[sp].mean(0), E[sn].mean(0)
            ev = np.array([i for i in np.where(hm)[0] if i not in shots])
            if len(np.unique(y[ev])) < 2:
                continue
            Pe = np.array([row(E[i], pp, pn, k, k) for i in ev])
            s = model.predict_proba(np.hstack([Xb[ev], Pe]))[:, 1]
            rows.append({"target": held, "k": k, "draw": r, "auroc": roc_auc_score(y[ev], s)})

d = pd.DataFrame(rows)
print("=== prototype support sweep (mean AUROC over %d draws) ===" % N_DRAWS)
print("   k      mean     SD     p10     p90    n_targets")
for k in KS:
    s = d[d.k == k]
    if not len(s):
        continue
    print("  %-4d  %.3f   %.3f  %.3f  %.3f   %d"
          % (k, s.auroc.mean(), s.auroc.std(), s.auroc.quantile(.1), s.auroc.quantile(.9), s.target.nunique()))

print("\n=== what the deployed thresholds claim, against what the sweep shows ===")
for k, claim in [(2, "few-shot regime, 'expect ~0.70 AUROC'"), (20, "in-domain, '~0.90-0.95 AUROC'")]:
    s = d[d.k == k]
    if len(s):
        print("  n_pos = %-3d  claimed: %-42s observed: %.3f (SD %.3f)"
              % (k, claim, s.auroc.mean(), s.auroc.std()))
    else:
        print("  n_pos = %-3d  claimed: %-42s observed: not reachable in this dataset" % (k, claim))

print("\n=== per-target test AUROC against training-binder count ===")
sp = fm.split.values
tr = (sp == "train") | (sp == "val")
P = proto_feats(tr)
X = np.hstack([Xb, P])
m = lgb.LGBMClassifier(**LGB).fit(X[tr], y[tr])
pr = m.predict_proba(X)[:, 1]
out = []
for t in np.unique(tgt):
    te_m = (tgt == t) & (sp == "test")
    if te_m.sum() < 10 or len(np.unique(y[te_m])) < 2:
        continue
    out.append({"target": t, "train_binders": int(((tgt == t) & tr & (y == 1)).sum()),
                "test_n": int(te_m.sum()), "test_binders": int(y[te_m].sum()),
                "test_auroc": round(roc_auc_score(y[te_m], pr[te_m]), 3)})
o = pd.DataFrame(out).sort_values("train_binders", ascending=False)
print(o.to_string(index=False))

d.to_csv(os.path.join(ROOT, "outputs/r16_support_sweep.csv"), index=False)
o.to_csv(os.path.join(ROOT, "outputs/r16_per_target_support.csv"), index=False)
print("\nsaved outputs/r16_support_sweep.csv, outputs/r16_per_target_support.csv")
