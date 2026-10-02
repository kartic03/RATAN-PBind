#!/usr/bin/env python3
"""Few-shot support-set variability and baselines (Reviewer 1, comment 3).

The reviewer asks (a) whether few-shot performance depends on WHICH support examples
are drawn, and (b) how much of the gain is specific to RATAN-PBind rather than to
simple prototype retrieval.

On (a): the canonical run src/r11_fewshot_final.py ALREADY resamples -- 10 independent
support draws per held-out target per k, positives and negatives sampled separately,
with the reported CI bootstrapping over those pooled cells. The manuscript describes
this as "a single deterministic run", meaning one script invocation, which is what
invited the criticism. This script exposes the per-draw distribution that r11 computes
and discards, at a larger number of draws.

On (b): three scorers are evaluated on the identical support/query partitions.
    full        the trained LightGBM over base + 7 prototype features (r11's method)
    centroid    cos(e, p+) - cos(e, p-) from the k-shot prototypes; NO trained model
    proto_ratio cos(e, p+) / cos(e, p-); NO trained model

Protocol otherwise identical to r11: candidate targets need >= 15 binders and >= 15
non-binders; k means k positives AND k negatives (2k support examples); support rows
are excluded from the query set.

    pixi run -e ml python src/r14_fewshot_variability.py
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-8
KS = [0, 1, 2, 5, 10]
N_DRAWS = 100            # r11 uses 10; more draws give a better variance estimate
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


def proto_feats(E, tgt, y, trm):
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


def rowfeat(e, pp, pn, a, b):
    ne = np.linalg.norm(e)
    cp = float(e @ pp) / (ne * np.linalg.norm(pp) + EPS)
    cn = float(e @ pn) / (ne * np.linalg.norm(pn) + EPS)
    d = pp - pn
    return [cp, cn, np.linalg.norm(e - pp), float(e @ d) / (np.linalg.norm(d) + EPS), cp / (cn + EPS), a, b], cp, cn


cand = [t for t in np.unique(tgt)
        if ((tgt == t) & (y == 1)).sum() >= 15 and ((tgt == t) & (y == 0)).sum() >= 15]
print("candidate targets (>=15 binders and >=15 non-binders): %d  %s" % (len(cand), cand))
print("draws per target per k: %d\n" % N_DRAWS)

rows = []
for held in cand:
    hm = tgt == held
    trm = ~hm
    Ptr = proto_feats(E, tgt, y, trm)
    model = lgb.LGBMClassifier(**LGB).fit(np.hstack([Xb, Ptr])[trm], y[trm])
    pidx, nidx = np.where(hm & (y == 1))[0], np.where(hm & (y == 0))[0]
    for k in KS:
        for r in range(N_DRAWS if k > 0 else 1):
            rs = np.random.RandomState(r)
            if k == 0:
                pp = pn = np.zeros(E.shape[1]); shots = set()
            else:
                sp = rs.choice(pidx, min(k, len(pidx)), False)
                sn = rs.choice(nidx, min(k, len(nidx)), False)
                shots = set(sp) | set(sn)
                pp, pn = E[sp].mean(0), E[sn].mean(0)
            ev = np.array([i for i in np.where(hm)[0] if i not in shots])
            if len(np.unique(y[ev])) < 2:
                continue
            feats, cps, cns = [], [], []
            for i in ev:
                f, cp, cn = rowfeat(E[i], pp, pn, k, k)
                feats.append(f); cps.append(cp); cns.append(cn)
            Pe = np.array(feats)
            scores = {
                "full": model.predict_proba(np.hstack([Xb[ev], Pe]))[:, 1],
                "centroid": np.array(cps) - np.array(cns),
                "proto_ratio": np.array(cps) / (np.array(cns) + EPS),
            }
            for sname, sc in scores.items():
                if k == 0 and sname != "full":
                    continue      # undefined without support
                rows.append({"target": held, "k": k, "draw": r, "scorer": sname,
                             "n_eval": len(ev), "auroc": roc_auc_score(y[ev], sc),
                             "auprc": average_precision_score(y[ev], sc)})

d = pd.DataFrame(rows)
d.to_csv(os.path.join(ROOT, "outputs/r14_fewshot_draws.csv"), index=False)

print("=== (a) support-set variability, full model ===")
f = d[d.scorer == "full"]
print("  k    mean    SD(all cells)   min     max     SD within target (mean)")
for k in KS:
    s = f[f.k == k]
    within = s.groupby("target").auroc.std().mean() if k > 0 else 0.0
    print("  %-3d  %.3f   %.3f           %.3f   %.3f   %.3f"
          % (k, s.auroc.mean(), s.auroc.std(), s.auroc.min(), s.auroc.max(), within))

print("\n=== per-target spread, full model (mean +/- SD over draws) ===")
pt = f[f.k > 0].groupby(["target", "k"]).auroc.agg(["mean", "std"]).round(3).reset_index()
print(pt.pivot(index="target", columns="k", values="mean").to_string())
print("\n  SD over draws:")
print(pt.pivot(index="target", columns="k", values="std").to_string())

print("\n=== (b) baselines on identical support/query partitions (mean AUROC) ===")
piv = d.groupby(["scorer", "k"]).auroc.mean().unstack().round(3)
print(piv.to_string())
print("\n  gain of full model over centroid baseline:")
for k in [kk for kk in KS if kk > 0]:
    a = d[(d.scorer == "full") & (d.k == k)].auroc.mean()
    b = d[(d.scorer == "centroid") & (d.k == k)].auroc.mean()
    print("    k=%-3d full %.3f  centroid %.3f  delta %+.3f" % (k, a, b, a - b))

print("\nreference, r11_fewshot_final.py (10 draws): k=0 0.574, k=1 0.648, k=2 0.700, k=5 0.704, k=10 0.728")
d.groupby(["scorer", "k"]).agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                               auprc_mean=("auprc", "mean")).round(4).to_csv(
    os.path.join(ROOT, "outputs/r14_fewshot_summary.csv"))
print("saved outputs/r14_fewshot_draws.csv, outputs/r14_fewshot_summary.csv")
