#!/usr/bin/env python3
"""R11 - few-shot recovery, FINAL canonical run: AUROC AND AUPRC with bootstrap CIs
from a SINGLE deterministic run (fixes the spliced-run inconsistency flagged in review).
CPU, deterministic. Writes outputs/r11_fewshot_final.csv."""
import os, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LGB = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.9,
           colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0, is_unbalance=True,
           verbosity=-1, n_jobs=1, num_threads=1, deterministic=True, force_row_wise=True,
           random_state=42, bagging_seed=42, feature_fraction_seed=42)
fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)
y = fm["binding_label"].values.astype(int); tgt = fm["target"].values
Xb = fm[base_cols].values.astype(np.float32); eps = 1e-8

def proto_feats(E, tgt, y, trm):
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t; pos = E[m & trm & (y == 1)]; neg = E[m & trm & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1]); pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn; dn = np.linalg.norm(diff) + eps; ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + eps)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + eps)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn,
                           cp / (cn + eps), np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out

def _n(v): return float(np.sqrt(np.sum(np.asarray(v, float) ** 2)))
def prow(ei, pp, pn, a, b):
    ei = np.asarray(ei, float).ravel(); pp = np.asarray(pp, float).ravel(); pn = np.asarray(pn, float).ravel()
    nei = _n(ei)
    cp = float(ei @ pp) / ((nei * _n(pp)) + eps)
    cn = float(ei @ pn) / ((nei * _n(pn)) + eps); d = pp - pn
    return [cp, cn, _n(ei - pp), float(ei @ d) / (_n(d) + eps), cp / (cn + eps), a, b]

cand = [t for t in np.unique(tgt) if ((tgt == t) & (y == 1)).sum() >= 15 and ((tgt == t) & (y == 0)).sum() >= 15]
KS = [0, 1, 2, 5, 10]; cellA = {k: [] for k in KS}; cellP = {k: [] for k in KS}
for held in cand:
    hm = tgt == held; trm = ~hm
    Ptr = proto_feats(E, tgt, y, trm)
    m = lgb.LGBMClassifier(**LGB); m.fit(np.hstack([Xb, Ptr])[trm], y[trm])
    pidx = np.where(hm & (y == 1))[0]; nidx = np.where(hm & (y == 0))[0]
    for k in KS:
        for r in range(10):
            rs = np.random.RandomState(r)
            if k == 0: pp = np.zeros(1280); pn = np.zeros(1280); shots = set()
            else:
                sp = rs.choice(pidx, min(k, len(pidx)), False); sn = rs.choice(nidx, min(k, len(nidx)), False)
                shots = set(sp) | set(sn); pp = E[sp].mean(0); pn = E[sn].mean(0)
            ev = np.array([i for i in np.where(hm)[0] if i not in shots])
            if len(np.unique(y[ev])) < 2: continue
            Pe = np.array([prow(E[i], pp, pn, k, k) for i in ev]); pr = m.predict_proba(np.hstack([Xb[ev], Pe]))[:, 1]
            cellA[k].append(roc_auc_score(y[ev], pr)); cellP[k].append(average_precision_score(y[ev], pr))

def ci(v):
    v = np.array(v); boot = [np.mean(np.random.RandomState(s).choice(v, len(v), True)) for s in range(2000)]
    return float(np.mean(v)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

rows = []
print(f"few-shot targets (n={len(cand)}): {cand}")
print(f"  {'k':>3} {'AUROC [95% CI]':>22} {'AUPRC [95% CI]':>22}")
for k in KS:
    am, alo, ahi = ci(cellA[k]); pm, plo, phi = ci(cellP[k])
    print(f"  {k:3d}  {am:.3f} [{alo:.3f}, {ahi:.3f}]   {pm:.3f} [{plo:.3f}, {phi:.3f}]")
    rows.append({"k": k, "auroc": round(am, 3), "auroc_ci_lo": round(alo, 3), "auroc_ci_hi": round(ahi, 3),
                 "auprc": round(pm, 3), "auprc_ci_lo": round(plo, 3), "auprc_ci_hi": round(phi, 3)})
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs/r11_fewshot_final.csv"), index=False)
print("\nsaved outputs/r11_fewshot_final.csv")
