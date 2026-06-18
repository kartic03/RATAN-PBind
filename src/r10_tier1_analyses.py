#!/usr/bin/env python3
"""R10 - Tier-1 analyses for the revision (deterministic, CPU, research/pixi-ml).
A. Budget-vs-yield: precision/recall/enrichment at top 1/5/10/20% on held-out test.
B. ECE (10 bins) of the 470-feature headline model.
C. Operational applicability-domain support: ESM-2 novelty (proto_cos_pos) distribution
   in training, the in-domain threshold, and within-test near- vs far-prototype AUROC.
D. Author-blind prototype gain: base(463) vs base+proto(470) under leave-author-out
   GroupKFold (does the prototype gain survive when whole campaigns are held out?).
E. Few-shot recovery with AUPRC alongside AUROC (deterministic)."""
import os, json, warnings, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
           colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0, is_unbalance=True,
           verbosity=-1, n_jobs=1, num_threads=1, deterministic=True, force_row_wise=True,
           random_state=42, bagging_seed=42, feature_fraction_seed=42)
LGB_FS = dict(LGB); LGB_FS["n_estimators"] = 300  # match the original few-shot config

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
pw = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))[["protein_id", "target", "author"]]
fm = fm.merge(pw, on=["protein_id", "target"], how="left")
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)
y = fm["binding_label"].values.astype(int); tgt = fm["target"].values; split = fm["split"].values
Xb = fm[base_cols].values.astype(np.float32); eps = 1e-8
tr = (split == "train") | (split == "val"); te = split == "test"

def proto_feats(E, tgt, y, trmask):
    out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t; pos = E[m & trmask & (y == 1)]; neg = E[m & trmask & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1]); pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn; dn = np.linalg.norm(diff) + eps; ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + eps)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + eps)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn,
                           cp / (cn + eps), np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out

P = proto_feats(E, tgt, y, tr)
X = np.hstack([Xb, P])
clf = lgb.LGBMClassifier(**LGB); clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]; yte = y[te]
out = {}

# --- A. budget vs yield ---
base_rate = float(yte.mean()); order = np.argsort(p)[::-1]; bvy = []
for frac in (0.01, 0.05, 0.10, 0.20):
    k = max(1, int(round(frac * len(yte)))); sel = order[:k]
    prec = float(yte[sel].mean()); rec = float(yte[sel].sum() / yte.sum()); enr = prec / base_rate
    bvy.append({"top_frac": frac, "n": k, "precision": round(prec, 3), "recall": round(rec, 3), "enrichment": round(enr, 2)})
pd.DataFrame(bvy).to_csv(os.path.join(ROOT, "outputs/r10_budget_yield.csv"), index=False)
out["base_rate_test"] = round(base_rate, 3); out["budget_yield"] = bvy
print("A. budget-vs-yield (test base rate %.3f):" % base_rate)
for r in bvy: print(f"   top {int(r['top_frac']*100):2d}%  prec={r['precision']:.3f}  recall={r['recall']:.3f}  enrich={r['enrichment']:.1f}x")

# --- B. ECE (10 bins) ---
def ece(yy, pp, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (pp >= b[i]) & (pp < b[i + 1] if i < nb - 1 else pp <= b[i + 1])
        if m.sum(): e += abs(pp[m].mean() - yy[m].mean()) * m.sum() / len(yy)
    return e
out["ece_test_10bin"] = round(float(ece(yte, p)), 3)
print(f"B. ECE (10 bins) = {out['ece_test_10bin']:.3f}")

# --- C. AD support: ESM-2 novelty (proto_cos_pos) + near/far test split ---
cos_pos_tr = P[tr, 0]; thr = float(np.percentile(cos_pos_tr, 5))  # in-domain lower bound
cos_pos_te = P[te, 0]; frac_in = float((cos_pos_te >= thr).mean())
med = float(np.median(cos_pos_tr)); near = cos_pos_te >= med; far = ~near
au_near = roc_auc_score(yte[near], p[near]) if len(np.unique(yte[near])) > 1 else float("nan")
au_far = roc_auc_score(yte[far], p[far]) if len(np.unique(yte[far])) > 1 else float("nan")
out["ad"] = {"cos_pos_5pct_threshold": round(thr, 3), "test_frac_in_domain": round(frac_in, 3),
             "auroc_near_prototype": round(au_near, 3), "auroc_far_prototype": round(au_far, 3),
             "n_near": int(near.sum()), "n_far": int(far.sum())}
print(f"C. AD novelty: in-domain cos_pos>= {thr:.3f}; test in-domain {frac_in:.3f}; "
      f"AUROC near={au_near:.3f} (n={near.sum()}) vs far={au_far:.3f} (n={far.sum()})")

# --- D. author-blind prototype gain: base vs base+proto under leave-author-out ---
g = fm["author"].fillna("NA").astype(str).values; gkf = GroupKFold(5)
base_aucs, proto_aucs = [], []
for a, b in gkf.split(X, y, g):
    if len(np.unique(y[b])) < 2: continue
    Pa = proto_feats(E, tgt, y, np.isin(np.arange(len(y)), a)); Xa = np.hstack([Xb, Pa])
    mb = lgb.LGBMClassifier(**LGB); mb.fit(Xb[a], y[a]); base_aucs.append(roc_auc_score(y[b], mb.predict_proba(Xb[b])[:, 1]))
    mp = lgb.LGBMClassifier(**LGB); mp.fit(Xa[a], y[a]); proto_aucs.append(roc_auc_score(y[b], mp.predict_proba(Xa[b])[:, 1]))
out["author_blind"] = {"base_auroc": round(float(np.mean(base_aucs)), 3), "proto_auroc": round(float(np.mean(proto_aucs)), 3),
                       "delta": round(float(np.mean(proto_aucs) - np.mean(base_aucs)), 3),
                       "delta_sd": round(float(np.std(np.array(proto_aucs) - np.array(base_aucs))), 3)}
print(f"D. author-blind: base {out['author_blind']['base_auroc']:.3f} vs base+proto "
      f"{out['author_blind']['proto_auroc']:.3f}  delta {out['author_blind']['delta']:+.3f} +/- {out['author_blind']['delta_sd']:.3f}")

# --- E. few-shot AUROC + AUPRC (deterministic) ---
def prow(ei, pp, pn, a, b):
    cp = (ei @ pp) / ((np.linalg.norm(ei) * np.linalg.norm(pp)) + eps)
    cn = (ei @ pn) / ((np.linalg.norm(ei) * np.linalg.norm(pn)) + eps); d = pp - pn
    return [cp, cn, np.linalg.norm(ei - pp), (ei @ d) / (np.linalg.norm(d) + eps), cp / (cn + eps), a, b]
cand = [t for t in np.unique(tgt) if ((tgt == t) & (y == 1)).sum() >= 15 and ((tgt == t) & (y == 0)).sum() >= 15]
KS = [0, 1, 2, 5, 10]; cellA = {k: [] for k in KS}; cellP = {k: [] for k in KS}
for held in cand:
    hm = tgt == held; trm = ~hm
    Ptr = proto_feats(E, tgt, y, trm)
    m = lgb.LGBMClassifier(**LGB_FS); m.fit(np.hstack([Xb, Ptr])[trm], y[trm])
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
fs = []
for k in KS:
    fs.append({"k": k, "auroc": round(float(np.mean(cellA[k])), 3), "auprc": round(float(np.mean(cellP[k])), 3)})
pd.DataFrame(fs).to_csv(os.path.join(ROOT, "outputs/r10_fewshot_auprc.csv"), index=False)
out["fewshot"] = fs
print("E. few-shot (deterministic):")
for r in fs: print(f"   k={r['k']:2d}  AUROC={r['auroc']:.3f}  AUPRC={r['auprc']:.3f}")

with open(os.path.join(ROOT, "outputs/r10_tier1.json"), "w") as fh: json.dump(out, fh, indent=2)
print("\nsaved outputs/r10_budget_yield.csv, r10_fewshot_auprc.csv, r10_tier1.json")
