#!/usr/bin/env python3
"""R9 - persist release artifacts that the manuscript cites but were not saved:
1. the 470-feature headline model (463 base + 7 prototype), saved as models/lgb_proto_470.pkl,
   with a manifest confirming it reproduces the reported test AUROC 0.946 [0.919-0.968];
2. the grouped-leakage (leave-author-out / leave-method-out) results, saved per fold to
   outputs/r9_grouped_leakage.csv, so the 0.77/0.73 "new campaign" figures are reproducible
   artifacts rather than hardcoded plot constants.
CPU, LightGBM, research env. Deterministic (seed 42, no shuffle in GroupKFold)."""
import os, json, warnings, numpy as np, pandas as pd
import joblib, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.RandomState(42)
# fully deterministic: single thread + fixed seed + deterministic histogram so the
# released model and every reported number reproduce exactly on re-run.
LGB = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
           colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0, is_unbalance=True,
           verbosity=-1, n_jobs=1, num_threads=1, deterministic=True, force_row_wise=True,
           random_state=42, bagging_seed=42, feature_fraction_seed=42)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
pw = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))[["protein_id", "target", "author", "design_method"]]
fm = fm.merge(pw, on=["protein_id", "target"], how="left")
emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
pids = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
p2r = {str(p): i for i, p in enumerate(pids)}
E = np.stack([emb[p2r[str(p)]] for p in fm["protein_id"]]).astype(np.float32)
y = fm["binding_label"].values.astype(int); tgt = fm["target"].values; split = fm["split"].values
tr = (split == "train") | (split == "val"); te = split == "test"

def proto_feats(E, tgt, y, trmask):
    eps = 1e-8; out = np.zeros((len(E), 7), np.float32)
    for t in np.unique(tgt):
        m = tgt == t
        pos = E[m & trmask & (y == 1)]; neg = E[m & trmask & (y == 0)]
        pp = pos.mean(0) if len(pos) else np.zeros(E.shape[1])
        pn = neg.mean(0) if len(neg) else np.zeros(E.shape[1])
        diff = pp - pn; dn = np.linalg.norm(diff) + eps; ei = E[m]
        cp = (ei @ pp) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pp)) + eps)
        cn = (ei @ pn) / ((np.linalg.norm(ei, axis=1) * np.linalg.norm(pn)) + eps)
        out[m] = np.stack([cp, cn, np.linalg.norm(ei - pp, axis=1), (ei @ diff) / dn,
                           cp / (cn + eps), np.full(m.sum(), len(pos)), np.full(m.sum(), len(neg))], 1)
    return out

proto_names = ["proto_cos_pos", "proto_cos_neg", "proto_l2_pos", "proto_disc_proj",
               "proto_ratio", "proto_n_pos", "proto_n_neg"]
P = proto_feats(E, tgt, y, tr)
feat_names = base_cols + proto_names
X = np.hstack([fm[base_cols].values.astype(np.float32), P])
assert X.shape[1] == 470, f"expected 470 features, got {X.shape[1]}"

# --- 1. headline 470-feature model ---
clf = lgb.LGBMClassifier(**LGB); clf.fit(X[tr], y[tr])
p = clf.predict_proba(X[te])[:, 1]; yte = y[te]
def boot_ci(yy, pp, fn, n=1000):
    vals = []; idx = np.arange(len(yy))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yy[b])) < 2: continue
        try: vals.append(fn(yy[b], pp[b]))
        except Exception: pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
au = roc_auc_score(yte, p); ap = average_precision_score(yte, p)
au_lo, au_hi = boot_ci(yte, p, roc_auc_score); ap_lo, ap_hi = boot_ci(yte, p, average_precision_score)
f1 = f1_score(yte, (p >= 0.5).astype(int)); mcc = matthews_corrcoef(yte, (p >= 0.5).astype(int))
print(f"[470 headline] n_features_in_={clf.n_features_in_}  test AUROC={au:.3f} [{au_lo:.3f}-{au_hi:.3f}]  "
      f"AUPRC={ap:.3f} [{ap_lo:.3f}-{ap_hi:.3f}]  F1={f1:.3f}  MCC={mcc:.3f}")

joblib.dump({"model": clf, "feature_names": feat_names, "n_features": 470}, os.path.join(ROOT, "models/lgb_proto_470.pkl"))
manifest = {"model_file": "models/lgb_proto_470.pkl", "n_features": int(clf.n_features_in_),
            "feature_groups": {"base": len(base_cols), "prototype": 7},
            "test_auroc": round(au, 3), "test_auroc_ci": [round(au_lo, 3), round(au_hi, 3)],
            "test_auprc": round(ap, 3), "test_auprc_ci": [round(ap_lo, 3), round(ap_hi, 3)],
            "test_f1": round(f1, 3), "test_mcc": round(mcc, 3), "seed": 42}
with open(os.path.join(ROOT, "models/lgb_proto_470_manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)

# --- 2. grouped leakage: leave-author-out & leave-method-out (5-fold GroupKFold, prototypes per train fold) ---
rows = []
for gcol in ["author", "design_method"]:
    g = fm[gcol].fillna("NA").astype(str).values
    gkf = GroupKFold(5); fold_aucs = []
    for k, (a, b) in enumerate(gkf.split(X, y, g)):
        Pa = proto_feats(E, tgt, y, np.isin(np.arange(len(y)), a))
        Xa = np.hstack([fm[base_cols].values.astype(np.float32), Pa])
        if len(np.unique(y[b])) < 2: continue
        mm = lgb.LGBMClassifier(**LGB); mm.fit(Xa[a], y[a])
        auc = roc_auc_score(y[b], mm.predict_proba(Xa[b])[:, 1]); fold_aucs.append(auc)
        rows.append({"split": f"leave-{gcol}-out", "fold": k, "n_test": int(b.size),
                     "n_test_binders": int(y[b].sum()), "auroc": round(auc, 4)})
    print(f"[leave-{gcol}-out] mean AUROC={np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}  "
          f"({fm[gcol].nunique()} groups, {len(fold_aucs)} folds)")
    rows.append({"split": f"leave-{gcol}-out", "fold": "MEAN", "n_test": fm[gcol].nunique(),
                 "n_test_binders": "", "auroc": round(float(np.mean(fold_aucs)), 4)})
    rows.append({"split": f"leave-{gcol}-out", "fold": "SD", "n_test": "", "n_test_binders": "",
                 "auroc": round(float(np.std(fold_aucs)), 4)})
# --- 3. leave-one-design-method-out (top methods >=40 pairs), the across-method axis upper end ---
dm = fm["design_method"].fillna("unknown").values  # design_method already merged into fm above
top = pd.Series(dm).value_counts(); top = top[top >= 40].index.tolist()[:8]
lmo = []
for meth in top:
    hm = dm == meth
    if len(np.unique(y[hm])) < 2 or hm.sum() < 20: continue
    Ph = proto_feats(E, tgt, y, ~hm)
    Xh = np.hstack([fm[base_cols].values.astype(np.float32), Ph])
    mo = lgb.LGBMClassifier(**LGB); mo.fit(Xh[~hm], y[~hm])
    auc = roc_auc_score(y[hm], mo.predict_proba(Xh[hm])[:, 1])
    lmo.append({"split": "leave-one-method-out", "fold": str(meth)[:32], "n_test": int(hm.sum()),
                "n_test_binders": int(y[hm].sum()), "auroc": round(auc, 4)})
lmo_mean = float(np.mean([r["auroc"] for r in lmo]))
lmo.append({"split": "leave-one-method-out", "fold": "MEAN", "n_test": len(top), "n_test_binders": "", "auroc": round(lmo_mean, 4)})
rows += lmo
print(f"[leave-one-method-out] mean AUROC={lmo_mean:.3f} over {len(top)} top methods")

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs/r9_grouped_leakage.csv"), index=False)
print("\nsaved: models/lgb_proto_470.pkl, models/lgb_proto_470_manifest.json, outputs/r9_grouped_leakage.csv")
