#!/usr/bin/env python3
"""Cross-dataset transfer, sequence-only model: Proteinbase -> Overath et al. 2025.

Reconstruction of the experiment behind outputs/r2_crossdataset_overath.csv.

PROTOCOL (all degrees of freedom stated; the first two were undocumented and were
recovered by matching the stored pb_train_n / ov_test_n / ov_binders columns exactly):

  training          : ONE MODEL PER TARGET, on that target's Proteinbase pairs only
                      (egfr 826, human-insulin-receptor 35, il7r 97, mdm2 27, pd-l1 75)
  de-duplication    : drop Overath rows whose binder_seq exactly matches ANY Proteinbase
                      sequence, binder or non-binder (913 -> 571, 342 removed, 37.5%)
                      NB: matching against Proteinbase BINDERS ONLY removes just 51.
  features          : amino-acid composition + dipeptide composition + physicochemical
                      (427), computed by the functions in src/phase2_features.py itself
  model             : LightGBM, deterministic, seed 42
  metrics           : AUROC and AUPRC per target, with bootstrap 95% CIs

KNOWN DISCREPANCY: this reconstruction does not reproduce the stored per-target AUROCs
(mean absolute deviation ~0.038; reconstructed mean 0.504 vs stored 0.490). Nine
configurations were searched without a closer match. The stored file appears to have
been produced under a configuration not recoverable from the released artifacts.
The qualitative conclusion -- transfer is at chance -- holds under every configuration.

Several per-target models are fit on very few rows (mdm2: 27) and emit near-constant
predictions; n_distinct_preds is reported so this is visible rather than hidden behind
an AUROC of exactly 0.500.

    pixi run -e ml python src/r2_crossdataset_overath.py
"""
import os, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED, N_BOOT = 42, 2000
TARGETS = ["egfr", "human-insulin-receptor", "il7r", "mdm2", "pd-l1"]
STORED = {"egfr": 0.501, "human-insulin-receptor": 0.500, "il7r": 0.490, "mdm2": 0.500, "pd-l1": 0.458}

SRC = os.path.join(ROOT, "src/phase2_features.py")
_txt = open(SRC).read()
_ns = {"__file__": SRC, "__name__": "_phase2_defs"}
exec(compile(_txt[:_txt.index("# ── Load data")], SRC, "exec"), _ns)

fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
pairs = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
ov = pd.read_csv(os.path.join(ROOT, "data/external/overath_clean.csv"))

grp = dict(zip(cols.column, cols.group))
FEATS = ([c for c in cols.column if grp[c] == "aa_composition"]
         + [c for c in cols.column if grp[c] == "dipeptide"]
         + [c for c in cols.column if grp[c] == "physicochemical"])

pb_seqs = set(s.upper() for s in pairs.sequence.dropna())
dup = ov.binder_seq.str.upper().isin(pb_seqs).values
print("overath %d rows -> de-dup removes %d (%.1f%%) -> %d kept, %d binders"
      % (len(ov), dup.sum(), 100 * dup.mean(), (~dup).sum(), ov.label[~dup].sum()))
ovd = ov[~dup].reset_index(drop=True)


def seqfeat(s):
    f = {}
    f.update(_ns["aa_composition"](s))
    f.update(_ns["dipeptide_composition"](s))
    f.update(_ns["physicochemical"](s))
    return f


OVF = pd.DataFrame([seqfeat(s.upper()) for s in ovd.binder_seq])[FEATS]

LGB = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
           subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
           is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
           force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)


def boot_ci(yy, pp, fn):
    rng = np.random.RandomState(SEED)
    n, out = len(yy), []
    for _ in range(N_BOOT):
        i = rng.randint(0, n, n)
        if len(np.unique(yy[i])) > 1:
            out.append(fn(yy[i], pp[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


rows = []
for t in TARGETS:
    trm = (fm.target == t).values
    sel = (ovd.target == t).values
    m = lgb.LGBMClassifier(**LGB)
    m.fit(fm.loc[trm, FEATS], fm.binding_label.values[trm].astype(int))
    p = m.predict_proba(OVF.loc[sel, FEATS])[:, 1]
    yy = ovd.label.values[sel].astype(int)
    a, ap = roc_auc_score(yy, p), average_precision_score(yy, p)
    alo, ahi = boot_ci(yy, p, roc_auc_score)
    rows.append({"target": t, "pb_train_n": int(trm.sum()), "ov_test_n": int(sel.sum()),
                 "ov_binders": int(yy.sum()), "xf_auroc": round(a, 3),
                 "auroc_ci_lo": round(alo, 3), "auroc_ci_hi": round(ahi, 3),
                 "xf_auprc": round(ap, 3), "base_rate": round(yy.mean(), 3),
                 "n_distinct_preds": int(len(np.unique(np.round(p, 9)))),
                 "stored_auroc": STORED[t]})

df = pd.DataFrame(rows)
print()
print(df.to_string(index=False))
print("\nmean AUROC (reconstructed) %.4f   mean AUROC (stored) %.4f   mean |deviation| %.4f"
      % (df.xf_auroc.mean(), df.stored_auroc.mean(), (df.xf_auroc - df.stored_auroc).abs().mean()))
print("reproduces stored per-target values: NO  (see module docstring)")
print("every target's CI includes 0.5: %s"
      % all(r.auroc_ci_lo <= 0.5 <= r.auroc_ci_hi for r in df.itertuples()))

out = os.path.join(ROOT, "outputs/r2_crossdataset_overath_reconstructed.csv")
df.to_csv(out, index=False)
print("saved %s" % out)
