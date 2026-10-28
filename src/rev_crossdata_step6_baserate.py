#!/usr/bin/env python3
"""Is the pooled cross-dataset AUROC driven by target identity rather than by
discriminating binders within a target?

The de-duplicated Overath targets have very different binding base rates
(egfr 0.089 ... mdm2 0.564). A pooled AUROC rewards ranking any mdm2 candidate
above any egfr candidate, which requires no binder-level signal at all.

Control: score every candidate by its TARGET'S base rate only -- a predictor with
zero within-target information -- and compute the pooled AUROC.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pairs = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
ov = pd.read_csv(os.path.join(ROOT, "data/external/overath_clean.csv"))
dup = ov.binder_seq.str.upper().isin(set(s.upper() for s in pairs.sequence.dropna())).values
d = ov[~dup].reset_index(drop=True)
y = d.label.values.astype(int)

br = d.groupby("target").label.mean()
print("de-duplicated Overath base rates by target:")
for t, v in br.sort_values().items():
    n = int((d.target == t).sum())
    print("    %-24s n=%3d  base rate %.3f" % (t, n, v))

# oracle base-rate predictor: zero within-target information
p_oracle = d.target.map(br).values
print("\npooled AUROC from TARGET BASE RATE ALONE : %.3f" % roc_auc_score(y, p_oracle))

# Proteinbase-derived base rate, i.e. what a model could actually learn in advance
fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
pb_br = fm.groupby("target").binding_label.mean()
p_pb = d.target.map(pb_br).fillna(fm.binding_label.mean()).values
print("pooled AUROC from PROTEINBASE base rate  : %.3f" % roc_auc_score(y, p_pb))

print("\nfull-model pooled (de-dup, primary config): 0.644   [from r8_crossdata_full.py]")
print("per-target full-model values              : 0.475 - 0.575 (all ~chance)")
print("\nInterpretation: pooled AUROC above chance alongside per-target AUROC at chance")
print("is what a base-rate effect looks like. Pooled figures across targets with")
print("unequal base rates should not be read as evidence of transfer.")
