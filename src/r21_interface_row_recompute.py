#!/usr/bin/env python3
"""Recompute the "LightGBM + interface" row of Table 2 with and without the chain filter.

phase6a_interface.py parsed Boltz-2 interface positions as
    [int(r["residue"]) for r in data if "residue" in r]
with no chain filter, then indexed them into the BINDER sequence. Chain A is the target, so target
residues numbered below the binder length were treated as binder interface residues (mean 14.8%
of kept positions). This script runs the same model twice:

  BUGGY  positions from both chains   (what the published row used)
  FIXED  positions from chain B only  (the binder)

with the interface_features() function copied verbatim from phase6a_interface.py, the same
463 base features plus the 39 interface features, the same hyperparameters and the same
train / validation / test split. The original used the GPU build of LightGBM; this runs on CPU,
so the BUGGY run is also a check that the published value is reproducible at all.

    pixi run -e ml python src/r21_interface_row_recompute.py
"""
import os, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- verbatim from phase6a_interface.py ------------------------------------------------------
KD_HYDRO = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5}
CHARGE = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}
AROMATIC = set('FWY'); HBOND_D = set('NQSTKRHWY'); HBOND_A = set('NQDEST')
VOLUME = {'G': 60, 'A': 89, 'S': 96, 'P': 112, 'V': 117, 'T': 116, 'C': 114, 'I': 166, 'L': 166, 'N': 114, 'D': 111,
          'Q': 144, 'K': 168, 'E': 138, 'M': 162, 'H': 153, 'F': 190, 'R': 173, 'Y': 194, 'W': 228}
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

def interface_features(sequence, positions):
    seq = sequence.upper(); n = len(seq)
    if not positions or n == 0: return {}
    pos_0 = [p - 1 for p in positions if 0 <= p - 1 < n]
    if not pos_0: return {}
    iface = [seq[p] for p in pos_0]; n_if = len(iface); feat = {}
    cnt = {aa: 0 for aa in AA20}
    for aa in iface:
        if aa in cnt: cnt[aa] += 1
    for aa in AA20: feat[f"if_aac_{aa}"] = cnt[aa] / n_if
    feat["if_n_residues"] = n_if; feat["if_coverage"] = n_if / n
    if len(pos_0) > 1:
        gaps = [pos_0[i + 1] - pos_0[i] for i in range(len(pos_0) - 1)]
        feat["if_span"] = (max(pos_0) - min(pos_0)) / n; feat["if_mean_gap"] = np.mean(gaps) / n
        feat["if_max_gap"] = max(gaps) / n; feat["if_n_segments"] = sum(1 for g in gaps if g > 3)
    else:
        feat["if_span"] = feat["if_mean_gap"] = feat["if_max_gap"] = 0.0; feat["if_n_segments"] = 0.0
    feat["if_nterm_frac"] = sum(1 for p in pos_0 if p < n * 0.33) / n_if
    feat["if_cterm_frac"] = sum(1 for p in pos_0 if p > n * 0.67) / n_if
    hydro = [KD_HYDRO.get(aa, 0.0) for aa in iface]; charge = [CHARGE.get(aa, 0.0) for aa in iface]
    vol = [VOLUME.get(aa, 130.0) for aa in iface]
    feat["if_mean_hydro"] = np.mean(hydro); feat["if_std_hydro"] = np.std(hydro)
    feat["if_net_charge"] = sum(charge); feat["if_mean_charge"] = np.mean(charge)
    feat["if_pos_frac"] = sum(1 for c in charge if c > 0) / n_if; feat["if_neg_frac"] = sum(1 for c in charge if c < 0) / n_if
    feat["if_aromatic_frac"] = sum(1 for aa in iface if aa in AROMATIC) / n_if
    feat["if_hbond_donor_frac"] = sum(1 for aa in iface if aa in HBOND_D) / n_if
    feat["if_hbond_acc_frac"] = sum(1 for aa in iface if aa in HBOND_A) / n_if
    feat["if_mean_volume"] = np.mean(vol)
    feat["if_hydro_delta"] = feat["if_mean_hydro"] - np.mean([KD_HYDRO.get(aa, 0.0) for aa in seq])
    return feat
# ------------------------------------------------------------------------------------------------

pairs = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
base_cols = list(pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))["column"])
ev = pd.read_parquet(os.path.join(ROOT, "data/evaluations_flat.parquet"))
ir = ev[ev.metric == "interface_residues"].copy()

def recs(v):
    try:
        d = json.loads(v) if isinstance(v, str) else v
        return d if isinstance(d, list) else []
    except Exception:
        return []

ir["recs"] = ir.value.apply(recs)
seq_map = dict(zip(pairs.protein_id, pairs.sequence))

def build(chain_filter):
    pos_map = {}
    for r in ir.itertuples():
        pos = [int(x["residue"]) for x in r.recs if "residue" in x and (not chain_filter or x.get("chain") == "B")]
        if pos: pos_map[(r.protein_id, r.target)] = pos
    rows = []
    for r in fm.itertuples():
        pos = pos_map.get((r.protein_id, r.target), []); s = seq_map.get(r.protein_id, "")
        rows.append(interface_features(s, pos) if pos and s else {})
    df = pd.DataFrame(rows); cols = [c for c in df.columns if c.startswith("if_")]
    df = df[cols]
    med = df.median()
    return df.fillna(med), cols, int(df.notna().all(axis=1).sum())

y = fm.binding_label.values.astype(int); split = fm.split.values
tr, va, te = split == "train", split == "val", split == "test"

def fit_eval(X, seed):
    prm = dict(objective="binary", metric="auc", verbosity=-1, n_estimators=1000, learning_rate=0.05, num_leaves=63,
               subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=10,
               random_state=seed, n_jobs=1, num_threads=1, deterministic=True, force_row_wise=True)
    m = lgb.LGBMClassifier(**prm)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(X[te])[:, 1]; b = (p >= 0.5).astype(int)
    return (roc_auc_score(y[te], p), average_precision_score(y[te], p), f1_score(y[te], b), matthews_corrcoef(y[te], b))

SEEDS = [42, 1, 7, 2024, 99]
res = {}
Xbase = fm[base_cols].values.astype(np.float32)
runs = {"463 base only (no interface)": Xbase}
for name, cf in [("BUGGY parse (both chains)", False), ("FIXED parse (binder chain only)", True)]:
    dfi, cols, n_full = build(cf)
    runs[name] = np.hstack([Xbase, dfi[cols].values.astype(np.float32)])
    print("%s: %d interface features, %d of %d pairs have a real (non-imputed) annotation"
          % (name, len(cols), n_full, len(fm)))

print("\n%-34s %s" % ("configuration", "seed-42 run: AUROC AUPRC  F1    MCC   |  5-seed mean AUROC +/- SD"))
for name, X in runs.items():
    allr = np.array([fit_eval(X, s) for s in SEEDS])
    r42 = allr[0]
    res[name] = {"seed42": [round(float(v), 4) for v in r42], "mean": [round(float(v), 4) for v in allr.mean(0)],
                 "sd": [round(float(v), 4) for v in allr.std(0)]}
    print("%-34s  %.3f  %.3f  %.3f %.3f  |  %.3f +/- %.3f" % (name, r42[0], r42[1], r42[2], r42[3], allr[:, 0].mean(), allr[:, 0].std()))

print("\npublished Table 2 row: AUROC 0.894  AUPRC 0.702  F1 0.624  MCC 0.588")
json.dump(res, open(os.path.join(ROOT, "outputs/r21_interface_row_recompute.json"), "w"), indent=2)
print("saved outputs/r21_interface_row_recompute.json")
