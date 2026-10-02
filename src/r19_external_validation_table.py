#!/usr/bin/env python3
"""External-validation reconciliation (Reviewer 1, comment 4): every row with its n, AUROC
and AUPRC, regenerated from the released data.

Covers
  A  Overath benchmark, all targets: 8 folding-derived scores (zero-shot, single-feature)
     and a sequence model trained by 5-fold CV WITHIN Overath (supervised). Same rows and labels.
  B  Overath rows on the 5 targets shared with Proteinbase, before/after exact-sequence
     de-duplication: base-rate null (no candidate information), pooled.
  C  SKEMPI 2.0 single mutations, GroupKFold by complex (a model fitted on SKEMPI, NOT a
     transfer from Proteinbase).

AUROC values for A and C are checked against outputs/r8_benchmark.csv and r8_skempi.csv.
Cross-dataset transfer models themselves come from r2_crossdataset_overath.py and
r8_crossdata_full.py.

    pixi run -e ml python src/r19_external_validation_table.py
"""
import os, re, json, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
out = {}

# ---------------------------------------------------------------- A: Overath benchmark
d = pd.read_csv("data/external/overath_prepared_training_dataset.csv", low_memory=False)
y = d.binder.values.astype(int)
mets = ["colab_iptm_avg", "boltz1_iptm_avg", "af3_iptm_avg", "colab_actifptm_avg",
        "boltz1_ipSAE_avg", "af3_ipSAE_avg", "af2_pae_interaction", "boltz1_complex_iplddt_avg"]
A = []
for m in mets:
    v = pd.to_numeric(d[m], errors="coerce").values
    ok = ~np.isnan(v)
    a = roc_auc_score(y[ok], v[ok])
    sign = 1 if a >= 0.5 else -1                    # orientation, as in r8_benchmark.py
    A.append({"method": m, "n": int(ok.sum()), "binders": int(y[ok].sum()),
              "auroc": max(a, 1 - a), "auprc": average_precision_score(y[ok], sign * v[ok])})

AAs = "ACDEFGHIKLMNPQRSTVWY"; AID = {a: i for i, a in enumerate(AAs)}
aa3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

def parse(ss):
    if not isinstance(ss, str): return None
    r = {}
    for tok in ss.split(":"):
        mm = re.match(r'([A-Z]{3})_(\d+)=', tok)
        if mm: r[int(mm.group(2))] = aa3.get(mm.group(1), 'X')
    return "".join(r[i] for i in sorted(r)) if r else None

def feat(s):
    s = "".join(c for c in str(s).upper() if c in AAs); n = max(len(s), 1)
    aac = np.zeros(20); dpc = np.zeros(400)
    for c in s: aac[AID[c]] += 1
    for i in range(len(s) - 1): dpc[AID[s[i]] * 20 + AID[s[i + 1]]] += 1
    return np.concatenate([aac / n, dpc / max(n - 1, 1), [n]])

seqs = d.input_pymol_binder_ss.map(parse)
ok = seqs.notna().values
X = np.stack([feat(s) for s in seqs[ok]]); yo = y[ok]
P = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
         is_unbalance=True, verbosity=-1, n_jobs=1, deterministic=True, force_row_wise=True)
oof = np.zeros(len(yo)); aucs = []
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, yo):
    m = lgb.LGBMClassifier(**P).fit(X[a], yo[a])
    oof[b] = m.predict_proba(X[b])[:, 1]; aucs.append(roc_auc_score(yo[b], oof[b]))
A.append({"method": "sequence_ML_5foldCV", "n": int(ok.sum()), "binders": int(yo.sum()),
          "auroc": float(np.mean(aucs)), "auprc": average_precision_score(yo, oof)})
base_rate_A = float(yo.mean())
stored = pd.read_csv("outputs/r8_benchmark.csv").set_index("method").auroc
print("A. Overath benchmark, n=%d rows, base rate %.3f" % (len(d), base_rate_A))
for r in A:
    print("   %-28s n=%d  AUROC %.3f (stored %.3f)  AUPRC %.3f"
          % (r["method"], r["n"], r["auroc"], stored.get(r["method"], float("nan")), r["auprc"]))
out["A"] = A; out["A_base_rate"] = base_rate_A

# ---------------------------------------------------------------- B: base-rate null
ov = pd.read_csv("data/external/overath_clean.csv")
pairs = pd.read_parquet("data/pairs_with_splits.parquet")
dup = ov.binder_seq.str.upper().isin(set(s.upper() for s in pairs.sequence.dropna())).values
B = {}
for name, sub in [("with_overlap", ov), ("de_duplicated", ov[~dup])]:
    yy = sub.label.values.astype(int)
    null = sub.target.map(sub.groupby("target").label.mean()).values   # each row gets its target's binding rate
    B[name] = {"n": int(len(sub)), "binders": int(yy.sum()), "targets": int(sub.target.nunique()),
               "base_rate": float(yy.mean()),
               "null_auroc": roc_auc_score(yy, null), "null_auprc": average_precision_score(yy, null)}
    print("B. %-14s n=%d binders=%d  pooled AUROC of a target-binding-rate-only score %.3f, AUPRC %.3f"
          % (name, len(sub), yy.sum(), B[name]["null_auroc"], B[name]["null_auprc"]))
rates = ov[~dup].groupby("target").label.mean().round(3).to_dict()
print("   de-duplicated per-target binding rates:", rates)
out["B"] = B; out["B_target_rates"] = rates

# ---------------------------------------------------------------- C: SKEMPI
sk = pd.read_csv("data/external/skempi_v2.csv", sep=";", low_memory=False)
KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
      'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
VOL = {'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2,
       'I': 166.7, 'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8,
       'Y': 193.6, 'V': 140.0}
CHG = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}
aw = pd.to_numeric(sk['Affinity_wt_parsed'], errors='coerce').values
am = pd.to_numeric(sk['Affinity_mut_parsed'], errors='coerce').values
mut = sk['Mutation(s)_cleaned'].astype(str).values
loc = sk['iMutation_Location(s)'].astype(str).values
pdbv = sk['#Pdb'].astype(str).values
rows = []
for i in range(len(sk)):
    if mut[i].count(",") > 0: continue
    mm = re.match(r'^([A-Z])[A-Z]?(\d+)([A-Z])$', mut[i])
    if not mm: continue
    w, mt = mm.group(1), mm.group(3)
    if w not in KD or mt not in KD or np.isnan(am[i]) or np.isnan(aw[i]): continue
    ab = 1 if am[i] > 1e-5 else (0 if am[i] < 1e-7 else None)
    if ab is None: continue
    rows.append(dict(pdb=pdbv[i].split("_")[0], y=ab, d_hydro=KD[mt] - KD[w], d_vol=VOL[mt] - VOL[w],
                     d_chg=CHG.get(mt, 0) - CHG.get(w, 0), abs_dhydro=abs(KD[mt] - KD[w]),
                     abs_dvol=abs(VOL[mt] - VOL[w]), to_gly_pro=int(mt in "GP"),
                     from_hydrophobic=int(w in "AILMFWV"), core=int("COR" in loc[i]),
                     rim=int("RIM" in loc[i]), support=int("SUP" in loc[i])))
df = pd.DataFrame(rows)
fe = ["d_hydro", "d_vol", "d_chg", "abs_dhydro", "abs_dvol", "to_gly_pro", "from_hydrophobic", "core", "rim", "support"]
Xs, ys, gs = df[fe].values, df.y.values, df.pdb.values
oofs = np.zeros(len(ys)); fa = []
for a, b in GroupKFold(5).split(Xs, ys, gs):
    m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15,
                           is_unbalance=True, verbosity=-1, n_jobs=1, deterministic=True, force_row_wise=True)
    m.fit(Xs[a], ys[a]); oofs[b] = m.predict_proba(Xs[b])[:, 1]; fa.append(roc_auc_score(ys[b], oofs[b]))
C = {"n": int(len(ys)), "abolishing": int(ys.sum()), "complexes": int(df.pdb.nunique()),
     "auroc": float(np.mean(fa)), "auprc": average_precision_score(ys, oofs), "base_rate": float(ys.mean())}
print("C. SKEMPI single mutations n=%d (abolishing %d, %.2f) complexes=%d  AUROC %.3f (stored 0.692)  AUPRC %.3f"
      % (C["n"], C["abolishing"], C["base_rate"], C["complexes"], C["auroc"], C["auprc"]))
out["C"] = C

json.dump(out, open("outputs/r19_external_validation_table.json", "w"), indent=2)
print("\nsaved outputs/r19_external_validation_table.json")
