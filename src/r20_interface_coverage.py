#!/usr/bin/env python3
"""Can a novel-interface (binding-site) split be run on these data? (Reviewer 2, comment 2)

The evaluations table carries Boltz-2-predicted interface residues for each modelled pair, as
{chain, residue} records. Chain B is the binder (indices fit within the binder sequence in every
record); chain A is the target. This script answers three questions from that annotation:

  1. Coverage: how many pairs and targets carry an annotation at all?
  2. For the one target with enough annotated pairs (nipah glycoprotein G), do the binders
     contact distinct sites on the target, or one site?
  3. If sites existed, how many binders would a held-out-site test set contain?

Epitope = the set of chain-A (target) residues a binder contacts. Distance = 1 - Jaccard.
Average-linkage clustering, cut at several thresholds.

    pixi run -e ml python src/r20_interface_coverage.py
"""
import os, json, collections
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ev = pd.read_parquet(os.path.join(ROOT, "data/evaluations_flat.parquet"))
pairs = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
ir = ev[ev.metric == "interface_residues"].copy()

def parse(v):
    try:
        return json.loads(v) if isinstance(v, str) else v
    except Exception:
        return []

ir["recs"] = ir.value.apply(parse)
ir["target_side"] = ir.recs.apply(lambda r: frozenset(int(x["residue"]) for x in r if x.get("chain") == "A" and "residue" in x))
ir["binder_side"] = ir.recs.apply(lambda r: [int(x["residue"]) for x in r if x.get("chain") == "B" and "residue" in x])
ir = ir[(ir.target_side.apply(len) > 0) & (ir.binder_side.apply(len) > 0)]
have = set(zip(ir.protein_id, ir.target))
pairs["annotated"] = [(p, t) in have for p, t in zip(pairs.protein_id, pairs.target)]

# ---- 1. coverage
cov = (pairs.groupby("target").agg(pairs=("annotated", "size"), annotated=("annotated", "sum"),
                                    binders=("binding_label", "sum"),
                                    annotated_binders=("binding_label", lambda s: int(s[pairs.loc[s.index, "annotated"]].sum())))
       .reset_index().sort_values("annotated", ascending=False))
cov["coverage_pct"] = (100 * cov.annotated / cov.pairs).round(1)
tot_a, tot_n = int(pairs.annotated.sum()), len(pairs)
print("modelled pairs %d; with interface annotation %d (%.1f%%); targets with any: %d of %d"
      % (tot_n, tot_a, 100 * tot_a / tot_n, int((cov.annotated > 0).sum()), len(cov)))
print(cov.to_string(index=False))
cov.to_csv(os.path.join(ROOT, "outputs/r20_interface_coverage.csv"), index=False)

# ---- 2/3. epitope structure on nipah
TGT = "nipah-glycoprotein-g"
sub = pairs[(pairs.target == TGT) & pairs.annotated].copy()
ep = {r.protein_id: r.target_side for r in ir[ir.target == TGT].itertuples()}
sub = sub[sub.protein_id.isin(ep)]
ids = list(sub.protein_id); S = [ep[i] for i in ids]; n = len(S)
y = sub.binding_label.values.astype(int)
share = n / tot_a
print("\nnipah annotated pairs %d (%.1f%% of all annotated), binders %d" % (n, 100 * share, y.sum()))

D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        u = len(S[i] & S[j]); v = len(S[i] | S[j]); D[i, j] = D[j, i] = 1 - (u / v if v else 0)
Z = linkage(squareform(D, checks=False), method="average")

freq = collections.Counter()
for s in S: freq.update(s)
n_res, n_half, maxf = len(freq), sum(1 for c in freq.values() if c / n >= 0.5), max(freq.values()) / n
print("distinct target residues contacted %d; contacted by >=50%% of binders %d; max frequency %.2f"
      % (n_res, n_half, maxf))

rows = []
for cut in [0.5, 0.6, 0.7, 0.8, 0.9]:
    cl = fcluster(Z, cut, criterion="distance"); cnt = collections.Counter(cl)
    order = [c for c, _ in cnt.most_common()]
    big = order[0]
    others = order[1:]
    sec = max(others, key=lambda c: cnt[c]) if others else None
    max_b_other = max((int(y[cl == c].sum()) for c in others), default=0)
    usable = [c for c in cnt if cnt[c] >= 20 and int(y[cl == c].sum()) >= 5]
    rows.append({"cut": cut, "clusters": len(cnt), "largest_pairs": cnt[big], "largest_binders": int(y[cl == big].sum()),
                 "secondary_pairs": cnt[sec] if sec else 0, "secondary_binders": int(y[cl == sec].sum()) if sec else 0,
                 "max_binders_any_other_cluster": max_b_other, "usable_clusters": len(usable)})
    if cut == 0.7:
        def consensus(c):
            mem = [S[i] for i in range(n) if cl[i] == c]; f = collections.Counter()
            for m in mem: f.update(m)
            return {r for r, k in f.items() if k / len(mem) >= 0.5}
        c1, c2 = consensus(big), consensus(sec)
        inter = len(c1 & c2)
        cons = {"cut": 0.7, "dominant_consensus": len(c1), "secondary_consensus": len(c2),
                "shared": inter, "jaccard": round(inter / len(c1 | c2), 3),
                "dominant_residues_inside_secondary": inter}
ep_df = pd.DataFrame(rows)
print("\n", ep_df.to_string(index=False))
print("\nconsensus epitopes at cut 0.7:", cons)
ep_df.to_csv(os.path.join(ROOT, "outputs/r20_epitope_clusters.csv"), index=False)
json.dump({"annotated_pairs": tot_a, "modelled_pairs": tot_n, "targets_with_annotation": int((cov.annotated > 0).sum()),
           "targets_total": int(len(cov)), "nipah_annotated": int(n), "nipah_share": round(share, 4),
           "nipah_binders": int(y.sum()), "distinct_residues": n_res, "residues_at_least_half": n_half,
           "max_residue_frequency": round(maxf, 3), "consensus_cut_0.7": cons},
          open(os.path.join(ROOT, "outputs/r20_epitope_summary.json"), "w"), indent=2)
print("\nsaved outputs/r20_interface_coverage.csv, r20_epitope_clusters.csv, r20_epitope_summary.json")
