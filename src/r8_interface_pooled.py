#!/usr/bin/env python3
"""Interface-pooled vs whole-sequence binder representation on nipah glycoprotein G.

Reconstruction of the experiment behind outputs/r8_interface_pooled.json.

PROTOCOL (all degrees of freedom stated explicitly; two of these were previously
undocumented and the sign of the effect depends on the first):

  prototype rows    : split in {train, val}, RESTRICTED to interface-annotated rows
  evaluation rows   : split == test, RESTRICTED to interface-annotated rows  (n = 153)
  score             : proto_ratio = cos(e, p+) / cos(e, p-)
  embedding         : ESM-2 esm2_t33_650M_UR50D, layer 33
  whole pooling     : mean over residue positions 1..L  (BOS/EOS EXCLUDED)
  interface pooling : mean over binder interface positions only
  interface source  : data/external/binder_iface_residues.json
                      chain-B (binder) filtered; verified 3816/3816 against
                      data/evaluations_flat.parquet, zero unfiltered entries

Because the `gpu` pixi environment carries only torch/fair-esm/numpy (no pandas),
the run is staged. Do NOT add pandas to that feature: it would force a relock and
break the tie to the versions behind the submitted results.

    pixi run -e ml  python src/r8_interface_pooled.py export
    pixi run -e gpu python src/r8_interface_pooled.py embed
    pixi run -e ml  python src/r8_interface_pooled.py score

Writes outputs/r8_interface_pooled_reconstructed.json.
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TGT = "nipah-glycoprotein-g"
LAYER = 33
EPS = 1e-8
N_BOOT = 5000
SEED = 42

SEQS = os.path.join(ROOT, "outputs/r8_interface_pooled_seqs.json")
POOLED = os.path.join(ROOT, "outputs/r8_interface_pooled_embeddings.npz")
RESULT = os.path.join(ROOT, "outputs/r8_interface_pooled_reconstructed.json")


def stage_export():
    """ml env: nipah sequences + binder interface positions -> JSON for the gpu stage."""
    import pandas as pd
    fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
    pairs = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
    seqmap = dict(zip(pairs.protein_id, pairs.sequence))
    iface = json.load(open(os.path.join(ROOT, "data/external/binder_iface_residues.json")))
    out = {}
    for u in sorted({str(p) for p in fm[fm.target == TGT].protein_id}):
        s = seqmap.get(u)
        if isinstance(s, str) and s:
            out[u] = {"seq": s.upper(), "iface": [int(p) for p in iface.get(u, [])]}
    json.dump(out, open(SEQS, "w"))
    print("exported %d sequences (%d with interface) -> %s"
          % (len(out), sum(1 for v in out.values() if v["iface"]), SEQS))


def stage_embed():
    """gpu env: per-residue ESM-2, pooled whole and interface."""
    import numpy as np, torch, esm
    data = json.load(open(SEQS))
    uniq = sorted(data)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s %s" % (dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else ""))
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(dev).eval()
    bc = alphabet.get_batch_converter()
    W, I = {}, {}
    with torch.no_grad():
        for s in range(0, len(uniq), 8):
            chunk = uniq[s:s + 8]
            _, _, toks = bc([(u, data[u]["seq"]) for u in chunk])
            rep = model(toks.to(dev), repr_layers=[LAYER])["representations"][LAYER]
            for bi, u in enumerate(chunk):
                L = len(data[u]["seq"])
                res = rep[bi].float()[1:L + 1].cpu().numpy()
                W[u] = res.mean(0)
                pos = [p for p in data[u]["iface"] if 1 <= p <= L]
                if pos:
                    I[u] = res[[p - 1 for p in pos]].mean(0)
            if (s // 8) % 25 == 0:
                print("  %d/%d" % (s, len(uniq)), flush=True)
    wk, ik = list(W), list(I)
    np.savez_compressed(POOLED,
                        ids=np.array(wk, dtype=object), whole=np.stack([W[u] for u in wk]),
                        iface_ids=np.array(ik, dtype=object), iface=np.stack([I[u] for u in ik]))
    print("pooled whole %d  interface %d -> %s" % (len(wk), len(ik), POOLED))


def stage_score():
    """ml env: proto_ratio AUROC for both arms + paired bootstrap CI on the delta."""
    import numpy as np, pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score

    fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
    sub = fm[fm.target == TGT].reset_index(drop=True)
    y = sub.binding_label.values.astype(int)
    split = sub.split.values
    pid = [str(p) for p in sub.protein_id]

    z = np.load(POOLED, allow_pickle=True)
    W = {str(u): v.astype(np.float64) for u, v in zip(z["ids"], z["whole"])}
    I = {str(u): v.astype(np.float64) for u, v in zip(z["iface_ids"], z["iface"])}

    has_if = np.array([u in I for u in pid])
    proto_m = np.isin(split, ["train", "val"]) & has_if     # protocol, see module docstring
    eval_m = (split == "test") & has_if

    def scores(vecs):
        P = np.stack([vecs[u] for u, k in zip(pid, proto_m) if k])
        ytr = y[proto_m]
        pos, neg = P[ytr == 1].mean(0), P[ytr == 0].mean(0)
        E = np.stack([vecs[u] for u, k in zip(pid, eval_m) if k])
        nei = np.linalg.norm(E, axis=1)
        cp = (E @ pos) / (nei * np.linalg.norm(pos) + EPS)
        cn = (E @ neg) / (nei * np.linalg.norm(neg) + EPS)
        return cp / (cn + EPS)

    yev = y[eval_m]
    sw, si = scores(W), scores(I)
    aw, ai = roc_auc_score(yev, sw), roc_auc_score(yev, si)
    pw, pi = average_precision_score(yev, sw), average_precision_score(yev, si)

    rng = np.random.RandomState(SEED)
    n = len(yev)
    d = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(yev[idx])) < 2:
            continue
        d.append(roc_auc_score(yev[idx], si[idx]) - roc_auc_score(yev[idx], sw[idx]))
    d = np.array(d)
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))

    print("prototype rows %d   evaluation rows %d (binders %d)" % (proto_m.sum(), n, yev.sum()))
    print("  whole      AUROC %.4f   AUPRC %.4f" % (aw, pw))
    print("  interface  AUROC %.4f   AUPRC %.4f" % (ai, pi))
    print("  delta      %+.4f   95%% CI [%+.4f, %+.4f]   P(delta>0)=%.3f"
          % (ai - aw, lo, hi, float((d > 0).mean())))
    print("  significant at 95%%: %s" % ("YES" if lo > 0 or hi < 0 else "NO"))

    res = {"nipah_whole": round(aw, 3), "nipah_interface": round(ai, 3), "n_test": int(n),
           "nipah_whole_auprc": round(pw, 4), "nipah_interface_auprc": round(pi, 4),
           "delta": round(ai - aw, 4), "delta_ci95": [round(lo, 4), round(hi, 4)],
           "p_delta_gt_0": round(float((d > 0).mean()), 3),
           "n_binders_eval": int(yev.sum()), "n_prototype_rows": int(proto_m.sum())}
    json.dump(res, open(RESULT, "w"), indent=2)
    print("saved %s" % RESULT)

    prior = os.path.join(ROOT, "outputs/r8_interface_pooled.json")
    if os.path.exists(prior):
        old = json.load(open(prior))
        same = all(res[k] == old[k] for k in ("nipah_whole", "nipah_interface", "n_test"))
        print("\nstored : %s" % old)
        print("matches stored headline values: %s" % ("YES" if same else "NO"))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"export": stage_export, "embed": stage_embed, "score": stage_score}.get(stage)
    if fn is None:
        sys.exit(__doc__)
    fn()
