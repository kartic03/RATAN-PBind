#!/usr/bin/env python3
"""Cross-dataset transfer, full model (prototype-augmented): Proteinbase -> Overath.

Reconstruction of the experiment behind outputs/r8_crossdata_full.json
(full_all 0.728, full_dedup 0.620, n_dedup 571).

PROTOCOL:
  training        : one model on ALL Proteinbase pairs
  prototypes      : per target, from Proteinbase rows; applied to Overath candidates
  de-duplication  : drop Overath rows matching ANY Proteinbase sequence (913 -> 571)
  embedding       : ESM-2 esm2_t33_650M_UR50D layer 33, mean over residues 1..L,
                    BOS/EOS excluded (convention verified bit-exact against
                    features/esm2_embeddings.npy)
  metrics         : pooled AUROC with and without de-duplication, per-target AUROC
                    after de-duplication, AUPRC throughout, bootstrap 95% CIs

Four feature compositions are reported because the released artifacts never state
which one "full model" denotes. PRIMARY is seq(427)+proto(7): it is closest to the
stored pair on both metrics. None reproduces the stored values exactly.

KNOWN DISCREPANCY: stored 0.728 / 0.620; primary reconstruction 0.737 / 0.644.
What is robust across all four compositions is the de-duplication effect itself:
pooled AUROC 0.72-0.77 with overlap, 0.48-0.64 after removing it.

    pixi run -e ml  python src/r8_crossdata_full.py export
    pixi run -e gpu python src/r8_crossdata_full.py embed
    pixi run -e ml  python src/r8_crossdata_full.py score
"""
import os, sys, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQS = os.path.join(ROOT, "outputs/r8_crossdata_overath_seqs.json")
EMB = os.path.join(ROOT, "outputs/r8_crossdata_overath_esm2.npz")
RESULT = os.path.join(ROOT, "outputs/r8_crossdata_full_reconstructed.json")
LAYER, SEED, N_BOOT, EPS = 33, 42, 2000, 1e-8
PRIMARY = "seq(427)+proto(7)"


def stage_export():
    import pandas as pd
    ov = pd.read_csv(os.path.join(ROOT, "data/external/overath_clean.csv"))
    json.dump({str(b): str(s).upper() for b, s in zip(ov.binder_id, ov.binder_seq)}, open(SEQS, "w"))
    print("exported %d overath sequences -> %s" % (len(ov), SEQS))


def stage_embed():
    import numpy as np, torch, esm
    d = json.load(open(SEQS))
    keys = sorted(d)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % dev)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(dev).eval()
    bc = alphabet.get_batch_converter()
    out = {}
    with torch.no_grad():
        for s in range(0, len(keys), 8):
            ch = keys[s:s + 8]
            _, _, toks = bc([(k, d[k]) for k in ch])
            rep = model(toks.to(dev), repr_layers=[LAYER])["representations"][LAYER]
            for bi, k in enumerate(ch):
                out[k] = rep[bi].float()[1:len(d[k]) + 1].mean(0).cpu().numpy()
            if (s // 8) % 20 == 0:
                print("  %d/%d" % (s, len(keys)), flush=True)
    ks = list(out)
    np.savez_compressed(EMB, ids=np.array(ks, dtype=object), emb=np.stack([out[k] for k in ks]))
    print("embedded %d -> %s" % (len(ks), EMB))


def stage_score():
    import numpy as np, pandas as pd, lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    SRC = os.path.join(ROOT, "src/phase2_features.py")
    txt = open(SRC).read()
    ns = {"__file__": SRC, "__name__": "_phase2_defs"}
    exec(compile(txt[:txt.index("# ── Load data")], SRC, "exec"), ns)

    fm = pd.read_parquet(os.path.join(ROOT, "features/feature_matrix.parquet"))
    cols = pd.read_csv(os.path.join(ROOT, "features/feature_columns.csv"))
    pairs = pd.read_parquet(os.path.join(ROOT, "data/pairs_with_splits.parquet"))
    ov = pd.read_csv(os.path.join(ROOT, "data/external/overath_clean.csv"))
    grp = dict(zip(cols.column, cols.group))
    SEQF = ([c for c in cols.column if grp[c] == "aa_composition"]
            + [c for c in cols.column if grp[c] == "dipeptide"]
            + [c for c in cols.column if grp[c] == "physicochemical"])

    emb = np.load(os.path.join(ROOT, "features/esm2_embeddings.npy"))
    pid = np.load(os.path.join(ROOT, "features/esm2_protein_ids.npy"), allow_pickle=True)
    p2r = {str(p): i for i, p in enumerate(pid)}
    E_pb = np.stack([emb[p2r[str(p)]] for p in fm.protein_id]).astype(np.float64)
    z = np.load(EMB, allow_pickle=True)
    om = {str(k): v.astype(np.float64) for k, v in zip(z["ids"], z["emb"])}
    E_ov = np.stack([om[str(b)] for b in ov.binder_id])

    y = fm.binding_label.values.astype(int)
    tgt = fm.target.values
    dup = ov.binder_seq.str.upper().isin(set(s.upper() for s in pairs.sequence.dropna())).values

    protos = {}
    for t in np.unique(tgt):
        m = tgt == t
        pos, neg = E_pb[m & (y == 1)], E_pb[m & (y == 0)]
        protos[t] = (pos.mean(0), neg.mean(0), len(pos), len(neg)) if len(pos) and len(neg) else None

    def p7(E, tl):
        out = np.zeros((len(E), 7))
        for i, t in enumerate(tl):
            pr = protos.get(t)
            if pr is None:
                continue
            pp, pn, a, b = pr
            e = E[i]; ne = np.linalg.norm(e)
            cp = float(e @ pp) / (ne * np.linalg.norm(pp) + EPS)
            cn = float(e @ pn) / (ne * np.linalg.norm(pn) + EPS)
            d = pp - pn
            out[i] = [cp, cn, np.linalg.norm(e - pp), float(e @ d) / (np.linalg.norm(d) + EPS), cp / (cn + EPS), a, b]
        return out

    P_pb, P_ov = p7(E_pb, tgt), p7(E_ov, ov.target.values)
    OVF = pd.DataFrame([{**ns["aa_composition"](s.upper()), **ns["dipeptide_composition"](s.upper()),
                         **ns["physicochemical"](s.upper())} for s in ov.binder_seq])[SEQF].values

    configs = {
        PRIMARY:                 (np.hstack([fm[SEQF].values, P_pb]), np.hstack([OVF, P_ov])),
        "ESM2(1280)+proto(7)":   (np.hstack([E_pb, P_pb]), np.hstack([E_ov, P_ov])),
        "proto(7) only":         (P_pb, P_ov),
        "seq+ESM2+proto":        (np.hstack([fm[SEQF].values, E_pb, P_pb]), np.hstack([OVF, E_ov, P_ov])),
    }
    LGB = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
               subsample=0.9, colsample_bytree=0.8, min_child_samples=10, reg_lambda=1.0,
               is_unbalance=True, verbosity=-1, n_jobs=1, num_threads=1, deterministic=True,
               force_row_wise=True, random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED)

    def ci(yy, pp):
        rng = np.random.RandomState(SEED); n, o = len(yy), []
        for _ in range(N_BOOT):
            i = rng.randint(0, n, n)
            if len(np.unique(yy[i])) > 1:
                o.append(roc_auc_score(yy[i], pp[i]))
        return round(float(np.percentile(o, 2.5)), 3), round(float(np.percentile(o, 97.5)), 3)

    yl = ov.label.values.astype(int)
    print("overath %d   duplicates %d   de-dup %d   binders(de-dup) %d\n"
          % (len(ov), dup.sum(), (~dup).sum(), yl[~dup].sum()))
    res = {}
    for name, (Xtr, Xte) in configs.items():
        m = lgb.LGBMClassifier(**LGB).fit(Xtr.astype(np.float32), y)
        p = m.predict_proba(Xte.astype(np.float32))[:, 1]
        a_all, a_ded = roc_auc_score(yl, p), roc_auc_score(yl[~dup], p[~dup])
        lo, hi = ci(yl[~dup], p[~dup])
        per = {t: round(roc_auc_score(yl[(ov.target == t).values & ~dup], p[(ov.target == t).values & ~dup]), 3)
               for t in sorted(ov.target.unique())
               if len(np.unique(yl[(ov.target == t).values & ~dup])) > 1}
        res[name] = {"all": round(a_all, 3), "dedup": round(a_ded, 3), "dedup_ci95": [lo, hi],
                     "auprc_all": round(average_precision_score(yl, p), 3),
                     "auprc_dedup": round(average_precision_score(yl[~dup], p[~dup]), 3),
                     "per_target_dedup": per, "n_all": int(len(ov)), "n_dedup": int((~dup).sum())}
        mark = "  <- PRIMARY" if name == PRIMARY else ""
        print("%-22s all %.3f  dedup %.3f [%.3f, %.3f]  AUPRC_dedup %.3f%s"
              % (name, a_all, a_ded, lo, hi, res[name]["auprc_dedup"], mark))
        print("     per-target (de-dup): %s" % per)

    res["_stored"] = {"full_all": 0.728, "full_dedup": 0.620, "n_dedup": 571}
    res["_reproduces_stored"] = False
    json.dump(res, open(RESULT, "w"), indent=2)
    print("\nstored: full_all 0.728  full_dedup 0.620  n_dedup 571")
    print("reproduces stored values: NO  (see module docstring)")
    print("saved %s" % RESULT)


if __name__ == "__main__":
    fn = {"export": stage_export, "embed": stage_embed, "score": stage_score}.get(
        sys.argv[1] if len(sys.argv) > 1 else "")
    if fn is None:
        sys.exit(__doc__)
    fn()
