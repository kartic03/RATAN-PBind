#!/usr/bin/env python3
"""R1 support - export pair metadata + base feature matrix to npy/csv so the
base env (no pyarrow) can train. Run in `research` env. Portable ROOT."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/targets"); os.makedirs(OUT, exist_ok=True)
fm = pd.read_parquet(os.path.join(ROOT,"features/feature_matrix.parquet"))
fm[["protein_id","target","binding_label","split"]].to_csv(os.path.join(OUT,"pairs_meta.csv"), index=False)
fc = pd.read_csv(os.path.join(ROOT,"features/feature_columns.csv"))["column"].tolist()
np.save(os.path.join(OUT,"base_features.npy"), np.nan_to_num(fm[fc].values.astype("float32"), nan=0.0))
json.dump(fc, open(os.path.join(OUT,"base_feature_cols.json"),"w"))
print(f"pairs={len(fm)} base_features={len(fc)} | splits={fm['split'].value_counts().to_dict()}", flush=True)
print(f"saved -> {OUT}/pairs_meta.csv, base_features.npy", flush=True)
