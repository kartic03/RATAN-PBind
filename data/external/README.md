# External data

Files that the analysis scripts read from `data/external/`. Three small files are included. The two larger
or separately licensed datasets are not redistributed; where to get them is listed below.

## Included

| File | What it holds | Source and licence | Read by |
|---|---|---|---|
| `binder_iface_residues.json` | Boltz-2 predicted interface residues on the binder chain, 3,796 entries keyed by Proteinbase protein id | Derived from Proteinbase (Adaptyv Bio), ODC-BY | `src/r8_interface_pooled.py` |
| `r5_test_instances.csv` | 24 Proteinbase instances used to check the language-model interpretation against the SHAP evidence | Derived from Proteinbase (Adaptyv Bio), ODC-BY | `src/r5_llm_faithfulness.py` |
| `overath_clean.csv` | 913 binders on the five targets shared with Proteinbase (`egfr`, `human-insulin-receptor`, `il7r`, `mdm2`, `pd-l1`), with columns `binder_id`, `target`, `binder_seq`, `label` | Derived from the Overath et al. dataset, CC BY 4.0 | `src/r2_crossdataset_overath.py`, `src/r8_crossdata_full.py`, `src/r19_external_validation_table.py` |

For `overath_clean.csv`, the `binder_id` and `label` values match the Overath records for all 913 rows. The sequence
column was extracted by the authors, and that extraction step is not part of this repository.

## Not included: download these

**Overath et al. dataset.** Overath MD, Rygaard AS, Jacobsen CP, Brasas V, Morell O, Sormanni P, et al. Predicting experimental success in
de novo binder design: a meta-analysis of 3,766 experimentally characterised binders. bioRxiv 2025.
Data: https://doi.org/10.5281/zenodo.15722219 (`final_dataset.csv`, 81,981,455 bytes, md5 `3a69ee9b0fecf53924a8c6479bac146e`, CC BY 4.0).
Read by `src/r8_benchmark.py`, `src/r8_robustness.py` and `src/r19_external_validation_table.py`, which expect the file at
`data/external/overath_prepared_training_dataset.csv` with the columns `binder_id`, `target_id` and `binder`.
The local file the scripts were run on has 85,466,098 bytes and md5 `aa9ef865f2cd0bb24ca08813402d03de`, which differs from
the Zenodo `final_dataset.csv`. Check that those three columns exist in the file you download before running the scripts.

**SKEMPI 2.0.** Jankauskaitė J, Jiménez-García B, Dapkūnas J, Fernández-Recio J, Moal IH. SKEMPI 2.0: an updated
benchmark of changes in protein-protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 2019;35(3):462-469.
Download from https://life.bsc.es/pid/skempi2 and save the semicolon-separated table as `data/external/skempi_v2.csv`.
The page displays a CC BY 4.0 licence. Read by `src/r8_skempi.py` and `src/r19_external_validation_table.py`.

**Proteinbase.** The raw dataset is not redistributed here. See the main README.

## Attribution

This work used Proteinbase by Adaptyv Bio under the ODC-BY licence.
