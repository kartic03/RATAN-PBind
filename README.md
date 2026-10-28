# RATAN-PBind

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20656437.svg)](https://doi.org/10.5281/zenodo.20656437)

**Retrieval-Augmented, Target-Aware Nomination for Protein Binding** - prioritisation of de novo binders, with a characterised applicability domain and few-shot extension to new targets.

A machine-learning pre-screen for de novo binder campaigns, trained on 2,630 experimentally labelled protein-target pairs across 24 human and viral targets from the [Proteinbase dataset](https://proteinbase.com) (Adaptyv Bio, ODC-BY licence). It ranks candidate binders and reports where its predictions can and cannot be trusted.

Current release: **v1.3** (the version reported in the paper).

![Graphical Abstract](graphical_abstract.svg)

<sub>Vector, so it stays sharp at any zoom. `figures/graphical_abstract.svg` is the same figure with the type kept as text and under the 150 KB the journal allows; `graphical_abstract.png` is the 920 x 300 raster.</sub>

## What it does

De novo binder design produces thousands of candidates per campaign. In the Proteinbase data, 17.8% of the labelled pairs are binders. RATAN-PBind ranks candidates using a target-conditioned **prototype-similarity** feature (how closely a candidate resembles a target's known binders vs non-binders in ESM-2 embedding space) on top of composition, physicochemical, design-method, and structure-derived features.

The main model uses 470 features, including ESMFold, ProteinMPNN and Boltz-2 scores. A 437-feature configuration that needs only the candidate sequence and the target's stored prototypes reaches AUROC 0.933 on the random held-out split, against 0.946 for the 470-feature model (Table S2 of the paper).

The main result is a measured **applicability domain**: performance is reported along four generalisation axes (in-distribution, across design methods and authors, across targets, and across datasets).

## Key results

| Evaluation | AUROC |
|---|---|
| In-distribution, held-out test | **0.946** (95% CI 0.920-0.969) |
| In-distribution, nested cross-validation (leakage-free) | **0.895 ± 0.006** |
| Across design methods (leave-author/method-out, prototypes recomputed per fold) | 0.70-0.76 |
| Fresh campaign on a known target | 0.70-0.76 (leave-author-out 0.761, leave-design-method-out 0.702, leave-one-method-out 0.706) |
| Zero-shot to a novel target (LOTO) | 0.54-0.57 |
| Independent dataset (after de-duplication), per target | at chance; mean 0.503 over five targets, every 95% interval includes 0.5 |

- **470 features** (463 base + 7 prototype-similarity). `proto_ratio` is the top SHAP feature.
- **Practical utility:** top-10% ranking enriches binders **4.65×** over the 17.1% test-set binding rate (top-5% reaches 90% precision, 5.27×). Those figures are the random held-out split; under grouped splits the top decile enriches 1.7-3.6× (Table S9).
- **Few-shot:** adding ~2 known binders of a new target recovers AUROC from near chance to **~0.70**.
- **Validated:** label-shuffle control, sequence- and batch-level leakage audits, nearest-neighbour and single-feature baselines, external (Overath 2025) and natural-PPI (SKEMPI 2.0) checks, and independent structural validation (Boltz-2 ipTM, MM-GBSA).
- The model is a **binder/non-binder classifier** and does not rank affinity.

## Installation

```bash
git clone https://github.com/kartic03/RATAN-PBind.git
cd RATAN-PBind
pip install -r requirements.txt
```

For a fully reproducible environment, use the pixi lockfile (`pixi.toml` / `pixi.lock`):

```bash
pixi run -e ml python src/r3_robust_eval.py   # CPU analysis
pixi run -e gpu python src/r1_embed_targets.py # ESM-2 embeddings (CUDA)
```

### Optional: Groq LLM interpretation
Create a `.env` file with `GROQ_API_KEY=...` (free key at [console.groq.com](https://console.groq.com)). The LLM module is a faithfulness-bounded convenience (~87% grounded in the SHAP evidence), not a mechanistic claim.

The faithfulness figures in the paper (87% of cited features grounded, 104 of 120) were measured with `llama-3.3-70b-versatile`. Groq has since withdrawn that model, so the app defaults to `openai/gpt-oss-120b`; set `GROQ_MODEL` in `.env` to use another (`GET https://api.groq.com/openai/v1/models` lists what the service currently serves). The interpretation text is the only part of this release that the pinned environment does not reproduce, and a different model will change it.

## Usage

### Web app
```bash
python3 app.py    # open http://localhost:7860
```

### Python API
```python
from protbind import ProtBind
pb = ProtBind()
result = pb.predict("MASWKELLVQNKNQFNLERSELTNGFLKPIVKVVKKLPEEVLAERIRKAFG",
                    target="nipah-glycoprotein-g")
print(f"Binding probability: {result['probability']:.1%}")
explanation = pb.explain(result, top_n=10)
mutations   = pb.suggest_mutations(sequence, target="egfr", top_n=10)
```

Targets with few known binders fall in the few-shot regime, so interpret scores accordingly and calibrate on a first experimental batch.

Each prediction reports a regime derived from the target's prototype support: 20 or more known binders (in-domain), 2 to 19 (few-shot), fewer than 2 (extrapolation). These cut points are conventions chosen for usability, not transitions found in the data (Table S5 of the paper).

The web app also prints a binder/non-binder verdict. For that verdict it applies a per-target decision threshold chosen on the validation split (the value in 0.05-0.94 that maximises validation F1, and 0.5 for a target with fewer than five validation pairs or no class variation). The thresholds are stored in `models/ensemble_meta_6b.pkl` and run from 0.05 (IL7R) to 0.5; PD-L1 is 0.28 and EGFR 0.17. Every metric in the paper uses a fixed 0.5 threshold, and the ranking the tool is meant for does not depend on any threshold.

## Reproducing the analysis

All experiments are scripted under `src/` and regenerate from the released artefacts:

- `src/r1_*` target-aware modelling / leave-one-target-out
- `src/r3_robust_eval.py` bootstrap CIs, per-target reliability, shared-vs-single
- `src/r8_*`, `src/r8b_*` significance, leakage audits, few-shot, baselines, calibration, external/SKEMPI/MM-GBSA
- `src/r7_figures_final.py`, `src/r22_revision2_figures.py` the manuscript figure set
- `src/r22_seed_and_calibration.py` per-seed metrics and the measured reliability curve (Table S7, Figure 7)
- `src/r12_deployment_matched.py` configurations ordered by what each needs at inference (Table S2)
- `src/r13_grouped_prospective.py` grouped prospective splits and per-method results (Tables S9, S10)
- `src/r14_fewshot_variability.py` few-shot support-set variability
- `src/r15_feature_block_bias.py`, `src/r15b_noise_floor.py`, `src/r15c_loao_noise_floor.py` feature-block contributions and run-to-run noise (Table S1)
- `src/r16_applicability_thresholds.py` target-support sweep (Table S5)
- `src/r2_crossdataset_overath.py`, `src/r8_crossdata_full.py`, `src/r19_external_validation_table.py` external validation (Tables S11, S12)
- `src/r20_interface_coverage.py` interface annotation coverage and binding-site structure (Tables S13, S14)
- `src/r17_verification_pass.py`, `src/r17b_verification_fixes.py`, `src/r18_regenerate_S2_S3.py`, `src/r21_interface_row_recompute.py` re-computation of reported figures (Tables S6, S7 and the interface row of Table 2)

The main model (`models/lgb_proto_470.pkl`), the feature matrix, feature columns, and the train/val/test splits are in the repo (`models/`, `features/`, `data/`). The large artefacts (the ESM-2 embeddings and the heavier baseline models: random forest, extra trees, SVM, fine-tuned ESM-2) are archived on Zenodo ([10.5281/zenodo.20656437](https://doi.org/10.5281/zenodo.20656437)) to keep the repo lightweight; they are also regenerable from `src/`. Each analysis script in `src/` writes its results to `outputs/` as CSV/JSON, so every reported number is regenerable.

The ablation models `models/lgb_interface_hc.pkl` and `models/xgb_interface_hc.pkl` use the 39-feature interface block and were trained before the binder-chain filter was added to `src/phase6a_interface.py` and `src/phase6b_target_embeddings.py`. The main model does not use that block.

## Supported targets (24)

`egfr` · `nipah-glycoprotein-g` · `pd-l1` · `mdm2` · `il7r` · `spcas9` · `human-insulin-receptor` · `human-pdgfr-beta` · `human-mzb1-perp1` · `ifnar2` · `fgf-r1` · `fcrn` · `der21` · `der7` · `human-ambp` · `human-idi2` · `human-rfk` · `hnmt` · `human-pmvk` · `human-phyh` · `human-serum-albumin` · `human-tnfa` · `human-orm2` · `human-gm2a`

## Data

Training data from **Proteinbase** by Adaptyv Bio (ODC-BY licence). The raw dataset is not redistributed here; download the pinned snapshot used here, `proteinbase_all_data_28_01_2026.csv` (version of 28 January 2026), from https://storage.proteinbase.com/proteinbase_all_data_28_01_2026.csv

The external files read by the analysis scripts, their sources and licences are listed in [data/external/README.md](data/external/README.md).

> *This work used Proteinbase by Adaptyv Bio under the ODC-BY licence.*

## Citation

> Kartic, Choi J, Park T-S. RATAN-PBind: Retrieval-Augmented, Target-Aware Nomination of de novo protein binders within a characterised applicability domain.
> Code: https://github.com/kartic03/RATAN-PBind

## Authors

Kartic and Jiwon Choi (equal contribution); Tae-Sik Park (corresponding).
Department of Life Sciences, Gachon University, Seongnam, Republic of Korea.

## Licence

Three licences apply to different parts of this release:

| Component | Licence |
|---|---|
| Source code in this repository (`src/`, `protbind/`, `app.py`) | MIT, see [LICENSE](LICENSE) |
| Zenodo deposit [10.5281/zenodo.20656437](https://doi.org/10.5281/zenodo.20656437) as a whole | CC BY 4.0 |
| Proteinbase-derived artefacts inside both (feature matrices, ESM-2 embeddings, split indices) | ODC-BY 1.0, from Adaptyv Bio |

Downstream users of the feature matrices and embeddings remain bound by ODC-BY attribution regardless of the licence on the container.

