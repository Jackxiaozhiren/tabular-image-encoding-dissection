# Reproducibility boundary — DMKD manuscript-active release

## Canonical manuscript state

The audited local submission package is `DMKD_ready_to_submit` (2026-08-20), using Springer Nature `sn-jnl`. The public software release is aligned to the scientific claims in that package, not to the earlier Neurocomputing repository state.

## Design

- 15 public datasets.
- Seeds: 42, 123, 2024.
- Adult uses its official split after duplicate handling; the other datasets use stratified 75/25 splits after duplicate removal.
- All imputers/scalers/encoders and all label-derived feature orderings are fitted on training data only.
- Main CNN probe plus a ResNet-style second backbone.
- Baselines: XGBoost, LightGBM, CatBoost, same-feature MLP, FT-Transformer, and published IGTD encoding.

## Statistical convention

Per-seed tables report mean ± population standard deviation across the three seeds. McNemar/DeLong tests and ROC/PR curves use the three-seed-averaged predictions. The cross-dataset Wilcoxon family pools paired per-seed AUC differences across seed×dataset observations and applies Holm–Bonferroni correction; because seeds within a dataset are not independent datasets, this pooled analysis is explicitly treated as exploratory/supporting. Dataset-median and Heart-excluded sensitivities are recorded in `frozen_results/extended_wilcoxon.json`.

## Audited frozen claims

Independent checks performed on the archived research bundle:

- `python -m compileall -q .` succeeds for the research code;
- `08_extended_evaluate.py`, `11_resnet_analysis.py`, and `make_tables.py` execute successfully from the frozen result artifacts;
- Adult M3-full vs M3-noG AUC: 0.9091 vs 0.8471;
- Bank M3-full vs M3-noG AUC: 0.9249 vs 0.8665;
- categorical-channel pooled Wilcoxon: p = 3.801573526884496e-05, Holm-reject = true;
- image-form MLP vs M3-full: p = 0.12559677728090787;
- ResNet dataset-median categorical sensitivity: p = 0.064453125.

## Full reproduction sequence

```bash
python3 01_download_clean.py --datasets adult,heart,wine,bank,credit,german,telco,sick,australian,cmc,ilpd,segment,vehicle,spambase,magic
python3 03_train_baseline.py --dataset adult   # repeat for every dataset
python3 07_train_modern_baselines.py --datasets adult,heart,wine,bank,credit,german,telco,sick,australian,cmc,ilpd,segment,vehicle,spambase,magic
# Run 04_train_cnn.py for the documented M1/M1c/M2/M3 variants on each dataset.
bash batch_train_new7.sh
python3 10_published_encodings.py
python3 08_extended_evaluate.py
python3 06_ordering_analysis.py --datasets adult,wine
python3 09_make_figures.py
python3 make_tables.py
bash batch_train_resnet.sh
python3 11_resnet_analysis.py
```

The exact per-dataset `04_train_cnn.py` command matrix is preserved in the repository README/history and the audited research README. Numeric-only datasets omit degenerate categorical variants by construction.

## Environment

The manuscript-reported environment is:

- Python 3.9 (the archived research guide specifies Python 3.9.6);
- PyTorch 2.8, Apple MPS;
- XGBoost 2.1;
- LightGBM 4.6;
- CatBoost 1.2;
- Matplotlib 3.9.4.

`requirements.txt` deliberately remains a compatibility specification because a complete historical `pip freeze` was not present in the audited bundle. Do not invent a lock file. Cross-device bit-exact reproduction is not claimed.

## Public-release exclusions

Do not commit:

- cached raw UCI/OpenML files;
- processed `*_arrays.npz` files;
- row-level prediction/result NPZ files;
- `.pt` model checkpoints;
- large confusion-matrix image collections;
- local cache folders or machine-specific absolute paths;
- journal submission-management prompts, cover letters, reviewer suggestions, or editorial correspondence.

The small aggregate `frozen_results/` files are sufficient to audit the manuscript's headline numerical claims while the full scripts regenerate training artifacts from public data.

## Known manuscript wording issue

One sentence in the frozen `DMKD_ready_to_submit/manuscript.tex` says the scalability section reports training/inference time and throughput, while the actual scalability section explicitly states that no new timing benchmark is introduced and reports analytic complexity/parameter counts. This is a wording inconsistency, not a result-file mismatch. It should be corrected in the manuscript at the next editable stage; the software release does not fabricate timing results.
