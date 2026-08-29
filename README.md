# Tabular-to-Image Encoding Dissection

[![Release](https://img.shields.io/github/v/release/Jackxiaozhiren/tabular-image-encoding-dissection?label=release)](https://github.com/Jackxiaozhiren/tabular-image-encoding-dissection/releases/tag/v1.0.0)

Reproducibility repository for the Data Mining and Knowledge Discovery (DMKD) manuscript:

**Dissecting tabular-to-image encodings for CNNs: categorical features, not the image form, are the primary driver of accuracy gains**

The frozen scientific snapshot is **[`v1.0.0`](https://github.com/Jackxiaozhiren/tabular-image-encoding-dissection/releases/tag/v1.0.0)**.

## What the study finds

Across 15 public tabular datasets and three fixed seeds (42, 123, 2024), the manuscript-active analysis isolates four design choices in tabular-to-image classifiers: categorical inclusion, channel separation, feature reordering, and the image form itself.

The frozen manuscript evidence shows:

- removing the categorical channel from M3 lowers AUC across the 10 mixed-type datasets (pooled seed×dataset Wilcoxon `n=30`, `p=3.80e-05`, Holm-significant; dataset-median sensitivity `p=0.00390625`);
- Adult: M3-full AUC `0.9091` vs M3-noG `0.8471`;
- Bank: M3-full AUC `0.9249` vs M3-noG `0.8665`;
- the same-feature MLP vs M3-full contrast does not reject parity (`p=0.1256`);
- feature-reordering/B-channel contrasts are small and not Holm-significant;
- the second ResNet-style backbone preserves the categorical-effect direction, while the dataset-median sensitivity weakens to `p=0.06445`.

These are model-level benchmark results under the stated protocol, not claims of universal superiority or deployment performance.

## Repository map

- `01_download_clean.py` — download/clean/preprocess the 15 datasets.
- `03_train_baseline.py` — XGBoost baseline and feature importance.
- `04_train_cnn.py` — CNN probe and ResNet-style backbone for M1/M1c/M2/M3 variants.
- `07_train_modern_baselines.py` — LightGBM, CatBoost, MLP, FT-Transformer and validation-ES XGBoost.
- `08_extended_evaluate.py` — manuscript tables, DeLong/McNemar tests, pooled Wilcoxon and Holm correction.
- `10_published_encodings.py` + `igtd_encoder.py` — published IGTD control.
- `11_resnet_analysis.py` — second-backbone robustness analysis.
- `frozen_results/` — small aggregate evidence files used to audit the manuscript-active claims without committing raw data, row-level predictions, or model checkpoints.
- `number_mapping.md` — manuscript number → script/artifact crosswalk.
- `docs/REPRODUCIBILITY.md` — exact release boundary and reproduction guidance.

## Quick verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python 08_extended_evaluate.py   # after training/frozen NPZ artifacts are present under data/
python 11_resnet_analysis.py
python make_tables.py
```

For a lightweight repository integrity check used by CI:

```bash
pip install -r requirements-ci.txt
python -m unittest discover -s tests -v
python tools/verify_frozen_results.py
```

## Full experiment protocol

The full command sequence is documented in `docs/REPRODUCIBILITY.md`. The analysis uses 15 public datasets, fixed seeds `{42, 123, 2024}`, training-split-only preprocessing and ordering weights, and two CNN backbones. Statistical tests and aggregation conventions are documented in `number_mapping.md`.

## Data and artifact boundary

This repository does **not** redistribute raw UCI/OpenML downloads, cached raw files, row-level predictions, processed arrays, or `.pt` checkpoints. The scripts retrieve the public datasets and regenerate the experiment artifacts. Small aggregate JSON/CSV evidence is committed under `frozen_results/` so that headline values and statistical claims can be independently checked.

## Environment

The manuscript reports Python 3.9, PyTorch 2.8 on Apple MPS, XGBoost 2.1, LightGBM 4.6, CatBoost 1.2, and Matplotlib 3.9.4. `requirements.txt` is a compatibility specification, not a claim of bit-exact cross-device reproduction. See `docs/REPRODUCIBILITY.md`.

## Historical note

The repository originally accompanied a Neurocomputing submission and contained only the earlier 5-dataset / scripts-01–09 state. The DMKD manuscript expanded the study to 15 datasets, added FT-Transformer/IGTD controls, a second CNN backbone, and additional sensitivity analyses. The immutable `v1.0.0` release freezes the audited DMKD manuscript-active state; earlier Git history remains available for provenance. Documentation-only changes on `main` after release do not alter that frozen scientific snapshot.

## Citation

Use GitHub's **Cite this repository** metadata from `CITATION.cff`. After journal publication, add the article as `preferred-citation` without rewriting the frozen software release.

## License

MIT for project-authored code and documentation. Third-party datasets and reference implementations remain subject to their original terms.