# Release notes

## v1.0.0 — 2026-08-29

First frozen reproducibility release for the Data Mining and Knowledge Discovery manuscript:

**Dissecting tabular-to-image encodings for CNNs: categorical features, not the image form, are the primary driver of accuracy gains**

### Scientific scope

- 15 public tabular datasets.
- Fixed seeds: 42, 123, 2024.
- Controlled dissection of categorical inclusion, channel separation, feature reordering, and image formulation.
- Modern tabular baselines including XGBoost, LightGBM, CatBoost, MLP, and FT-Transformer.
- Published-encoding IGTD control.
- Second ResNet-style backbone robustness analysis.

### Frozen headline evidence

- Adult: M3-full AUC 0.9091 vs M3-noG 0.8471.
- Bank: M3-full AUC 0.9249 vs M3-noG 0.8665.
- Categorical-channel pooled Wilcoxon p=3.8016e-05; dataset-median sensitivity p=0.00390625.
- Same-feature MLP vs M3-full p=0.1256.
- ResNet dataset-median sensitivity p=0.064453125.

### Reproducibility additions

- manuscript-active scripts 01–11 and expanded dataset registry;
- FT-Transformer and IGTD implementation/control files;
- small frozen aggregate CSV/JSON evidence under `frozen_results/`;
- encoder tests and frozen-result verifier;
- lightweight GitHub Actions CI;
- deterministic SHA-256 release-manifest generation;
- `CITATION.cff` and public release boundary documentation.

### Excluded artifacts

Raw public datasets, row-level prediction arrays, processed NPZ files, model checkpoints, local virtual environments/caches, and editorial/submission-management material are intentionally excluded from the public release.

This release freezes the DMKD manuscript-active reproducibility state. Earlier Git history preserves the original Neurocomputing/5-dataset repository state for provenance.
