# Frozen aggregate evidence

These files are small, non-row-level summaries derived from the audited manuscript experiment artifacts.

- `auc_summary.csv` — per-dataset/per-method AUC mean, population std, and the three seed AUCs.
- `dataset_summary.json` — final train/test sizes and feature counts.
- `extended_significance.json` — per-dataset significance outputs.
- `extended_wilcoxon.json` — the 10 planned cross-dataset contrasts, Heart-excluded sensitivity, dataset-median sensitivity, and Holm decisions.
- `ordering_divergence.json` — feature-ordering diagnostics.
- `resnet_robustness.json` — second-backbone AUCs and robustness statistics.

Raw downloads, processed arrays, row-level predictions, and checkpoints are intentionally excluded from Git. They can be regenerated from the public datasets with the repository scripts.
