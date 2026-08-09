# Number ↔ Script mapping table

Every number in the manuscript is produced by the scripts below. "Sources"
give the exact artifact (`data/*.npz` / `*.json`) and the script that writes it.

| Manuscript item | Source artifact | Produced by |
|---|---|---|
| Dataset counts (Adult 32,537/16,276; Heart 227/76; Wine 2,970/991; Bank 33,908/11,303; Credit 22,473/7,492) | `data/{name}_arrays.npz` shapes; `data/dataset_summary.json` | `01_download_clean.py` |
| XGBoost table rows (3 datasets) | `data/{name}_xgb_results.npz` | `03_train_baseline.py` (test-ES) / `07_train_modern_baselines.py` (val-ES) |
| LightGBM / CatBoost / MLP rows | `data/{name}_{lgb,cat,mlp}_results.npz` | `07_train_modern_baselines.py` |
| CNN-M1/M1c/M2/M3-* table rows | `data/{name}_cnn_{tag}_results.npz` | `04_train_cnn.py` |
| per-seed mean±std (all tables) | `per_seed_metrics` arrays in the above | recomputed by `08_extended_evaluate.py` |
| McNemar / DeLong per-dataset tests | `data/extended_significance.json` | `08_extended_evaluate.py` |
| Cross-dataset Wilcoxon + Holm | `data/extended_wilcoxon.json` | `08_extended_evaluate.py` |
| Kendall τ (Adult 0.60, Wine 0.49) | `data/ordering_divergence.json` | `06_ordering_analysis.py` |
| ROC / PR curves, figures | `figures/fig_*.pdf` | `05_evaluate_compare.py` (curves), `09_make_figures.py` (vector) |
| Architecture / encoding / AUC-summary figures | `figures/fig1_architecture.pdf`, `fig_encodings.pdf`, `fig_auc_summary.pdf` | `09_make_figures.py` |

## Aggregation convention (important)

- **Tables**: mean±std over the three per-seed metrics.
- **Significance tests (McNemar, DeLong) and ROC/PR**: computed on the
  predictions averaged over the three seeds (the `probs` key of each
  `*_results.npz`), exactly as in `05`/`08`.
- **Cross-dataset Wilcoxon**: per-seed AUC pooled across seeds and datasets
  (`n = 5 datasets × 3 seeds = 15`), Holm–Bonferroni corrected.

To reproduce a single table cell from scratch: train with the given seed,
threshold at 0.5 for accuracy/precision/recall/F1, use soft predictions for AUC.
