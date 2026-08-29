# Number ↔ Script mapping table

Every number in the manuscript is produced by the scripts below. "Sources"
give the exact artifact (`data/*.npz` / `*.json`) and the script that writes it.

| Manuscript item | Source artifact | Produced by |
|---|---|---|
| Dataset counts (15 datasets; 8 core + 7 OpenML; Adult 32,537/16,276; Heart 227/76; Wine 2,970/991; Bank 33,908/11,303; Credit 22,473/7,492; German 750/250; Telco 5,265/1,756; Sick 2,064/689; australian/cmc/ilpd/segment/vehicle/spambase/magic via OpenML) | `data/{name}_arrays.npz` shapes; `data/dataset_summary.json` | `01_download_clean.py` |
| XGBoost table rows | `data/{name}_xgb_results.npz` | `03_train_baseline.py` (test-ES) / `07_train_modern_baselines.py` (val-ES) |
| LightGBM / CatBoost / MLP / FT-Transformer rows | `data/{name}_{lgb,cat,mlp,ft}_results.npz` | `07_train_modern_baselines.py` (FT via `ft_transformer.py`) |
| CNN-M1/M1c/M2/M3-* table rows (incl. M3-RG two-channel) | `data/{name}_cnn_{tag}_results.npz` | `04_train_cnn.py` |
| CNN-IGTD (published encoder) rows | `data/{name}_cnn_IGTD_results.npz` | `10_published_encodings.py` (layout in `{name}_igtd_layout.npz` via `igtd_encoder.py`) |
| per-seed mean±std (all tables) | `per_seed_metrics` arrays in the above | recomputed by `08_extended_evaluate.py` |
| Per-dataset McNemar / DeLong tests (significance matrix, 14 methods × 15 datasets) | `data/extended_significance.json` | `08_extended_evaluate.py` |
| Cross-dataset Wilcoxon + Holm (Table 12, 10 contrasts, n=45/30) | `data/extended_wilcoxon.json` (`pairs[]`) | `08_extended_evaluate.py` |
| **Architecture robustness** (second ResNet-style backbone): per-encoding AUC × dataset, pooled Wilcoxon + DeLong on key contrasts | `data/*_cnn_{tag}_resnet_results.npz` + `data/resnet_robustness.json` | `04_train_cnn.py --backbone resnet`（`batch_train_resnet.sh`）+ `11_resnet_analysis.py` |
| Wilcoxon sensitivity: excl-Heart / per-dataset-median (n=10) | `data/extended_wilcoxon.json` (`p_excl_heart` / `p_dataset_median`) | `08_extended_evaluate.py` |
| Kendall τ (Adult 0.47, Wine 0.53) | `data/ordering_divergence.json` | `06_ordering_analysis.py` |
| ROC / PR curves, AUC-summary, figures | `figures/fig_*.pdf` | `05_evaluate_compare.py` (curves), `09_make_figures.py` (vector) |
| Table/LaTeX cell generation (for manuscript tables) | — | `make_tables.py` / `gen_tables_latex.py` (helper, not in paper) |

## Aggregation convention (important)

- **Tables**: mean±std over the three per-seed metrics.
- **Significance tests (McNemar, DeLong) and ROC/PR**: computed on the
  predictions averaged over the three seeds (the `probs` key of each
  `*_results.npz`), exactly as in `05`/`08`.
- **Cross-dataset Wilcoxon**: per-seed AUC pooled across seeds and datasets
  (`n = 15 datasets × 3 seeds = 45`), Holm–Bonferroni corrected over the
  10-contrast pre-specified family. Contrasts where a variant is degenerate on
  fully numeric datasets (M1c, M3-noG) use `n = 30` (the five numeric-only
  datasets excluded: wine, segment, vehicle, spambase, magic; M1c ≡ M1 and
  M3-noG ≡ M3-full there). Sensitivity: `p_excl_heart` re-runs each contrast
  without Heart; `p_dataset_median` re-runs on per-dataset medians (n = 10).
- **15-dataset expansion**: 7 additional OpenML datasets (australian, cmc,
  ilpd, segment, vehicle, spambase, magic; all UCI-sourced or OpenML), loaded
  via `datasets.py` `openml_id` branches with majority-vs-rest binarization
  for multiclass tasks. All preprocessing (MinMax/mode/LabelEncoder/weights)
  is fit on the training split only; ordering weights (|corr|, XGBoost
  importance, SHAP) are also training-split only.
- **Known honest exceptions**: spambase image-encoding CNNs are unstable
  (per-seed std up to 0.13); segment/vehicle saturate at ~1.0 AUC after
  binarization; ilpd and credit each have a single categorical feature and
  show no categorical-channel effect; sick image encoding significantly
  outperforms the MLP and FT-Transformer controls.

To reproduce a single table cell from scratch: train with the given seed,
threshold at 0.5 for accuracy/precision/recall/F1, use soft predictions for AUC.
