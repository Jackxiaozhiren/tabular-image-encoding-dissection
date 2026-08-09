# README — Reproduction guide

Reproduces every number in **"Dissecting image encodings of tabular data:
categorical features drive gains, while channel separation, feature reordering,
and the image form do not"** (Neurocomputing submission).

## Environment

```
pip install -r requirements.txt        # Python 3.9+
```

## Scripts and pipeline

| Script | Purpose |
|---|---|
| `01_download_clean.py` | Download + clean + preprocess datasets into `data/{name}_arrays.npz` |
| `02_visualize_encoding.py` | Generate encoding example figures |
| `03_train_baseline.py` | Train XGBoost baseline + feature/SHAP importances |
| `04_train_cnn.py` | Train CNN on encodings (M1/M1c/M2/M3-variants), 3 seeds |
| `05_evaluate_compare.py` | Aggregate metrics, McNemar/DeLong tests, ROC/PR |
| `06_ordering_analysis.py` | Kendall-tau ordering divergence diagnostic |
| `07_train_modern_baselines.py` | LightGBM/CatBoost/MLP + XGBoost (val-ES) baselines |
| `08_extended_evaluate.py` | Full comparison + cross-dataset Wilcoxon/Holm |
| `09_make_figures.py` | Vector (PDF) publication figures |

Supporting modules: `datasets.py` (dataset registry + preprocessing),
`tabular_to_image.py` (M1/M1c/M2/M3 encoders).

## Full reproduction (all datasets)

```bash
python3 01_download_clean.py --datasets adult,heart,wine,bank,credit
python3 03_train_baseline.py --dataset adult   # (repeat for heart, wine, bank, credit)
python3 07_train_modern_baselines.py --datasets adult,heart,wine,bank,credit
python3 04_train_cnn.py --dataset adult --tags M1,M1c,M2,M3-full,M3-noG,M3-noB,M3-corrB,M3-shapB
python3 04_train_cnn.py --dataset heart --tags M1,M1c,M2,M3-full,M3-noG,M3-noB
python3 04_train_cnn.py --dataset wine --tags M1,M1c,M2,M3-full,M3-noG,M3-noB,M3-corrB,M3-shapB
python3 04_train_cnn.py --dataset bank --tags M1,M1c,M2,M3-full,M3-noG,M3-noB
python3 04_train_cnn.py --dataset credit --tags M1,M1c,M2,M3-full,M3-noG,M3-noB
python3 08_extended_evaluate.py          # comparison tables + Wilcoxon/Holm
python3 06_ordering_analysis.py --datasets adult,wine
python3 09_make_figures.py               # vector figures
```

## Conventions

- **Seeds**: {42, 123, 2024}; tables report mean±std of per-seed metrics.
- **Aggregation**: statistical tests (McNemar, DeLong) and ROC/PR curves use
  the three-seed-averaged predictions; tables report per-seed mean±std.
- **Early stopping**: on a stratified 10% validation split (patience 7 for
  CNN/MLP, 20 for trees).
- **Heart** uses class-weighted loss; Heart and Wine use stratified 75/25 splits
  after duplicate removal (Wine: 4,898 → 3,961 rows).

## Data

All datasets are public from the UCI Machine Learning Repository:
Adult (Census Income), Heart Disease (Cleveland), Wine Quality (White),
Bank Marketing, Credit Card Default. `01_download_clean.py` downloads and
caches them into `data/`.
