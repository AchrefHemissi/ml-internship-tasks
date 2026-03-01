# Task 7 — Sales Forecasting

Predict future weekly department sales for 45 Walmart stores using historical data,
time-based features, and gradient boosting models.

## Dataset

**Walmart Sales Forecast** — Kaggle: `aslanahmedov/walmart-sales-forecast`

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 421,570 | Weekly sales per store-department |
| `features.csv` | 8,190 | Store-level exogenous features |
| `stores.csv` | 45 | Store type (A/B/C) and size |
| `test.csv` | 115,064 | Future weeks for Kaggle submission |

Download and place all CSVs in the `data/` folder:
```bash
kaggle datasets download -d aslanahmedov/walmart-sales-forecast -p data/ --unzip
```

## Project Structure

```
Task_7_Sales_Forecasting/
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA, trends, seasonality
│   ├── 02_preprocessing.ipynb      # Merge, clean, split preview
│   ├── 03_time_features.ipynb      # Lag, rolling, correlation
│   ├── 04_prophet_baseline.ipynb   # Prophet time-series model
│   ├── 05_xgboost_lgbm.ipynb       # XGBoost + LightGBM (bonus)
│   └── 06_evaluation.ipynb         # Final comparison & insights
├── src/
│   ├── __init__.py
│   ├── config.py          # Paths, feature lists, hyperparameters
│   ├── data_loader.py     # Load & merge the 4 CSV files
│   ├── preprocessing.py   # Feature engineering pipeline
│   └── models.py          # Model factories + evaluation helpers
├── data/                  # CSV files (not tracked by git)
├── models/                # Saved model checkpoints (.pkl)
├── requirements.txt
├── setup_env.sh
└── README.md
```

## Quick Start

```bash
# 1. Set up environment
bash setup_env.sh
source sales-forecast/bin/activate   # Linux/Mac
# sales-forecast\Scripts\activate    # Windows

# 2. Download dataset
kaggle datasets download -d aslanahmedov/walmart-sales-forecast -p data/ --unzip

# 3. Launch notebooks
jupyter notebook notebooks/
```

## Pipeline

| Notebook | Description | Key output |
|----------|-------------|------------|
| 01 · EDA | Load data, seasonal decomposition, store/holiday analysis | Trend + seasonality plots |
| 02 · Preprocessing | Merge, fill NaN, encode store type | Clean merged DataFrame |
| 03 · Time Features | Lag-1/4/8/52, MA-4/8/12, holiday flags | 24 model features |
| 04 · Prophet | Yearly seasonality + custom holidays baseline | RMSE, forecast plot |
| 05 · XGBoost/LightGBM | Early stopping, feature importance | RMSE, R² ≈ 0.99 |
| 06 · Evaluation | All models compared, actual vs predicted | Final insights |

## Configuration

All paths and hyperparameters are set in `src/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_END_DATE` | `"2012-06-01"` | End of training window |
| `VAL_END_DATE` | `"2012-08-01"` | Start of test window |
| `LAG_WEEKS` | `[1, 4, 8, 52]` | Lag offsets |
| `ROLLING_WINDOWS` | `[4, 8, 12]` | Rolling stat windows |
| `MODEL_FEATURES` | (24 features) | Full feature list |
| `RANDOM_STATE` | `42` | Reproducibility seed |

## Results

| Model | Granularity | Test R² | Notes |
|-------|-------------|---------|-------|
| Prophet | Weekly aggregate | ~0.08 | Captures trend/seasonality |
| XGBoost | Per store-dept | ~0.99 | Best per-dept accuracy |
| LightGBM | Per store-dept | ~0.99 | Faster training, similar R² |

> Tree-based models dominate because lag features encode nearly all of the variance at store-dept level.
> Prophet is more interpretable for aggregate business reporting.

## Covered Topics

- **Time series EDA**: trend, seasonality, rolling averages, additive decomposition
- **Feature engineering**: calendar features, lag features, rolling statistics
- **Regression models**: Prophet (baseline), XGBoost, LightGBM
- **Time-aware validation**: chronological train/val/test split (no data leakage)
- **Evaluation**: RMSE, MAE, R², MAPE, residual diagnostics
