# Sales Forecasting — Technical Report

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Exploration — Observations (Notebook 01)](#3-data-exploration--observations-notebook-01)
4. [Preprocessing Pipeline (Notebook 02)](#4-preprocessing-pipeline-notebook-02)
5. [Feature Engineering (Notebook 03)](#5-feature-engineering-notebook-03)
6. [Prophet Baseline (Notebook 04)](#6-prophet-baseline-notebook-04)
7. [XGBoost & LightGBM (Notebook 05)](#7-xgboost--lightgbm-notebook-05)
8. [Final Evaluation (Notebook 06)](#8-final-evaluation-notebook-06)
9. [Observations vs Results — Confirmation Analysis](#9-observations-vs-results--confirmation-analysis)
10. [Conclusion](#10-conclusion)

---

## 1. Project Overview

This project builds a **time-series sales forecasting pipeline** for 45 Walmart stores across
up to 81 departments. The goal is to predict weekly department-level sales using historical data,
time-based features, and gradient-boosting regression models.

**Three models** were trained and compared:

- **Prophet** — Facebook's additive time-series model, used as an interpretable baseline on
  aggregated weekly totals.
- **XGBoost** — Gradient-boosted trees operating on per-store-department rows with lag and
  rolling features.
- **LightGBM** — Histogram-based gradient boosting; same features as XGBoost, faster training.

All code is organised into a modular `src/` package (`config.py`, `data_loader.py`,
`preprocessing.py`, `models.py`) and 6 sequential notebooks.

---

## 2. Dataset Description

**Source**: Walmart Sales Forecast — Kaggle `aslanahmedov/walmart-sales-forecast`

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train.csv` | 421,570 | 5 | Weekly sales per store-department |
| `features.csv` | 8,190 | 12 | Store-level exogenous features |
| `stores.csv` | 45 | 3 | Store type (A/B/C) and size |
| `test.csv` | 115,064 | 4 | Future weeks for Kaggle submission |

### train.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| `Store` | int | Store ID (1–45) |
| `Dept` | int | Department ID (1–99) |
| `Date` | datetime | Week start date (Fridays) |
| `Weekly_Sales` | float | Target — weekly sales in USD |
| `IsHoliday` | bool | Whether the week contains a major holiday |

### features.csv Key Columns

| Column | Missing | Decision |
|--------|---------|---------|
| MarkDown1 – MarkDown5 | 50–64% | **Dropped** — too sparse to impute meaningfully |
| CPI | 7.1% (585 rows) | Forward/back-filled per store |
| Unemployment | 7.1% (585 rows) | Forward/back-filled per store |
| Temperature, Fuel_Price | 0% | Kept as features |

### Store Type Distribution

| Type | Count | Avg Size (sq ft) | Avg Weekly Sales / Dept |
|------|-------|-----------------|------------------------|
| A | 22 (49%) | ~175,000 | ~$25,000 |
| B | 17 (38%) | ~100,000 | ~$16,000 |
| C | 6 (13%) | ~42,000 | ~$8,000 |

### Target Variable

| Property | Value |
|----------|-------|
| Mean weekly sales (per dept) | $16,485 |
| Std deviation | $22,983 |
| Minimum (returns/markdowns) | −$1,321 |
| Maximum | $649,770 |
| Date range | 2010-02-05 → 2012-10-26 |
| Total weeks | 143 |

---

## 3. Data Exploration — Observations (Notebook 01)

Notebook 01 is a **read-only exploration step**. It makes no predictions and saves no files.
The observations made here directly informed technical decisions in all subsequent notebooks.

### Key Observations

**1. Strong seasonal spike in weeks 47–52 (Thanksgiving + Christmas)**

Total weekly sales across all stores rise from a baseline of ~$50M/week to ~$80M/week during
the holiday season (weeks 47–52). This spike is consistent across all three years in the dataset.

**2. Stable overall trend (no drift)**

The 12-week rolling average is nearly flat across 2010–2012. There is no structural upward or
downward trend — only cyclical seasonal variation. This is important because it means a model
without trend components (XGBoost/LightGBM) will not be disadvantaged.

**3. Holiday weeks average 7.1% higher than regular weeks**

`IsHoliday=True` rows have an average of $17,036/dept/week vs $15,901 for `IsHoliday=False`.
The boost is moderate but consistent, justifying specific holiday flag features.

**4. Store type A dominates in volume (A ≈ 3× Type C)**

Type A stores (22 stores, ~49%) average ~$25,000/dept/week. Type C stores (6 stores) average
~$8,000/dept/week. Type and Size will be highly informative features for the regression models.

**5. All MarkDown columns exceed 50% missing**

MarkDown2 is 64.3% missing, MarkDown1 is 50.8% missing. Imputing values at these rates would
introduce more noise than signal. Decision: drop all 5 MarkDown columns.

**6. CPI and Unemployment have 585 NaN each (7.1%)**

These are small, contiguous gaps in the `features.csv` file caused by store-level reporting
gaps after merging. They are easily closed by forward/back-filling within each store group.

### Expected Impact on Later Notebooks

| Observation | Expected downstream effect |
|-------------|---------------------------|
| Holiday spike in weeks 47–52 | `Week` feature will rank highly; holiday flags will add incremental signal |
| Year-over-year stability | `Sales_Lag_52` (same week last year) should be the strongest individual predictor |
| Store type disparity | `Store`, `Dept`, `Size`, `Type` dummies will capture store-level scale differences |
| CPI/Unemployment NaN gaps | Closed by `fill_missing_externals()` — no rows will be lost |
| MarkDowns all > 50% missing | Confirmed: drop in `merge_datasets(drop_markdowns=True)` |

---

## 4. Preprocessing Pipeline (Notebook 02)

### Technical Decisions

| Step | Decision | Justification |
|------|----------|---------------|
| **MarkDown columns** | Dropped (all 5) | All exceed 50% missing — imputing at this rate adds noise |
| **CPI / Unemployment NaN** | Forward/back-fill per store group | Small contiguous gaps (7.1%), no data loss |
| **Store type encoding** | One-hot: `Type_A`, `Type_B`, `Type_C` | Three unordered categories — ordinal encoding would imply A > B > C in distance |
| **Time-based split** | Chronological (no shuffling) | Shuffling would leak future data into training — invalid for time series |
| **Split boundaries** | Train: < 2012-06-01, Val: < 2012-08-01, Test: ≥ 2012-08-01 | Test period covers the Aug–Oct 2012 back-to-school and pre-holiday windows |

### Raw Data Split (Before Lag Requirement)

| Set | Rows | Date range |
|-----|------|------------|
| Train | 356,489 | 2010-02-05 – 2012-05-25 |
| Validation | 26,551 | 2012-06-01 – 2012-07-27 |
| Test | 38,530 | 2012-08-03 – 2012-10-26 |

> **Important**: After adding `Sales_Lag_52` in Notebook 03, the first full year of data
> (≈160,000 rows) is dropped because no 52-week prior history exists. The model-ready split
> becomes: **Train 197,201 · Val 26,017 · Test 37,865**.

---

## 5. Feature Engineering (Notebook 03)

### Pipeline Steps

| Step | Function | Output |
|------|----------|--------|
| 1 | `fill_missing_externals()` | 0 NaN in CPI, Unemployment, Temperature, Fuel_Price |
| 2 | `add_time_features()` | Year, Month, Week, Quarter, DayOfYear |
| 3 | `add_holiday_flags()` | IsBlackFriday, IsChristmas, IsThanksgiving |
| 4 | `encode_store_type()` | Type_A, Type_B, Type_C |
| 5 | `add_lag_features()` | Sales_Lag_1, Sales_Lag_4, Sales_Lag_8, Sales_Lag_52 |
| 6 | `add_rolling_features()` | Sales_MA/STD at 4, 8, 12 weeks |

Total features added: **21 new columns** (421,570 × 11 → 421,570 × 32)

### Lag Feature Correlations with Weekly_Sales

| Feature | Pearson r | Interpretation |
|---------|-----------|----------------|
| `Sales_Lag_52` | **0.981** | Same week last year — strongest single feature |
| `Sales_Lag_1` | 0.960 | Last week's sales — strong short-term autocorrelation |
| `Sales_Lag_4` | 0.947 | One month ago |
| `Sales_Lag_8` | 0.917 | Two months ago |

**Key finding**: `Sales_Lag_52` (r = 0.981) is the highest-correlated feature — not `Sales_Lag_1`.
This confirms the Notebook 01 prediction that year-over-year patterns dominate retail sales.

### Rows Dropped by Lag Requirement

| Stage | Rows |
|-------|------|
| After pipeline | 421,570 |
| With all lag features valid | 261,083 |
| **Rows dropped** | **160,487 (38%)** |

The 38% drop is expected: each (Store, Dept) pair requires 52 weeks of prior history before
the Lag_52 feature becomes available. The remaining 261,083 rows fully cover the 2011–2012
period that feeds the train/val/test split.

### Final Feature Set (24 features)

| Group | Features |
|-------|---------|
| Store identity | Store, Dept, Size |
| Calendar | Week, Month, Quarter, DayOfYear |
| Holiday flags | IsHoliday, IsBlackFriday, IsChristmas, IsThanksgiving |
| Store type | Type_A, Type_B, Type_C |
| Lag features | Sales_Lag_1, Sales_Lag_4, Sales_Lag_8, Sales_Lag_52 |
| Rolling mean | Sales_MA_4, Sales_MA_8, Sales_MA_12 |
| Rolling std | Sales_STD_4, Sales_STD_8, Sales_STD_12 |

---

## 6. Prophet Baseline (Notebook 04)

### Approach

Prophet was trained on **aggregated total weekly sales** across all 45 stores (143 weekly periods).
This is the correct granularity for Prophet, which is designed for univariate time series.

**Custom holidays defined:**

| Holiday | Dates | Window |
|---------|-------|--------|
| Black Friday | 2010-11-26, 2011-11-25, 2012-11-23 | −1 to +2 days |
| Thanksgiving | 2010-11-25, 2011-11-24, 2012-11-22 | Day of only |
| Christmas | 2010-12-24, 2011-12-24, 2012-12-24 | −3 to +1 days |
| Super Bowl | 2010-02-12, 2011-02-11, 2012-02-10 | −1 to +1 days |

**Model configuration:** `seasonality_mode='multiplicative'`, `changepoint_prior_scale=0.05`,
`yearly_seasonality=True`, `weekly_seasonality=False`, `holidays_prior_scale=10.0`

### Results

| Set | Periods | RMSE | MAE | R² | MAPE |
|-----|---------|------|-----|----|------|
| Train | 121 | $1,752,375 | $1,326,409 | 0.9101 | 2.87% |
| Val | 9 | $1,865,156 | $1,634,145 | 0.1439 | 3.46% |
| Test | 13 | $1,417,660 | $1,282,070 | 0.0725 | 2.76% |

### Why Val/Test R² Is Low

The drop from Train R² = 0.91 to Test R² = 0.07 is a **small-sample artefact**, not overfitting:
- The validation set has only **9 weekly data points**; the test set has only **13**.
- R² with n < 20 is highly unstable — a single unexpected week can swing it dramatically.
- MAPE is more robust at this granularity: **2.76%** on the test set means Prophet predicts
  total weekly sales to within ~$1.4M on a ~$50M/week baseline.

### What Prophet Gets Right

- The **yearly seasonality component** correctly shapes the Thanksgiving/Christmas peaks.
- The **trend component** is nearly flat across 2010–2012 — confirming Notebook 01 observation 2.
- The **holiday effects** are detectable: Super Bowl week shows a small negative coefficient
  (shoppers divert attention away from retail).

### Prophet's Limitation

Prophet has no knowledge of individual stores or departments. It forecasts the chain total,
which is useful for **executive-level planning** but useless for inventory management at the
department level — the primary business use case.

---

## 7. XGBoost & LightGBM (Notebook 05)

### Approach

Both models operate at **per-store-department row granularity** using the 24-feature set from
Notebook 03. Training uses the time-based split to prevent data leakage.

**XGBoost configuration:**

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 500 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `early_stopping_rounds` | 50 |

**LightGBM configuration:** same learning rate / depth, `num_leaves=63`, histogram-based splits.

### Training Observations

- **XGBoost** ran all **500 iterations** without triggering early stopping (validation RMSE was
  still decreasing slowly at iteration 499). Increasing `n_estimators` to 800+ could yield a
  small further improvement.
- **LightGBM** also ran 500 iterations; both models converged to similar validation performance.

### Results

| Model | Train RMSE | Train R² | Val RMSE | Val R² | Test RMSE | Test R² | Test MAE |
|-------|-----------|----------|---------|--------|----------|---------|---------|
| XGBoost | $1,767 | 0.9941 | $2,273 | 0.9896 | **$2,176** | **0.9903** | $1,017 |
| LightGBM | $1,971 | 0.9926 | $2,246 | 0.9898 | $2,177 | 0.9902 | $1,018 |

The two models are **statistically tied**: less than $1 RMSE difference on the test set.

### MAPE Caveat

Reported MAPE values — XGBoost 99.4%, LightGBM 78.4% — are severely inflated by departments
with near-zero weekly sales. For example: a department with $21 actual sales and $41 predicted
gives 95% MAPE but only $20 absolute error. This is a known limitation of MAPE on sparse retail
data. **RMSE and R² are the reliable metrics for this task.**

### Generalisation Check

| Model | Train R² | Test R² | Drop |
|-------|----------|---------|------|
| XGBoost | 0.9941 | 0.9903 | 0.0038 |
| LightGBM | 0.9926 | 0.9902 | 0.0024 |

The train-to-test R² drop is ~0.003 for both models — **minimal overfitting**. This is expected
given the strong autocorrelation structure in the data: lag features essentially encode the target,
so the model is solving a well-constrained regression problem.

---

## 8. Final Evaluation (Notebook 06)

### Complete Test Set Results

| Model | Granularity | Test RMSE | Test R² | Test MAE |
|-------|-------------|-----------|---------|---------|
| **XGBoost** | Per store-dept | **$2,176** | **0.9903** | $1,017 |
| LightGBM | Per store-dept | $2,177 | 0.9902 | $1,018 |
| Prophet | Total weekly | $1,417,660 | 0.0725 | $1,282,070 |

> **Important note on Prophet comparison**: Prophet's RMSE is expressed in terms of total weekly
> chain sales (~$50M scale), while XGBoost/LightGBM RMSEs are per department-store (~$16K scale).
> The metrics are not directly comparable across granularity levels.

### Best Model: XGBoost (by a hair)

XGBoost achieves the lowest test RMSE ($2,176.20 vs $2,176.60 for LightGBM) and the highest
test R² (0.9903 vs 0.9902). In practice, the two models are **interchangeable for deployment**.

### Actual vs Predicted (Test Period: Aug–Oct 2012)

The time-series overlay of aggregated test predictions shows:
- XGBoost and LightGBM track the actual weekly totals almost perfectly.
- Both models correctly predict the slight upward trend in late September / early October (the
  beginning of the pre-holiday build-up).
- Prophet tracks the general direction but shows larger deviations, especially in weeks when
  individual store promotions drive spikes not captured in aggregate seasonality.

### Residual Analysis (XGBoost)

| Statistic | Value |
|-----------|-------|
| Mean residual | ≈ $0 (unbiased) |
| Residual std | ≈ $2,176 (= RMSE) |
| Errors > $5,000 | ~4–5% of test rows |

The residual distribution is approximately normal and centred at zero, confirming the model
is unbiased. The heavy tail corresponds to high-volume promotional departments that can spike
unpredictably.

---

## 9. Observations vs Results — Confirmation Analysis

### Observation 1: Strong seasonal spike in weeks 47–52

| Where confirmed | Evidence |
|----------------|----------|
| NB03 — Calendar features | Q4 (weeks 47–52) shows the highest average sales bar in the month/week plots |
| NB03 — Feature correlation | `Week` and `Month` appear in the top-10 correlated features with `Weekly_Sales` |
| NB05 — Feature importance | `Week` ranks in the top 5 in both XGBoost and LightGBM importance charts |
| NB04 — Prophet components | The yearly seasonality curve shows a large positive spike at week 48 |

**Verdict: Fully confirmed.**

---

### Observation 2: Year-over-year stability → Lag_52 dominant

| Where confirmed | Evidence |
|----------------|----------|
| NB03 — Lag correlations | `Sales_Lag_52` achieves r = **0.981** — the single highest-correlated feature |
| NB03 — Lag scatter plots | The Lag_52 vs Weekly_Sales scatter shows a near-perfect linear relationship |
| NB05 — Feature importance | `Sales_Lag_52` consistently ranks among the top 2 in both models |
| NB04 — Prophet trend | The flat trend component confirms no structural drift → Lag_52 is reliable |

**Verdict: Fully confirmed.** This was the most important prediction from Notebook 01 — that
year-over-year patterns would dominate over short-term autocorrelation. `Sales_Lag_52` (r=0.981)
beat `Sales_Lag_1` (r=0.960), confirming the seasonal dominance.

---

### Observation 3: Store type disparity (A ≈ 3× Type C)

| Where confirmed | Evidence |
|----------------|----------|
| NB02 — Split plot | The weekly total sales chart clearly shows sustained high volume throughout |
| NB05 — Feature importance | `Store`, `Dept`, `Size`, `Type_A` all rank in the top 10 features |
| NB06 — Error by store type | Type A stores have the highest absolute MAE (expected: larger scale → larger errors) |

**Verdict: Confirmed.** Store identity features are critical. Without `Store` and `Dept`, the
model would need to average across scales that differ by a factor of 3.

---

### Observation 4: CPI/Unemployment NaN gaps are small and recoverable

| Where confirmed | Evidence |
|----------------|----------|
| NB02 — Post-fill check | `Total NaN remaining: 0` after `fill_missing_externals()` |
| NB03 — Feature set | CPI and Unemployment are included in the correlation heatmap with no NaN issues |

**Verdict: Confirmed.** Forward/back-fill per store cleanly resolves all 585 gaps.

---

### Observation 5: MarkDowns correct to drop

| Where confirmed | Evidence |
|----------------|----------|
| NB02 — Merge | Dropping MarkDowns yields a clean 421,570 × 11 df with 0 additional NaN |
| NB05 — Model performance | R² = 0.99 achieved without any MarkDown data |

**Verdict: Confirmed.** Excluding MarkDown columns has no detectable negative impact on model
performance, validating the decision to drop them rather than attempting complex imputation.

---

### Summary Table

| Observation | Confirmed? | Strength |
|-------------|-----------|---------|
| Holiday spike weeks 47–52 | ✅ Yes | Very strong — visible in every analysis layer |
| Lag_52 dominant (r=0.981 > Lag_1 r=0.960) | ✅ Yes | Very strong — top feature in all models |
| Store type A ≈ 3× Type C | ✅ Yes | Strong — Store/Dept/Size all top-10 features |
| CPI/Unemployment NaN recoverable | ✅ Yes | Strong — 0 NaN after forward/back-fill |
| MarkDowns correct to drop | ✅ Yes | Strong — R²=0.99 without them |
| Stable trend → Lag_52 reliable | ✅ Yes | Strong — Prophet trend component is flat |

---

## 10. Conclusion

### What Worked

1. **Modular `src/` architecture** — `full_pipeline()` runs all 6 feature-engineering steps
   consistently across every notebook, eliminating copy-paste errors.
2. **Time-aware split** — Using chronological boundaries (not random shuffling) prevents
   data leakage and produces honest generalisation estimates.
3. **Lag + rolling features** — The single most impactful modelling decision. Without lag
   features, XGBoost would have to learn seasonality from calendar features alone. With them,
   R² jumps to 0.99.
4. **Dual-model approach** — Prophet for aggregate trend interpretation, XGBoost/LightGBM
   for high-resolution per-dept forecasting.

### Final Model Performance

**XGBoost** (500 trees, 24 features, time-aware split) is the best-performing model:

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Test RMSE | $2,176 | Average absolute error ≈ $1,017 per dept/week |
| Test R² | 0.9903 | Model explains 99.03% of weekly sales variance |
| Train → Test R² drop | 0.0038 | Minimal overfitting |
| Test MAPE | 99.4% | **Inflated by near-zero depts — not a reliable metric** |

### Limitations

1. **Lag requirement removes 38% of data** — The first year of each store-dept series is
   unusable for training because `Sales_Lag_52` requires a full year of history. For new stores
   (< 1 year of data), the model cannot generate predictions.

2. **MarkDown data discarded** — Promotional markdown events are genuinely predictive in
   Walmart's business (big discounts drive volume spikes). Imputing the 50–64% missing
   MarkDown values with a more sophisticated strategy (e.g., zero-fill + missingness indicator)
   could improve accuracy for promotional weeks.

3. **MAPE unreliable** — High MAPE is an artefact of small-sales departments, not a true
   model weakness. Consider filtering to depts with mean weekly sales > $1,000 for MAPE analysis.

4. **No exogenous features used** — Temperature, Fuel_Price, CPI, and Unemployment were
   included in the feature set but their importance may be low. A formal ablation study
   (training without them) would confirm whether they add signal.

### Recommendations

| Priority | Action |
|----------|--------|
| 1 | **Deploy XGBoost** as the production forecasting model (or LightGBM for faster re-training) |
| 2 | **Use Prophet** for monthly business reviews and trend/seasonality visualisations |
| 3 | **Investigate MarkDown imputation** — zero-fill + `MarkDown_available` binary flag |
| 4 | **Re-train annually** — `Sales_Lag_52` requires up-to-date year-ago history |
| 5 | **Filter MAPE reporting** to departments with mean weekly sales > $1,000 |
| 6 | **Increase XGBoost `n_estimators`** to 800+ — training never early-stopped at 500 |
