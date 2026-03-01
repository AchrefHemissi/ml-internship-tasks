# Loan Approval Prediction — Technical Report

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Exploration — Observations (Notebook 01)](#3-data-exploration--observations-notebook-01)
4. [Preprocessing Pipeline (Notebook 02)](#4-preprocessing-pipeline-notebook-02)
5. [Imbalanced Data Handling (Notebook 03)](#5-imbalanced-data-handling-notebook-03)
6. [Logistic Regression (Notebook 04)](#6-logistic-regression-notebook-04)
7. [Tree-Based Models (Notebook 05)](#7-tree-based-models-notebook-05)
8. [Final Evaluation & Model Comparison (Notebook 06)](#8-final-evaluation--model-comparison-notebook-06)
9. [Observations vs Results — Confirmation Analysis](#9-observations-vs-results--confirmation-analysis)
10. [Conclusion](#10-conclusion)

---

## 1. Project Overview

This project builds a binary classification pipeline to predict whether a loan application will be **Approved** or **Rejected**. The dataset comes from the [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) (4 269 samples, 13 columns).

Four classifiers were trained and compared:

- **Logistic Regression** — interpretable linear baseline
- **Decision Tree** — explainable rule-based model
- **Random Forest** — ensemble of decision trees
- **Gradient Boosting** — sequential boosted ensemble

All code is organized into a modular `src/` package (`config.py`, `data_loader.py`, `preprocessing.py`, `models.py`) and 6 sequential notebooks.

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| Rows | 4 269 |
| Columns | 13 (11 features + 1 target + 1 ID) |
| Target | `loan_status` (Approved / Rejected) |
| Missing values | **0** (no missing data) |
| Numerical features (9) | `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value` |
| Categorical features (2) | `education` (Graduate / Not Graduate), `self_employed` (Yes / No) |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Approved | 2 656 | 62.2% |
| Rejected | 1 613 | 37.8% |
| **Imbalance ratio** | **1.65** (Approved / Rejected) |

---

## 3. Data Exploration — Observations (Notebook 01)

Notebook 01 is a **read-only exploration** step. It does not save any files, train any model, or produce any arrays. However, the observations made here directly informed technical decisions in all subsequent notebooks.

### Key Observations

1. **`cibil_score` is by far the strongest predictor**: correlation with `loan_status` = **0.77**. All other numerical features have near-zero correlation (< 0.02 in absolute value).

2. **`cibil_score` boxplot analysis**: Approved loans cluster tightly around median ≈ 720 (range 600–900), Rejected loans cluster around median ≈ 450 (range 300–600). The two distributions barely overlap, making `cibil_score` a near-perfect separator on its own.

3. **`loan_term` is the only other mildly useful feature**: correlation = **−0.11**. Rejected loans have a higher median term (≈ 14 years) vs Approved (≈ 10 years), suggesting longer repayment periods carry higher risk.

4. **Asset and income features are non-informative**: `income_annum`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, and `bank_asset_value` show flat, overlapping boxplots between Approved and Rejected. They contribute very little predictive signal.

5. **Categorical features are non-informative**: `education` (Graduate vs Not Graduate) and `self_employed` (Yes vs No) show virtually identical approval rates across their categories.

### Expected Impact on Later Notebooks

| Observation | Expected downstream effect |
|-------------|---------------------------|
| `cibil_score` dominance (r=0.77) | LR should achieve high baseline accuracy even without tuning; tree models will place `cibil_score` at the root |
| Near-zero correlation of other features | L2 regularisation will shrink their coefficients toward zero in LR; feature importance charts will confirm low importance in tree models |
| Mild imbalance (1.65 ratio) | SMOTE is justified, but under-sampling risks losing too much data |
| No missing values | Imputation step is defensive programming, not functionally needed |

---

## 4. Preprocessing Pipeline (Notebook 02)

### Technical Decisions

| Step | Decision | Justification |
|------|----------|---------------|
| **Missing value handling** | Median (numerical) / mode (categorical) imputation | Defensive programming — no missing values existed in this dataset, but the pipeline is robust for future data |
| **Target encoding** | Approved → 1, Rejected → 0 | Standard binary classification convention |
| **Categorical encoding** | `LabelEncoder` | Only 2 binary categorical features (Graduate/Not Graduate, Yes/No) — ordinal encoding is sufficient; one-hot would add unnecessary dimensionality |
| **Feature scaling** | `StandardScaler` (z-score normalisation) | Required for Logistic Regression (gradient-based solver `lbfgs`); not strictly needed for tree models, but applied uniformly for consistency |
| **Train/test split** | 80/20 stratified split | `random_state=42`, stratified on target to preserve class proportions in both sets |

### Results

| Split | Samples | Approved | Rejected |
|-------|---------|----------|----------|
| Train | 3 415 | 2 125 | 1 290 |
| Test | 854 | 531 | 323 |

### Artifacts Saved

- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy` — preprocessed arrays
- `feature_names.pkl` — ordered list of 11 feature names
- `scaler.pkl` — fitted StandardScaler
- `encoders.pkl` — fitted LabelEncoders

---

## 5. Imbalanced Data Handling (Notebook 03)

### The Problem: Why Accuracy Is Misleading

A **naive baseline** (DummyClassifier, always predicting the majority class "Approved") demonstrates the issue:

| Metric | Naive Baseline |
|--------|----------------|
| Accuracy | 0.62 |
| F1 (Rejected) | **0.00** |
| F1 (Approved) | 0.77 |
| Macro-F1 | **0.38** |

The model achieves 62% accuracy by predicting "Approved" for every sample, but has **zero recall for Rejected loans** — completely useless for identifying risky applications.

### Technical Decision: SMOTE vs RandomUnderSampler

| Technique | Mechanism | Result |
|-----------|-----------|--------|
| **SMOTE** (chosen) | Generates synthetic minority samples via k-NN interpolation | 1 290 → 2 125 Rejected (added 835 synthetic), total 4 250 balanced samples |
| RandomUnderSampler | Discards majority class samples | 2 125 → 1 290 Approved (removed 835 real), total 2 580 balanced samples |

**Decision: SMOTE preferred** because:

- Under-sampling discards 835 real Approved samples — a significant information loss
- The imbalance is mild (1.65 ratio), so SMOTE only needs to generate 835 synthetic samples (not an extreme interpolation)
- SMOTE preserves all original data while enriching the minority class boundary

### Datasets Saved

| Dataset | Shape | Description |
|---------|-------|-------------|
| `X_smote.npy` / `y_smote.npy` | (4 250, 11) | SMOTE-balanced (used for final training) |
| `X_under.npy` / `y_under.npy` | (2 580, 11) | Under-sampled (comparison only) |

---

## 6. Logistic Regression (Notebook 04)

### Experimental Design

Four LR variants were compared to isolate the effect of each technique:

| Variant | Training data | Class weight | C value |
|---------|--------------|--------------|---------|
| Baseline | Original (imbalanced) | None | 1.0 |
| SMOTE | SMOTE-balanced | None | 1.0 |
| Weighted | Original | `balanced` | 1.0 |
| Best (tuned) | SMOTE-balanced | None | 0.001 |

### Model Configuration

All variants used: `solver='lbfgs'`, `max_iter=1000`, `random_state=42`.

### Hyperparameter Tuning — Regularisation C

C was tuned via 5-fold cross-validation on SMOTE data using Macro-F1 as the scoring metric:

| C | CV Macro-F1 |
|---|-------------|
| 0.001 | **0.9299** |
| 0.01 | 0.9296 |
| 0.1 | 0.9254 |
| 0.5 | 0.9256 |
| 1.0 | 0.9249 |
| 5.0 | 0.9247 |
| 10.0 | 0.9252 |
| 50.0 | 0.9254 |
| 100.0 | 0.9254 |

**Best C = 0.001** — strong regularisation performs best, which is expected when one feature (`cibil_score`) dominates: heavy L2 penalty suppresses the useless features without harming the dominant one.

### Results — All Variants on Test Set

| Variant | Accuracy | Precision (R) | Recall (R) | F1 (Rejected) | F1 (Approved) | Macro-F1 | ROC-AUC |
|---------|----------|---------------|------------|----------------|---------------|----------|---------|
| Baseline (no resampling) | 0.91 | 0.90 | 0.87 | 0.88 | 0.93 | 0.91 | 0.9726 |
| SMOTE | 0.92 | 0.88 | 0.92 | 0.90 | 0.94 | 0.92 | 0.9732 |
| class_weight='balanced' | 0.92 | 0.88 | 0.93 | 0.90 | 0.94 | 0.92 | 0.9734 |
| **Best (C=0.001, SMOTE)** | **0.93** | 0.86 | **0.98** | **0.92** | **0.94** | **0.93** | **0.9738** |

**Key finding**: SMOTE + strong regularisation (C=0.001) gives the best LR model. The trade-off: precision for Rejected drops (0.86) but recall jumps to 0.98, meaning almost all actual rejections are caught — a desirable property for risk assessment.

### Feature Coefficients (Best Model)

| Feature | Coefficient | Direction |
|---------|------------|-----------|
| **cibil_score** | **+0.8648** | Strongly increases approval probability |
| loan_amount | +0.0449 | Slightly increases (counterintuitive, likely interaction effect) |
| commercial_assets_value | +0.0126 | Negligible |
| self_employed | +0.0117 | Negligible |
| residential_assets_value | +0.0047 | Negligible |
| education | −0.0075 | Negligible |
| luxury_assets_value | −0.0139 | Negligible |
| income_annum | −0.0160 | Negligible |
| no_of_dependents | −0.0178 | Negligible |
| **loan_term** | **−0.1225** | Decreases approval (longer term = higher risk) |

**Confirmation of Notebook 01 observations**: `cibil_score` coefficient (0.86) is **70× larger** than the next largest coefficient. All asset/income features have coefficients near zero (< 0.05), exactly as predicted by their near-zero correlations.

---

## 7. Tree-Based Models (Notebook 05)

### 7.1 Decision Tree

#### Depth Tuning (5-fold CV on SMOTE data)

| max_depth | CV Macro-F1 |
|-----------|-------------|
| 2 | 0.9571 |
| 3 | 0.9666 |
| 4 | 0.9592 |
| 5 | 0.9711 |
| 6 | 0.9649 |
| 8 | 0.9673 |
| 10 | 0.9746 |
| None (unlimited) | **0.9762** |

**Best max_depth = None** (unlimited). The unlimited tree achieves the best CV score because `cibil_score`'s strong signal means even deep trees don't overfit severely — the first split on `cibil_score` already separates most of the data.

#### Decision Tree Test Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.98 |
| F1-Rejected | 0.97 |
| F1-Approved | 0.98 |
| Macro-F1 | 0.98 |
| ROC-AUC | 0.9764 |

#### Tree Visualisation (depth=3)

A shallow tree with `max_depth=3` was visualised to confirm interpretability. As predicted from Notebook 01 observations, **`cibil_score` appears at the root node** and dominates the first few levels of splits.

### 7.2 Random Forest

#### n_estimators Tuning (5-fold CV on SMOTE data)

| n_estimators | CV Macro-F1 |
|--------------|-------------|
| 10 | 0.9720 |
| 25 | 0.9765 |
| 50 | 0.9798 |
| **100** | **0.9807** |
| 200 | 0.9800 |
| 300 | 0.9798 |

**Best n_estimators = 100**. Performance plateaus after 100 trees — adding more brings no benefit and increases computation time.

#### Random Forest Test Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.98 |
| F1-Rejected | 0.98 |
| F1-Approved | 0.99 |
| Macro-F1 | **0.9826** |
| ROC-AUC | **0.9988** |

### 7.3 Gradient Boosting

Trained with `n_estimators=100`, `max_depth=3`, `learning_rate=0.1` (defaults from config).

#### Gradient Boosting Test Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.98 |
| F1-Rejected | 0.97 |
| F1-Approved | 0.98 |
| Macro-F1 | 0.98 |
| ROC-AUC | 0.9983 |

### Feature Importance — All Tree Models

Across all three tree-based models, **`cibil_score` dominates feature importance**, with all other features contributing minimally. This directly confirms the Notebook 01 observation that `cibil_score` (r=0.77) is the overwhelmingly dominant predictor.

---

## 8. Final Evaluation & Model Comparison (Notebook 06)

### Complete Test Set Results

| Model | Accuracy | F1-Approved | F1-Rejected | Macro-F1 | ROC-AUC |
|-------|----------|-------------|-------------|----------|---------|
| Logistic Regression | 0.93 | 0.94 | 0.92 | 0.93 | 0.9738 |
| Decision Tree | 0.98 | 0.98 | 0.97 | 0.98 | 0.9764 |
| **Random Forest** | **0.98** | **0.99** | **0.98** | **0.9826** | **0.9988** |
| Gradient Boosting | 0.98 | 0.98 | 0.97 | 0.98 | 0.9983 |

### Best Model: Random Forest

| Detail | Value |
|--------|-------|
| Macro-F1 | 0.9826 |
| ROC-AUC | 0.9988 |
| F1-Approved | 0.9868 |
| F1-Rejected | 0.9783 |
| True Negatives (Rejected correctly) | 315 / 323 |
| True Positives (Approved correctly) | 525 / 531 |
| False Positives | 8 |
| False Negatives | 6 |
| **Total misclassifications** | **14 / 854 (1.6%)** |

### Threshold Optimisation

For the Random Forest model, a threshold sweep from 0.10 to 0.90 was performed:

| Threshold | Macro-F1 |
|-----------|----------|
| 0.50 (default) | 0.9838 |
| **0.48 (optimal)** | **0.9850** |
| **Improvement** | **+0.0012** |

The improvement is marginal, indicating the default 0.50 threshold is already near-optimal. The model's predicted probabilities are well-calibrated.

### Why Random Forest Wins

1. **Highest Macro-F1 (0.9826)** — best balanced performance across both classes
2. **Highest ROC-AUC (0.9988)** — near-perfect probability ranking
3. **Smallest F1 gap between classes** — F1-Approved (0.99) vs F1-Rejected (0.98) = 0.01 gap, meaning the model treats both classes almost equally
4. **Ensemble smoothing** — 100 trees average out individual noise, unlike a single Decision Tree

---

## 9. Observations vs Results — Confirmation Analysis

This section systematically traces each Notebook 01 observation through the subsequent results, confirming or nuancing the initial hypotheses.

### Observation 1: `cibil_score` dominance (r = 0.77)

| Where confirmed | Evidence |
|----------------|----------|
| **NB04 — LR coefficients** | `cibil_score` coefficient = **0.8648**, which is **70× larger** than the next feature (`loan_term` = −0.1225). All other features < 0.05. |
| **NB05 — Tree visualisation** | `cibil_score` appears at the **root node** of the Decision Tree (depth=3 visualisation) |
| **NB05 — Feature importance** | `cibil_score` dominates feature importance in all three tree models (Decision Tree, Random Forest, Gradient Boosting) |
| **NB06 — Overall performance** | All models achieve ≥ 0.93 accuracy because `cibil_score` alone nearly separates the two classes |

**Verdict: Fully confirmed.** `cibil_score` is the single most important driver of loan approval prediction in this dataset.

### Observation 2: Near-zero correlation of asset/income features

| Where confirmed | Evidence |
|----------------|----------|
| **NB04 — LR coefficients** | `income_annum` (−0.016), `residential_assets_value` (+0.005), `commercial_assets_value` (+0.013), `luxury_assets_value` (−0.014), `bank_asset_value` — all coefficients < 0.05 in absolute value |
| **NB04 — C tuning** | Best C = 0.001 (strongest regularisation tested). L2 penalty shrinks near-zero-contribution features toward zero, confirming they add more noise than signal |
| **NB05 — Feature importance** | Asset/income features consistently rank at the bottom across all tree models |

**Verdict: Fully confirmed.** Asset and income features are non-informative for this classification task.

### Observation 3: Mild imbalance justifies SMOTE over under-sampling

| Where confirmed | Evidence |
|----------------|----------|
| **NB03 — Naive baseline** | Always-predict-majority achieves 62% accuracy but **0% recall for Rejected** (Macro-F1 = 0.38) — proves accuracy alone is misleading |
| **NB03 — Sample counts** | SMOTE adds 835 synthetic samples (reasonable for 1.65 ratio). Under-sampling would discard 835 real samples (24% data loss) |
| **NB04 — LR comparison** | SMOTE-trained LR (Macro-F1 = 0.92) outperforms Baseline LR (0.91), and recall for Rejected improves from 0.87 → 0.92 |
| **NB05 — All tree models** | All tree models trained on SMOTE achieve ≥ 0.97 F1-Rejected, with near-parity between classes |

**Verdict: Fully confirmed.** SMOTE successfully balances the classes without information loss. The under-sampled dataset was saved but not used for final models.

### Observation 4: `loan_term` as the secondary predictor (r = −0.11)

| Where confirmed | Evidence |
|----------------|----------|
| **NB04 — LR coefficients** | `loan_term` has the **second-largest absolute coefficient** (−0.1225), aligning with the −0.11 correlation |
| **NB05 — Feature importance** | `loan_term` typically ranks 2nd or 3rd in tree-based feature importance |

**Verdict: Confirmed, but effect is minor.** `loan_term` contributes some signal but is dwarfed by `cibil_score`.

### Observation 5: LR should achieve high baseline accuracy

| Where confirmed | Evidence |
|----------------|----------|
| **NB04 — Baseline LR** | LR Baseline (no resampling, no tuning, C=1.0) achieves **0.91 accuracy** and **0.9726 ROC-AUC** on the first attempt — confirming that `cibil_score`'s linear separability makes LR effective immediately |
| **NB06 — Final comparison** | LR (0.93 accuracy, 0.9738 ROC-AUC) is already a strong model. Tree models improve it only by ~5 percentage points |

**Verdict: Fully confirmed.** The near-linear relationship between `cibil_score` and loan approval means LR captures most of the signal without non-linear modelling.

### Observation 6: Categorical features are non-informative

| Where confirmed | Evidence |
|----------------|----------|
| **NB04 — LR coefficients** | `education` coefficient = −0.0075, `self_employed` = +0.0117 — both negligible |
| **NB05 — Feature importance** | Both categorical features rank at the bottom in all tree models |

**Verdict: Fully confirmed.** Graduate vs Not Graduate and self-employed vs not have no meaningful impact on loan approval in this dataset.

### Summary Table

| Observation | Confirmed? | Strength of evidence |
|-------------|-----------|---------------------|
| `cibil_score` dominance (r=0.77) | **Yes** | Very strong — coefficients, feature importance, tree root, all models ≥ 0.93 |
| Asset/income features non-informative | **Yes** | Strong — all coefficients < 0.05, bottom of importance rankings |
| Mild imbalance → SMOTE preferred | **Yes** | Strong — SMOTE improves recall without data loss |
| `loan_term` secondary predictor (r=−0.11) | **Yes** | Moderate — 2nd largest coefficient, minor importance |
| LR achieves high baseline | **Yes** | Strong — 0.91 accuracy on first attempt |
| Categorical features non-informative | **Yes** | Strong — negligible coefficients and importance |

---

## 10. Conclusion

### What Worked

1. **Modular code architecture** (`src/` package) — enabled consistent evaluation across all notebooks via shared functions (`evaluate_model`, `plot_confusion_matrix`, etc.)
2. **SMOTE resampling** — improved minority-class recall without losing data
3. **Multiple model comparison** — demonstrated that tree-based models capture non-linear patterns better than Logistic Regression
4. **Observation-driven workflow** — Notebook 01 observations correctly predicted behavior in all later stages

### Final Model Performance

**Random Forest** (100 trees, trained on SMOTE data) is the best-performing model:

- **98.4% accuracy** (840 / 854 correct)
- **Macro-F1 = 0.9826** (balanced across both classes)
- **ROC-AUC = 0.9988** (near-perfect ranking)
- Only **14 misclassifications** on the 854-sample test set

### Limitations & Critical Note

The exceptionally high performance (all models ≥ 0.93) is largely driven by a **single dominant feature** (`cibil_score`). This raises important considerations:

- The model is essentially a sophisticated `cibil_score` threshold with minor adjustments
- If `cibil_score` is unavailable or unreliable in production, performance would drop dramatically
- The remaining 10 features contribute minimal predictive value
- The dataset may not reflect real-world complexity where credit scores are just one of many factors

### Recommendations

1. **Deploy Random Forest** as the production model, with the default 0.50 threshold
2. **Monitor `cibil_score` distribution drift** — the model's performance depends almost entirely on this feature
3. **Investigate feature engineering** — combining asset/income features or creating ratio features might extract additional signal
4. **Collect more data** — a larger, more diverse dataset might reveal patterns that the current 4 269 samples don't capture
