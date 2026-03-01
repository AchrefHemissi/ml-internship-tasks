# Loan Approval Prediction

A professional, modular machine-learning project that predicts whether a loan application will be **approved or rejected** using the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) from Kaggle.

The project compares **Logistic Regression** vs **tree-based models** (Decision Tree, Random Forest, Gradient Boosting), addresses **class imbalance with SMOTE**, and evaluates performance with precision, recall, and F1-score.

---

## Project Structure

```text
Task_4_Loan_Approval_Prediction/
│
├── notebooks/                          # Step-by-step notebooks (run in order)
│   ├── 01_data_exploration.ipynb       # Dataset inspection · class balance · distributions
│   ├── 02_preprocessing.ipynb          # Missing values · encoding · scaling · save arrays
│   ├── 03_imbalanced_data.ipynb        # SMOTE · RandomUnderSampler · imbalance analysis
│   ├── 04_logistic_regression.ipynb    # LR baseline · C tuning · feature coefficients
│   ├── 05_tree_models.ipynb            # Decision Tree · Random Forest · Gradient Boosting
│   └── 06_evaluation.ipynb             # Final comparison · ROC/PR curves · best model
│
├── src/                                # Importable Python package
│   ├── __init__.py
│   ├── config.py                       # All paths & hyperparameters in one place
│   ├── data_loader.py                  # CSV loader · stratified train/test split
│   ├── preprocessing.py                # Imputation · encoding · scaling pipeline
│   └── models.py                       # Classifier factories + evaluation utilities
│
├── data/                               # Dataset files (not tracked by git)
│   └── loan_approval_dataset.csv       # ← place your downloaded CSV here
│
├── models/                             # Saved model checkpoints (auto-created)
│
├── RAPPORT.md                          # Full technical report with all findings
├── requirements.txt                    # Python dependencies
├── setup_env.sh                        # One-click Linux / macOS / WSL venv setup script
└── README.md
```

---

## Quick Start

### 1. Clone the project

```bash
git clone https://github.com/AchrefHemissi/ml-internship-tasks.git
cd ml-internship-tasks/Task_4_Loan_Approval_Prediction
```

### 2. Set up the virtual environment

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This will:

- Create a `.venv` virtual environment
- Install all dependencies from `requirements.txt`
- Register a Jupyter kernel named **`loan-approval`**

### 3. Download the dataset

**Option A — Kaggle CLI** (recommended):

```bash
kaggle datasets download \
  -d architsharma01/loan-approval-prediction-dataset \
  -p data/ --unzip
```

**Option B — Manual download**:

1. Go to [kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
2. Download and extract into `data/`

Make sure `data/loan_approval_dataset.csv` is present before running the notebooks.

### 4. Launch Jupyter

```bash
source .venv/bin/activate
jupyter notebook
```

Open the notebooks **in order** and select the `loan-approval` kernel.

---

## Notebook Pipeline

| # | Notebook | What it does | Output |
|---|---|---|---|
| 01 | Data Exploration | Inspect CSV, class distribution (62/38 split), correlation heatmap — identifies `cibil_score` as dominant predictor (r=0.77) | Charts, observations |
| 02 | Preprocessing | Impute missing values, LabelEncode categoricals, StandardScaler, stratified 80/20 split | `data/*.npy`, `models/scaler.pkl`, `models/encoders.pkl` |
| 03 | Imbalanced Data | Quantify imbalance (ratio 1.65), naive baseline (Macro-F1=0.38), apply SMOTE (+835 synthetic) and RUS | `data/X_smote.npy`, `data/X_under.npy` |
| 04 | Logistic Regression | Train 4 LR variants, tune C (best=0.001), analyse coefficients — `cibil_score` coeff 70× larger than others | `models/logistic_regression.pkl` |
| 05 | Tree Models | Decision Tree (depth tuning), Random Forest (100 trees), Gradient Boosting, feature importance | `models/decision_tree.pkl`, `models/random_forest.pkl`, `models/gradient_boosting.pkl` |
| 06 | Evaluation | Compare all 4 models, confusion matrices, ROC/PR curves, threshold optimisation — **Random Forest wins** (Macro-F1=0.9826, ROC-AUC=0.9988) | Final metrics |

---

## Dataset

### Loan Approval Prediction Dataset (architsharma01)

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | int | Unique identifier |
| `no_of_dependents` | int | Number of dependents (0–5) |
| `education` | str | Graduate / Not Graduate |
| `self_employed` | str | Yes / No |
| `income_annum` | int | Annual income |
| `loan_amount` | int | Requested loan amount |
| `loan_term` | int | Loan term in years |
| `cibil_score` | int | Credit score (300–900) |
| `residential_assets_value` | int | Residential assets |
| `commercial_assets_value` | int | Commercial assets |
| `luxury_assets_value` | int | Luxury assets |
| `bank_asset_value` | int | Bank assets |
| `loan_status` | str | **Target**: Approved / Rejected |

- **4 269 rows** · 13 columns · **0 missing values**
- Class distribution: 2 656 Approved (62.2%) / 1 613 Rejected (37.8%) — imbalance ratio 1.65
- **Key insight**: `cibil_score` alone has 0.77 correlation with the target; all other features < 0.02

---

## Preprocessing Pipeline

Each sample goes through 5 steps before entering a classifier:

1. **Missing value imputation** — median for numerical, mode for categorical (defensive — no missing values in this dataset)
2. **Target encoding** — `Approved → 1`, `Rejected → 0`
3. **Label encoding** — categorical columns (`education`: Graduate→0, Not Graduate→1; `self_employed`: No→0, Yes→1)
4. **Train/test split** — 80/20 stratified by target (3 415 train / 854 test)
5. **StandardScaler** — zero mean, unit variance for all 11 features

---

## Handling Class Imbalance

A naive baseline (always predict "Approved") achieves 62% accuracy but **0% recall for Rejected** (Macro-F1 = 0.38), demonstrating why accuracy alone is misleading.

| Strategy | Technique | Result |
|----------|-----------|--------|
| None (baseline) | No resampling | Original 3 415 samples |
| `class_weight='balanced'` | Built-in LR loss weighting | Original size, adjusted loss |
| **SMOTE** *(chosen)* | Synthetic minority over-sampling | 4 250 balanced samples (+835 synthetic) |
| RandomUnderSampler | Discard majority samples | 2 580 samples (−835 real data lost) |

**SMOTE is preferred** because the imbalance is mild (1.65 ratio) — under-sampling would discard 835 real samples (24% data loss), while SMOTE preserves all original data.

---

## Models & Results

### Logistic Regression (Notebook 04)

| Variant | Accuracy | Macro-F1 | ROC-AUC |
|---------|----------|----------|---------|
| Baseline (C=1.0, no SMOTE) | 0.91 | 0.91 | 0.9726 |
| SMOTE (C=1.0) | 0.92 | 0.92 | 0.9732 |
| class_weight='balanced' | 0.92 | 0.92 | 0.9734 |
| **Best (C=0.001, SMOTE)** | **0.93** | **0.93** | **0.9738** |

- Best C = 0.001 (strongest regularisation) — L2 penalty silences noisy features, improving generalisation
- `cibil_score` coefficient = **0.8648** (70× larger than next feature)
- All asset/income features < 0.05 — confirmed non-informative

### Tree-Based Models (Notebook 05)

| Model | Best HP | Accuracy | Macro-F1 | ROC-AUC |
|-------|---------|----------|----------|---------|
| Decision Tree | max_depth=None | 0.98 | 0.98 | 0.9764 |
| **Random Forest** | **n_estimators=100** | **0.98** | **0.9826** | **0.9988** |
| Gradient Boosting | n_est=100, depth=3, lr=0.1 | 0.98 | 0.98 | 0.9983 |

- `cibil_score` dominates feature importance across all three models
- Random Forest achieves the best Macro-F1 and ROC-AUC

### Best Model: Random Forest

- **14 misclassifications** out of 854 test samples (315 TN, 525 TP, 8 FP, 6 FN)
- Near-parity between classes: F1-Approved = 0.99, F1-Rejected = 0.98
- Threshold optimisation: optimal at 0.48 (+0.0012 vs default 0.50)

---

## Configuration

All paths and hyperparameters live in [`src/config.py`](src/config.py) — change them there and every notebook picks up the new values automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_STATE` | `42` | Global random seed |
| `TEST_SIZE` | `0.20` | Fraction reserved for testing |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `TARGET_COL` | `"loan_status"` | Target column name |

---

## Requirements

- Python 3.11+
- scikit-learn ≥ 1.4
- imbalanced-learn ≥ 0.12 (SMOTE)
- See [`requirements.txt`](requirements.txt) for the full list
