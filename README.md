# Machine Learning Internship — Elevvo

**Intern:** Mohamed Achref Hemissi
**Duration:** 1 month
**Company:** Elevvo

---

## Overview

This repository contains four end-to-end machine learning projects completed during a one-month internship at Elevvo. Each project covers a distinct ML domain: classification, audio analysis, time-series forecasting, and computer vision, and follows a consistent structure: a modular `src/` package, six sequential Jupyter notebooks, and a detailed technical report.

---

## Repository Structure

```
stage Elevvo/
├── Task_4_Loan_Approval_Prediction/    ← Binary classification
├── Task_6_Music_Genre_Classification/  ← Audio classification (tabular + CNN + TL)
├── Task_7_Sales_Forecasting/           ← Time-series regression
└── Task_8_Traffic_Sign_Recognition/    ← Image classification (CNN + MobileNetV2)
```

Each task folder follows the same layout:

```
Task_N_Name/
├── notebooks/   01_ … 06_.ipynb
├── src/         config.py  data_loader.py  preprocessing.py  models.py
├── data/        (dataset files — not tracked by git)
├── models/      (saved checkpoints — not tracked by git)
├── requirements.txt
├── setup_env.sh
├── README.md
└── RAPPORT.md   ← full technical report
```

---

## Task 4 — Loan Approval Prediction

**Type:** Binary classification
**Dataset:** [Kaggle — Loan Approval Prediction](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) · 4,269 samples · 13 columns
**Report:** [Task_4_Loan_Approval_Prediction/RAPPORT.md](Task_4_Loan_Approval_Prediction/RAPPORT.md)

### Objective

Predict whether a loan application will be **Approved** or **Rejected** based on applicant financial profile (credit score, assets, income, loan term).

### Pipeline

| Notebook | Description |
|----------|-------------|
| 01 | Data exploration — class distribution, feature correlations |
| 02 | Preprocessing — encoding, scaling, 80/20 stratified split |
| 03 | Imbalance handling — SMOTE vs RandomUnderSampler |
| 04 | Logistic Regression — 4 variants, C tuning via 5-fold CV |
| 05 | Tree-based models — Decision Tree, Random Forest, Gradient Boosting |
| 06 | Final evaluation — model comparison, threshold optimisation |

### Models & Results

| Model | Accuracy | Macro-F1 | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.93 | 0.93 | 0.9738 |
| Decision Tree | 0.98 | 0.98 | 0.9764 |
| **Random Forest** | **0.98** | **0.9826** | **0.9988** |
| Gradient Boosting | 0.98 | 0.98 | 0.9983 |

**Best model: Random Forest** (100 trees, SMOTE-balanced) — 98.4% accuracy, only 14 misclassifications out of 854 test samples.

**Key finding:** `cibil_score` (credit score) is the overwhelmingly dominant predictor — its LR coefficient is 70× larger than any other feature. All asset and income features contribute negligible signal.

---

## Task 6 — Music Genre Classification

**Type:** Multi-class classification (10 genres)
**Dataset:** [Kaggle — GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) · 1,000 WAV files · 30 s each
**Report:** [Task_6_Music_Genre_Classification/RAPPORT.md](Task_6_Music_Genre_Classification/RAPPORT.md)

### Objective

Identify the genre of a 30-second audio clip (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using three distinct approaches.

### Pipeline

| Notebook | Description |
|----------|-------------|
| 01 | Data exploration — waveforms, spectrograms, MFCCs per genre |
| 02 | Feature extraction — 70 audio features + 128×128 spectrogram images |
| 03 | Feature analysis — correlation matrix, MFCC heatmap, pairplot |
| 04 | Tabular models — 6 scikit-learn classifiers on 70-dimensional vectors |
| 05 | CNN & transfer learning — custom 4-block CNN + VGG16 frozen base |
| 06 | Final evaluation — cross-approach comparison |

### Three Approaches Compared

| Approach | Input | Best Model | Test Accuracy |
|----------|-------|------------|---------------|
| **Tabular ML** | 70 hand-crafted audio features | **SVM (RBF)** | **75.5%** |
| Transfer Learning | 128×128 Mel-spectrogram images | VGG16 (frozen) | 57.5% |
| Custom CNN | 128×128 Mel-spectrogram images | 4-block CNN | 12.5% |

**Best model: SVM (RBF)** — 75.5% accuracy at 0.02 s inference, no GPU required.

**Key findings:**

- Hand-crafted features outperform deep learning when training data is small (800 images)
- Transfer learning bridges the gap but still falls short of classical ML on GTZAN
- Classical ↔ Jazz is the hardest confusion pair across all models and approaches
- Metal is the easiest genre (extreme spectral energy, never misclassified)

---

## Task 7 — Sales Forecasting

**Type:** Time-series regression
**Dataset:** [Kaggle — Walmart Sales Forecast](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) · 421,570 weekly rows · 45 stores · up to 81 departments
**Report:** [Task_7_Sales_Forecasting/RAPPORT.md](Task_7_Sales_Forecasting/RAPPORT.md)

### Objective

Predict weekly department-level sales for 45 Walmart stores using historical data, lag features, and gradient-boosting models.

### Pipeline

| Notebook | Description |
|----------|-------------|
| 01 | Data exploration — seasonal patterns, store type analysis, missing-value audit |
| 02 | Preprocessing — merge, MarkDown drop, CPI/Unemployment fill, time-aware split |
| 03 | Feature engineering — 24 features: calendar, holiday flags, lags, rolling stats |
| 04 | Prophet baseline — aggregate weekly forecast with custom holidays |
| 05 | XGBoost & LightGBM — per store-department rows, 500 trees |
| 06 | Final evaluation — cross-model comparison, residual analysis |

### Models & Results

| Model | Granularity | Test RMSE | Test R² |
|-------|-------------|-----------|---------|
| **XGBoost** | Per store-dept | **$2,176** | **0.9903** |
| LightGBM | Per store-dept | $2,177 | 0.9902 |
| Prophet | Total weekly chain | $1,417,660* | 0.0725* |

*Prophet's metrics are at chain-aggregate scale (~$50M/week), not comparable to per-dept metrics.

**Best model: XGBoost** (500 trees, 24 features) — explains 99.03% of weekly sales variance with minimal overfitting (train → test R² drop = 0.004).

**Key findings:**

- `Sales_Lag_52` (same week last year, r = 0.981) is the single most predictive feature — stronger than `Sales_Lag_1`
- Year-over-year patterns dominate; no structural trend drift across 2010–2012
- Weeks 47–52 (Thanksgiving + Christmas) produce a consistent ~60% sales spike
- MarkDown columns (50–64% missing) were dropped; R² = 0.99 achieved without them

---

## Task 8 — Traffic Sign Recognition

**Type:** Multi-class image classification (43 classes)
**Dataset:** [Kaggle — GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) · 39,209 training / 12,630 test images
**Report:** [Task_8_Traffic_Sign_Recognition/RAPPORT.md](Task_8_Traffic_Sign_Recognition/RAPPORT.md)

### Objective

Classify German traffic signs (43 categories: speed limits, stop, yield, pedestrian, etc.) from real-world photographs taken under varying lighting and distances.

### Pipeline

| Notebook | Description |
|----------|-------------|
| 01 | Data exploration — class distribution, imbalance analysis |
| 02 | Preprocessing — ROI crop, CLAHE, resize 64×64, normalise |
| 03 | Data augmentation — rotation ±15°, shift ±10%, zoom ±20% |
| 04 | Custom CNN — 3-block architecture trained from scratch |
| 05 | MobileNetV2 — two-phase transfer learning (freeze → fine-tune) |
| 06 | Final evaluation — per-class metrics, generalisation analysis |

### Preprocessing Pipeline

```
Raw image → BGR→RGB → ROI Crop → CLAHE → Resize 64×64 → Normalize [0,1]
```

### Models & Results

| Model | Val Accuracy | Test Accuracy | Parameters |
|-------|-------------|---------------|------------|
| **Custom CNN** | **99.90%** | **98.75%** | 4,629,067 |
| MobileNetV2 | 89.10% | 80.03% | 2,597,995 |

**Best model: Custom CNN** — 3 convolutional blocks (32→64→128 filters), BatchNorm + Dropout regularisation, trained entirely from scratch on RTX 4060.

**Key findings:**

- For domain-specific tasks with sufficient data (~30k images), training from scratch can outperform transfer learning
- MobileNetV2's ImageNet features (natural photos) do not transfer as effectively to highly stylized traffic signs
- Val → test accuracy gap of only 1.15 pp confirms strong generalization (no overfitting)
- Hardest classes are all low-frequency ones (60–120 test samples) with visually similar counterparts

---

## Tech Stack

| Domain | Libraries |
|--------|-----------|
| Data manipulation | pandas, numpy |
| Classical ML | scikit-learn, imbalanced-learn |
| Gradient boosting | XGBoost, LightGBM |
| Time-series | Prophet (Meta) |
| Audio | librosa |
| Deep learning | TensorFlow / Keras |
| Computer vision | OpenCV |
| Visualisation | matplotlib, seaborn |

---

## Author

**Mohamed Achref Hemissi**
Internship at Elevvo — February/March 2026
