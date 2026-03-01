# Task 6 — Music Genre Classification

## Technical Report

**Date:** February 28, 2026
**Framework:** TensorFlow 2.20
**Hardware:** NVIDIA GeForce RTX 4060 (8 GB VRAM)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Notebook 01 — Data Exploration](#5-notebook-01--data-exploration)
6. [Notebook 02 — Feature Extraction](#6-notebook-02--feature-extraction)
7. [Notebook 03 — Feature Analysis](#7-notebook-03--feature-analysis)
8. [Notebook 04 — Tabular Models](#8-notebook-04--tabular-models)
9. [Notebook 05 — CNN & Transfer Learning](#9-notebook-05--cnn--transfer-learning)
10. [Notebook 06 — Final Evaluation](#10-notebook-06--final-evaluation)
11. [Source Code — `src/` Module](#11-source-code--src-module)
12. [Results Summary](#12-results-summary)
13. [Key Findings & Conclusions](#13-key-findings--conclusions)

---

## 1. Project Overview

This project tackles **automatic music genre classification** using the GTZAN dataset. The goal is to correctly identify the genre of a 30-second audio clip among 10 possible categories: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

Three distinct approaches are explored and compared:

| Approach | Input | Method |
| --- | --- | --- |
| **Tabular ML** | 70 hand-crafted audio features | 6 scikit-learn classifiers |
| **Custom CNN** | 128×128 Mel-spectrogram images | Convolutional neural network trained from scratch |
| **Transfer Learning** | 128×128 Mel-spectrogram images | VGG16 with ImageNet weights, frozen base |

The project is structured as a sequential pipeline of 6 numbered Jupyter notebooks backed by a modular `src/` Python package.

---

## 2. Dataset

| Property | Value |
| --- | --- |
| **Name** | GTZAN Music Genre Dataset |
| **Source** | Kaggle — `andradaolteanu/gtzan-dataset-music-genre-classification` |
| **Genres** | 10: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Files** | 1,000 WAV files (100 per genre × 10 genres) |
| **Duration** | 30 seconds per clip |
| **Sample rate** | ~22,050 Hz |
| **Class balance** | Perfectly balanced (100 files per genre) |
| **Known issues** | ~2–4% label noise in original GTZAN; 1 file fails extraction → 999 usable |

---

## 3. Project Structure

```text
Task_6_Music_Genre_Classification/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_feature_analysis.ipynb
│   ├── 04_tabular_models.ipynb
│   ├── 05_cnn_transfer.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py          — centralized paths & hyperparameters
│   ├── data_loader.py     — dataset discovery & CSV loading
│   ├── preprocessing.py   — feature extraction & spectrogram generation
│   └── models.py          — model factories, training helpers, evaluation
├── data/
│   └── genres_original/   — 10 genre folders, 100 WAV files each (not git-tracked)
├── models/                — saved model checkpoints (not git-tracked)
├── results/               — generated CSVs and metrics
├── spectrograms_data/     — 128×128 PNG spectrogram images
│   ├── train/<genre>/     — 799 training images (~80 per genre)
│   └── test/<genre>/      — 200 test images (20 per genre)
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## 4. Pipeline Overview

```text
Raw WAV files (1,000 × 30s)
        │
        ▼
  [NB01] Data Exploration
  — visualise waveforms, spectrograms, MFCCs
        │
        ▼
  [NB02] Feature Extraction
  ┌─────┴──────────────────────────────────┐
  │ Path A: 70-dimensional feature vectors │  Path B: 128×128 spectrogram PNGs
  └─────┬──────────────────────────────────┘──────────────────────┐
        │                                                          │
        ▼                                                          ▼
  [NB03] Feature Analysis                               [NB05] CNN & Transfer
  — EDA, correlations, heatmaps                        — Custom CNN (scratch)
        │                                              — VGG16 Transfer Learning
        ▼                                                          │
  [NB04] Tabular Models                                           │
  — 6 sklearn classifiers                                         │
        │                                                          │
        └─────────────────────────┬────────────────────────────────┘
                                  ▼
                      [NB06] Final Evaluation
                      — compare all 3 approaches
```

---

## 5. Notebook 01 — Data Exploration

**Goals:** understand the dataset before building any model — verify balance, listen to genre differences visually.

### What was done

- Located the GTZAN dataset at `data/genres_original/` using `find_audio_path()` from `src/data_loader.py`
- Verified the dataset: **10 genres × 100 files = 1,000 WAV clips**, all perfectly balanced
- For one sample per genre (10 files total):
  - Plotted the **raw waveform** (amplitude vs time)
  - Computed and plotted the **Mel spectrogram** (128 mel bins, log power in dB)
  - Extracted and plotted the **13 MFCCs** (Mel-Frequency Cepstral Coefficients)
- Generated a **2×5 spectrogram grid** comparing all 10 genres side-by-side in one figure

### Key observations

| Genre | Spectrogram Characteristics |
| --- | --- |
| Classical | Sparse, smooth energy; concentrated in low-mid frequencies |
| Metal | Dense energy across all frequencies; high variance throughout |
| Hiphop / Reggae | Strong bass (low frequencies); clear rhythmic patterns |
| Jazz | Harmonic, low energy; visually similar to Classical |
| Disco / Pop | Mid-range energy; regular rhythmic structure |

- Waveforms alone are not sufficient to distinguish genres — spectrograms reveal far more structure
- Classical and Jazz are the most visually similar genres across all visualization types

---

## 6. Notebook 02 — Feature Extraction

**Goals:** convert raw audio into two machine-learning-ready formats: a tabular feature matrix (Path A) and spectrogram images (Path B).

### Path A — 70-Dimensional Feature Vectors

Using `librosa`, the following features are extracted from each 30-second WAV file:

| Feature Group | Features | Count |
| --- | --- | --- |
| MFCCs mean | MFCC 1–13 mean across all frames | 13 |
| MFCCs std | MFCC 1–13 std across all frames | 13 |
| Chroma mean | 12 chroma bins mean | 12 |
| Chroma std | 12 chroma bins std | 12 |
| Spectral centroid | mean + std | 2 |
| Spectral bandwidth | mean + std | 2 |
| Spectral rolloff | mean + std | 2 |
| Spectral contrast | 7 sub-band means | 7 |
| Zero-Crossing Rate | mean + std | 2 |
| RMS energy | mean + std | 2 |
| Tempo | single BPM estimate | 1 |
| Harmony | mean harmonic component | 1 |
| Perceptual | mean percussive component | 1 |
| **Total** | | **70** |

**Result:** DataFrame of shape **(999, 72)** — 70 features + `genre` + `filename`.
One file failed to load (corrupted), leaving 999 usable samples. **Zero missing values.**

Saved to: `results/audio_features.csv`

### Path B — 128×128 Mel-Spectrogram Images

- Computed Mel spectrograms (128 mel bins, log scale in dB) for each WAV file
- Saved as **128×128 PNG images** using `librosa.display` + `matplotlib`
- Applied stratified 80/20 train/test split (`random_state=42`)

| Split | Images | Per genre |
| --- | --- | --- |
| Train | 799 | ~80 |
| Test | 200 | 20 |

Saved to: `spectrograms_data/train/<genre>/` and `spectrograms_data/test/<genre>/`

---

## 7. Notebook 03 — Feature Analysis

**Goals:** understand which features best discriminate between genres, detect correlations, and identify the hardest genre pairs before training.

### Analyses performed

#### 1. MFCC distributions (6 histograms)

- Plotted MFCC1–6 mean distributions with all 10 genres overlaid
- **MFCC1** (total energy / loudness) shows the strongest genre separation
- Metal has the highest MFCC1 values; Classical and Jazz have the lowest and most overlapping values

#### 2. Spectral feature boxplots (2×3 grid)

- Spectral centroid: Metal is the highest (bright, harsh sound); Classical is the lowest (dark, warm)
- Tempo: Disco and Hiphop have the highest BPM; Classical and Jazz have the lowest
- ZCR / RMS: Metal shows the highest energy and crossing rate; Classical is consistently low

#### 3. 70×70 feature correlation matrix

- Heatmap computed with Pearson correlation across all 999 samples
- MFCCs are internally correlated (r > 0.90 for adjacent coefficients)
- Chroma bins are internally correlated
- High multicollinearity → tree-based models (Random Forest) and regularized models (SVM, LR) are appropriate choices

#### 4. 10×13 MFCC heatmap by genre

- Average MFCC profile for each of the 10 genres
- Metal occupies the extreme high end; Classical and Jazz are nearly indistinguishable
- This heatmap confirms Classical ↔ Jazz as the hardest pair to separate

#### 5. Pairplot of 4 key features

- Variables: `mfcc1_mean`, `mfcc2_mean`, `spectral_centroid_mean`, `tempo`
- Clear cluster boundaries for Metal, Classical, Hiphop in the mfcc1 / spectral_centroid space
- Significant overlap in the centre: Rock, Country, Pop, Reggae all occupy similar regions

### Key findings

- **MFCC1** is the single most discriminative feature
- **Spectral centroid** cleanly separates high-energy (Metal) from low-energy (Classical) genres
- **Tempo** distinguishes rhythmic (Disco, Hiphop) from non-rhythmic (Classical, Jazz) genres
- Classical and Jazz are nearly inseparable using individual features — a combination of many features is required

---

## 8. Notebook 04 — Tabular Models

**Goals:** train and evaluate 6 classical machine learning classifiers on the 70-dimensional feature vectors and identify the best tabular model.

### Data preparation

```text
Load results/audio_features.csv  →  999 samples × 70 features
Apply StandardScaler (fit on train only)
Stratified 80/20 split → X_train (799, 70) / X_test (200, 70)
Save scaler → models/scaler.pkl
Save LabelEncoder → models/label_encoder.pkl
```

StandardScaler is critical: SVM and KNN degrade significantly on un-scaled features because they are distance-based.

### Models trained

| Model | Key Hyperparameters |
| --- | --- |
| Random Forest | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| Gradient Boosting | `n_estimators=200`, `random_state=42` |
| SVM (RBF kernel) | `C=10`, `gamma='scale'`, `random_state=42` |
| K-Nearest Neighbours | `n_neighbors=5` |
| Logistic Regression | `max_iter=2000`, L2 penalty, `random_state=42` |
| MLP Neural Network | `hidden_layer_sizes=(256, 128, 64)`, `max_iter=500`, `random_state=42` |

### Results

| Rank | Model | Test Accuracy | Inference Time |
| --- | --- | --- | --- |
| 1 | **SVM (RBF)** | **75.5%** | **0.02 s** |
| 1 | **MLP Neural Net** | **75.5%** | 1.10 s |
| 3 | Random Forest | 71.5% | 0.40 s |
| 4 | Logistic Regression | 70.5% | 0.04 s |
| 5 | Gradient Boosting | 68.0% | 20.21 s |
| 6 | KNN | 64.5% | 0.00 s |

**Winner: SVM (RBF)** — tied for best accuracy (75.5%) but **55× faster** than MLP (0.02 s vs 1.10 s). Preferred for any production or real-time use case.

### Confusion matrix observations

- **Most confused pair: Classical ↔ Jazz** — consistent across all 6 models
- **Well-classified: Metal** — almost never confused with any other genre
- **Moderate confusion:** Rock ↔ Country, Disco ↔ Pop (similar energy and tempo profiles)

All 6 models and their checkpoints saved under `models/`.

---

## 9. Notebook 05 — CNN & Transfer Learning

**Goals:** explore whether image-based deep learning on Mel-spectrogram images can match or surpass the tabular approach.

### Data generators

```python
ImageDataGenerator(
    rescale=1/255,
    rotation_range=5,        # mild augmentation for train only
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2     # 20% of train used for validation
)
```

| Split | Images | Purpose |
| --- | --- | --- |
| Train | 512 (~64/genre) | Gradient updates |
| Validation | ~160 (~16/genre) | Early stopping & checkpoint selection |
| Test | 200 (20/genre) | Final evaluation |

### Custom CNN (trained from scratch)

#### Architecture

```text
Input: (128, 128, 3)
  Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  Block 4: Conv2D(256) → BN → MaxPool → Dropout(0.25)
  Head: Flatten → Dense(512, relu) → BN → Dropout(0.5)
               → Dense(256, relu) → BN → Dropout(0.5)
               → Dense(10, softmax)
Parameters: ~2 million
```

**Training config:** Adam(lr=0.001), categorical_crossentropy, up to 40 epochs, EarlyStopping (patience=8), ReduceLROnPlateau, ModelCheckpoint.

Result: **12.5% test accuracy**

The model clearly overfits: training accuracy rises while validation accuracy stays near random chance (~10%). Root cause: 512 training images is far too few for a CNN to learn reliable convolutional filters from scratch. CNNs typically require tens of thousands of images.

### VGG16 Transfer Learning

#### VGG16 Architecture

```text
Base: VGG16 (ImageNet weights, include_top=False)
  → All 16 convolutional layers FROZEN (trainable=False)
  → ~14.7M frozen parameters
Head (trainable):
  GlobalAveragePooling2D()
  → Dense(512, relu) → BN → Dropout(0.5)
  → Dense(256, relu) → BN → Dropout(0.5)
  → Dense(10, softmax)
Total trainable parameters: ~790,000
```

**Training config:** Adam(lr=0.001), categorical_crossentropy, up to 30 epochs, same callbacks.

Result: **57.5% test accuracy**

VGG16's frozen convolutional base provides powerful low-level feature detectors (edges, textures, shapes) pre-trained on 1.2 million ImageNet images. These generic visual features transfer well to Mel spectrograms, which are 2D visual representations of audio frequency content. Freezing the base prevents overfitting on the small training set.

### Training curves summary

| Epoch | Custom CNN val_acc | VGG16 val_acc |
| --- | --- | --- |
| 1 | ~10% | ~11% |
| 5 | ~10% | ~20% |
| 10 | ~11% | ~26% |
| 18 | — | **~56% (best checkpoint)** |

Saved checkpoints: `models/cnn_best.keras`, `models/transfer_best.keras`

---

## 10. Notebook 06 — Final Evaluation

**Goals:** aggregate results from all three approaches, produce the final comparison, and draw actionable conclusions.

### Final model comparison

| Rank | Approach | Input | Test Accuracy | Notes |
| --- | --- | --- | --- | --- |
| 1 | **Tabular — SVM (RBF)** | 70 audio features | **75.5%** | Best accuracy, fastest inference (0.02 s) |
| 2 | VGG16 Transfer Learning | 128×128 spectrogram | 57.5% | Benefits from ImageNet pre-training |
| 3 | Custom CNN | 128×128 spectrogram | 12.5% | Data-starved; near-random performance |

### Per-genre performance (best tabular model — SVM)

| Genre | Difficulty | Notes |
| --- | --- | --- |
| Metal | Easy | Extreme spectral energy; rarely confused |
| Classical | Medium | Low energy, but confused with Jazz |
| Jazz | Hard | Most confused with Classical |
| Blues | Medium | Occasional overlap with Rock and Country |
| Country | Hard | Confused with Rock and Pop |
| Disco | Hard | Confused with Pop and Hiphop |
| Hiphop | Medium | Strong bass separates it from most genres |
| Pop | Hard | Generic sound; overlaps with multiple genres |
| Reggae | Medium | Distinctive rhythm, but overlaps with Hiphop |
| Rock | Hard | Confused with Metal, Country, Blues |

### Visualisations produced

- Final bar chart comparing all three approaches (tabular, VGG16, CNN)
- Per-genre F1-score chart for the best model (steelblue ≥ 0.70 F1, coral < 0.70 F1)

---

## 11. Source Code — `src/` Module

All reusable logic is centralised in a modular Python package, imported by every notebook via:

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))
from src.config import *
from src.data_loader import ...
from src.preprocessing import ...
from src.models import ...
```

### `src/config.py` — Central Configuration

Single source of truth for all paths and hyperparameters:

```python
# Paths
BASE_DIR    = <project root>
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SPEC_DIR    = BASE_DIR / "spectrograms_data"

# Dataset
GENRES            = [blues, classical, ..., rock]   # 10 genres
N_FEATURES        = 70
AUDIO_DURATION    = 30    # seconds

# Deep learning
IMG_SIZE          = (128, 128)
BATCH_SIZE        = 32
CNN_EPOCHS        = 40
TRANSFER_EPOCHS   = 30
LEARNING_RATE     = 0.001
PATIENCE          = 8

# Style
PRIMARY_COLOR     = "steelblue"
SECONDARY_COLOR   = "coral"
```

### `src/data_loader.py`

| Function | Description |
| --- | --- |
| `find_audio_path()` | Locates `genres_original/` or `genres/` inside `data/`; raises `FileNotFoundError` with Kaggle download instructions if not found |
| `discover_dataset(audio_path)` | Returns list of genres and per-genre file counts |
| `get_audio_files(audio_path, genre)` | Returns sorted list of `.wav` / `.au` files for a genre |
| `load_features(csv_path)` | Loads pre-extracted feature CSV and returns a DataFrame |

### `src/preprocessing.py`

| Function | Description |
| --- | --- |
| `extract_features(file_path)` | Extracts the 70-dimensional feature vector from one WAV file using `librosa`; returns `None` on error |
| `extract_all_features(audio_path, genres)` | Loops over all genres and files; returns DataFrame + optionally saves to `results/audio_features.csv` |
| `generate_spectrogram_images(audio_path, genres, spec_dir)` | Generates 128×128 PNG Mel-spectrogram images; creates `train/` and `test/` sub-folders with stratified 80/20 split |
| `prepare_tabular_data(df)` | Fits `StandardScaler` and `LabelEncoder` on training data; returns `X_train, X_test, y_train, y_test, scaler, le` |

### `src/models.py`

| Function | Description |
| --- | --- |
| `get_tabular_models()` | Returns a dict of 6 unfitted scikit-learn classifiers |
| `evaluate_classifier(model, ...)` | Fits a model, measures training time, evaluates on test set; returns accuracy, time, predictions, classification report |
| `build_cnn(num_classes)` | Returns the custom 4-block Keras CNN (~2M parameters) |
| `build_transfer_model(num_classes)` | Returns VGG16 with frozen base + trainable classification head (~790K trainable parameters) |
| `get_callbacks(checkpoint_path)` | Returns `[EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]` |
| `make_generators(spec_dir)` | Creates `ImageDataGenerator` flows for train (with augmentation), validation, and test |
| `plot_training_history(history, title)` | Plots accuracy and loss curves side-by-side |
| `plot_confusion_matrix(y_true, y_pred, ...)` | Plots a 10×10 confusion matrix heatmap (raw counts or row-normalized) |

---

## 12. Results Summary

### All model accuracies at a glance

```text
Tabular Models (on 70 audio features, test set n=200)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SVM (RBF)           ████████████████████████  75.5%  ← BEST
  MLP Neural Net      ████████████████████████  75.5%
  Random Forest       ███████████████████████   71.5%
  Logistic Regression ██████████████████████    70.5%
  Gradient Boosting   █████████████████████     68.0%
  KNN                 ████████████████████      64.5%

Deep Learning (on 128×128 spectrogram images, test set n=200)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VGG16 Transfer      ██████████████████        57.5%
  Custom CNN          ████                      12.5%  ← data-starved
```

### Model recommendation by use case

| Use Case | Recommended Model | Reason |
| --- | --- | --- |
| Production / real-time / CPU-only | **SVM (RBF)** | 75.5% accuracy, 0.02 s inference |
| GPU available, larger dataset (>10k images) | VGG16 or EfficientNet (fine-tuned) | CNNs shine with more data |
| Interpretability / feature importance | Random Forest + SHAP | Tree-based, inspectable |
| Maximum accuracy (ensemble) | SVM + VGG16 combined | Complementary error patterns |

---

## 13. Key Findings & Conclusions

### 1. Hand-crafted features outperform deep learning on small datasets

The 70 carefully engineered audio features yield **75.5% accuracy**, beating both deep learning approaches (57.5% and 12.5%). This is a direct consequence of dataset size: GTZAN has only 800 training samples — far too few for CNNs to learn meaningful representations from scratch.

### 2. Transfer learning bridges the gap, but still falls short

VGG16 with frozen ImageNet weights achieves **57.5%**, far better than the custom CNN (12.5%), because the pre-trained convolutional filters generalize from natural images to Mel spectrograms. However, it still underperforms tabular models because:

- 512 effective training images remains small for fine-tuning a deep head
- Mel spectrograms at 128×128 px lose some temporal resolution compared to raw audio features

### 3. Classical and Jazz are the hardest genre pair — in every approach

Classical and Jazz share: low overall energy (low MFCC1), rich harmonic content (similar chroma profiles), low spectral centroid, and complex tonal structures. No model reliably separates them on GTZAN. This is a known limitation of the dataset (some tracks are debatably mislabelled).

### 4. Metal is the easiest genre to classify

Metal occupies an extreme corner of the feature space: high energy, high spectral centroid, dense frequency content, high ZCR and RMS. It is almost never misclassified by any model.

### 5. StandardScaler is critical for distance-based models

SVM and KNN degrade dramatically on un-scaled features because MFCC values span –200 to +50 while tempo values span 60 to 180 BPM. Standardising all features to zero mean and unit variance is a prerequisite for these models.

### 6. SVM (RBF) is the best overall model for this task

Given the small dataset and the need for practical deployment, SVM with an RBF kernel offers the best accuracy (75.5%) at the lowest inference cost (0.02 s per sample). It requires no GPU, no warm-up time, and scales well to real-time classification.

---

Report — Task 6: Music Genre Classification | Internship project — Elevvo
