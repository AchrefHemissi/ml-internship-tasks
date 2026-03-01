# Traffic Sign Recognition using CNN

A professional, modular deep-learning project that classifies **43 traffic sign categories** from the [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/) dataset using both a custom CNN and MobileNetV2 transfer learning.

---

## Project Structure

```text
Task_8_Traffic_Sign_Recognition/
│
├── notebooks/                      # Step-by-step notebooks (run in order)
│   ├── 01_data_exploration.ipynb   # Dataset inspection & class distribution
│   ├── 02_preprocessing.ipynb      # ROI crop · CLAHE · resize · normalise
│   ├── 03_data_augmentation.ipynb  # Train/val split · augmentation preview
│   ├── 04_custom_cnn.ipynb         # Build & train 3-block CNN from scratch
│   ├── 05_transfer_learning.ipynb  # MobileNetV2 fine-tuning (2-phase)
│   └── 06_evaluation.ipynb         # Test set evaluation · confusion matrix · reports
│
├── src/                            # Importable Python package
│   ├── __init__.py
│   ├── config.py                   # All paths & hyperparameters in one place
│   ├── preprocessing.py            # Image preprocessing pipeline
│   ├── data_loader.py              # CSV loaders · train/val split · augmentation
│   └── models.py                   # CNN & MobileNetV2 definitions + callbacks
│
├── data/                           # Dataset files (not tracked by git)
│   ├── Train.csv
│   ├── Meta.csv
│   ├── Test.csv
│   └── <class folders with images>
│
├── models/                         # Saved model checkpoints (auto-created)
│
├── RAPPORT.md                      # Full technical report with all findings
├── requirements.txt                # Python dependencies
├── setup_env.sh                    # One-click Linux / macOS / WSL venv setup script
└── README.md
```

---

## Quick Start

### 1. Clone the project

```bash
git clone https://github.com/AchrefHemissi/ml-internship-tasks.git
cd ml-internship-tasks/Task_8_Traffic_Sign_Recognition
```

### 2. Set up the virtual environment

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This will:

- Create a `.venv` virtual environment
- Install all dependencies from `requirements.txt` (including CUDA packages for GPU support)
- Register a Jupyter kernel named **`traffic-signs`**

### 3. Download the dataset

**Option A — Kaggle CLI** (recommended):

```bash
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p data/ --unzip
```

**Option B — Manual download**:

1. Go to [kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
2. Download and extract into `data/`

Make sure `data/Train.csv`, `data/Test.csv`, `data/Meta.csv`, and the image folders are present.

### 4. Launch Jupyter

```bash
source .venv/bin/activate
jupyter notebook
```

Open the notebooks **in order** and select the `traffic-signs` kernel.

---

## Dataset: GTSRB

| Property | Value |
|----------|-------|
| Classes | 43 different traffic signs |
| Training images | 39,209 |
| Test images | 12,630 |
| Image sizes | Variable (15×15 to 250×250 px) |
| Class imbalance | 210 to 2,250 images per class (ratio ~10.7×) |

The dataset includes CSV files with **bounding box coordinates** (ROI) for each image, used to crop out the sign from its background.

---

## Notebook Pipeline

| # | Notebook | What it does | Output |
| --- | --- | --- | --- |
| 01 | Data Exploration | Inspect CSV files, class distribution (43 classes, 10.7× imbalance), raw image sizes (15px to 250px) | Charts |
| 02 | Preprocessing | ROI crop → CLAHE → resize (64×64) → normalise [0,1]; saves arrays | `data/X.npy`, `data/y.npy` |
| 03 | Data Augmentation | Stratified 80/20 train/val split (31,367 / 7,842); augmentation preview | `data/X_train.npy`, `data/X_val.npy` … |
| 04 | Custom CNN | Build 3-block CNN (4.6M params), train with augmentation — **val accuracy 99.90%** | `models/custom_cnn_final.keras` |
| 05 | Transfer Learning | MobileNetV2 Phase 1 (frozen) + Phase 2 (fine-tune from layer 100) — val accuracy 89.10% | `models/mobilenet_final.keras` |
| 06 | Evaluation | Test both models on 12,630 held-out images — **Custom CNN 98.75%, MobileNetV2 80.03%** | Confusion matrices, classification reports |

> **Caching**: notebooks 02 and 06 save preprocessed arrays to `data/` on first run and reload instantly on subsequent runs.

---

## Architecture

### Custom CNN (98.75% test accuracy)

A 3-block convolutional network trained from scratch:

```text
Block 1 : Conv(32)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
Block 2 : Conv(64)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
Block 3 : Conv(128) × 2 → BatchNorm → MaxPool → Dropout(0.25)
Head    : Dense(512) → BatchNorm → Dropout(0.5)
          Dense(256) → BatchNorm → Dropout(0.5)
          Dense(43, softmax)
```

- **4,629,067 parameters** (~17.65 MB)
- Dense head accounts for ~94% of all parameters
- Filters double at each block (32→64→128) to capture progressively complex patterns

### MobileNetV2 Transfer Learning (80.03% test accuracy)

Two-phase fine-tuning:

- **Phase 1** — freeze the MobileNetV2 base (ImageNet weights), train only the classification head (`lr = 1e-3`)
- **Phase 2** — unfreeze layers from index 100 onwards, fine-tune with lower learning rate (`lr = 1e-5`)

**Why it underperforms**: MobileNetV2 was pre-trained on ImageNet (natural photographs). Traffic signs are small, stylized, symbolic images — very different from the source domain. The fine-tuning phase was not long enough to fully adapt.

### Training Configuration

| Parameter | Value |
| --- | --- |
| Optimizer | Adam (lr = 1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 64 |
| Max epochs | 40 |
| Train / Val split | 80% / 20% stratified (31,367 / 7,842 images) |

### Callbacks

| Callback | Behaviour |
| --- | --- |
| `EarlyStopping` | Stops after 8 epochs without `val_accuracy` improvement; restores best weights |
| `ReduceLROnPlateau` | Halves the learning rate after 3 stagnant epochs |
| `ModelCheckpoint` | Saves the best checkpoint to `models/` |

---

## Data Augmentation

Applied on-the-fly during training (never saved to disk):

| Transform | Value | Why |
| --- | --- | --- |
| Rotation | ±15° | Signs seen at different angles |
| Width shift | ±10% | Sign off-center in field of view |
| Height shift | ±10% | Same |
| Zoom | ±20% | Signs at different distances |

**No horizontal flip** — "Turn right" and "Turn left" are distinct classes; flipping would create false labels.

---

## Results — Detailed

### Custom CNN Test Performance

- **Test accuracy: 98.75%** (12,472 / 12,630 correct)
- Val/test gap: only 1.15 points → model generalizes well
- Most classes achieve **F1 = 1.00** (perfect classification)

**Hardest classes** (all low-frequency with visual similarity to other signs):

| Class | F1 | Support | Issue |
|-------|-----|---------|-------|
| End of no passing | 0.91 | 60 | Similar to "End of speed limit" |
| Bumpy road | 0.92 | 120 | Low frequency |
| End of no passing (>3.5t) | 0.92 | 90 | Similar to base "End of no passing" |
| No vehicles | 0.93 | 210 | Visual overlap with other circular signs |

### MobileNetV2 Test Performance

- **Test accuracy: 80.03%** — significantly below Custom CNN
- ImageNet features (edges, textures of natural photos) don't transfer well to symbolic traffic signs
- Could be improved with extended fine-tuning, wider unfreezing, and domain-specific augmentation

---

## Known Limitations & Possible Improvements

| Limitation | Severity | Fix | Priority |
| --- | --- | --- | --- |
| Class imbalance (10.7× ratio) not handled | Low (98.75% achieved) | `class_weight='balanced'` or targeted augmentation | Medium |
| MobileNetV2 underperforms (80%) | Medium | Extend fine-tuning epochs, unfreeze more layers, cosine decay LR | High |
| No cross-validation | Medium | Stratified K-Fold (5×) for reliable estimates | Medium |
| No skip connections in CNN | Low | Add residual connections for deeper variants | Low |
| Limited augmentation | Low | Add brightness/contrast, Gaussian noise, perspective distortion | Low |

---

## GPU Support

The project is configured for GPU training. `requirements.txt` installs CUDA 12 and cuDNN automatically via pip (no system-level install needed on Linux / WSL):

```text
tensorflow[and-cuda]==2.20.0
```

To verify GPU is detected after setup:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For WSL users, GPU passthrough requires **WSL2** and an up-to-date NVIDIA Windows driver.

---

## Configuration

All hyperparameters live in [`src/config.py`](src/config.py) — change them there and every notebook picks up the new values automatically.

| Parameter | Default | Description |
| --- | --- | --- |
| `IMG_SIZE` | `64` | Image dimension (px) |
| `BATCH_SIZE` | `64` | Training batch size |
| `EPOCHS` | `40` | Maximum training epochs |
| `LEARNING_RATE` | `1e-3` | Initial Adam learning rate |
| `VAL_SPLIT` | `0.2` | Fraction reserved for validation |
| `AUG_ROTATION` | `15` | Max rotation degrees |
| `NUM_CLASSES` | `43` | Number of traffic sign categories |

---

## Requirements

- Python 3.13+
- TensorFlow 2.20 (with CUDA 12 via `tensorflow[and-cuda]`)
- OpenCV 4.10+
- See [`requirements.txt`](requirements.txt) for the full list

---

## Dataset

### GTSRB — German Traffic Sign Recognition Benchmark

- 43 traffic sign classes
- ~39 000 training images · ~12 000 test images
- Images vary from 15×15 to 250×250 px
- CSV columns: `Width`, `Height`, `Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`, `ClassId`, `Path`
