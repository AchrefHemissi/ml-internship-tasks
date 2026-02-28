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
├── requirements.txt                # Python dependencies
├── setup_env.sh                    # One-click Linux / macOS / WSL venv setup script
└── README.md
```

---

## Quick Start

### 1. Clone the project

```bash
git clone https://github.com/YOUR_USERNAME/ml-internship-projects.git
cd ml-internship-projects/Task_8_Traffic_Sign_Recognition
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

## Notebook Pipeline

| # | Notebook | What it does | Output |
| --- | --- | --- | --- |
| 01 | Data Exploration | Inspect CSV files, class counts, raw image sizes | Charts |
| 02 | Preprocessing | ROI crop → CLAHE → resize (64×64) → normalise; saves arrays | `data/X.npy`, `data/y.npy` |
| 03 | Data Augmentation | Train/val split; augmentation preview; saves split arrays | `data/X_train.npy` … |
| 04 | Custom CNN | Build + train 3-block CNN; plot training curves | `models/custom_cnn_final.keras` |
| 05 | Transfer Learning | MobileNetV2 Phase 1 (frozen base) + Phase 2 (fine-tune) | `models/mobilenet_final.keras` |
| 06 | Evaluation | Evaluate both models on the **held-out test set**; confusion matrices; classification report; per-class accuracy | Charts / metrics |

> **Caching**: notebooks 02 and 06 save preprocessed arrays to `data/` on first run and reload instantly on subsequent runs.

---

## Architecture

### Custom CNN

A 3-block convolutional network trained from scratch:

```text
Block 1 : Conv(32)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
Block 2 : Conv(64)  × 2 → BatchNorm → MaxPool → Dropout(0.25)
Block 3 : Conv(128) × 2 → BatchNorm → MaxPool → Dropout(0.25)
Head    : Dense(512) → Dense(256) → Dense(43, softmax)
```

### MobileNetV2 Transfer Learning

Two-phase fine-tuning:

- **Phase 1** — freeze the MobileNetV2 base, train only the classification head (`lr = 1e-3`)
- **Phase 2** — unfreeze layers from index 100 onwards and fine-tune with a lower learning rate (`lr = 1e-5`)

### Training Callbacks

All training runs use three callbacks (configured in `src/models.py`):

| Callback | Behaviour |
| --- | --- |
| `EarlyStopping` | Stops after 8 epochs without `val_accuracy` improvement; restores best weights |
| `ReduceLROnPlateau` | Halves the learning rate after 3 stagnant epochs |
| `ModelCheckpoint` | Saves the best checkpoint to `models/` |

---

## Image Preprocessing Pipeline

Each image goes through 5 steps before entering the model:

1. **BGR → RGB** colour conversion
2. **ROI crop** — bounding box from the CSV removes background clutter
3. **CLAHE** — contrast enhancement on the L-channel in LAB colour space
4. **Resize to 64×64** using Lanczos4 interpolation
5. **Normalise** pixel values to \[0, 1\]

---

## Data Augmentation

Applied on-the-fly during training (never saved to disk):

| Transform | Value |
| --- | --- |
| Rotation | ±15° |
| Width shift | ±10% |
| Height shift | ±10% |
| Zoom | ±20% |

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
