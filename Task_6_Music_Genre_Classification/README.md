# Music Genre Classification

A professional, modular machine-learning project that classifies music into **10 genres** using the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) from Kaggle.

The project compares **three approaches**: tabular models on extracted audio features (MFCCs, chroma, spectral features), a **custom CNN** on Mel-spectrogram images, and **Transfer Learning with VGG16**.

---

## Project Structure

```text
Task_6_Music_Genre_Classification/
│
├── notebooks/                          # Step-by-step notebooks (run in order)
│   ├── 01_data_exploration.ipynb       # Dataset discovery · waveforms · spectrograms · MFCCs
│   ├── 02_feature_extraction.ipynb     # Extract 70 audio features per file → CSV
│   ├── 03_feature_analysis.ipynb       # EDA · distributions · correlations · MFCC heatmap
│   ├── 04_tabular_models.ipynb         # RF · SVM · KNN · GB · LR · MLP comparison
│   ├── 05_cnn_transfer.ipynb           # Custom CNN + VGG16 transfer learning on spectrograms
│   └── 06_evaluation.ipynb             # Final comparison · best model · recommendations
│
├── src/                                # Importable Python package
│   ├── __init__.py
│   ├── config.py                       # All paths & hyperparameters in one place
│   ├── data_loader.py                  # Dataset discovery · CSV loader
│   ├── preprocessing.py                # Feature extraction · spectrogram generation · data prep
│   └── models.py                       # Model factories · training helpers · evaluation utils
│
├── data/                               # Dataset files (not tracked by git)
│
├── models/                             # Saved model checkpoints (auto-created)
│   ├── scaler.pkl                      # Fitted StandardScaler
│   ├── label_encoder.pkl               # Fitted LabelEncoder
│   ├── cnn_best.keras                  # Best CNN checkpoint
│   └── transfer_best.keras             # Best VGG16 checkpoint
│
├── results/                            # Auto-generated CSV results
│   ├── audio_features.csv              # Extracted 70-feature dataset
│   └── tabular_model_comparison.csv
│
├── spectrograms_data/                  # Spectrogram PNGs for CNN (auto-generated)
│   ├── train/<genre>/
│   └── test/<genre>/
│
├── requirements.txt                    # Python dependencies
├── setup_env.sh                        # One-click Linux / macOS / WSL venv setup
└── README.md
```

---

## Quick Start

```bash
# 1. Setup environment (Linux / macOS / WSL)
chmod +x setup_env.sh
./setup_env.sh

# 2. Activate and launch
source .venv/bin/activate
jupyter notebook

# 3. Run notebooks in order (select 'music-genre' kernel)
#    01 → 02 → 03 → 04 → 05 → 06
```

---

## Dataset

| Property | Value |
| --- | --- |
| Source | GTZAN — Kaggle `andradaolteanu/gtzan-dataset-music-genre-classification` |
| Genres | 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) |
| Files | 1,000 × 30-second WAV clips (100 per genre) |
| Split | 80% train / 20% test (stratified) |

---

## Pipeline

| Notebook | Input | Output |
| --- | --- | --- |
| 01 · Data Exploration | Raw audio files | Waveform / spectrogram / MFCC plots |
| 02 · Feature Extraction | Raw audio files | `results/audio_features.csv` (70 features) |
| 03 · Feature Analysis | `audio_features.csv` | EDA plots, correlation matrix |
| 04 · Tabular Models | `audio_features.csv` | Trained RF/SVM/KNN/GB/LR/MLP + comparison |
| 05 · CNN & Transfer | Spectrogram PNGs | `cnn_best.keras`, `transfer_best.keras` |
| 06 · Evaluation | All model results | Final comparison table + recommendations |

---

## GPU Training

Notebook 05 (`05_cnn_transfer.ipynb`) trains two deep learning models — a custom CNN and VGG16 — which benefit significantly from GPU acceleration.

### How it works

The setup cell in NB05 automatically detects and enables GPU:

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU : {[gpu.name for gpu in gpus]}")
else:
    print("GPU : None detected — training on CPU")
```

`set_memory_growth` prevents TensorFlow from reserving all GPU memory at once, avoiding out-of-memory crashes.

### Requirements

TensorFlow is installed with bundled CUDA — **no system CUDA installation needed**:

```text
tensorflow[and-cuda]>=2.13
```

This is handled automatically by `setup_env.sh` and `requirements.txt`.

### Expected output

| Scenario | Output |
| --- | --- |
| NVIDIA GPU found | `GPU : ['/physical_device:GPU:0']` |
| No GPU / no CUDA | `GPU : None detected — training on CPU` |

### Training time estimate

| Model | GPU | CPU |
| --- | --- | --- |
| Custom CNN (40 epochs) | ~5–10 min | ~30–60 min |
| VGG16 Transfer (30 epochs) | ~10–20 min | ~60–120 min |

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `GPU : None detected` | Ensure you have an NVIDIA GPU and reinstall: `pip install "tensorflow[and-cuda]>=2.13"` |
| CUDA library errors in logs | These are warnings only — training continues on CPU automatically |
| Out-of-memory crash | Reduce `BATCH_SIZE` in `src/config.py` (default: 32 → try 16) |

---

## Config Reference

| Parameter | Default | Location |
| --- | --- | --- |
| `AUDIO_DURATION` | 30 s | `src/config.py` |
| `N_MFCC` | 13 | `src/config.py` |
| `N_FEATURES` | 70 | `src/config.py` |
| `IMG_SIZE` | (128, 128) | `src/config.py` |
| `BATCH_SIZE` | 32 | `src/config.py` |
| `CNN_EPOCHS` | 40 | `src/config.py` |
| `TRANSFER_EPOCHS` | 30 | `src/config.py` |
| `TEST_SIZE` | 0.20 | `src/config.py` |
| `RANDOM_STATE` | 42 | `src/config.py` |
