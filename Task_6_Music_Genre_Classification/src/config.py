"""
config.py — Central configuration for Task 6: Music Genre Classification.
All paths, constants, and hyperparameters are defined here.
"""
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SPEC_DIR    = os.path.join(BASE_DIR, "spectrograms_data")

FEATURES_CSV = os.path.join(RESULTS_DIR, "audio_features.csv")

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
# Download from Kaggle: andradaolteanu/gtzan-dataset-music-genre-classification
KAGGLE_DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
N_GENRES        = 10
FILES_PER_GENRE = 100   # GTZAN: 100 files × 10 genres = 1,000 total

# ─────────────────────────────────────────────
# Audio feature extraction
# ─────────────────────────────────────────────
AUDIO_DURATION = 30     # seconds per clip loaded by librosa
N_MFCC         = 13     # number of MFCC coefficients
N_FEATURES     = 70     # total feature vector length per file

FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)] +
    [f"mfcc{i}_std"  for i in range(1, N_MFCC + 1)] +
    [f"chroma{i}_mean" for i in range(1, 13)] +
    [f"chroma{i}_std"  for i in range(1, 13)] +
    ["spectral_centroid_mean", "spectral_centroid_std",
     "spectral_bandwidth_mean", "spectral_bandwidth_std",
     "spectral_rolloff_mean",   "spectral_rolloff_std"] +
    [f"spectral_contrast{i}" for i in range(1, 8)] +
    ["zcr_mean", "zcr_std", "rms_mean", "rms_std",
     "tempo", "harmony", "perceptr"]
)

# ─────────────────────────────────────────────
# Spectrogram images (CNN input)
# ─────────────────────────────────────────────
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
CNN_EPOCHS      = 40
TRANSFER_EPOCHS = 30
LEARNING_RATE   = 0.001
PATIENCE        = 8

# ─────────────────────────────────────────────
# Model checkpoint filenames
# ─────────────────────────────────────────────
SCALER_FILE   = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_FILE  = os.path.join(MODELS_DIR, "label_encoder.pkl")
CNN_FILE      = os.path.join(MODELS_DIR, "cnn_best.keras")
TRANSFER_FILE = os.path.join(MODELS_DIR, "transfer_best.keras")

# ─────────────────────────────────────────────
# Visualisation palette
# ─────────────────────────────────────────────
PRIMARY_COLOR   = "steelblue"
SECONDARY_COLOR = "coral"
