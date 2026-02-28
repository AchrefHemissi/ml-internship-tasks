"""
config.py — Central configuration file.
All paths, hyperparameters, and constants are defined here.
"""
import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATA_DIR, "Train.csv")
META_CSV  = os.path.join(DATA_DIR, "Meta.csv")
TEST_CSV  = os.path.join(DATA_DIR, "Test.csv")

# ─────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────
IMG_SIZE    = 64
NUM_CLASSES = 43

# ─────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 40
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.2
RANDOM_STATE  = 42

# ─────────────────────────────────────────────
# Data augmentation settings
# ─────────────────────────────────────────────
AUG_ROTATION      = 15
AUG_WIDTH_SHIFT   = 0.10
AUG_HEIGHT_SHIFT  = 0.10
AUG_ZOOM          = 0.20

# ─────────────────────────────────────────────
# Model checkpoint filenames
# ─────────────────────────────────────────────
CUSTOM_CNN_CHECKPOINT  = os.path.join(MODELS_DIR, "best_custom_cnn.keras")
MOBILENET_CHECKPOINT   = os.path.join(MODELS_DIR, "best_mobilenet.keras")
CUSTOM_CNN_FINAL       = os.path.join(MODELS_DIR, "custom_cnn_final.keras")
MOBILENET_FINAL        = os.path.join(MODELS_DIR, "mobilenet_final.keras")

# ─────────────────────────────────────────────
# Class names (GTSRB 43 classes)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Speed limit (20km/h)",        # 0
    "Speed limit (30km/h)",        # 1
    "Speed limit (50km/h)",        # 2
    "Speed limit (60km/h)",        # 3
    "Speed limit (70km/h)",        # 4
    "Speed limit (80km/h)",        # 5
    "End of speed limit (80km/h)", # 6
    "Speed limit (100km/h)",       # 7
    "Speed limit (120km/h)",       # 8
    "No passing",                  # 9
    "No passing (>3.5t)",          # 10
    "Right-of-way at intersection",# 11
    "Priority road",               # 12
    "Yield",                       # 13
    "Stop",                        # 14
    "No vehicles",                 # 15
    "No vehicles (>3.5t)",         # 16
    "No entry",                    # 17
    "General caution",             # 18
    "Dangerous curve left",        # 19
    "Dangerous curve right",       # 20
    "Double curve",                # 21
    "Bumpy road",                  # 22
    "Slippery road",               # 23
    "Road narrows right",          # 24
    "Road work",                   # 25
    "Traffic signals",             # 26
    "Pedestrians",                 # 27
    "Children crossing",           # 28
    "Bicycles crossing",           # 29
    "Beware of ice/snow",          # 30
    "Wild animals crossing",       # 31
    "End of all restrictions",     # 32
    "Turn right ahead",            # 33
    "Turn left ahead",             # 34
    "Ahead only",                  # 35
    "Go straight or right",        # 36
    "Go straight or left",         # 37
    "Keep right",                  # 38
    "Keep left",                   # 39
    "Roundabout mandatory",        # 40
    "End of no passing",           # 41
    "End of no passing (>3.5t)",   # 42
]
