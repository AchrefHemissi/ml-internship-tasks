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

# Primary dataset file
# Download from Kaggle: architsharma01/loan-approval-prediction-dataset
DATASET_FILE = os.path.join(DATA_DIR, "loan_approval_dataset.csv")

# ─────────────────────────────────────────────
# Dataset schema
# ─────────────────────────────────────────────
TARGET_COL  = "loan_status"
LOAN_ID_COL = "loan_id"

CATEGORICAL_COLS = [
    "education",
    "self_employed",
]

NUMERICAL_COLS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# Target label mapping  (strip whitespace before mapping)
TARGET_MAP = {"Approved": 1, "Rejected": 0}

# ─────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80/20 split
CV_FOLDS     = 5      # cross-validation folds

# ─────────────────────────────────────────────
# Model checkpoint filenames
# ─────────────────────────────────────────────
LR_MODEL_FILE = os.path.join(MODELS_DIR, "logistic_regression.pkl")
DT_MODEL_FILE = os.path.join(MODELS_DIR, "decision_tree.pkl")
RF_MODEL_FILE = os.path.join(MODELS_DIR, "random_forest.pkl")
GB_MODEL_FILE = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
SCALER_FILE   = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_FILE  = os.path.join(MODELS_DIR, "encoders.pkl")
