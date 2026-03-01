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

# Dataset files
# Download from Kaggle: aslanahmedov/walmart-sales-forecast
TRAIN_FILE    = os.path.join(DATA_DIR, "train.csv")
TEST_FILE     = os.path.join(DATA_DIR, "test.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
STORES_FILE   = os.path.join(DATA_DIR, "stores.csv")

# ─────────────────────────────────────────────
# Dataset schema
# ─────────────────────────────────────────────
TARGET_COL = "Weekly_Sales"
DATE_COL   = "Date"

# MarkDown columns have >50 % missing — dropped during merge
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

# ─────────────────────────────────────────────
# Feature engineering settings
# ─────────────────────────────────────────────
LAG_WEEKS       = [1, 4, 8, 52]
ROLLING_WINDOWS = [4, 8, 12]

# All features fed to regression models
MODEL_FEATURES = [
    "Store", "Dept", "Size",
    "Week", "Month", "Quarter", "DayOfYear",
    "IsHoliday", "IsBlackFriday", "IsChristmas", "IsThanksgiving",
    "Type_A", "Type_B", "Type_C",
    "Sales_Lag_1", "Sales_Lag_4", "Sales_Lag_8", "Sales_Lag_52",
    "Sales_MA_4",  "Sales_MA_8",  "Sales_MA_12",
    "Sales_STD_4", "Sales_STD_8", "Sales_STD_12",
]

# ─────────────────────────────────────────────
# Time-based train / val / test split
# ─────────────────────────────────────────────
# Data range: 2010-02-05 → 2012-10-26
# Lag-52 removes the first year, so modelling starts ~2011-02
TRAIN_END_DATE = "2012-06-01"   # Train : Feb 2011 – May 2012  (~16 months)
VAL_END_DATE   = "2012-08-01"   # Val   : Jun – Jul 2012       (~2 months)
                                # Test  : Aug – Oct 2012       (~3 months)

# ─────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# Model checkpoint filenames
# ─────────────────────────────────────────────
PROPHET_FILE = os.path.join(MODELS_DIR, "prophet_model.pkl")
XGB_FILE     = os.path.join(MODELS_DIR, "xgboost_model.pkl")
LGBM_FILE    = os.path.join(MODELS_DIR, "lgbm_model.pkl")

# ─────────────────────────────────────────────
# Visualisation palette
# ─────────────────────────────────────────────
PRIMARY_COLOR   = "steelblue"
SECONDARY_COLOR = "coral"
