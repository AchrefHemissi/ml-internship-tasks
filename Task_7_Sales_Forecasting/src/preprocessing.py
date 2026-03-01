"""
preprocessing.py — Feature engineering for sales forecasting.

Pipeline steps:
  1. fill_missing_externals()  — forward/back-fill CPI, Unemployment per store
  2. add_time_features()       — Year, Month, Week, Quarter, DayOfYear
  3. add_holiday_flags()       — IsBlackFriday, IsChristmas, IsThanksgiving
  4. encode_store_type()       — Type A/B/C → one-hot columns
  5. add_lag_features()        — Sales_Lag_1 / 4 / 8 / 52
  6. add_rolling_features()    — Sales_MA/STD_4 / 8 / 12
  7. time_based_split()        — chronological train / val / test
  8. full_pipeline()           — runs steps 1–6, returns clean modelling df
"""
import pandas as pd
import numpy as np

from src.config import (
    DATE_COL, TARGET_COL,
    LAG_WEEKS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,
    MODEL_FEATURES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Fill missing external features
# ─────────────────────────────────────────────────────────────────────────────

def fill_missing_externals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then back-fill CPI, Unemployment, Temperature, Fuel_Price
    within each Store group (small gaps introduced by the merge).
    """
    df = df.copy()
    for col in ["CPI", "Unemployment", "Temperature", "Fuel_Price"]:
        if col not in df.columns:
            continue
        n_before = df[col].isnull().sum()
        if n_before == 0:
            continue
        df[col] = df.groupby("Store")[col].transform(
            lambda x: x.ffill().bfill()
        )
        n_after = df[col].isnull().sum()
        print(f"  {col:<15}: {n_before:>4} → {n_after} NaN")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Time-based features
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the Date column."""
    df = df.copy()
    df["Year"]      = df[DATE_COL].dt.year
    df["Month"]     = df[DATE_COL].dt.month
    df["Week"]      = df[DATE_COL].dt.strftime("%U").astype(int)
    df["Quarter"]   = df[DATE_COL].dt.quarter
    df["DayOfYear"] = df[DATE_COL].dt.dayofyear
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Holiday flags
# ─────────────────────────────────────────────────────────────────────────────

def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary columns for key retail holidays."""
    df = df.copy()
    df["IsBlackFriday"]  = ((df["Month"] == 11) & (df["Week"] >= 47)).astype(int)
    df["IsChristmas"]    = ((df["Month"] == 12) & (df["Week"] >= 51)).astype(int)
    df["IsThanksgiving"] = ((df["Month"] == 11) & (df["Week"] == 46)).astype(int)
    df["IsHoliday"]      = df["IsHoliday"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Store type one-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_store_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the Store Type (A / B / C)."""
    df = df.copy()
    if "Type" in df.columns:
        df["Type_A"] = (df["Type"] == "A").astype(int)
        df["Type_B"] = (df["Type"] == "B").astype(int)
        df["Type_C"] = (df["Type"] == "C").astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Lag features
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Add lagged sales per (Store, Dept) group.

    Parameters
    ----------
    lags : list of int week offsets. Defaults to LAG_WEEKS from config.
    """
    df   = df.copy()
    lags = lags or LAG_WEEKS
    df   = df.sort_values(["Store", "Dept", DATE_COL]).reset_index(drop=True)
    for lag in lags:
        df[f"Sales_Lag_{lag}"] = (
            df.groupby(["Store", "Dept"])[TARGET_COL].shift(lag)
        )
        print(f"  Sales_Lag_{lag}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Rolling statistics
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """
    Add rolling mean and std per (Store, Dept) group.

    Parameters
    ----------
    windows : list of int window sizes. Defaults to ROLLING_WINDOWS from config.
    """
    df      = df.copy()
    windows = windows or ROLLING_WINDOWS
    for w in windows:
        df[f"Sales_MA_{w}"] = df.groupby(["Store", "Dept"])[TARGET_COL].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        df[f"Sales_STD_{w}"] = df.groupby(["Store", "Dept"])[TARGET_COL].transform(
            lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
        )
        print(f"  Sales_MA_{w}, Sales_STD_{w}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Chronological split
# ─────────────────────────────────────────────────────────────────────────────

def time_based_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    val_end:   str = VAL_END_DATE,
    features:  list = None,
):
    """
    Chronological train / val / test split.

    Rows with NaN in feature columns are dropped before splitting.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    features  = features or MODEL_FEATURES
    df_clean  = df.dropna(subset=features).copy()

    train = df_clean[df_clean[DATE_COL] < train_end]
    val   = df_clean[(df_clean[DATE_COL] >= train_end) & (df_clean[DATE_COL] < val_end)]
    test  = df_clean[df_clean[DATE_COL] >= val_end]

    print(f"Train : {len(train):>7,} samples  "
          f"{train[DATE_COL].min().date()} – {train[DATE_COL].max().date()}")
    print(f"Val   : {len(val):>7,} samples  "
          f"{val[DATE_COL].min().date()} – {val[DATE_COL].max().date()}")
    print(f"Test  : {len(test):>7,} samples  "
          f"{test[DATE_COL].min().date()} – {test[DATE_COL].max().date()}")

    return (
        train[features], val[features], test[features],
        train[TARGET_COL], val[TARGET_COL], test[TARGET_COL],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature-engineering pipeline on a merged DataFrame.

    Steps: fill_missing_externals → add_time_features → add_holiday_flags
           → encode_store_type → add_lag_features → add_rolling_features

    Returns
    -------
    pd.DataFrame with all model features added.
    """
    print("Step 1/6 — Fill missing externals")
    df = fill_missing_externals(df)
    print("Step 2/6 — Time features")
    df = add_time_features(df)
    print("Step 3/6 — Holiday flags")
    df = add_holiday_flags(df)
    print("Step 4/6 — Store type encoding")
    df = encode_store_type(df)
    print("Step 5/6 — Lag features")
    df = add_lag_features(df)
    print("Step 6/6 — Rolling statistics")
    df = add_rolling_features(df)
    print(f"\nPipeline complete. Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df
