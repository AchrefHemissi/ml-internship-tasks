"""
preprocessing.py — Missing-value imputation, encoding, and scaling.

Pipeline steps:
  1. handle_missing_values()  — median / mode imputation
  2. encode_target()          — map Approved/Rejected → 1/0
  3. encode_features()        — LabelEncoder for categoricals
  4. get_features_and_target()— separate X and y
  5. scale_features()         — StandardScaler on numerical columns
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    TARGET_COL, LOAN_ID_COL,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    TARGET_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Missing value imputation
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in-place (works on a copy):
      - Numerical columns  → median
      - Categorical columns → mode (most frequent)

    Returns
    -------
    pd.DataFrame (copy, imputed)
    """
    df = df.copy()
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print("No missing values detected — skipping imputation.")
        return df

    print(f"Total missing cells: {total_missing}")

    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        n = df[col].isnull().sum()
        if n > 0:
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            print(f"  [numerical]   {col:<35} {n:>3} missing → filled with median={fill:.2f}")

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        n = df[col].isnull().sum()
        if n > 0:
            fill = df[col].mode()[0]
            df[col] = df[col].fillna(fill)
            print(f"  [categorical] {col:<35} {n:>3} missing → filled with mode='{fill}'")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Target encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Loan_Status string → integer (Approved=1, Rejected=0)."""
    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)
        if df[TARGET_COL].isnull().any():
            unknown = df.loc[df[TARGET_COL].isnull(), TARGET_COL].unique()
            raise ValueError(
                f"Unknown target values after mapping: {unknown}. "
                f"Expected one of {list(TARGET_MAP.keys())}."
            )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Categorical feature encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    LabelEncode each categorical column.

    Parameters
    ----------
    df       : DataFrame with raw categorical columns.
    encoders : dict of {col: LabelEncoder}. Pass fitted encoders for transform-only mode.
    fit      : True → fit_transform (training); False → transform (inference).

    Returns
    -------
    (df_encoded, encoders)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature / target split
# ─────────────────────────────────────────────────────────────────────────────

def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop id and target columns; return (X, y)."""
    drop_cols = [c for c in [LOAN_ID_COL, TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Feature scaling
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
):
    """
    StandardScaler applied to all columns of X_train (and optionally X_test).

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)  if X_test is provided
    (X_train_scaled, scaler)                 otherwise
    """
    if scaler is None:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train) if fit else scaler.transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Convenience — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Run steps 1–4 (impute → target encode → feature encode → split X/y).
    Scaling is handled separately so the same scaler can be reused.

    Returns
    -------
    (X, y, encoders)
    """
    df = handle_missing_values(df)
    df = encode_target(df)
    df, encoders = encode_features(df, encoders=encoders, fit=fit)
    X, y = get_features_and_target(df)
    return X, y, encoders
