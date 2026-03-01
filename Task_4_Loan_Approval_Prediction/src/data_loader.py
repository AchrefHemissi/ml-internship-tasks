"""
data_loader.py — Dataset loading utilities.

Handles:
  - Reading the loan approval CSV
  - Stratified train / test split
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_FILE, TARGET_COL, LOAN_ID_COL,
    TARGET_MAP, TEST_SIZE, RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath: str | None = None) -> pd.DataFrame:
    """
    Load the loan approval dataset from CSV.

    Parameters
    ----------
    filepath : path to the CSV file. Defaults to DATASET_FILE from config.

    Returns
    -------
    pd.DataFrame
    """
    path = filepath or DATASET_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Kaggle:\n"
            "  kaggle datasets download -d architsharma01/loan-approval-prediction-dataset "
            "-p data/ --unzip"
        )

    df = pd.read_csv(path)

    # Normalise column names: strip whitespace
    df.columns = df.columns.str.strip()

    # Strip whitespace from string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    print(f"Dataset  : {len(df):>5} rows  |  {df.shape[1]} columns")
    print(f"Columns  : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Stratified train / test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train    : {len(X_train):>5} samples  "
          f"(approved={y_train.sum()}, rejected={( y_train == 0).sum()})")
    print(f"Test     : {len(X_test):>5} samples  "
          f"(approved={y_test.sum()}, rejected={(y_test == 0).sum()})")
    return X_train, X_test, y_train, y_test
