"""
data_loader.py — Dataset loading utilities.

Handles:
  - Reading the four Walmart CSV files (train, test, features, stores)
  - Merging into a single modelling dataframe
  - Aggregating to total weekly sales for Prophet
"""
import os
import pandas as pd

from src.config import (
    TRAIN_FILE, TEST_FILE, FEATURES_FILE, STORES_FILE,
    MARKDOWN_COLS, DATE_COL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_files(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Dataset file(s) not found:\n  " + "\n  ".join(missing) + "\n\n"
            "Download from Kaggle:\n"
            "  kaggle datasets download -d aslanahmedov/walmart-sales-forecast "
            "-p data/ --unzip"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(
    train_path=None,
    features_path=None,
    stores_path=None,
):
    """
    Load the three core CSV files.

    Parameters
    ----------
    train_path, features_path, stores_path : optional override paths.
      Defaults to values in config.py.

    Returns
    -------
    (train_df, features_df, stores_df)
    """
    t = train_path    or TRAIN_FILE
    f = features_path or FEATURES_FILE
    s = stores_path   or STORES_FILE
    _check_files(t, f, s)

    train_df    = pd.read_csv(t, parse_dates=[DATE_COL])
    features_df = pd.read_csv(f, parse_dates=[DATE_COL])
    stores_df   = pd.read_csv(s)

    print(f"train_df    : {train_df.shape[0]:>7,} rows × {train_df.shape[1]} cols")
    print(f"features_df : {features_df.shape[0]:>7,} rows × {features_df.shape[1]} cols")
    print(f"stores_df   : {stores_df.shape[0]:>7,} rows × {stores_df.shape[1]} cols")

    return train_df, features_df, stores_df


def merge_datasets(
    train_df: pd.DataFrame,
    features_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    drop_markdowns: bool = True,
) -> pd.DataFrame:
    """
    Merge train + features + stores into a single DataFrame.

    Steps
    -----
    1. Optionally drop MarkDown columns (>50 % missing).
    2. Merge features onto train by [Store, Date, IsHoliday].
    3. Merge stores onto result by [Store].
    4. Sort by Date and reset index.

    Returns
    -------
    pd.DataFrame
    """
    if drop_markdowns:
        cols_to_drop = [c for c in MARKDOWN_COLS if c in features_df.columns]
        features_df  = features_df.drop(columns=cols_to_drop)
        print(f"Dropped MarkDown columns (>50% missing): {cols_to_drop}")

    df = train_df.merge(features_df, on=["Store", DATE_COL, "IsHoliday"], how="left")
    df = df.merge(stores_df, on="Store", how="left")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    print(f"Merged shape : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Date range   : {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")
    return df


def get_aggregated_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to total weekly sales across all stores and departments.

    Returns a DataFrame with columns [ds, y] ready for Prophet.
    """
    weekly = (
        df.groupby(DATE_COL)["Weekly_Sales"]
        .sum()
        .reset_index()
        .rename(columns={DATE_COL: "ds", "Weekly_Sales": "y"})
    )
    print(f"Weekly aggregation: {len(weekly)} time periods")
    return weekly
