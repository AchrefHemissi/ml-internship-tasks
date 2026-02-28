"""
data_loader.py — Dataset loading utilities.

Handles:
  - Reading Train.csv / Meta.csv / Test.csv
  - Stratified train/validation split
  - Building a Keras ImageDataGenerator for augmentation
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    TRAIN_CSV, META_CSV, TEST_CSV,
    AUG_ROTATION, AUG_WIDTH_SHIFT, AUG_HEIGHT_SHIFT,
    AUG_ZOOM,
    VAL_SPLIT, RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(data_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Train.csv and Meta.csv.

    Parameters
    ----------
    data_dir : Data directory path. Defaults to DATA_DIR from config.

    Returns
    -------
    (train_df, meta_df)
    """
    train_path = TRAIN_CSV if data_dir is None else os.path.join(data_dir, "Train.csv")
    meta_path  = META_CSV  if data_dir is None else os.path.join(data_dir, "Meta.csv")

    train_df = pd.read_csv(train_path)
    meta_df  = pd.read_csv(meta_path)

    print(f"Train CSV  : {len(train_df):>6} rows  |  columns: {list(train_df.columns)}")
    print(f"Meta  CSV  : {len(meta_df):>6} rows  |  columns: {list(meta_df.columns)}")
    return train_df, meta_df


def load_test_csv(data_dir: str | None = None) -> pd.DataFrame:
    """Load Test.csv (used for final evaluation if available)."""
    path = TEST_CSV if data_dir is None else os.path.join(data_dir, "Test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test CSV not found at {path}")
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validation split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = VAL_SPLIT,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/validation split.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train : {X_train.shape[0]:>6} images")
    print(f"Val   : {X_val.shape[0]:>6} images")
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation generator
# ─────────────────────────────────────────────────────────────────────────────

def build_augmentation_generator() -> ImageDataGenerator:
    """Return an ImageDataGenerator configured with augmentation parameters from config."""
    datagen = ImageDataGenerator(
        rotation_range=AUG_ROTATION,
        width_shift_range=AUG_WIDTH_SHIFT,
        height_shift_range=AUG_HEIGHT_SHIFT,
        zoom_range=AUG_ZOOM,
        fill_mode="nearest",
    )
    return datagen
