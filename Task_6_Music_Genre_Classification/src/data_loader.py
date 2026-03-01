"""
data_loader.py — Audio dataset discovery and feature CSV loading.
"""
import os
import pandas as pd

from .config import DATA_DIR, FEATURES_CSV


def find_audio_path(base_dir=None):
    """
    Locate the genres_original (or genres) folder under base_dir (defaults to data/).
    Raises FileNotFoundError with a clear download instruction if not found.
    """
    search_root = base_dir or DATA_DIR

    for root, dirs, _ in os.walk(search_root):
        if "genres_original" in dirs:
            return os.path.join(root, "genres_original")
    for root, dirs, _ in os.walk(search_root):
        if "genres" in dirs:
            return os.path.join(root, "genres")

    raise FileNotFoundError(
        f"Audio data not found under '{search_root}'.\n\n"
        "Download the dataset first:\n"
        "  kaggle datasets download "
        "-d andradaolteanu/gtzan-dataset-music-genre-classification "
        "-p data/ --unzip\n\n"
        "Then re-run this cell."
    )


def discover_dataset(audio_path):
    """Return (genres list, per-genre file counts dict)."""
    genres = sorted(
        g for g in os.listdir(audio_path)
        if os.path.isdir(os.path.join(audio_path, g))
    )
    counts = {
        g: len([
            f for f in os.listdir(os.path.join(audio_path, g))
            if f.endswith(".wav") or f.endswith(".au")
        ])
        for g in genres
    }
    return genres, counts


def get_audio_files(audio_path, genre):
    """Return sorted list of audio file paths for a given genre."""
    folder = os.path.join(audio_path, genre)
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".wav") or f.endswith(".au")
    )


def load_features(csv_path=None):
    """Load pre-extracted audio features CSV."""
    path = csv_path or FEATURES_CSV
    df = pd.read_csv(path)
    print(f"Loaded features: {df.shape[0]} samples × {df.shape[1]} columns")
    return df
