"""
preprocessing.py — Audio feature extraction, spectrogram image generation,
and tabular data preparation.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from .config import (
    AUDIO_DURATION, N_MFCC, FEATURE_NAMES, SPEC_DIR,
    FEATURES_CSV, RANDOM_STATE, TEST_SIZE,
)


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

def extract_features(file_path, duration=AUDIO_DURATION):
    """
    Extract a 70-dimensional feature vector from a single audio file.

    Features (70 total):
      - 13 MFCCs mean + 13 std              = 26
      - 12 Chroma mean + 12 std             = 24
      - Spectral centroid, bandwidth, rolloff (mean + std each) = 6
      - 7 Spectral contrast bands (mean)    =  7
      - ZCR mean + std                      =  2
      - RMS mean + std                      =  2
      - Tempo, harmony, perceptr            =  3
                                        total = 70
    """
    try:
        import librosa

        y, sr = librosa.load(file_path, duration=duration)

        mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
        sc      = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb      = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)
        scon    = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr     = librosa.feature.zero_crossing_rate(y)
        rms     = librosa.feature.rms(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        harmony  = np.mean(librosa.effects.harmonic(y))
        perceptr = np.mean(librosa.effects.percussive(y))

        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        return np.concatenate([
            np.mean(mfccs, axis=1),   np.std(mfccs, axis=1),   # 26
            np.mean(chroma, axis=1),  np.std(chroma, axis=1),  # 24
            [np.mean(sc),  np.std(sc)],                         #  2
            [np.mean(sb),  np.std(sb)],                         #  2
            [np.mean(sr_feat), np.std(sr_feat)],                #  2
            np.mean(scon, axis=1),                              #  7
            [np.mean(zcr), np.std(zcr)],                        #  2
            [np.mean(rms), np.std(rms)],                        #  2
            [tempo_val, harmony, perceptr],                     #  3
        ])
    except Exception:
        return None


def extract_all_features(audio_path, genres, save=True):
    """
    Extract features for every audio file across all genres.
    Returns a DataFrame with shape (n_files, 72) — 70 features + genre + filename.
    """
    import pandas as pd

    rows, labels, filenames, errors = [], [], [], []

    for genre in genres:
        folder = os.path.join(audio_path, genre)
        files  = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".wav") or f.endswith(".au")
        )
        print(f"  {genre}: {len(files)} files")

        for fname in files:
            feats = extract_features(os.path.join(folder, fname))
            if feats is not None and len(feats) == len(FEATURE_NAMES):
                rows.append(feats)
                labels.append(genre)
                filenames.append(fname)
            else:
                errors.append((genre, fname))

    print(f"\n  Done — success: {len(rows)}, errors: {len(errors)}")

    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df["genre"]    = labels
    df["filename"] = filenames

    if save:
        os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
        df.to_csv(FEATURES_CSV, index=False)
        print(f"  Saved: {FEATURES_CSV}")

    return df


# ──────────────────────────────────────────────────────────────
# Spectrogram image generation
# ──────────────────────────────────────────────────────────────

def generate_spectrogram_images(audio_path, genres, spec_dir=SPEC_DIR):
    """
    Generate 128×128 Mel-spectrogram PNG images for each audio file.
    Images are written to spec_dir/train/<genre>/ and spec_dir/test/<genre>/.
    Returns spec_dir.
    """
    from sklearn.model_selection import train_test_split
    import librosa
    import librosa.display

    for split in ("train", "test"):
        for genre in genres:
            os.makedirs(os.path.join(spec_dir, split, genre), exist_ok=True)

    total = {"train": 0, "test": 0}

    for genre in genres:
        folder = os.path.join(audio_path, genre)
        files  = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".wav") or f.endswith(".au")
        )
        train_files, test_files = train_test_split(
            files, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        for flist, split in [(train_files, "train"), (test_files, "test")]:
            for fname in flist:
                dst = os.path.join(
                    spec_dir, split, genre,
                    os.path.splitext(fname)[0] + ".png"
                )
                if os.path.exists(dst):
                    total[split] += 1
                    continue
                try:
                    y, sr = librosa.load(os.path.join(folder, fname), duration=30)
                    S = librosa.power_to_db(
                        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128),
                        ref=np.max
                    )
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.axis("off")
                    librosa.display.specshow(S, sr=sr, ax=ax)
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.savefig(dst, bbox_inches="tight", pad_inches=0, dpi=72)
                    plt.close()
                    total[split] += 1
                except Exception:
                    pass

        print(f"  ✓ {genre}")

    print(f"\n  Images: train={total['train']}, test={total['test']}")
    return spec_dir


# ──────────────────────────────────────────────────────────────
# Tabular data preparation
# ──────────────────────────────────────────────────────────────

def prepare_tabular_data(df, feature_cols=None):
    """
    Scale features and encode labels.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    scaler : fitted StandardScaler
    le     : fitted LabelEncoder
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    cols = feature_cols or [c for c in df.columns if c not in ("genre", "filename")]

    X = df[cols].values
    y = df["genre"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    return X_train, X_test, y_train, y_test, scaler, le
