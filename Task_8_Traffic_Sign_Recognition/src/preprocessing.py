"""
preprocessing.py — Image preprocessing pipeline.

Steps applied to each image:
  1. Load from disk (BGR → RGB)
  2. Crop to Region of Interest (optional)
  3. CLAHE contrast enhancement (optional)
  4. Resize to target size with Lanczos interpolation
  5. Normalise pixel values to [0, 1]
"""
import os
import cv2
import numpy as np
from src.config import IMG_SIZE


# Shared CLAHE instance (created once, reused for all images)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Enhance contrast by applying CLAHE to the L-channel in LAB colour space."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def preprocess_image(
    img_path: str,
    roi_coords: dict | None = None,
    use_clahe: bool = True,
    target_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a single image.

    Parameters
    ----------
    img_path    : Path to the image file.
    roi_coords  : Optional dict with keys 'x1', 'y1', 'x2', 'y2' for cropping.
    use_clahe   : Whether to apply CLAHE contrast enhancement.
    target_size : Output spatial dimension (square).

    Returns
    -------
    np.ndarray of shape (target_size, target_size, 3), dtype float32, range [0, 1].
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Convert colour space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop to region of interest
    if roi_coords is not None:
        x1 = max(0, int(roi_coords["x1"]))
        y1 = max(0, int(roi_coords["y1"]))
        x2 = min(img.shape[1], int(roi_coords["x2"]))
        y2 = min(img.shape[0], int(roi_coords["y2"]))
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    # Contrast enhancement
    if use_clahe:
        img = _apply_clahe(img)

    # Resize
    img = cv2.resize(img, (target_size, target_size),
                     interpolation=cv2.INTER_LANCZOS4)

    # Normalise
    img = img.astype("float32") / 255.0

    return img


def load_dataset(
    df,
    data_dir: str,
    use_roi: bool = True,
    use_clahe: bool = True,
    target_size: int = IMG_SIZE,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess all images listed in a DataFrame.

    Parameters
    ----------
    df          : DataFrame with columns Path, ClassId, Roi.X1/Y1/X2/Y2.
    data_dir    : Root directory that image paths are relative to.
    use_roi     : Whether to crop images to their ROI bounding box.
    use_clahe   : Whether to apply CLAHE enhancement.
    target_size : Output image size (square).
    verbose     : Print progress every 1 000 images.

    Returns
    -------
    X : float32 array of shape (N, target_size, target_size, 3)
    y : int array of shape (N,)
    """
    images, labels, errors = [], [], []
    total = len(df)

    for idx, row in df.iterrows():
        if verbose and idx % 1000 == 0:
            print(f"  [{idx:>5}/{total}] processed …")

        try:
            roi = None
            if use_roi:
                roi = {
                    "x1": row["Roi.X1"],
                    "y1": row["Roi.Y1"],
                    "x2": row["Roi.X2"],
                    "y2": row["Roi.Y2"],
                }

            full_path = os.path.join(data_dir, row["Path"])
            img = preprocess_image(full_path, roi, use_clahe=use_clahe,
                                   target_size=target_size)
            images.append(img)
            labels.append(int(row["ClassId"]))

        except Exception as exc:
            errors.append((row["Path"], str(exc)))

    if errors and verbose:
        print(f"\n  {len(errors)} images could not be loaded:")
        for path, msg in errors[:5]:
            print(f"    {path}: {msg}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more.")

    if verbose:
        print(f"\n  Loaded {len(images)}/{total} images successfully.")

    return np.array(images, dtype="float32"), np.array(labels, dtype="int32")
