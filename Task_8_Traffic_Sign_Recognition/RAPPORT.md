# Report — Traffic Sign Recognition

## GTSRB Dataset · Deep Learning Project

**Date:** February 27, 2026
**Framework:** TensorFlow 2.20 · Python 3.12
**Hardware:** NVIDIA GeForce RTX 4060 (8 GB VRAM)

---

## 1. Context and Objective

The goal of this project is to build an **automatic traffic sign classification system** from images. This type of system is at the heart of autonomous driving technology: a vehicle must be able to recognize a Stop sign, a 50 km/h speed limit, or a pedestrian crossing in real time and with high reliability.

We used the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset, a standard benchmark in the deep learning community for this problem.

**Chosen approach:** two models were built and compared:

1. A **custom CNN**, trained entirely from scratch to understand the fundamentals
2. **MobileNetV2** with transfer learning to leverage pre-learned knowledge

---

## 2. Dataset: GTSRB

### General Description

| Property | Value |
| --- | --- |
| Number of classes | 43 different traffic signs |
| Training images | 39,209 |
| Test images | 12,630 |
| Image sizes | Variable (15×15 px to 250×250 px) |
| Format | PNG color (RGB) |

The dataset includes a CSV file (`Train.csv`) containing for each image: its size, the **bounding box coordinates** (ROI — Region of Interest) that precisely delimits the sign, and the class ID.

### Class Distribution

The distribution is **imbalanced**: the most represented class has 2,250 images while the least represented has only 210. Some signs (30/50 km/h speed limits) are much more frequent than others (double curve, end of no-passing zone).

```text
Most frequent class  : 2,250 images
Least frequent class :   210 images
Max/min ratio        : ~10.7×
```

This imbalance is typical of real-world data on the road, speed limit signs are far more common than level crossing signs.

---

## 3. Preprocessing Pipeline

Before entering a model, each image goes through **5 steps**:

```text
Raw image  →  [1] BGR→RGB  →  [2] ROI Crop  →  [3] CLAHE  →  [4] Resize 64×64  →  [5] Normalize  →  Final image
```

### Step 1 — Color Conversion (BGR → RGB)

OpenCV loads images in BGR by default. We convert to RGB so colors match the standard display format and what the network expects.

### Step 2 — ROI Crop

The CSV provides the exact coordinates of the sign within the image. We crop the image to keep only the sign, eliminating the background (sky, road, buildings) that carries no useful information for classification.

**Why?** Without this crop, the model could learn characteristics of the background rather than the sign itself.

### Step 3 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

Many images are taken in difficult lighting conditions: sign in shadow, overexposed, foggy. CLAHE locally improves the image contrast by working in the LAB color space (on the L = luminosity channel only).

**Why?** Without CLAHE, a well-lit sign and the same sign in shadow would have very different representations, making learning harder.

### Step 4 — Resize to 64×64 px

All images are resized to a uniform size of **64×64 pixels** using Lanczos4 interpolation (which preserves fine details better than bilinear).

**Why 64×64?** A compromise between quality and computational cost. Too low a resolution (32×32) loses important details; too high (128×128) quadruples the computational cost.

### Step 5 — Normalization [0, 1]

Pixel values (initially integers between 0 and 255) are divided by 255 to obtain floats between 0 and 1.

**Why?** Neural networks converge much faster and more stably when inputs are in numerical ranges close to zero.

---

## 4. Data Augmentation

Augmentation is applied **on the fly during training** (via `ImageDataGenerator`), never saved to disk. For each batch, each image is randomly transformed:

| Transform | Parameter |
| --- | --- |
| Rotation | ±15° |
| Horizontal shift | ±10% |
| Vertical shift | ±10% |
| Zoom | ±20% |

**Why augment?** The model never sees the exact same image twice, which forces it to learn robust features (the shape of a Stop sign) rather than photo-specific artifacts (a particular angle or lighting).

**Why these specific transforms?** In real conditions, a sign can be seen at different angles (rotation), at different distances (zoom), and slightly off-center in the field of view (shift). We do **not** apply horizontal flip because "Turn right" and "Turn left" are two different classes — flipping them would create false labels.

---

## 5. Architecture — Custom CNN

### Overview

The custom CNN is a **sequential architecture with 3 convolutional blocks**, followed by a fully-connected classification head. It is trained **from scratch**, with no pre-learned weights.

```text
INPUT (64×64×3)
       │
  ┌────▼────────────────────────────────────┐
  │  BLOCK 1                                │
  │  Conv2D(32, 3×3, relu, padding=same)    │  →  64×64×32
  │  BatchNormalization                     │
  │  Conv2D(32, 3×3, relu, padding=same)    │  →  64×64×32
  │  BatchNormalization                     │
  │  MaxPooling2D(2×2)                      │  →  32×32×32
  │  Dropout(0.25)                          │
  └────┬────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────┐
  │  BLOCK 2                                │
  │  Conv2D(64, 3×3, relu, padding=same)    │  →  32×32×64
  │  BatchNormalization                     │
  │  Conv2D(64, 3×3, relu, padding=same)    │  →  32×32×64
  │  BatchNormalization                     │
  │  MaxPooling2D(2×2)                      │  →  16×16×64
  │  Dropout(0.25)                          │
  └────┬────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────┐
  │  BLOCK 3                                │
  │  Conv2D(128, 3×3, relu, padding=same)   │  →  16×16×128
  │  BatchNormalization                     │
  │  Conv2D(128, 3×3, relu, padding=same)   │  →  16×16×128
  │  BatchNormalization                     │
  │  MaxPooling2D(2×2)                      │  →   8×8×128
  │  Dropout(0.25)                          │
  └────┬────────────────────────────────────┘
       │
  Flatten  →  8×8×128 = 8,192 values
       │
  Dense(512, relu) → BatchNorm → Dropout(0.5)
       │
  Dense(256, relu) → BatchNorm → Dropout(0.5)
       │
  Dense(43, softmax)
       │
  OUTPUT: probabilities for 43 classes
```

### Detailed Explanation of Each Component

#### Conv2D Layers

Each convolutional layer applies a set of **filters** (small 3×3 matrices) that slide across the entire image and detect local patterns.

- **Block 1 (32 filters):** detects low-level elements — edges, contours, color transitions
- **Block 2 (64 filters):** detects intermediate patterns — corners, angles, simple shapes
- **Block 3 (128 filters):** detects complex patterns — digits, arrows, sign shapes

Doubling the filters at each block allows the network to have more "detectors" as patterns become increasingly abstract.

`padding=same` means the output keeps the same spatial dimensions as the input (zeros are added at the borders).

#### ReLU Activation

After each convolution, ReLU (Rectified Linear Unit) is applied: `f(x) = max(0, x)`. It introduces the **non-linearity** that allows the network to learn complex decision boundaries. Without activation, all layers would behave as a single linear transformation.

#### BatchNormalization

After each convolution, BatchNorm normalizes the activations so they have a mean close to 0 and a standard deviation close to 1. It **stabilizes and accelerates training** by preventing values from becoming too large or too small across layers. It also acts as a mild regularizer.

#### MaxPooling2D(2×2)

Reduces spatial dimensions by half (64→32→16→8) by keeping only the maximum value in each 2×2 window. This:

- Reduces computational cost
- Makes detections robust to small positional shifts
- Forces the network to identify the most salient features

#### Dropout

- **Dropout(0.25)** after each convolutional block: randomly disables 25% of neurons at each forward pass during training
- **Dropout(0.5)** in the classification head: disables 50% of neurons (higher rate because Dense layers carry a greater risk of overfitting)

This forces the network to learn redundant and robust representations, **preventing overfitting**.

#### Classification Head

```text
Flatten    → 8,192 values (flattens the 8×8×128 feature map into a 1D vector)
Dense(512) → summarizes features into 512 high-level representations
Dense(256) → compresses further into 256 representations
Dense(43)  → produces 43 scores (one per class)
Softmax    → converts scores into probabilities (sum = 1)
```

#### Total Parameters

| Section | Parameters |
| --- | --- |
| Block 1 | ~10,400 |
| Block 2 | ~55,900 |
| Block 3 | ~222,000 |
| Dense Head | ~4,337,000 |
| **Total** | **4,629,067** (~17.65 MB) |

> The Dense head accounts for ~94% of all parameters — a classic characteristic of CNNs: convolutional layers are lightweight, the decision head is heavy.

---

## 6. Training Strategy

### Optimizer and Loss

- **Optimizer:** Adam with initial learning rate = `1e-3`
- **Loss:** Sparse Categorical Crossentropy (suited for multi-class classification with integer labels)
- **Monitored metric:** accuracy

### Callbacks

Three callbacks automate training management:

| Callback | Behavior |
| --- | --- |
| `EarlyStopping` | Stops training if `val_accuracy` does not improve for 8 epochs; automatically restores the best weights |
| `ReduceLROnPlateau` | Halves the learning rate if `val_accuracy` stagnates for 3 epochs (min = `1e-7`) |
| `ModelCheckpoint` | Saves only the best model (maximum `val_accuracy`) |

### Training Parameters

| Parameter | Value |
| --- | --- |
| Batch size | 64 |
| Max epochs | 40 |
| Train/val split | 80% / 20% (stratified) |
| Training images | 31,367 |
| Validation images | 7,842 |

---

## 7. Results

### Overall Performance

| Model | Val Accuracy | Test Accuracy | Parameters |
| --- | --- | --- | --- |
| **Custom CNN** | **99.90%** | **98.75%** | 4,629,067 |
| MobileNetV2 | 89.10% | 80.03% | 2,597,995 |

The Custom CNN significantly outperforms MobileNetV2. This may seem counterintuitive, but it is explained by the fact that MobileNetV2 was pre-trained on ImageNet (natural photographs) and its fine-tuning phase on traffic signs — which are small and highly stylized — was not long enough to fully adapt its features to this very specific domain.

### Detailed Classification Report (Custom CNN, Test Set)

The Custom CNN achieves an **overall accuracy of 99%** on the test set of 12,630 images.

**Classes with perfect or near-perfect performance (F1 = 1.00):**

- Speed limit (20km/h), Speed limit (70km/h), No passing, No passing (>3.5t)
- Stop, No entry, Dangerous curve left, Double curve
- Wild animals crossing, Turn right/left ahead, Ahead only, Keep right/left, etc.

**Hardest classes:**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| End of no passing | 0.86 | 0.98 | 0.91 | 60 |
| Bumpy road | 1.00 | 0.86 | 0.92 | 120 |
| End of no passing (>3.5t) | 0.96 | 0.89 | 0.92 | 90 |
| No vehicles | 0.88 | 1.00 | 0.93 | 210 |
| Traffic signals | 0.96 | 0.97 | 0.96 | 180 |

The hardest classes share two characteristics:

- A **low number of test examples** (60 to 120 images)
- **Visual similarity** with other classes (e.g., "End of no passing" vs "End of speed limit")

### Generalization Analysis

The gap between val accuracy (99.90%) and test accuracy (98.75%) is only **1.15 points**. This minimal gap indicates that the model **generalizes well** and does not suffer from significant overfitting. The regularization techniques (BatchNorm + Dropout + augmentation) fulfilled their role effectively.

---

## 8. Custom CNN vs MobileNetV2 Comparison

| Criterion | Custom CNN | MobileNetV2 |
| --- | --- | --- |
| Starting point | Random weights | ImageNet pre-trained |
| Test accuracy | **98.75%** | 80.03% |
| Parameters | 4.6M | 2.6M |
| Training | 1 phase, all layers | 2 phases (head only → fine-tune) |
| Domain adaptability | Strong (trained specifically) | Limited (ImageNet ≠ traffic signs) |

**Comparative conclusion:** For highly specific domains (traffic signs, medical imaging, etc.) with sufficient data (~30,000+ images), a CNN trained from scratch can outperform transfer learning. Transfer learning is most advantageous when data is scarce (<5,000 images) or when the source and target domains are close.

---

## 9. Limitations and Possible Improvements

### 9.1 Class Imbalance

**Current state:** The imbalance (210 to 2,250 images per class, ratio ~10.7×) was only partially addressed via a stratified split — the proportions are preserved in train/val, but the imbalance itself is not corrected.

**Why it matters:** The model sees rare classes far less often during training, which can hurt performance on those specific classes. This is confirmed by the results — the hardest classes ("End of no passing" with F1=0.91, "Bumpy road" with F1=0.92) are all low-frequency ones.

**How to fix it:**

| Technique | How | Expected gain |
| --- | --- | --- |
| `class_weight='balanced'` | Pass to `model.fit()` — penalizes errors on rare classes more | Easy to add, moderate gain |
| Targeted augmentation | Apply heavier augmentation specifically to minority classes | Moderate complexity |
| Oversampling (duplicate images) | Replicate images from minority classes before training | Simple but adds redundancy |

---

### 9.2 No Cross-Validation

**Current state:** A single fixed 80/20 train/val split was used. The reported val accuracy (99.90%) depends on which images ended up in the validation set — a different random seed could give a slightly different number.

**How to fix it:** Use **Stratified K-Fold** (e.g., k=5). Train 5 models on 5 different splits and average the results:

```python
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # train model on X[train_idx], evaluate on X[val_idx]
```

**Trade-off:** 5× longer training time, but a much more reliable performance estimate.

---

### 9.3 MobileNetV2 Fine-Tuning

**Current state:** MobileNetV2 was trained using a standard two-phase transfer learning strategy: the base was first frozen to train the classification head, then partially unfrozen for fine-tuning. This is a well-established approach that achieved 80.03% test accuracy.

**Proposed enhancements to further improve performance:**

- **Extended fine-tuning budget:** Increasing `patience_es` from 5 to 15 would allow the model more epochs to converge during the fine-tuning phase, which is particularly beneficial when adapting a large pretrained backbone to a new domain.
- **Higher epoch ceiling for Phase 2:** Phase 2 is currently capped at 40 epochs, but with `patience_es=5` the training often stops well before that. Raising the ceiling to **100 epochs** and increasing `patience_es` to 10 gives the backbone layers more opportunity to fully adapt to the traffic sign domain, with EarlyStopping still acting as the safety guard.
- **Wider unfreezing range:** Unfreezing from layer 80 instead of 100 would expose more of the backbone to domain-specific adaptation, potentially improving the model's ability to extract traffic sign features.
- **Cosine decay learning rate schedule:** Replacing `ReduceLROnPlateau` with a cosine decay schedule would provide a smoother, more principled learning rate annealing throughout fine-tuning, which is known to improve final performance in transfer learning settings.
- **Domain-specific augmentation:** Adding brightness and contrast augmentation would help reduce the distribution gap between ImageNet (natural photographs) and the GTSRB dataset (road sign images taken under varied lighting conditions).

---

### 9.4 Architecture Improvements

**Current state:** The Custom CNN uses a straightforward 3-block design with no skip connections.

**Possible enhancements:**

| Change | What it does | Complexity |
| --- | --- | --- |
| Add a 4th block Conv(256) | Deeper feature extraction | Low |
| Residual connections (ResNet-style) | Prevents gradient vanishing in deeper networks | Medium |
| Global Average Pooling instead of Flatten | Reduces parameters from 4.2M to ~100K in the head | Low — likely same accuracy |
| Label smoothing in loss | Prevents overconfident predictions, better calibration | Very low |
| Learning rate warmup | Stabilizes the first few epochs | Low |

---

### 9.5 Data Improvements

**Current state:** Augmentation is limited to rotation, shift, and zoom.

**Additional transforms that would help:**

| Transform | Why useful |
| --- | --- |
| Random brightness/contrast | Traffic signs vary in lighting conditions |
| Gaussian noise | Simulates camera sensor noise |
| Random perspective distortion | Simulates viewing the sign at an angle from a moving car |

---

### 9.6 Summary Table

| Limitation | Severity | Effort to fix | Priority |
| --- | --- | --- | --- |
| Class imbalance not handled | Low (98.75% still achieved) | Very low (`class_weight`) | Medium |
| No cross-validation | Medium (single split reliability) | Medium (5× training time) | Medium |
| MobileNetV2 fine-tuning can be extended | Medium (80% vs 99%) | Low (adjust callbacks + epoch ceiling) | High |
| No skip connections in CNN | Low (already at 98.75%) | Medium | Low |
| Limited augmentation | Low | Very low | Low |

---

## 10. Conclusion

This project demonstrates that a relatively simple CNN (3 convolutional blocks, ~4.6M parameters) can achieve **98.75% accuracy** on 43 traffic sign classes, starting from scratch.

The key elements that enabled this result:

1. **Careful preprocessing**: ROI crop + CLAHE ensure the network works on clean, well-contrasted images
2. **Augmentation**: prevents overfitting by multiplying the apparent diversity of the data
3. **Progressive architecture**: doubling filters (32→64→128) to capture patterns of increasing complexity
4. **BatchNorm + Dropout**: effective regularization that allows training a deep network without overfitting
5. **Smart callbacks**: EarlyStopping and ReduceLROnPlateau automatically optimize training

The most impactful next step would be improving the MobileNetV2 fine-tuning strategy, as its current 80% test accuracy leaves significant room for improvement and transfer learning remains a powerful approach when properly tuned.

The final model is saved in `models/custom_cnn_final.keras` and ready for deployment.
