# How the Tabular Model Works
## From Raw Audio to 70 Numbers to a Genre Prediction

---

## The Idea

Instead of converting audio to an image (spectrogram),
the tabular approach asks a different question:

> **"What can a human expert measure from a sound that
> would help distinguish genres?"**

A music researcher knows that Metal is loud, fast, and harsh.
Classical is quiet, harmonic, and slow. Jazz is complex and tonal.
These intuitions can be turned into **concrete numbers** using
signal processing formulas — no image needed.

The result: one row of **70 numbers** per audio clip,
fed into a standard ML classifier.

---

## The Pipeline

```
1,000 WAV files  (30s each)
        ↓  librosa reads each file
660,000 raw amplitude numbers
        ↓  apply signal processing formulas
70 meaningful features per file
        ↓  stack all files
999 × 70 table  (one corrupted file skipped)
        ↓  StandardScaler + train/test split
Train (799 rows) / Test (200 rows)
        ↓  fit 6 classifiers
Genre prediction  (1 of 10 classes)
```

---

## The 70 Features — What Each One Measures

### Group 1 — MFCCs (26 features)

**MFCC = Mel-Frequency Cepstral Coefficient**

MFCCs capture the **timbre** of a sound — what makes a guitar
sound like a guitar even when playing the same note as a violin.

librosa computes 13 MFCC values per 20ms frame across the
entire 30-second clip. Then two statistics are taken:

```
MFCC 1 mean   ← average value across all frames
MFCC 1 std    ← how much it varies over time
MFCC 2 mean
MFCC 2 std
...
MFCC 13 mean
MFCC 13 std
─────────────
26 numbers total
```

**What each MFCC coefficient captures:**

| Coefficient | What it mainly captures |
|-------------|------------------------|
| MFCC 1 | Overall loudness / energy of the clip |
| MFCC 2–4 | Broad spectral shape (bright vs dark sound) |
| MFCC 5–8 | Mid-level timbral texture |
| MFCC 9–13 | Fine-grained timbral details |

**MFCC 1** is the single most discriminative feature in the dataset:
Metal has the highest values, Classical and Jazz the lowest.

---

### Group 2 — Chroma (24 features)

Chroma features capture the **harmonic / pitch content** of the sound.
The musical octave is divided into 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
For each, librosa measures how strongly that pitch class is present:

```
Chroma C   mean + std
Chroma C#  mean + std
Chroma D   mean + std
...
Chroma B   mean + std
──────────────────────
24 numbers total
```

**Why it helps:**
- Classical and Jazz have rich, complex chroma profiles (many pitch classes active)
- Metal has strong power chords (fewer distinct pitch classes but very loud)
- Hiphop has repetitive bass notes (one or two dominant pitch classes)

---

### Group 3 — Spectral Features (6 features)

These describe the **shape and brightness** of the frequency spectrum.

| Feature | What it measures | Mean + Std = |
|---------|-----------------|--------------|
| Spectral centroid | "Centre of mass" of the spectrum — high = bright/harsh, low = dark/warm | 2 numbers |
| Spectral bandwidth | How wide the spectrum is spread around the centroid | 2 numbers |
| Spectral rolloff | Frequency below which 85% of the energy sits | 2 numbers |

```
Example values:
  Metal     centroid ≈ 3,500 Hz   (bright, harsh)
  Classical centroid ≈   800 Hz   (dark, warm)
```

---

### Group 4 — Spectral Contrast (7 features)

The spectrum is divided into **7 sub-bands**. For each band,
spectral contrast measures the difference between the peaks
(loud parts) and valleys (quiet parts):

```
Sub-band 1 (low)    →  contrast value
Sub-band 2          →  contrast value
...
Sub-band 7 (high)   →  contrast value
────────────────────
7 numbers total
```

High contrast = clear separation between loud and quiet parts (Metal, Disco).
Low contrast = uniform, blurry spectrum (Classical, Ambient).

---

### Group 5 — Energy Features (4 features)

| Feature | What it measures | Count |
|---------|-----------------|-------|
| Zero-Crossing Rate (ZCR) mean + std | How often the wave crosses zero — high = noisy/percussive, low = tonal | 2 |
| RMS energy mean + std | Root Mean Square amplitude — the average loudness | 2 |

```
Metal:     ZCR high (lots of noise), RMS high (very loud)
Classical: ZCR low  (tonal, smooth), RMS low  (quiet)
```

---

### Group 6 — Rhythm & Harmony (3 features)

| Feature | What it measures | Count |
|---------|-----------------|-------|
| Tempo | Estimated BPM (beats per minute) | 1 |
| Harmony mean | Average strength of the harmonic component | 1 |
| Percussive mean | Average strength of the percussive component | 1 |

```
Disco / Hiphop:  tempo ≈ 120–130 BPM, percussive high
Classical / Jazz: tempo ≈ 60–90 BPM,  harmonic high
```

---

## The Full 70-Feature Table

```
Feature group        │ Features                    │ Count
─────────────────────┼─────────────────────────────┼──────
MFCCs                │ mfcc1_mean … mfcc13_std     │  26
Chroma               │ chroma_C_mean … chroma_B_std│  24
Spectral centroid    │ mean + std                  │   2
Spectral bandwidth   │ mean + std                  │   2
Spectral rolloff     │ mean + std                  │   2
Spectral contrast    │ 7 sub-band means            │   7
Zero-crossing rate   │ mean + std                  │   2
RMS energy           │ mean + std                  │   2
Tempo                │ BPM estimate                │   1
Harmony              │ mean                        │   1
Percussive           │ mean                        │   1
─────────────────────┼─────────────────────────────┼──────
Total                │                             │  70
```

One row in the final table looks like:

```
genre    mfcc1_mean  mfcc1_std  mfcc2_mean  ...  tempo  harmony  percussive
metal      -82.3       31.2       120.4     ...  142.0    0.003     0.021
classical  -312.1      18.7        47.1     ...   72.0    0.015     0.004
jazz       -298.4      22.3        51.8     ...   88.0    0.014     0.005
```

---

## Why StandardScaler is Critical

The 70 features are on completely different scales:

```
mfcc1_mean  ranges from  −400  to  −50
tempo       ranges from    60  to  180  BPM
chroma_C    ranges from     0  to    1
```

Distance-based models (SVM, KNN) compute distances between rows.
Without scaling, `mfcc1_mean` dominates completely because its
numbers are hundreds of times larger than `chroma` values.

StandardScaler fixes this by transforming every feature to
**mean = 0, standard deviation = 1**:

```
x_scaled = (x − mean) / std

Before scaling:  mfcc1_mean = −82.3,  tempo = 142.0
After scaling:   mfcc1_mean =  0.73,  tempo =   1.21
```

Now all features contribute equally to distance calculations.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit only on train
X_test_scaled  = scaler.transform(X_test)        # apply same scale to test
```

**Important:** the scaler is fitted on the training set only —
never on the test set. Fitting on the test set would leak
information about its distribution into the model.

---

## The 6 Classifiers Compared

| Model | How it classifies | Test Accuracy | Inference time |
|-------|------------------|---------------|----------------|
| **SVM (RBF)** | Finds the best boundary in 70D space | **75.5%** | **0.02 s** |
| MLP Neural Net | 3-layer network (256→128→64) | 75.5% | 1.10 s |
| Random Forest | 200 decision trees, majority vote | 71.5% | 0.40 s |
| Logistic Regression | Linear boundary in 70D space | 70.5% | 0.04 s |
| Gradient Boosting | 200 sequential trees | 68.0% | 20.21 s |
| KNN | 5 nearest neighbours by distance | 64.5% | 0.00 s |

---

## Why SVM (RBF) Wins

SVM with an RBF kernel finds the **maximum-margin boundary**
between all 10 genre classes in the 70-dimensional feature space.

```
In 2D (simplified):

  Metal  ●●●                         ●●● Classical
         ●●●   ←── SVM boundary ──→  ●●●
         ●●●     (maximum gap)        ●●●
```

The RBF kernel allows **curved, non-linear boundaries** —
genres do not separate with straight lines in feature space.

**Why it beats MLP despite identical accuracy:**

```
SVM  →  0.02 seconds per prediction  (no GPU needed)
MLP  →  1.10 seconds per prediction  (55× slower)
```

For any real-time or production use case, SVM is the clear winner.

---

## Where the Tabular Approach Fails

The 70 features collapse the **time dimension** into statistics
(mean and std). This means:

```
A song that goes:   quiet → loud → quiet   (3 acts)
A song that goes:   medium throughout

Both could have the same mean and std → same feature values
→ model cannot tell them apart
```

The spectrogram approach keeps the full time evolution and
avoids this problem — but requires far more training data
for the CNN to learn from it.

---

## Summary

> The tabular approach extracts **70 hand-crafted numbers** from each
> audio clip using signal processing formulas (MFCCs, chroma, spectral
> features, energy, tempo). These numbers summarise the timbre, harmony,
> brightness, energy, and rhythm of the clip. A StandardScaler normalises
> all features to the same scale, then an SVM with an RBF kernel finds
> the best curved boundary between the 10 genre clusters in
> 70-dimensional space — achieving **75.5% accuracy** at **0.02 s
> per prediction**, with no GPU required.
