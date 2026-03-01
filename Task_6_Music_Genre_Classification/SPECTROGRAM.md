# How a Mel Spectrogram Works
## From Raw Audio to an Image the CNN Can Read

---

## The Problem

A CNN expects a **2D image** as input — a grid of pixel values.
A WAV file is a **1D list of numbers** — air pressure measured over time.

```
WAV file:   [0.02, 0.08, 0.15, 0.09, -0.03, -0.12, ...]
                ↑ one number every 1/22,050th of a second
                ↑ 30 seconds × 22,050 = ~660,000 numbers total
```

You cannot feed that directly to a CNN. The numbers describe
**when** the air moves — not **which frequencies** are present.

---

## Step 1 — Chop the Audio into 20ms Slices

The full 30-second signal is cut into tiny overlapping windows,
each 20 milliseconds long:

```
|──────────────── 30 seconds ─────────────────|
 |slice| |slice| |slice| |slice| ...  (≈1,300 slices)
   20ms    20ms    20ms    20ms
```

Each slice contains ~441 numbers (22,050 × 0.02).

**Why 20ms?** Short enough to capture one "snapshot" of sound,
long enough to detect the lowest audible frequencies (~50 Hz needs
at least 20ms to complete one full cycle).

---

## Step 2 — Apply the Fourier Transform on Each Slice

The raw 441 numbers in one slice tell you the air pressure over time.
They do **not** tell you which frequencies are playing.

The Fourier Transform answers: **"which frequencies are present,
and how loud is each one?"**

### The core idea

Any complex wave is a sum of simple sine waves at different frequencies.
Fourier decomposes the mixture back into its components — like separating
the instruments in a recording by how fast they vibrate.

```
Combined messy wave  (one 20ms slice)
        ↓  Fourier Transform
= 50  Hz sine wave  ×  0.80   (loud bass)
+ 440 Hz sine wave  ×  0.30   (medium guitar note)
+ 4000 Hz sine wave ×  0.05   (quiet cymbal)
+ ...
```

### Output: a frequency spectrum

For each slice, Fourier outputs one amplitude per frequency:

```
Frequency │ Amplitude
──────────┼───────────
   50 Hz  │  0.80   ← loud
  100 Hz  │  0.60
  440 Hz  │  0.30   ← medium
 2000 Hz  │  0.10
 8000 Hz  │  0.02   ← nearly silent
  ...
```

---

## Step 3 — Map to the Mel Scale (128 bins)

Human ears do not hear all frequencies equally:
- We easily hear the difference between 100 Hz and 200 Hz
- We barely notice the difference between 8,000 Hz and 9,000 Hz

The **Mel scale** compresses the frequency axis to match human perception.
Instead of thousands of raw frequency values, we keep only **128 Mel bins**
— more resolution at low frequencies, less at high frequencies.

```
Linear scale:  50  100  200  400  800 1600 3200 6400 11000 Hz
                |    |    |    |    |    |    |    |    |
Mel scale:     [bin1][bin2][bin3]...[bin64]...[bin127][bin128]
               ← more bins here →          ← fewer bins here →
```

---

## Step 4 — Convert to Decibels (log scale)

Raw amplitude values span a huge range:

```
Loud bass guitar  →  0.800
Quiet whisper     →  0.000001
```

This is impossible to visualise or learn from. Converting to
**decibels** (log scale) compresses the range:

```
Loud bass guitar  →  −10 dB
Quiet whisper     →  −70 dB
```

Now all values fall in a manageable range, typically −80 dB to 0 dB.

---

## Step 5 — Each Slice Becomes One Column

After steps 2–4, one 20ms slice has been transformed into
**128 dB values** — one per Mel frequency bin:

```
Mel bin 1   (50 Hz)    →  −10 dB   ← loud bass
Mel bin 2   (100 Hz)   →  −15 dB
Mel bin 3   (150 Hz)   →  −22 dB
...
Mel bin 64  (1000 Hz)  →  −38 dB   ← medium guitar
...
Mel bin 127 (8000 Hz)  →  −65 dB   ← quiet
Mel bin 128 (11000 Hz) →  −72 dB   ← nearly silent
```

That is one **column** of the final image — 128 pixels tall.

---

## Step 6 — Stack All Columns Side by Side

Repeat steps 2–5 for every slice. Stack the ~1,300 columns
left to right in time order:

```
                 slice1  slice2  slice3  ...  slice1300
                 (0ms)   (20ms)  (40ms)       (30s)
                   │       │       │             │
Mel bin 128 (high) │ −72   │ −70   │ −68   ...  │  −74
Mel bin 127        │ −65   │ −63   │ −67   ...  │  −66
...                │  ...  │  ...  │  ...  ...  │  ...
Mel bin 64  (mid)  │ −38   │ −25   │ −40   ...  │  −30
...                │  ...  │  ...  │  ...  ...  │  ...
Mel bin 1   (low)  │ −10   │ −12   │ −8    ...  │  −11
```

Result: a **128 × 1,300 grid of dB values**.

---

## Step 7 — Colour the Grid → PNG Image

`matplotlib` maps dB values to colours and saves as a PNG:

```
−80 dB  →  darkest  (black / dark blue)
  0 dB  →  brightest (yellow / white)
  in between → gradient
```

The grid becomes a real image:

```
high freq  ████████████████████████████  ← dark  (quiet up here)
           ████████▓▓▓▓████████████████
mid freq   ▓▓▓▓▓▓▓▓░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← medium
           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░
low freq   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← bright (loud bass)
           time ──────────────────────→
```

Finally, the image is resized to **128×128 pixels** and saved as PNG.

---

## What Each Axis Means

```
┌─────────────────────────────────────────┐
│ high freq (11,000 Hz)                   │
│                                         │
│ mid  freq  (1,000 Hz)                   │  ← Y-axis = frequency (which row)
│                                         │     each row = one Mel bin = one frequency
│ low  freq  (50 Hz)                      │
└─────────────────────────────────────────┘
  time 0s ──────────────────────→ time 30s
           X-axis = time (which column)
              each column = one 20ms slice
```

Every single pixel answers exactly one question:

> **"How loud was frequency X at time Y?"**

---

## What Different Genres Look Like

```
Metal      ░░░░░░░░░░░░░░░░░░░░   ← bright everywhere
(loud,     ░░░░░░░░░░░░░░░░░░░░   ← all frequencies active
dense)     ░░░░░░░░░░░░░░░░░░░░   ← all the time

Classical  ████████████████████   ← mostly dark (quiet)
(sparse,   ████▓▓▓▓████████████   ← only low/mid occasionally active
quiet)     ████████████████████

Hiphop     ████████████████████   ← dark at top (no high freq)
(bass-     ████████████████████
heavy)     ░░░░░░░░░░░░░░░░░░░░   ← bright at bottom (strong bass)
```

These visual differences are what the CNN learns to recognise.

---

## The Full Pipeline in One View

```
30-second WAV file  (~660,000 numbers)
        ↓
Chop into ~1,300 slices of 20ms
        ↓  for each slice:
Fourier Transform   →  which frequencies are present?
        ↓
Mel scale mapping   →  compress to 128 perceptually meaningful bins
        ↓
Convert to dB       →  compress loudness range to −80…0 dB
        ↓
128 dB values       →  one column of 128 pixels
        ↓
Stack 1,300 columns →  128 × 1,300 grid
        ↓
Colour + resize     →  128 × 128 PNG image
        ↓
Feed to CNN
```

---

## Key Numbers (Task 6)

| Parameter | Value | Why |
|-----------|-------|-----|
| Sample rate | 22,050 Hz | Standard for music audio |
| Slice duration | ~20 ms | Captures one frequency snapshot |
| Mel bins | 128 | Rows in the image |
| Slices per 30s clip | ~1,300 | Columns before resize |
| Final image size | 128 × 128 px | Resized for CNN input |
| Colour scale | dB (log) | Compresses the loudness range |

---

## Why Not Just Feed the Raw Wave to the CNN?

| Input | What it contains | Useful for genre? |
|-------|-----------------|-------------------|
| Raw waveform (660,000 numbers) | Air pressure over time | No — genre is in the frequencies, not the raw wave |
| **Mel spectrogram (128×128 image)** | Loudness of each frequency at each moment | Yes — genre patterns are visible |

The spectrogram is not a trick — it is the **right representation**
of audio for visual pattern recognition. A bass-heavy hiphop track
always has a bright bottom row; a metal track always has a fully bright
image; a classical track is always mostly dark. The CNN reads those
patterns exactly like it reads shapes in a photograph.

---

## Summary

> A Mel spectrogram is built by chopping audio into 20ms slices,
> applying a Fourier Transform to each slice to find its frequency content,
> mapping to 128 Mel bins that match human hearing, converting loudness
> to decibels, and stacking all slices side by side to form a 2D grid
> where **rows = frequencies** and **columns = time**.
> That grid is then coloured and saved as a PNG image for the CNN.
