# How SMOTE Works
## Synthetic Minority Over-sampling Technique

---

## The Problem SMOTE Solves

When your dataset has many more samples of one class than another,
the model learns to be **biased toward the majority class**.

```
Original training set:
  Approved  ████████████████████  2125  (62%)
  Rejected  █████████████         1290  (38%)
```

A model trained on this will learn:
> *"When in doubt, predict Approved — I'll be right 62% of the time."*

---

## The Naive Fix — Random Over-sampling ❌

The simplest idea: just **copy** existing Rejected rows until balanced.

```
Rejected row #1  →  copied 3 times
Rejected row #2  →  copied 3 times
...
```

**Problem**: the model memorises the same rows over and over →
**overfitting** on the minority class.

---

## What SMOTE Does Instead ✅

Instead of copying, SMOTE **creates brand new synthetic samples**
by interpolating between existing minority samples.

### Step-by-step:

**Step 1** — Pick a random minority sample (a real Rejected row)

```
Sample A  →  [cibil=420, income=3M, loan=10M, term=14, ...]
```

**Step 2** — Find its K nearest neighbours among other Rejected rows
(default K = 5)

```
Neighbour B  →  [cibil=460, income=4M, loan=12M, term=16, ...]
Neighbour C  →  [cibil=390, income=2M, loan=8M,  term=10, ...]
... (5 neighbours total)
```

**Step 3** — Pick one neighbour at random (say B)

**Step 4** — Generate a new point **somewhere on the line between A and B**

```
random weight t = 0.3   (random number between 0 and 1)

new_sample = A + t × (B - A)

cibil  = 420 + 0.3 × (460 - 420)  =  432
income = 3M  + 0.3 × (4M  - 3M)   =  3.3M
loan   = 10M + 0.3 × (12M - 10M)  =  10.6M
term   = 14  + 0.3 × (16  - 14)   =  14.6
```

**Step 5** — Label this new sample as Rejected (class 0)

**Step 6** — Repeat until both classes are equal size

---

## Visualised in 2D

```
     cibil_score
  |
  |    B(460)
  |      •
  |      |  ← new synthetic point lands here (t=0.3)
  |      ★  [432]
  |      |
  |      •
  |    A(420)
  |
  └──────────────── income_annum
```

The ★ is the synthetic sample — it is **plausible** because it lies
between two real Rejected applicants in feature space.

---

## In Our Dataset

```
Before SMOTE:
  Approved  2125   Rejected  1290   Total  3415

After SMOTE:
  Approved  2125   Rejected  2125   Total  4250
                             ↑
                    835 synthetic rows added
```

The 835 new Rejected rows are all artificial — interpolated from
real Rejected applicants. No real data was copied or deleted.

---

## Why This Works Better Than Copying

| Approach | What it does | Risk |
|----------|-------------|------|
| Random copy | Duplicates real rows | Model memorises exact rows (overfitting) |
| **SMOTE** | Creates new points in minority region | Model learns the **region**, not specific rows |
| Under-sampling | Deletes majority rows | Loses real information |

---

## Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k_neighbors` | 5 | How many neighbours to consider for interpolation |
| `random_state` | — | Seed for reproducibility |
| `sampling_strategy` | `'auto'` | How much to oversample (default: balance perfectly) |

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
```

---

## Important Rule

SMOTE must be applied **only on the training set** — never on the
test set. The test set must always reflect real-world distribution.

```
✅  smote.fit_resample(X_train, y_train)   ← correct
❌  smote.fit_resample(X_full,  y_full)    ← data leakage
```

---

## Summary

> SMOTE creates **synthetic minority samples** by drawing random
> points on straight lines connecting real minority samples in
> feature space — giving the model more diverse examples of the
> minority class without simply repeating the same rows.
