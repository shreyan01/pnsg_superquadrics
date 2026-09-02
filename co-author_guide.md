# SAGE — Co-author Guide

This document gets you from zero context to running your assigned experiments. You don't need to touch the robotics/ROS side of the codebase for any of this.

---

## 1. What SAGE actually is (30 seconds)

SAGE represents each object as a **superquadric** — a shape defined by ~5-13 real numbers (radius, height, "roundness" exponents, a color, a taper profile) — fit directly to a real depth-camera point cloud via nonlinear least-squares. There are no neural network weights anywhere. A learned "vocabulary" (the `Registry` class) stores, per category, the mean and variance of these numbers across confirmed real examples (Welford's online algorithm — the same math a running average uses, not gradient descent). Classifying a new object means comparing its fitted numbers against every learned category's mean/variance and picking the closest match.

This matters for your tasks specifically: **every feature you'll work with is a real, named physical measurement**, not a learned embedding. Column 0 really is a radius in meters. This is why the baseline comparison (your task #1) is interesting — it asks whether a *learned* classifier on these *same* interpretable numbers beats our simple statistical one, and separately, how much accuracy a full black-box model gains by working from raw points instead.

**Current locked result:** 78.4% top-1 accuracy on YCB-Video (5 categories: box, mug, bowl, can, bottle), on a held-out, video-level split of 1109 real object instances.

---

## 2. What you've been given: `baseline_data/`

You should have a folder with three files. You don't need our package installed to use them — plain `numpy` is enough.

### `features_val_sample.npz`
```python
import numpy as np
data = np.load('features_val_sample.npz', allow_pickle=True)
X = data['X']               # (N, 13) float array
y = data['y']                # (N,) string array — the true label
feature_names = data['feature_names']  # (13,) column names, in order
```

| Index | Name | Meaning |
|---|---|---|
| 0 | `a1` | primary radius/half-width (meters) |
| 1 | `a2` | secondary radius/half-width (meters) |
| 2 | `eps1` | vertical roundness exponent (0=sharp corner, 1=round) |
| 3 | `eps2` | horizontal roundness exponent (0=sharp corner, 1=round) |
| 4 | `a3` | half-height (meters) |
| 5 | `hue` | always 0 in this export (color excluded — geometry-only comparison) |
| 6 | `saturation` | always 0 in this export |
| 7-11 | `r_10`...`r_90` | radius at 10/30/50/70/90% of object height (meters). **Zero for non-round categories** (box, mug, bowl, bottle) — this isn't missing data, it's how our own model treats the concept as not applicable |
| 12 | `aspect_ratio` | height / max(a1,a2) |

Labels (`y`): `'box'`, `'mug'`, `'bowl'`, `'can'`, `'bottle'`

### `pointclouds_val_sample.npz`
```python
data = np.load('pointclouds_val_sample.npz', allow_pickle=True)
clouds = data['clouds']   # object array; clouds[i] is an (Mi, 3) float32 array of real 3D points
y = data['y']              # same order, same labels as above
```
Each cloud is a **real depth-camera point cloud** for one segmented object — variable number of points per instance (no padding/truncation done for you).

### `README.txt`
Same info as above, in plain text, in case you want it without opening this doc.

**Important:** both files use the same instance order and the same held-out split we used for our own reported 78.4% number. Anything you train/test on this data is directly, fairly comparable to our result — no need to re-derive a split yourself.

---

## 3. Your tasks

### Task 1 — Same-features baseline
Train a k-NN and/or SVM on `features_val_sample.npz` (`X`, `y`). This isolates one specific question: **is our scoring formula the bottleneck, or is the 13D feature set itself the real ceiling?** If a simple learned classifier on the identical numbers beats us significantly, that tells us where to actually focus. Use standard train/test splitting within this data (e.g. stratified k-fold) since this file itself is our *held-out* set — you're not meant to also train on our original training data for this specific comparison.

**Report:** overall accuracy, per-class accuracy, confusion matrix. Compare directly against our numbers: box 94.7%, can 94.9%, mug 32.6%, bottle 31.5%, bowl 18.2%, overall 78.4%.

### Task 2 — Raw point-cloud baseline
Train a small PointNet-style network directly on `pointclouds_val_sample.npz` (no feature engineering — raw points in, label out). This measures what accuracy we're giving up by choosing an interpretable representation over a learned embedding. Same reporting format as Task 1.

### Task 3 — Sample-efficiency curve
This one needs our training script, not just the exported data (flag below). The idea: retrain the registry with only 1, 2, 5, 10, and 20 confirmed examples per category, then evaluate each version, and plot accuracy vs. n. This is meant to demonstrate something a neural baseline structurally can't match — a usable category from a handful of real examples, no retraining architecture needed, just more confirmed examples.

**You'll need:** access to the real YCB-Video dataset locally, and `train_registry_multiview.py` (has a `--load_from` flag for continuing training if that's useful, though for this specific experiment you likely want five clean from-scratch runs at each n, not incremental ones). Ping me if you want the dataset access sorted out, or if you'd rather I run this one and hand you the resulting five accuracy numbers to plot and write up instead — genuinely fine either way given you're not set up with the full pipeline.

### Task 4 — Robustness to sensor degradation
Take the point clouds from `pointclouds_val_sample.npz` and re-evaluate after (a) progressively downsampling point count and (b) adding synthetic Gaussian noise at increasing σ. For (b), if you want to replicate our exact noise model rather than roll your own: `cloud + np.random.normal(0, sigma, cloud.shape)` is literally what we used throughout (sigma values we tested: 0.0015-0.004, i.e. 1.5-4mm, roughly matching real depth sensor noise). **Note:** re-evaluating our actual model at each noise level needs our fitting pipeline (not just the exported features, since noise changes the shape *fit*, not just the stored numbers) — so this one also likely needs either dataset+pipeline access from me, or you send me the noise levels you want tested and I return the accuracy numbers.

**Report:** accuracy vs. point count / accuracy vs. noise σ, ideally as two curves. Shows whether we degrade gracefully or fall off a cliff.

### Task 5 — Statistical rigor
Given you specifically flagged AUC/error-rate as important — this is squarely yours. Add confidence intervals (bootstrap resampling over the instances in `features_val_sample.npz` is enough — resample with replacement, recompute accuracy, repeat ~1000x, take the 2.5/97.5 percentiles) around every accuracy number we report. Separately, run a significance test (McNemar's test is the standard choice for comparing two classifiers' predictions on the same test set) specifically on the **can** (axisymmetric fitting) vs. **bottle** (flexible fitting) result, to confirm that gap is real and not just noise from a modestly-sized held-out set.

**Report:** CIs on every headline number, plus the McNemar's test result (statistic + p-value) for the can/bottle fitting-strategy comparison.

### Task 6 — Stretch, only if everything else is done early
Evaluate our already-trained model on real GraspNet-1Billion scenes with zero retraining — tests whether the learned prototypes generalize beyond YCB-Video. Lower priority than 1-5. Don't start this until the rest is solid.

---

## 4. If you want to look at the actual codebase

Not required for any of the above (Tasks 1, 2, 5 only need the exported `.npz` files), but if you're curious or want to verify something yourself:

| File | What it does |
|---|---|
| `registry.py` | The learned vocabulary itself — `Registry`, `Mode` (Welford mean/variance), `canonicalize()` (builds the 13D feature vector you're working with) |
| `superquadric.py` | Core shape-fitting math (nonlinear least-squares) |
| `radius_profile.py` | The 5-point radial profile computation (columns 7-11 in your feature file) |
| `train_registry_multiview.py` | Real training script — this is what Task 3 needs |
| `evaluate_on_ycbv.py` | Real evaluation script — produces the numbers we've reported |
| `export_baseline_data.py` | What generated your `baseline_data/` folder — read this if you want to see exactly how each feature was computed |

`pip install sage-superquadric` also gets you the whole package with a bundled model if you want to poke at live predictions:
```python
from sage_superquadric import SAGEModel
model = SAGEModel()   # loads the bundled 78.4%-accuracy model
result = model.predict(point_cloud)
```

---

## 5. Questions

Ping me directly rather than guessing on anything ambiguous — especially Tasks 3 and 4, where the right call (send you dataset access vs. I run it and hand you numbers) depends on how much time you want to spend on pipeline setup vs. analysis. Given your background, I'd lean toward you focusing on the analysis/stats side (Tasks 1, 2, 5) and me handling the pipeline-dependent runs (3, 4) — but your call.