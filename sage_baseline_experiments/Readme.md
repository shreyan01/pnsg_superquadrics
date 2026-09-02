# SAGE Baseline Experiments

Independent baselines against SAGE's locked YCB-Video result (78.4% top-1, five
categories, 1109 held-out object instances), run from the exported
`baseline_data/` files. Covers Tasks 1, 2, 3, 4 and 5 of `co-author_guide.md`.

---

## TL;DR

1. **The 13D feature set, not the scoring formula, is the ceiling on mug and
   bowl.** A linear SVM trained discriminatively on the *identical* 13 features
   reproduces SAGE's per-class failure profile almost exactly — mug 33.7% vs
   SAGE's 32.6%, bowl 25.0% vs 18.2% (GroupKFold). Where SAGE fails, a learned
   classifier on the same numbers fails the same way.

2. **Bottle is the exception, and it is a scoring problem.** Every same-feature
   baseline roughly doubles SAGE on bottle (55–83% vs 31.5%) while using the
   same inputs. That gap is recoverable without changing the representation.

3. **Raw points carry far more than the 13D projection retains.** PointNet on
   raw clouds reaches 93.2% overall / 91.8% balanced under GroupKFold, against
   84–86% overall and only 61–65% balanced for the feature baselines. The
   projection to 13 numbers is where most of the discriminative geometry goes.

4. **The original 5-fold numbers were inflated by leakage.** See below — this
   changed several conclusions and is the main correction in this repo. Results
   are reported under three CV regimes; the two leakage-controlled ones agree
   to within 0.7 pp, so the conclusions do not rest on that choice.

5. **SAGE's can-vs-bottle gap is real,** not small-sample noise: +63.4 pp
   [+55.3, +71.1], Fisher exact p = 7.6e-50.

---

## The leakage problem (read this before quoting any number)

`export_baseline_data.py` iterates over YCB-Video **frames** and emits one row
per object *per frame*. The 1109 rows are therefore repeated observations of a
much smaller set of physical objects, seen across the frames of ~12 validation
videos.

Evidence, from `data_utils.compute_object_groups`: in rotation-invariant size
space (principal extents of the cloud), the **median distance from an instance
to its nearest neighbour is 0.56 mm**. Clustering at a 3 mm tolerance collapses
1109 rows into **201 distinct objects**:

| Category | Distinct objects | Rows |
|---|---|---|
| box | 93 | 475 |
| can | 55 | 355 |
| mug | 22 | 89 |
| bottle | 58 | 146 |
| bowl | 26 | 44 |

A plain `StratifiedKFold` splits those rows independently, so near-identical
views of one physical object land in both the train and the test fold. The
classifier can retrieve a memorised neighbour instead of generalising.

SAGE's own 78.4% used a **video-level** split, so a random-split baseline is not
comparable to it.

**Every task therefore reports three regimes**, and the gap between them is
itself a result:

| Regime | Splitter | Objects kept whole? | Folds stratified? |
|---|---|---|---|
| `random` | `StratifiedKFold` | no — leaks | yes |
| `grouped` | `StratifiedGroupKFold` | yes | yes |
| `groupkfold` | `GroupKFold` | yes | no |

`groupkfold` is the number to quote. Stratifying uses the label distribution to
build the folds, and a real video-level split cannot do that — you hold out
whole videos and accept whatever class mix falls out. `GroupKFold` is therefore
the closer analogue of how SAGE's own 78.4% was measured; `grouped` is the
lower-variance companion estimate, and `random` is kept only to show the size
of the leak.

Both group-aware splitters were verified: no group appears in both train and
test in any fold. `StratifiedGroupKFold` gives near-flat folds (~222 instances
each); `GroupKFold` gives folds of 181–288 with `can` swinging 37–124, which is
what an unstratified held-out split of real videos actually looks like.

**The two agree to within a few tenths of a point on every model**, so no
conclusion here depends on which group-aware splitter is used.

**Caveat.** The grouping is a *proxy*. The export carries no `video_id` or
`frame_id`, so true object identity cannot be recovered from the `.npz` files.
Extent-clustering over-merges genuinely distinct objects of similar size and can
split one object whose visible portion changes under occlusion. Read the grouped
numbers as "leakage substantially reduced", not "leakage eliminated".

---

## Results

### Tasks 1 & 2 — same-feature and raw-point baselines

`results/task2/task1_task2_summary.csv`

| Model | Split | Overall | Balanced | Box | Can | Mug | Bottle | Bowl |
|---|---|---|---|---|---|---|---|---|
| **SAGE** | video-level | **78.4** | – | 94.7 | 94.9 | 32.6 | 31.5 | 18.2 |
| SVM (linear) | random | 84.6 | 60.9 | 97.9 | 99.7 | 32.6 | 56.2 | 18.2 |
| SVM (RBF) | random | 86.4 | 64.4 | 99.4 | 99.2 | 28.1 | 65.8 | 29.5 |
| k-NN | random | 92.6 | 83.3 | 96.4 | 98.9 | 74.2 | 85.6 | 61.4 |
| PointNet | random | 97.7 | 95.5 | 99.4 | 99.2 | 95.5 | 92.5 | 90.9 |
| SVM (linear) | grouped | 84.1 | 60.6 | 97.7 | 99.2 | 30.3 | 55.5 | 20.5 |
| SVM (RBF) | grouped | 85.3 | 64.8 | 96.4 | 98.9 | 36.0 | 63.0 | 29.5 |
| k-NN | grouped | 90.0 | 79.4 | 93.3 | 98.9 | 65.2 | 82.9 | 56.8 |
| PointNet | grouped | 93.9 | 92.8 | 93.1 | 97.5 | 94.4 | 88.4 | 90.9 |
| SVM (linear) | **groupkfold** | 84.2 | 61.6 | 97.9 | 99.2 | 33.7 | 52.1 | 25.0 |
| SVM (RBF) | **groupkfold** | 85.6 | 64.8 | 96.4 | 99.4 | 33.7 | 65.1 | 29.5 |
| k-NN | **groupkfold** | 90.3 | 81.1 | 92.8 | 98.9 | 70.8 | 81.5 | 61.4 |
| PointNet | **groupkfold** | **93.2** | **91.8** | 92.2 | 97.5 | 93.3 | 87.7 | 88.6 |

Per-model, across regimes (overall accuracy %):

| Model | random | grouped | groupkfold | leak (random − groupkfold) | stratification effect (grouped − groupkfold) |
|---|---|---|---|---|---|
| SVM (linear) | 84.58 | 84.13 | 84.22 | +0.36 | −0.09 |
| SVM (RBF) | 86.38 | 85.30 | 85.57 | +0.81 | −0.27 |
| k-NN | 92.61 | 89.99 | 90.26 | +2.35 | −0.27 |
| PointNet | 97.75 | 93.87 | 93.24 | **+4.51** | +0.63 |

**Stratification barely matters.** The `grouped` − `groupkfold` column is within
±0.7 pp for every model — an order of magnitude smaller than PointNet's leak.
McNemar on the two group-aware regimes confirms it: `svm_rbf` p = 0.65,
`pointnet` p = 0.39, i.e. no detectable difference. **So no conclusion here
depends on which group-aware splitter is used**, which is what running both was
meant to establish.

Leakage cost, against `groupkfold`: PointNet +4.5 pp, k-NN +2.4 pp, SVMs
+0.4–0.8 pp. k-NN and PointNet — the two models most able to memorise — are
the two most affected, which is the expected signature of leakage rather
than of a general difficulty change.

**Answering the Task 1 question directly** ("is the scoring formula the
bottleneck, or is the 13D feature set the ceiling?"):

The per-class profile of `svm_linear` tracks SAGE's at r ≈ 0.96 under every
regime: under `groupkfold`, mug 33.7% against SAGE's 32.6% and bowl 25.0%
against 18.2%. (The random-split run landed on mug 32.6% — matching SAGE to
the decimal.)
**A discriminatively trained classifier on the same 13 numbers fails on exactly
the same classes, by nearly the same amount.** For mug and bowl, the features
are the ceiling — better scoring will not help.

Bottle is the counter-example: 52–56% vs SAGE's 31.5% from identical inputs.
That one is worth chasing in the scoring/fitting path.

For contrast, k-NN's profile diverges from SAGE's most (mean absolute difference
25.6 pp) — and k-NN is also the model most sensitive to the leakage above, which
is consistent with it succeeding by retrieval rather than by generalising.

PointNet's balanced accuracy (91.8% under `groupkfold`) versus the best
feature baseline's (64.8%) is the size of what the 13D projection discards.

### Task 3 — sample efficiency

`results/task3/`, figure at `results/task3/figures/sample_efficiency_curve.png`

Accuracy (%) on held-out objects, mean over 20 random draws, counting *n* in
**distinct physical objects** (not rows) with train/eval objects disjoint:

| n per category | knn | svm_linear | svm_rbf |
|---|---|---|---|
| 1 | 53.4 | 53.4 | 53.4 |
| 2 | 54.1 | 60.0 | 53.8 |
| 5 | 56.5 | 63.0 | 60.7 |
| 10 | 68.8 | 69.0 | 67.0 |

n = 20 is not reachable: bowl only has 26 distinct objects, ~18 after holding
out the evaluation pool.

#### Running the SAGE half

`src/task3_sample_efficiency.py` implements **both** halves. The numbers above
are the baseline half, which needs only `baseline_data/`. The SAGE half —
retraining the real registry at each n and evaluating it on YCB-Video — runs as
soon as you point it at a dataset:

```bash
pip install opencv-python          # the YCB-Video loader needs cv2

python src/task3_sample_efficiency.py --dataset_root ~/ycb_dataset
```

Both curves then land on one axis in
`results/task3/figures/sample_efficiency_curve.png`, and the registry numbers in
`results/task3/sage_sample_efficiency_summary.csv`.

Check it works on a small slice before committing to a full run:

```bash
python src/task3_sample_efficiency.py --dataset_root ~/ycb_dataset \
    --sizes 1 2 --max-frames 200 --workers 8
```

Useful flags: `--sizes`, `--repeats` (independent draws of *which* examples get
confirmed), `--split` / `--val-split`, `--frame-stride`, `--max-frames`,
`--workers`, `--skip-baseline`. `--help` lists them all.

**How it works.** It drives the author's own code — `discover_video_classes`,
`_aggregate_and_fit` and `AXISYMMETRIC_WORDS` from
`ycbv_training/train_registry_multiview.py`, and `_eval_one_frame` from
`ycbv_training/evaluate_on_ycbv.py`. **Nothing in the parent repository is
modified.** `src/sage_pipeline.py` handles the import (those modules mix
package-relative and flat imports and the repo has no `__init__.py`, so a plain
import fails); it registers a synthetic parent package, so a clone under any
folder name works.

"n confirmed examples per category" means one (video, class) pair — exactly the
unit `train_registry_multiview.py` counts in `per_class_counts`. Registries are
built **from scratch** at each n, not incrementally, as the guide asks.

The fits are the expensive step, so each (video, class) pair is fitted **once**
per repeat and reused across every n — the n=1 registry is a prefix of the n=20
one. The whole sweep therefore costs about one n=20 training run rather than
five separate ones. Trained registries are saved to `models/task3/` and
instance-level predictions to `results/task3/`.

If the dataset is missing or `cv2` is not installed, the script says exactly
what to fix and still runs the baseline half.

**Not run here.** This checkout has the training and evaluation scripts but
`image_sets/train.txt` and `image_sets/val.txt` are empty stubs and there is no
dataset root, so the SAGE numbers in this README are absent rather than
estimated. The registry-construction path *was* verified end-to-end against real
exported point clouds; only the dataset-reading and evaluation loops are
untested here, since they need YCB-Video files.

### Task 4 — robustness to sensor degradation

`results/task4/`, figures in `results/task4/figures/`

Overall accuracy (%), `groupkfold` split (all three regimes are in
`results/task4/`, and all three degrade with the same shape):

| Points | 1024 | 512 | 256 | 128 | 64 |
|---|---|---|---|---|---|
| Accuracy | 93.2 | 93.0 | 92.9 | 91.0 | 87.8 |

| Noise σ (mm) | 0.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|---|
| Accuracy | 93.2 | 92.8 | 92.8 | 92.4 | 92.5 | 92.4 | 91.8 |

Graceful degradation, no cliff. Across the full sensor-realistic noise range
(1.5–4.0 mm, your noise model: `cloud + np.random.normal(0, sigma, cloud.shape)`)
the loss is 2.4 pp; a 16× point reduction costs 4.3 pp.

All three regimes reproduce their Task 2 baseline exactly at the undegraded
condition (97.75 / 93.87 / 93.24%), which confirms each object is scored only
by the fold model that held it out.

**Scope:** this degrades the clouds and re-runs the *PointNet* baseline. It does
not re-evaluate SAGE — as the guide notes, noise changes the superquadric *fit*,
not just the stored numbers, so that needs the fitting pipeline and the dataset.

### Task 5 — statistical rigor

`results/task5/`

95% percentile bootstrap CIs (1000 resamples) on every headline number:

| Model | Split | Accuracy | 95% CI |
|---|---|---|---|
| knn | random | 92.6 | [91.2, 94.0] |
| svm_linear | random | 84.6 | [82.4, 86.8] |
| svm_rbf | random | 86.4 | [84.4, 88.5] |
| pointnet | random | 97.7 | [96.8, 98.6] |
| knn | grouped | 90.0 | [88.3, 91.7] |
| svm_linear | grouped | 84.1 | [82.0, 86.4] |
| svm_rbf | grouped | 85.3 | [83.3, 87.4] |
| pointnet | grouped | 93.9 | [92.4, 95.2] |

Per-class CIs are in `bootstrap_ci_summary.csv`. Bowl's are wide (n = 44) —
e.g. svm_rbf grouped bowl is 29.5% [15.9, 43.2].

**Can vs bottle — the fitting-strategy comparison.** `can_vs_bottle_tests.csv`

A note on method: McNemar's test is the right tool for two classifiers scored on
the *same* instances, because pairing is what gives it its power. Cans and
bottles are different objects, so there are no pairs and McNemar does not apply.
The question — does axisymmetric fitting succeed at a higher rate than flexible
fitting? — is a comparison of two independent binomial proportions, which
**Fisher's exact test** answers directly.

| Model | can | bottle | difference | 95% CI | Fisher p |
|---|---|---|---|---|---|
| **SAGE** | 94.9 | 31.5 | **+63.4 pp** | [+55.3, +71.1] | **7.6e-50** |
| svm_linear (grouped) | 99.2 | 55.5 | +43.7 pp | [+35.3, +52.6] | 1.4e-36 |
| svm_rbf (grouped) | 98.9 | 63.0 | +35.9 pp | [+28.7, +43.7] | 5.1e-28 |
| knn (grouped) | 98.9 | 82.9 | +16.0 pp | [+10.1, +22.7] | 6.6e-11 |
| pointnet (grouped) | 97.5 | 88.4 | +9.1 pp | [+3.6, +14.9] | 9.0e-05 |

**SAGE's can/bottle gap is emphatically real** — the CI does not come close to
zero and p is 45 orders of magnitude below 0.05. Note also that the gap shrinks
monotonically as the representation gets richer (63.4 → 43.7 → 35.9 → 16.0 →
9.1 pp), which suggests it is largely a representational limitation rather than
something intrinsic to bottles.

*Caveat:* SAGE has no instance-level predictions in the export, so its 2×2 table
is reconstructed by rounding `rate × support`. I verified the reconstruction
reproduces the reported 78.4% exactly (78.42%), so the supports are right — but
the counts are reconstructed, not measured.

Paired McNemar tests between classifiers are in `mcnemar_model_comparisons.csv`.
Under the grouped split, `svm_linear` vs `svm_rbf` is the one comparison that is
**not** significant (p = 0.13); every other pair is.

---

## A second comparability caveat

Even with leakage controlled, the baselines and SAGE are not measured the same
way. The baselines cross-validate *within* the validation set. SAGE's registry
was built on the training videos and evaluated zero-shot on this set. Some of
the baseline margin is that distributional advantage, not a representational
one. The grouped numbers are an upper bound on the learned classifiers, not a
clean like-for-like estimate.

## What would sharpen this

1. **A re-export carrying `video_id` / `frame_key` per instance.** This is the
   single highest-value fix: it replaces the proxy grouping with a true
   video-level split, making the baselines directly comparable to 78.4%.
2. **Instance-level SAGE predictions on `val_sample`.** Enables McNemar of SAGE
   vs each baseline, and an exact rather than reconstructed can/bottle table.
3. **Task 4:** SAGE's accuracy at the σ values above, or pipeline access.
   (Task 3 no longer needs anything from you — it runs the registry itself
   given `--dataset_root`.)

Per the guide's suggestion — happy to take (1) and (2) and do the analysis side,
if you run (4).

---

## Running it

The `experiment/` virtualenv is bundled and already has everything (Python
3.13.9, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.1,
matplotlib 3.11.1, statsmodels 0.15.0, torch 2.13.0+cpu).

```bash
experiment/Scripts/python.exe run_all.py            # everything, ~30 min on CPU
experiment/Scripts/python.exe run_all.py --list     # show the steps
experiment/Scripts/python.exe run_all.py 1 5        # just tasks 1 and 5
```

Or one script at a time (they run from any working directory):

```bash
experiment/Scripts/python.exe src/task1_feature_baselines.py
```

Order matters: Task 4 reloads the fold models Task 2 trains, and Task 5 reads
the predictions Tasks 1 and 2 write. `run_all.py` stops at the first failure
rather than reporting results built on stale files.

To recreate the environment elsewhere: `pip install -r requirements.txt`.
That includes `opencv-python`, which only Task 3's SAGE half needs — the other
tasks run without it.

## Layout

```
src/
  data_utils.py               loaders + compute_object_groups (leakage control)
  evaluation.py               metrics, confusion matrices, result files
  plotting.py                 confusion-matrix figures
  inspect_data.py             data sanity checks (run this first)
  task1_feature_baselines.py  k-NN / linear SVM / RBF SVM on the 13D features
  task2_pointnet.py           small PointNet on raw clouds
  task3_sample_efficiency.py  n-shot curve: SAGE registry + baselines
  sage_pipeline.py            bridge to the parent repo's train/eval code
  task4_robustness.py         downsampling + Gaussian noise sweeps
  task5_statistics.py         bootstrap CIs, McNemar, Fisher exact
  results_io.py               loads saved results for the notebooks
  viz_style.py                shared palette + matplotlib defaults
notebooks/                    one per task, plus an overview (see below)
results/task{1..5}/           CSVs and figures
models/task2/                 PointNet fold checkpoints, per split regime
run_all.py                    pipeline runner
```

## Notebooks

`notebooks/` holds the write-up: one notebook per guide task plus an overview,
committed with outputs embedded so they read without being run.

| Notebook | Guide task |
|---|---|
| `00_overview.ipynb` | Headline comparison, the leakage correction, the findings |
| `01_task1_same_feature_baselines.ipynb` | Task 1 |
| `02_task2_pointnet_raw_points.ipynb` | Task 2 |
| `03_task3_sample_efficiency.ipynb` | Task 3 |
| `04_task4_robustness.ipynb` | Task 4 |
| `05_task5_statistics.ipynb` | Task 5 |

They summarise and plot the saved `results/` CSVs through `src/results_io.py` —
they never retrain, so a figure cannot disagree with the script that produced
its numbers. Use the **`Python (experiment)`** kernel; setup and chart
conventions are in `notebooks/README.md`.

Data is read from `../baseline_data/` (sibling of this directory), untouched.

### Conventions

- Random-split outputs are unsuffixed (`knn_metrics.csv`); grouped-split
  outputs carry `_grouped` (`knn_grouped_metrics.csv`).
- `RANDOM_STATE = 42` throughout; all splits and samplers are seeded.
- Class order is fixed in `evaluation.LABEL_ORDER` so every table, confusion
  matrix and test agrees.
- Physical scale is preserved — clouds are centred but never unit-normalised,
  since object size is genuinely discriminative here.
