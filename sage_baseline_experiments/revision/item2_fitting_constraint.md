# Item 2 — remove the category-specific fitting constraint

> *Ensure the fitting constraint cannot use the ground-truth category
> label; rerun affected evaluations.*

**Status: code complete. Needs a dataset run and a retrain.**
Commits `37a1eff`, `988aaab`.

This is the item that changes how other results must be read, so read
it before item 1 or item 5.

---

## What was wrong

The fitting strategy was selected from the ground-truth category, at
inference time, in three places:

| File | Line | Code |
|---|---|---|
| `export_baseline_data.py` | 64 | `axisym = vocab_word in AXISYMMETRIC_WORDS` |
| `ycbv_training/evaluate_on_ycbv.py` | 81 | `axisym = true_word in AXISYMMETRIC_WORDS` |
| `ycbv_training/train_registry_multiview.py` | 125 | `axisym = vocab_word in AXISYMMETRIC_WORDS` |

`AXISYMMETRIC_WORDS` is `{'can'}`, and `axisymmetric=True`
(`superquadric.py:76-88`) removes `a2` and `eps2` as free parameters,
fixing `a2 = a1` and `eps2 = 1.0`. Only `can` took that path, so the
fitted vector carried a signature of **which path was used** — and that
path was chosen by the answer key.

Measured on `baseline_data/features_val_sample.npz`:

| Diagnostic | Result |
|---|---|
| `a1 == a2` exactly | 100% of cans, 0% of everything else |
| any of `r_10..r_90` non-zero | 100% of cans, 0% of everything else |
| can-vs-rest accuracy from that one bit | **100.00%** |
| `svm_rbf` can-recall with `r_10..r_90` deleted | still **98.9%** |

That last row matters: **deleting the feature columns does not fix it.**
The `a1 == a2` signature survives on its own. Only re-fitting helps.

**What this contaminates:** every ~99% `can` figure in the repo. SAGE's
own `can = 100.0%` in `results/sage/sage_predictions.csv`. And the
can-vs-bottle comparison, since `can` is fitted axisymmetrically
*because it is a can*, while `bottle` is not.

---

## What was tried and rejected

The obvious fix is a label-free selector: impose the constraint when the
data supports it. **This was built, measured, and it selects backwards.**

Median `residual_ratio` (the cost of imposing `a2 = a1`) over 100 real
clouds, 20 per class:

```
can     3.271     <- the genuinely round class
box     2.394
bottle  1.479
mug     1.185
bowl    1.013     <- the least round
```

Cans pay the **highest** price for the circularity constraint, not the
lowest. The cause is the one the `axisymmetric` docstring already gives:
a single view sees only part of a round object's circumference, so an
unconstrained fit hugs that arc closely with an ellipse while a circular
fit must compromise.

Can-vs-rest AUROC: `residual_ratio` 0.841 (in the **inverse** sense),
`angular_variation` 0.603 (near useless on partial views).

So the physical prior genuinely cannot be recovered from a partial arc —
which is exactly why it was hardcoded in the first place.
`symmetry.looks_axisymmetric()` therefore **raises
`NotImplementedError`** carrying these numbers, rather than shipping a
plausible-looking rule that picks the wrong objects.

---

## What changed

### New `symmetry.py` (repo root)

- `fit_both(cloud, max_nfev)` — runs **both** fitting paths for every
  object; returns both parameter sets, both residuals normalised by
  object scale, and two symmetry measures. **No label reaches it.**
- `normalised_rmse(params, info)` — `info['rmse']` is volume-scaled
  (`superquadric.py:28`), so raw values are not comparable across object
  sizes. Divides by the geometric mean semi-axis.
- `angular_radius_variation(cloud)` — geometric symmetry measure, kept
  as a feature despite its weak AUROC.
- `looks_axisymmetric()` — raises, deliberately. See above.

### `registry.py` — additive only

- `FEATURE_KEYS_PAIR` (21 columns) and `canonicalize_pair(diagnostics,
  ...)`, plus `color_active_mask_pair` / `taper_active_mask_pair`.
- The radial profile is now computed for **every** object, not only for
  the one category permitted to have one.

**Why additive:** `canonicalize(` has **130 call sites**, and the
working 95.72% pipeline depends on it. Widening it in place mid-
submission was not worth the risk. `canonicalize()`, `FEATURE_KEYS` and
the 13D Welford machinery are untouched, so every existing
`trained_ycbv_*.json` keeps loading. The wider vector is consumed by the
ML path, which is already width-agnostic —
`import_ml_training_data()` accepts any `(N, D)`.

### New `sbe/src/export_instrumented.py`

Produces the label-free features over the real dataset. See
[item 1](item1_unified_protocol.md) for the rest of what it carries.

---

## Evidence the fix works

Acceptance test, 125 real clouds:

| Diagnostic | on can | on rest | |
|---|---|---|---|
| flexible fit `a1==a2` | 0.0% | 0.0% | PASS |
| radial profile non-zero | 100.0% | 100.0% | PASS |

Before: **100% / 0%**. The signature is now constant across classes, so
it carries no label information. The leak closes **by construction** —
there is no threshold to calibrate and no rule that can be wrong later.

Effect on accuracy, ExtraTrees + GroupKFold, same 125 instances:

| features | overall | balanced | macro-F1 | can |
|---|---|---|---|---|
| 13D (label-leaked) | 72.8% | 72.8% | 73.0% | **100.0%** |
| 21D (label-free pair) | **81.6%** | **81.6%** | **81.5%** | 84.0% |

Removing the leak costs 16 pp on `can` — that was the artifact — and
**gains 8.8 pp overall**, because the paired vector carries geometry the
single label-selected fit discarded.

---

## What is left

### 1. The three call sites still branch on the label

`export_instrumented.py` is label-free, but the original three are
unchanged. That is deliberate: changing
`train_registry_multiview.py` invalidates every trained model, so it
should happen together with the retrain rather than leaving the repo in
a state where the training and evaluation paths disagree.

### 2. Retrain — this is the blocking one

**Where:** Linux box. **Expect:** a full training run.

```bash
# after the corrected pipeline has produced the instrumented export
python3 ycbv_training/train_registry_multiview.py \
    --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --split train --out trained_ycbv_labelfree.json --workers 30
```

That command does **not** yet use `fit_both` — repointing
`_aggregate_and_fit` at it is the remaining code change, and it should
be done in one commit with the retrain so the two never disagree.

**What to expect:**

- `can` recall drops from 100% to roughly 84%. **This is the correct
  result, not a regression.** Anyone reading the paper will ask why can
  was perfect; "we were choosing the fitter using the label" is not an
  answer you want to give at review.
- Overall may go **up**, per the 125-instance check. Verify at full
  scale.
- Every `trained_ycbv_*.json` produced before this is leak-affected.
  Keep them for comparison; do not quote them.

### 3. Re-run everything downstream

Items 1, 3, 4, 5 and 6 all consume features. Their numbers are
provisional until this retrain lands.

---

## What to do, in order

1. Run the corrected pipeline (produces the label-free export).
2. Confirm the acceptance test passes at full scale — the pipeline
   prints it.
3. Repoint `train_registry_multiview._aggregate_and_fit` at
   `symmetry.fit_both`, retrain, commit both together.
4. Re-run `evaluate_on_ycbv.py` with the new model to get a SAGE number
   that is comparable to the baselines.
