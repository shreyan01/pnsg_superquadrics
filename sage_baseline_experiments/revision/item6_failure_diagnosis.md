# Item 6 — geometric failure diagnosis

> *Quantify whether the proposed geometric indicators actually
> distinguish reliable vs unreliable fits, including correct vs
> incorrect recognition cases.*

**Status: code complete. Needs a dataset run for real numbers.**
Commits `988aaab`, `7315e80`.

---

## What was wrong

**The signal existed and was thrown away.**

`iterative_segment.py` computes
`combined_rmse = info_a['rmse'] + info_b['rmse']` on every refinement
round and uses it to select `best_state` — then, **before this change**,
discarded it at all three return points. **Four of the five categories**
(box, mug, bowl, bottle) go through this path rather than the
axisymmetric one, so **no residual information reached the caller for
them at all**.

(Line numbers in that file shifted when the fix landed. The computation
is now at line 83; the returns are at 51, 92 and 96.)

`segmentation.py:43,53` *does* carry per-part `info`, so the data
existed upstream the whole time.

That left `is_physically_plausible()` (`superquadric.py:156`) as the
pipeline's **only** fit-quality gate — and it checks nothing but whether
the semi-axes are ≤ 0.5 m. It never looks at `rmse` or `success`.

---

## What changed

### `iterative_segment.py` — optional diagnostics

`iterative_two_part_segment(..., return_diagnostics=True)` appends a 4th
return value:

```python
{'combined_rmse', 'rmse_a', 'rmse_b',
 'n_parts', 'n_iters', 'converged', 'stale_exit'}
```

The single-part early return now reports `segment_and_fit`'s per-part
`info`, which was also being dropped.

**Default stays `False`**, so the existing 3-tuple contract and every
current caller are untouched. Verified both branches.

### `symmetry.py` — normalised residuals

`info['rmse']` is volume-scaled (`superquadric.py:28` returns
`(F**(e1/2) - 1) * sqrt(a1*a2*a3)`), so a large object gets a large rmse
for the same *relative* misfit. `normalised_rmse()` divides by the
geometric mean semi-axis, making residuals comparable across objects.

Both fits' residuals are carried into the feature vector as `ax_nrmse`
and `fl_nrmse`.

### `run_corrected_pipeline.stage_failure_diagnosis`

Frames the question as **detection**: predict "this instance will be
misclassified" from fit-quality signals alone, never from the label.

- AUROC per indicator, over `ax_nrmse`, `fl_nrmse`, `residual_ratio`,
  `angular_variation`.
- An indicator scoring **below** 0.5 is reported as informative with the
  sign flipped, rather than discarded — that is a real finding, not a
  failure.
- A **risk–coverage curve**: error rate as a function of how much of the
  data you keep. This is what a robot deciding whether to trust an
  estimate actually needs, and it is what [item 7](item7_corrective_multiview.md)
  triggers on.

Outputs: `results/corrected/failure_diagnosis.csv`,
`results/corrected/risk_coverage.csv`.

---

## Early result — fixture only

From a 120-instance fixture built from real exported clouds with
pseudo-video ids:

```
ax_nrmse             AUROC 0.683  (lower=worse)
fl_nrmse             AUROC 0.639  (lower=worse)
residual_ratio       AUROC 0.561  (higher=worse)
angular_variation    AUROC 0.558  (higher=worse)

risk-coverage (indicator: ax_nrmse)
  error at 100% coverage : 25.00%
  error at  50% coverage : 12.07%
  -> abstaining on the flagged half genuinely reduces error
```

**Abstaining on the flagged half roughly halves the error rate.** If
that survives at full scale, item 7 has something real to trigger on.

**Treat this as a signal, not a result:** 120 instances, pseudo-videos,
`max_nfev=300`. The direction is encouraging; the magnitude is not
trustworthy.

Note the direction — `lower=worse` on the residuals. Lower residual
associates with *higher* error, which is counter-intuitive and worth
understanding rather than just reporting. The likely explanation is the
same one behind
[item 2's negative result](item2_fitting_constraint.md#what-was-tried-and-rejected):
a small partial arc is easy to fit closely and also easy to misclassify.

---

## What is left

### 1. Real numbers

Run the corrected pipeline. `stage_failure_diagnosis` runs
automatically as stage 4.

**What to expect:** AUROC will likely fall relative to the fixture — 120
instances with 3 folds is optimistic. The number to watch is whether
**any** indicator clears roughly **0.65** on real data. Below that,
item 7 has nothing dependable to trigger on and should be reconsidered
rather than forced.

### 2. Wire `combined_rmse` into the export

`export_instrumented.py` currently records `seg_combined_rmse` as
`np.nan` — it calls `fit_both`, which uses `fit_superquadric` directly
rather than the two-part segmenter, so the segmenter's diagnostic is not
produced on that path.

**What to do:** decide whether the two-part segmenter should also run
during export. It roughly doubles export cost again, and its value is
unproven until the simpler indicators are measured on real data. **Do
the cheap measurement first.**

### 3. Add the confidence margin as an indicator

`classify_graph` returns a full ranked list, so `top1 − top2` is
computable — and nothing currently computes it. It is often the
strongest error predictor in a classifier, and costs nothing.

Caveat from the code: for `joint`/`ensembled` scoring, scores are
**unnormalised Gaussian memberships**, not probabilities, so the margin
is not calibrated. For `--scoring ml` it is a genuine
`predict_proba` simplex and the margin is meaningful.

### 4. Correct-vs-incorrect breakdown

The item asks for this explicitly. The current output gives AUROC and
risk–coverage over all instances. Add a per-class split so "which
categories are diagnosable" is answerable — bowl and mug are the ones
that matter, since they fail most.

---

## What to do

1. Run the corrected pipeline; read `failure_diagnosis.csv`.
2. If no indicator clears ~0.65 AUROC, say so and drop item 7 rather
   than building a policy on a detector that does not detect.
3. If one does, add the score margin (cheap) before adding the segmenter
   residual (expensive).
