# Item 5 — grouped statistical analysis

> *Recompute confidence intervals/significance using video-level
> grouping (e.g., grouped bootstrap).*

**Status: done, to the limit of the current data. Improves further once
[item 1](item1_unified_protocol.md)'s export lands.**
Commit `35f7c51`.

---

## What was wrong

`bootstrap_accuracy_ci` resampled the 1109 evaluation rows as if they
were independent draws. They are frames from ~13 validation videos, and
frames of one object in one video succeed or fail together. Treating
correlated observations as independent makes the effective sample size
far smaller than 1109, and produces intervals that are **too narrow** —
the standard cluster-sampling error.

`ycbv_training/metrics.py` could not have done better: it receives
`(true, pred, confidence)` tuples with no instance or video identifier,
so no existing metric could group by anything.

---

## What changed

**Where:** `sbe/src/task5_statistics.py`

- `cluster_bootstrap_ci(y_true, y_pred, clusters, ...)` — resamples
  whole clusters with replacement rather than rows. The resampled set
  varies in size between draws, which is correct for a cluster
  bootstrap.
- `clusters_for_predictions(predictions)` — recovers the clustering from
  whatever the prediction file carries:
  - `"video"` — `instance_id` is `"<video>/<frame>#<obj>"`, so real
    video identity is available. This is what `evaluate_on_ycbv.py`
    writes.
  - `"object_proxy"` — `instance_id` is a bare export row index, so it
    falls back to the inferred object grouping.
  - `None` — nothing recoverable.
- Both intervals are reported **side by side**, not one replacing the
  other, because the gap between them is itself the result.

New columns in `results/task5/bootstrap_ci_summary.csv`:
`ci_lower_clustered`, `ci_upper_clustered`, `cluster_kind`,
`n_clusters`, `ci_width_iid`, `ci_width_clustered`.

---

## The result

SAGE, `val_sample`:

| | 95% CI | width |
|---|---|---|
| i.i.d. bootstrap | [87.74, 91.25] | 3.52 pp |
| **cluster bootstrap over videos** | **[78.50, 97.79]** | **19.29 pp** |

**A 5.5× widening.** The headline is not 89.4% ± 1.8 but 89.4% ± 9.6,
because it rests on **13 videos**, not 1109 independent samples.

**The consequence that matters:** the clustered lower bound (78.5%) sits
**below several baseline point estimates**. Comparisons that looked
decisive under the i.i.d. interval need rechecking before they go in a
paper.

Every model widens, 1.1× to 5.5×. The full table prints on every run.

---

## The caveat, which the script prints as a warning

The clustering **levels differ between models**:

- SAGE → `video` (n = 13)
- baselines → `object_proxy` (n = 201)

because only SAGE's prediction file carries video identity. Coarser
clustering widens intervals mechanically, so **SAGE's larger widening
factor is partly granularity, not necessarily greater uncertainty**, and
the clustered intervals are **not yet comparable across models**.

`task5_statistics.py` emits this as an explicit warning rather than
leaving a reader to notice it.

---

## What is left

### Nothing, in this file — but it gets better after item 1

Once `run_corrected_pipeline.py` has run, `stage_grouped_stats` computes
the same intervals with **real video ids for every method**, at the same
clustering level. Those intervals *are* comparable across models, and
that is the version to put in the paper.

Already implemented, in `run_corrected_pipeline.stage_grouped_stats`;
it just needs the export.

### Re-run the significance tests on the clustered basis

`mcnemar_model_comparisons.csv` and `can_vs_bottle_tests.csv` are still
computed on the i.i.d. assumption. McNemar is paired per instance, so it
is less affected than the CIs — but the can-vs-bottle Fisher test
compares two independent groups of *correlated* observations and will
overstate significance.

**What to expect:** the can-vs-bottle gap (+31.5 pp, p = 6.8e-28) will
stay significant — it is enormous — but the p-value will move by many
orders of magnitude, and the CI will widen substantially. Worth
recomputing before quoting either.

---

## What to do

1. Run the corrected pipeline; `grouped_statistics.csv` appears with all
   methods clustered by real video.
2. Quote the **clustered** interval in the paper, and say it is
   clustered. An interval that says ±1.8 pp when the honest answer is
   ±9.6 pp is the kind of thing a reviewer finds.
3. Recompute the can-vs-bottle test on a clustered basis, and re-run it
   **after** [item 2](item2_fitting_constraint.md)'s retrain — `can` is
   currently at 100% because of the leak, which inflates that gap.
