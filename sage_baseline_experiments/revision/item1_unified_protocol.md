# Item 1 — unified evaluation protocol

> *Rerun SAGE and baselines using the same video-level splits,
> evaluation samples, and input information; report accuracy, balanced
> accuracy, macro-F1, per-category recall, and confusion matrices.*

**Status: code complete. Needs a dataset run.**
Commits `988aaab`, `7315e80`.

---

## What was wrong

Three separate things made the existing comparison unfair:

**1. Different splits.** SAGE's number came from a video-level hold-out.
The baselines cross-validated *within* the validation set. Not the same
question.

**2. No video identity in the export.** `export_baseline_data.py` emitted
one row per object per frame with no provenance — no `video_id`, no
`frame_id`. A video-level split was therefore impossible, and
`data_utils.compute_object_groups` had to *infer* object identity from
point-cloud size as a proxy. That proxy is also why grouped results do
not reproduce across machines
([item 0](item0_p0_repairs.md#reproducibility-finding)).

**3. Different input information.** SAGE trains on **multi-view**
aggregated clouds but evaluates **single-frame**. Worse, multi-view
aggregation uses ground-truth pose —
`ycb_pose_aggregation.py:63` does `cloud_object = (cloud_camera - t) @ R`
from `meta['poses']`. Multi-view numbers are therefore **not achievable
at test time without pose**, and must be labelled as such wherever they
appear.

---

## What changed

### `sbe/src/export_instrumented.py` — the enabling artifact

Replaces `export_baseline_data.py`'s output. Carries four things it
dropped, each blocking a different item:

| Carried | Unblocks |
|---|---|
| `video_id`, `frame_id`, `instance_id` | this item, and item 5 properly |
| both fits, chosen without the label | [item 2](item2_fitting_constraint.md) |
| abstention as a **row**, not a `continue` | [item 3](item3_fixed_population_noise.md) |
| fit residuals, success flags, nfev, plausibility | [item 6](item6_failure_diagnosis.md) |

### `sbe/src/run_corrected_pipeline.py` — `stage_unified`

- One `GroupKFold` fold definition over **real** `video_id`, shared by
  every method.
- **Asserts per fold** that no video appears on both sides. This is the
  property the whole protocol rests on, so it is checked rather than
  assumed.
- Every method reads the **same feature matrix**, so "same input
  information" holds by construction rather than by claim.
- Reports accuracy, balanced accuracy, macro-F1, per-class recall and a
  confusion matrix per method, plus per-instance predictions with
  `video_id` attached.

Methods evaluated: `knn`, `svm_linear`, `svm_rbf`, `extratrees`.

---

## How to run

```bash
PY=experiment/bin/python
$PY src/run_corrected_pipeline.py \
    --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --splits train val_sample --workers 30
```

Outputs land in `results/corrected/`:

```
unified_protocol.csv          one row per method
<method>_confusion_matrix.csv
<method>_predictions.csv      instance_id, video_id, true, predicted
```

---

## What to expect

**Baseline numbers will drop.** The current grouped/groupkfold regimes
use inferred object groups (~201 clusters). Real video-level splitting
is coarser (~13 videos in `val_sample`) and therefore harder. A drop is
the correct result, not a regression.

**Folds will be lumpy.** With ~13 videos and 5 folds there are only 2–3
videos per fold, and class balance will swing. `GroupKFold` balances
fold *size*, not class mix — which is exactly what a real video hold-out
looks like, and why it was chosen over `StratifiedGroupKFold`.

**If the pipeline reduces the fold count**, that is the guard working:
it drops to `n_videos` folds when there are fewer videos than folds.

---

## What is left

### 1. SAGE is not yet in the table

`stage_unified` evaluates the four feature-based methods. SAGE itself
needs `evaluate_on_ycbv.py` run against the **retrained, label-free**
model on the same folds. Until [item 2](item2_fitting_constraint.md)'s
retrain lands, SAGE's row would not be on equal terms anyway.

**What to do:** after the retrain, run `evaluate_on_ycbv.py
--predictions_out results/corrected/sage_predictions.csv`, then merge
that row into `unified_protocol.csv`. The prediction format already
matches — `instance_id` is `"<video>/<frame>#<obj>"` in both.

### 2. PointNet is not in the table

It consumes raw clouds rather than the feature matrix, so it cannot use
`stage_unified` directly. It needs `task2_pointnet.py` run against the
same fold definition.

**What to do:** have `task2` read `results/corrected/` fold assignments
instead of recomputing its own. Small change; not done because the fold
file format should settle first.

### 3. Decide the headline input modality

Single-frame is the honest default, because multi-view needs
ground-truth pose. If multi-view numbers appear in the paper they must
carry that caveat explicitly. See [item 4](item4_ablation.md), which
exists to separate these two axes.

### 4. `val_sample.txt` provenance

The 1109-instance split behind the original 78.4% is **not generated by
any script in the repo** — `build_splits.py` emits only `train` and
`val`. It appears to be a manually created subset.

**What to do:** either record how it was made, or regenerate it with
`build_splits.py` and re-run. A reviewer asking "what is val_sample?"
should get a reproducible answer.
