# Item 4 — classifier × observation aggregation ablation (2×2)

> *Compare classifier choice and per-frame vs multi-view/aggregated
> observations independently.*

**Status: not started. All four cells already have runners.**

---

## Why this experiment is needed

SAGE's headline moved 78.4% → 89.4% → 95.72% across recent work, and
**two things changed at once**: the classifier (Welford prototypes →
ExtraTrees) and the observation (per-frame → multi-view aggregated).
Nothing currently separates their contributions, so no one can say which
of the two the improvement came from.

That is precisely what a reviewer will ask.

---

## The 2×2

| | per-frame | multi-view aggregated |
|---|---|---|
| **registry prototype scoring** | `ycbv_training/evaluate_on_ycbv.py` | `evaluate_multiview_ycbv.py` |
| **learned classifier (ExtraTrees)** | `evaluate_on_ycbv.py --scoring ml` | `kfold_multiview_eval.py` |

**Every cell already has a working runner.** The work is not
implementation — it is running all four against **one shared fold
definition** and one metric writer, so the numbers are comparable.

---

## Two things that must be stated with the result

### 1. Multi-view uses ground-truth pose

`ycb_pose_aggregation.py:63` builds the fused cloud with
`cloud_object = (cloud_camera - t) @ R` taken from `meta['poses']`.

Multi-view numbers are therefore **not achievable at test time without
pose**. They are an upper bound under known pose, and must be labelled
that way. A 95.72% that quietly depends on ground-truth pose is not a
recognition result a robot can reproduce.

### 2. The cells differ in *n* by an order of magnitude

Multi-view granularity is one instance per **(video, class)** pair —
roughly 30 to 157 instances — against 1109 for per-frame. So the
multi-view cells have far wider intervals, and comparing point estimates
across the diagonal is meaningless without them.

`kfold_multiview_eval.py` already provides `wilson_ci`; use it per cell.

---

## What to do

1. **Run the corrected pipeline first.** `stage_unified` writes the fold
   definition to `results/corrected/`; all four cells should consume it
   rather than each computing its own.

2. **Run the two multi-view cells:**
   ```bash
   python3 kfold_multiview_eval.py --dataset_root <root> --scoring ml --workers 30
   python3 evaluate_multiview_ycbv.py --dataset_root <root> \
       --model ../trained_ycbv_ml_v2.json --scoring ml
   ```

3. **Run the two per-frame cells** with `evaluate_on_ycbv.py` at
   `--scoring joint` and `--scoring ml`.

4. **Assert which scorer actually ran in each cell.** This is not
   paranoia:
   - `classify_graph_ml` **silently falls back** to
     `classify_graph_ensembled` when `_ml_classifier` is `None`
     (`registry.py:617-622`).
   - `Registry.load` does **not** deserialise the sklearn model — it
     retrains it on every load (`registry.py:684`), and
     `rebuild_ml_classifier` is a no-op without sklearn or without raw
     examples.

   So a cell can quietly run the wrong classifier and report a plausible
   number. Log the classifier type per cell and fail loudly on a
   mismatch.

5. **Report a 2×2 table with intervals**, plus the pose caveat in the
   caption, not a footnote.

---

## What to expect

The interesting outcome is the **interaction**. If multi-view helps the
prototype scorer much more than the learned one, that says the fusion is
compensating for a weak classifier rather than adding information — and
the honest framing of the contribution changes accordingly.

Also expect the per-frame ExtraTrees cell to be the one most affected by
[item 2](item2_fitting_constraint.md)'s retrain, since it consumes the
feature vector directly.

**Do this after item 2's retrain**, or the 2×2 measures the leak as much
as it measures the two axes.
