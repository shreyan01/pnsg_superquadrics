# Runbook — how to run everything, in order

For whoever has the YCB-Video dataset. Follow top to bottom; each step
says what to check before moving on.

Nothing here needs a decision from you except **Step 5**, which is
flagged.

---

## 0. Setup (once)

```bash
cd ~/pnsg_superquadrics
git checkout task1 && git pull

cd sage_baseline_experiments
python3 -m venv experiment
experiment/bin/pip install -r requirements.txt
```

Everything below uses `PY=experiment/bin/python`.

**Check the thread fix before any long run.** There is a documented
incident where 30 workers each spawned ~30 BLAS threads — a load average
of 935, and 4 folds took 14 hours instead of 75 minutes:

```bash
cd ~/pnsg_superquadrics && $PY check_thread_limits.py --workers 8
```

---

## 1. Verify before spending hours — 30 seconds

```bash
cd sage_baseline_experiments
$PY src/run_corrected_pipeline.py --selftest
```

**Expect:**

```
flexible fit a1==a2              0.0%      0.0%   PASS
radial profile non-zero        100.0%    100.0%   PASS
PASS -- the fitting path carries no label information.
```

**If it FAILs, stop.** The feature path is broken and everything
downstream would be wrong. No dataset is touched by this step.

---

## 2. Smoke test — a minute or two

```bash
$PY src/run_corrected_pipeline.py \
    --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --limit 50 --workers 30
```

**Check three things before continuing:**

1. `Denominator reconciles: N fitted + M abstained = N+M` — if this line
   is absent or wrong, the abstention accounting is broken.
2. `Videos covered:` is more than 1. If it is 1, the frame keys are not
   parsing as `<video>/<frame>` and the video-level split will be a lie.
3. No fold-leak assertion fires.

**Common failure:** `ModuleNotFoundError: No module named 'cv2'` →
`experiment/bin/pip install opencv-python`.

---

## 3. The main run — items 1, 5, 6

```bash
$PY src/run_corrected_pipeline.py \
    --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --splits train val_sample --workers 30
```

Roughly `11/N` minutes per split for the export on N workers, plus a few
minutes of analysis. With 30 workers, budget under half an hour.

**Produces** `results/corrected/`:

| File | What |
|---|---|
| `unified_protocol.csv` | item 1 — every method, same video folds |
| `<method>_confusion_matrix.csv` | per-method confusion matrices |
| `<method>_predictions.csv` | per-instance, with `video_id` |
| `grouped_statistics.csv` | item 5 — clustered CIs, all methods |
| `failure_diagnosis.csv` | item 6 — AUROC per indicator |
| `risk_coverage.csv` | item 6 — error vs coverage |

**What to expect — numbers move, mostly down. That is the correction
working, not a regression:**

- **`can` recall falls from ~100% to ~84%.** The 100% was the label
  leak. Do not "fix" this.
- **Overall accuracy may go UP.** A 125-instance check showed
  72.8% → 81.6%, because carrying both fits preserves geometry the old
  single label-selected fit discarded. Confirm at full scale.
- **Confidence intervals widen ~5×.** SAGE's went from ±1.8 pp to
  ±9.6 pp once clustered by video. Some comparisons will stop being
  significant.
- **~30% of instances abstain.** They were always failing; they were
  just dropped silently before.

---

## 4. Fixed-population robustness — item 3

```bash
$PY src/evaluate_sage_robustness.py \
    --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --model ../trained_ycbv_ml_v2.json --workers 30
```

Fixed population is now the **default**. The first line of each sweep
prints `population fixed at N instances`; every later level scores that
same set, and instances that stop fitting count as **errors** rather
than vanishing.

**Expect SAGE's robustness curve to look worse than previously
reported.** That is the honest number — the old curve dropped the
instances it could no longer fit, so accuracy was computed over an
easier population as noise increased.

`--per-level-population` restores the old behaviour if you want the
comparison for the paper.

Then re-run task 4 so the head-to-head uses the fixed population:

```bash
$PY src/task4_robustness.py \
    --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --sage_model ../trained_ycbv_ml_v2.json --workers 30
```

---

## 5. The retrain — item 2. **This one needs a decision.**

Everything above still evaluates against models trained with the
label-conditioned fitter. Two paths exist, and they are not equivalent.

### 5a. The ML classifier — straightforward

The corrected pipeline's `extratrees` row **already is** the corrected
SAGE-ML number: same 21-column label-free features, same video folds. If
ML scoring is your headline, there is nothing further to retrain — read
it out of `unified_protocol.csv`.

To bake it into a distributable model:

```bash
cd ~/pnsg_superquadrics
$PY bake_ml_classifier.py \
    --model trained_ycbv_ml_v2.json \
    --features baseline_data/instrumented_train.npz \
    --out trained_ycbv_labelfree.json
```

`import_ml_training_data()` accepts any `(N, D)`, so the 21-column
vector needs no change there.

### 5b. The Welford prototype path — needs a choice

`classify_graph` and `graph_modes` are hardcoded to 13 columns, and
`canonicalize(` has **130 call sites**. Making that path label-free
means choosing one of:

- **Use the flexible fit for everyone.** 13 columns, no leak, no code
  churn. Costs the circularity prior — and the author's notes record
  that forcing the wrong path cost bottle 31.5% → 10.3%.
- **Widen the Welford path to 21 columns.** Principled, but touches
  `canonicalize`, `DEFAULT_INIT_STD`, both active masks, and invalidates
  every `trained_ycbv_*.json`.
- **Report ML scoring as the headline** and describe the prototype path
  as the interpretable-but-lower-accuracy variant, with the leak
  disclosed.

**Recommendation: the third.** ML scoring is already the headline
(89.4% vs 78.4%), it is already fixed by 5a, and the other two options
are a large change under deadline. Whichever you pick, say in the paper
which scoring produced which number.

---

## 6. Item 4 — the 2×2 ablation

Run after step 5, or the ablation measures the leak as much as the two
axes it is meant to separate.

```bash
cd ~/pnsg_superquadrics
$PY kfold_multiview_eval.py --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --scoring ml --workers 30
$PY evaluate_multiview_ycbv.py --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --model trained_ycbv_ml_v2.json --scoring ml
```

Two things **must** appear with the result:

1. **Multi-view uses ground-truth pose** (`ycb_pose_aggregation.py:63`).
   Those numbers are not achievable at test time without pose.
2. **Cells differ in n by an order of magnitude** — ~30–157 multi-view
   instances against 1109 per-frame. Report Wilson intervals per cell.

And **assert which scorer actually ran**: `classify_graph_ml` silently
falls back to `classify_graph_ensembled` when `_ml_classifier` is None
(`registry.py:617-622`), and `Registry.load` retrains the sklearn model
on every load rather than deserialising it.

---

## 7. Item 7 — corrective multi-view. **Gate first.**

```bash
cat results/corrected/failure_diagnosis.csv
```

**If no indicator clears ~0.65 AUROC, do not build this.** Write up the
negative result instead — "we tested whether geometric fit quality
predicts recognition failure, and it does not" is a real finding and a
defensible reason not to claim a closed loop.

If one does clear it, add the top1−top2 score margin as an indicator
first — it is free (`classify_graph` already returns the full ranked
list) and is often the strongest error predictor. Then see
[`item7_corrective_multiview.md`](item7_corrective_multiview.md).

---

## Sanity checks to run at the end

```bash
# every recorded SAGE constant vs what the data says
$PY src/sage_reference.py

# the non-dataset pipeline still passes
$PY run_all.py 0 1 5
```

`sage_reference.py` currently reports `joint_v1` (the 78.4%) as
**unverifiable** — there is no instance-level file backing it in this
repo. If that number goes in the paper, either regenerate its
predictions or state that it is a recorded figure from an earlier run.

---

## Two standing warnings

**Regenerate results only on this machine.** `knn_grouped` gives 0.8999
on Windows against 0.9089 here, with identical inputs, source and
package versions — `AgglomerativeClustering` breaks ties differently
across platforms. Commits from the revision work are deliberately
source-only for this reason. Once step 3 has run, real `video_id`
replaces that inferred grouping and the problem disappears.

**Do not quote a pre-retrain number for `can`.** Anything near 100% is
the leak. It will be the first thing a reviewer asks about.
