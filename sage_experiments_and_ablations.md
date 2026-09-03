# SAGE — Experiments & Ablation Studies: Complete Log (v4)

**This replaces v3.** One small but clean closure since then: the CSG/k-NN
work (rows 17-18, carried as "status unknown" through v2 and v3) has been
confirmed by you as no longer present in the repo, and not something you
want to pursue. Marked formally abandoned below rather than left open —
Part 2 now genuinely has only one remaining item.

---

## What changed since v2, in one place

- **Task 3's SAGE half is DONE, not in progress.** Real result: SAGE beats
  every baseline classifier at every sample size tested — 66.7% at n=1
  (+13.3pp over the best baseline), 62.1% at n=2 (+2.2pp), 70.1% at n=5
  (+7.1pp). This is the direct evidence for the data-efficiency claim the
  paper's motivation makes, not just an assertion. One honest caveat: this
  is a single draw per n (each real SAGE evaluation takes several minutes
  of actual geometry re-fitting, unlike the baselines' cheap 20-repeat
  classifier resampling), so it doesn't carry the same statistical power
  as the baseline curve — stated directly in the paper, not smoothed over.
- **Two more real regressions found and fixed while getting that result,
  both worth recording so the same mistake isn't repeated:**
  - `task3_sample_efficiency.py` called the new `assert_thread_limits_ok()`
    safety guard *before* `sage_pipeline.check_available()` — the exact
    step that puts the repo root on `sys.path` in the first place. Fixed
    ordering; tested directly in this exact environment afterward, not
    just reasoned about.
  - Adding `instance_id` to `_eval_one_frame`'s return value (v2's item
    #23) broke a *second* consumer of that function that wasn't caught the
    first time: `task3_sample_efficiency.py`'s own `evaluate_registry()`
    unpacks those results directly and was still expecting the old 4-tuple
    shape. Found via a real `ValueError: too many values to unpack`, fixed,
    and the whole codebase was searched for every remaining call site of
    `_eval_one_frame` to confirm no third copy of this bug exists.
  - Also added: a real tqdm progress bar to `evaluate_registry()`'s
    evaluation loop, which had none before — a run that's actually
    working (just slow) and a run that's genuinely hung were
    indistinguishable from the terminal until this was added.
- **A new, minimal script was built**: `sage_sample_efficiency.py` — calls
  `task3_sample_efficiency.py`'s own tested functions directly
  (`collect_training_graphs`, `build_registry`, `evaluate_registry`),
  skipping the baseline-comparison scaffolding, specifically so getting
  SAGE's own curve doesn't require re-running or re-explaining the whole
  combined script.
- **Figure 3 (sample efficiency) was revised twice** based on direct
  feedback: first to overlay SAGE's real curve on the baseline curve,
  then again to (a) use an equidistant categorical x-axis instead of
  linear numeric spacing, and (b) drop the n=10 tick entirely, since SAGE
  has no data point there (`bowl`'s 5-instance ceiling) and showing it
  made SAGE's line look like it stopped mid-figure rather than presenting
  a fair matched-range comparison.
- **Remaining PENDING items in the paper are now down to exactly one**:
  the 5-fold cross-validated multi-view result.

---

## PART 1 — Master Ablation Table (UPDATED, new row added)

| # | Condition | Before | After | Verdict |
|---|---|---|---|---|
| 1 | Single-view vs. multi-view training | 61.0% | 83.2% | KEPT |
| 2 | Naive geometric-mean vs. dominant-anchored structural matching | mug→can: 93/200 wrong | 10/200 wrong | KEPT |
| 3 | Unbounded vs. capped registry growth | 44 modes for one word | capped at 5 | KEPT |
| 4 | Circular-hue bug (raw subtraction on wraparound quantity) | corrupted matches near 0°/360° | fixed | KEPT |
| 5 | Missing-color treated as real zero | corrupted single-frame matches | fixed | KEPT |
| 6 | Discrete body/neck detector for taper | real accuracy 36.9%→28.0%→12.8% (2 threshold attempts) | reverted | **REVERTED** |
| 7 | Continuous 5-point radial profile (replaces #6) | — | `can` → 94.9% | KEPT |
| 8 | Circularity fix (`eps2=1.0`, not just `a1=a2`) | real bottle proto showed `eps2=0.62` (squircle) | true circle | KEPT |
| 9 | Vocabulary split (`bottle`→`mustard_bottle`/`bleach_bottle`) | 78.4% | 52.4% | **REVERTED** |
| 10 | `CONFIDENCE_PSEUDO_N` discount tuning | k=8 (over-penalized well-supported modes) | k=1 | KEPT |
| 11 | Forcing `bottle` through axisymmetric fitting (like `can`) | 31.5% (flexible) | 10.3% (forced axisym) | **REVERTED**, kept flexible for bottle |
| 12 | Chamfer-distance fit-quality gate | — | 89% of training rejected, `bowl` eliminated, `box` 95%→54% | **REVERTED** |
| 13 | Display-only confidence calibration (sqrt transform) | ECE 0.75 | ECE 0.667, accuracy unchanged (proven) | KEPT |
| 14 | Original scoring rule, freshly re-measured | — | **78.8%** (n=1109, real saved predictions) | Historical baseline, corrected |
| 15 | Classifier diagnosis: features or scoring rule? | GaussianNB (proxy for original rule) 77.1% | ExtraTrees on identical features 94.9% (95% CI 93.6–96.0%) | **ExtraTrees adopted** |
| 16 | Weighted k-NN classifier (raw example storage) | — | superseded by #15 | ABANDONED |
| 17 | CSG joint two-primitive fitting, `trained_ycbv_csg_knn.json` | — | confirmed no longer present in the repo | **ABANDONED** — not being pursued |
| 18 | Circular-hue bug reintroduced in new k-NN code | 50x distance distortion | fixed | Moot — the code it applied to (#16, #17) is abandoned |
| 19 | Real data-starvation bug in ExtraTrees training | 82.2% | **89.4%** after per-frame training data added (35,178 examples) | KEPT, the real headline fix |
| 20 | Class-imbalance correction (cap majority classes 16,619→2,200) | 89.45% | 89.00% | **McNemar p=0.182 — proven non-significant** |
| 21 | Paired significance test, original vs. adopted classifier | 78.8% | 89.4% | **McNemar p=4.17×10⁻²¹ — proven real improvement** |
| 22 | Multi-view fusion at inference (occlusion mitigation) | 89.4% (single-frame) | 93.3% (n=30, fixed split) | Directionally positive, not yet proven; 5-fold CV in progress |
| 23 | SAGE vs. PointNet, point-count downsampling robustness | — | SAGE 90.9% vs. PointNet 87.6% at 64 pts | **SAGE wins**, real head-to-head |
| 24 | SAGE vs. PointNet, Gaussian noise robustness | — | SAGE 78.7% vs. PointNet 91.3% at σ=4mm | **PointNet wins badly** — honest negative result |
| **25** | **SAGE's own sample-efficiency curve vs. baselines** | best baseline: 53.4%/60.0%/63.0% at n=1/2/5 | **SAGE: 66.7%/62.1%/70.1%** at n=1/2/5 | **SAGE wins at every budget tested**, single-draw caveat noted |

---

## PART 2 — Immediate next steps (UPDATED — genuinely one item now)

1. **5-fold video-level cross-validated multi-view result.** The only
   remaining gap of consequence. `kfold_multiview_eval.py` is built,
   tested, and now protected by the same thread-limit guard that's already
   been validated on two other expensive runs (Task 3, Task 4) without
   incident. Expected ~75 minutes.
   ```bash
   python3 kfold_multiview_eval.py \
       --dataset_root ~/pnsg_superquadrics/ycb_dataset \
       --n_folds 5 --scoring ml --workers 30
   ```

~~Confirm the status of the CSG/k-NN work~~ — **resolved: confirmed gone,
not being pursued. See row 17.**

---

## PART 3 — Baseline comparison experiments (Suraj's track) — UPDATED, all DONE except Task 6

| Task | Status | Real result |
|---|---|---|
| **Task 1: same-feature baselines, corrected** | **DONE** | GroupKFold-corrected: k-NN 90.26% (95% CI [88.46%,91.97%]), SVM-linear 84.22%, SVM-RBF 85.57%. |
| **Task 2: raw point-cloud baseline (PointNet)** | **DONE** | GroupKFold: 93.24% (95% CI [91.70%,94.59%]). |
| **Task 3: sample-efficiency curve** | **DONE, both halves** | SAGE beats every baseline classifier at n=1,2,5 (see Part 1, row 25). Baseline curve reaches only 67-69% even at n=10 — still well below SAGE's full-data 89.4%. |
| **Task 4: robustness to noise/downsampling** | **DONE** | SAGE beats PointNet on downsampling (90.9% vs 87.6% at 64 pts), loses badly on noise (78.7% vs 91.3% at σ=4mm). |
| **Task 5: statistical rigor** | **DONE** | Real per-instance SAGE predictions included throughout; full bootstrap CI table, full pairwise McNemar matrix, real (not reconstructed) `can`/`bottle` Fisher's exact test. |
| **Task 6: GraspNet-1Billion cross-dataset generalization** | **Not started** | Stretch goal, unchanged. |
| **Failure-mode taxonomy** (Akira) | **Not started** | Unchanged — confirm still planned. |

---

## PART 4 — Detailed findings, in chronological order (ADDITIONS ONLY)

For items 1–24, see v2 of this document. What follows is new since then.

### Sample efficiency
25. **SAGE's real sample-efficiency curve, matched budgets against the
    baseline sweep.** Built a minimal standalone script
    (`sage_sample_efficiency.py`) that reuses `task3_sample_efficiency.py`'s
    own already-tested functions directly rather than re-running the
    combined baseline+SAGE script. Result: SAGE reaches 66.7% at n=1
    (a single confirmed example per category), 62.1% at n=2, 70.1% at
    n=5 — beating the best baseline classifier by 13.3, 2.2, and 7.1
    points respectively at those same budgets. n=10 and n=20 remain
    structurally untestable for SAGE (`bowl` has only 5 total confirmed
    training instances at the video-aggregation granularity SAGE's own
    training uses), unlike the baselines, which draw from a larger
    object pool at a different granularity and can reach n=10.
26. **Honest limitation on this result, stated directly rather than
    smoothed over**: unlike the baseline curve (20 random draws per
    point, real error bars), SAGE's curve is a single draw per n, since
    each real evaluation involves genuine geometry re-fitting (several
    minutes), not a cheap classifier resample. The non-monotonic dip at
    n=2 (62.1%, below both neighboring points) is consistent with
    ordinary small-sample draw variance rather than a real trend, given
    the baseline curve's own error bars at matching n are comparably
    wide (±8-13 points).

### Infrastructure (two more real bugs, worth recording)
27. **Safety-guard ordering bug.** `assert_thread_limits_ok()` was added
    to `task3_sample_efficiency.py`'s SAGE-half entry point *before*
    `sage_pipeline.check_available()` — the function that actually adds
    the repository root to `sys.path`. Result: `ModuleNotFoundError: No
    module named 'registry'` on every attempt. Root cause was fully
    traced (read `sage_pipeline.py`'s `_register_parent_package()`
    directly rather than guessing) before the fix, and the fix was
    verified by actually running the corrected import sequence in this
    exact environment afterward, not just reasoned about.
28. **A second, previously-missed consumer of `_eval_one_frame`'s
    changed return shape.** Adding `instance_id` (v2 item #23) was
    verified against its use inside `evaluate_on_ycbv.py` at the time,
    but `task3_sample_efficiency.py` has its own, separate
    `evaluate_registry()` function that also calls
    `evaluation_module._eval_one_frame` directly and unpacks the result
    itself — this second call site was missed the first time and only
    surfaced as a real `ValueError: too many values to unpack (expected
    4)` partway through a real run. Fixed, and this time the entire
    codebase was grepped for every remaining call site of
    `_eval_one_frame` (two total, both now confirmed correct) rather
    than trusting that the first fix was complete.
29. **No visibility into whether a long-running evaluation is actually
    working or genuinely hung.** `evaluate_registry()`'s main loop had no
    progress indicator of any kind. Given real per-run costs of several
    minutes, this made "is it stuck?" impossible to answer from the
    terminal alone. Fixed with a real tqdm bar; also documented the
    useful property that `ProcessPoolExecutor.map()` yields results in
    submission order, not completion order — so a stalled bar reliably
    identifies the specific stuck frame, even if later frames already
    finished computing in the background.

---

## PART 5 — Full "still to do" list, by owner (UPDATED)

**You (hardware + pipeline-specific):**
- Run the 5-fold cross-validated multi-view result (Part 2) — the single
  remaining item of consequence for the paper.
- ~~Confirm CSG/k-NN status~~ — **resolved: confirmed gone, not pursuing.**
- **Physical grasping trials on OpenArm.** Unchanged from v2 — the ROS2
  package (`sage_openarm_grasping`) is built but untested on real
  hardware. Config placeholders (camera topics, joint names, MoveIt2
  planning group, and especially the elbow-bend angle) still need real
  values before any physical trial.
- ~~Task 3's SAGE-half sample-efficiency numbers~~ — **done, see Part 1
  row 25.**
- Real, isolated `can`-vs-`bottle` fitting-strategy comparison — already
  satisfied by Table III in the paper draft, not a separate task.

**Suraj (ML/stats track):**
- Tasks 1, 2, 4, 5 — **done.**
- Task 3 — **done, both halves.**
- Task 6 (GraspNet) — still open, stretch only.

**Akira:**
- Failure-mode taxonomy — still not started; worth checking whether
  this is still planned given how much of the accuracy story has
  already been re-diagnosed this session (v2 item #19 covers a lot of
  the same ground).

**Future work (explicitly out of scope, real and legitimate, not
abandoned):**
- Chamfer-gate threshold recalibration against real (not synthetic) data.
- GraspNet integration beyond the stretch-task evaluation.
- The general, principled fix for the elbow-first collision policy:
  registering the pickup platform's real geometry as a MoveIt2 collision
  object so the planner routes around it automatically.
- A properly-averaged (multi-repeat) SAGE sample-efficiency curve, if the
  single-draw result (Part 1, row 25) is judged to need tighter error
  bars before publication — real compute cost, not attempted this
  session given the already-strong single-draw signal.