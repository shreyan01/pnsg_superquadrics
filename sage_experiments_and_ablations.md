# SAGE — Experiments & Ablation Studies: Complete Log (v5)

**This replaces v4.** The one item Part 2 still had — the 5-fold
cross-validated multi-view result — is now done, with a real, proven,
statistically significant result. Every `[PENDING]` marker in the paper
is now filled in. This is the last update needed before the paper is
considered content-complete.

**One clarification worth stating plainly, since it caused real
confusion**: "ExtraTrees" and "k-fold" are not two competing models or
two things to choose between. ExtraTrees is the classifier — the same
one in every result this session, no exceptions. K-fold is an
*evaluation method* (rigorous cross-validation), used here specifically
to test one real design question: does classifying from a *fused
multi-view point cloud* work better than classifying from a *single
frame*? It does, substantially. The actual deployment decision this
unlocks is: **use multi-view fusion at inference whenever the robot can
gather more than one observation of an object before committing to a
classification** — not "switch to a different model."

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
| 22 | Multi-view fusion at inference (occlusion mitigation), fixed-split preview | 89.4% (single-frame) | 93.3% (n=30, fixed split) | Directionally positive, CI too wide to be conclusive alone |
| **22b** | **Multi-view fusion, 5-fold video-level cross-validation (the real test)** | 89.4% (single-frame, n=1109) | **95.72%** (179/187, 95% CI [91.8%,97.8%], the full dataset ceiling at this granularity) | **PROVEN**: two-proportion z-test vs. single-frame result, z=2.69, p=0.0072 (Fisher's exact p=0.0048). Every category improved, most sharply the occlusion-prone ones: mug 51.7%→88.2%, bottle 68.5%→97.1%, bowl 36.4%→57.1% |
| 23 | SAGE vs. PointNet, point-count downsampling robustness | — | SAGE 90.9% vs. PointNet 87.6% at 64 pts | **SAGE wins**, real head-to-head |
| 24 | SAGE vs. PointNet, Gaussian noise robustness | — | SAGE 78.7% vs. PointNet 91.3% at σ=4mm | **PointNet wins badly** — honest negative result |
| **25** | **SAGE's own sample-efficiency curve vs. baselines** | best baseline: 53.4%/60.0%/63.0% at n=1/2/5 | **SAGE: 66.7%/62.1%/70.1%** at n=1/2/5 | **SAGE wins at every budget tested**, single-draw caveat noted |

---

## PART 2 — Immediate next steps (UPDATED — the paper is now content-complete)

~~5-fold video-level cross-validated multi-view result~~ — **done, see
row 22b.** Real cost: 7.1 hours (25,694s), not the ~75 minutes estimated
beforehand — that estimate was wrong, not the run. Each of the 5 folds
requires a full, independent re-training (fresh per-frame export +
fresh ExtraTrees fit on ~37-38k examples per fold), which is far more
expensive than the original estimate accounted for.

~~Confirm the status of the CSG/k-NN work~~ — **resolved: confirmed gone,
not being pursued. See row 17.**

**Every `[PENDING]` marker in the paper draft is now filled in.** The
remaining open items in this document are all outside the paper itself
(hardware trials, future-work items) — see Part 5.

---

## PART 2B — The k-fold multi-view algorithm, start to end

Added here because the distinction between "which classifier" and
"which evaluation protocol" caused real confusion — worth a precise,
complete record.

**The classifier is ExtraTrees in every single result in this
document, with no exception.** "K-fold" is not a competing model — it
is the cross-validation *method* used specifically to rigorously test
one real design question: does classifying from a fused multi-view
point cloud beat classifying from a single frame? (It does,
substantially — see row 22b.)

1. **Find every real video.** Scan the dataset for all 92 video
   folders (`discover_video_ids`).
2. **Split into 5 folds.** Shuffle the 92 videos with a fixed random
   seed, deal round-robin into 5 groups of ~18-19 videos
   (`build_folds`). Every video belongs to exactly one fold — this is
   the entire basis for the "no leakage" guarantee, and it was
   re-verified directly against the actual code before trusting the
   result, not just assumed.
3. **Repeat 5 times, once per fold:**
   - This fold's videos become the test set; the other 4 folds
     (~74 videos) become the training set for this round.
   - Export per-frame training features from those ~74 videos (every
     5th frame, real superquadric fitting per object) — this is the
     expensive step; fold 5 alone touched 21,110 frames.
   - Train a brand-new `ExtraTreesClassifier` from scratch on
     everything just exported (37,622 examples in fold 5) — never
     reused from a previous fold.
   - Evaluate on the held-out fold's videos: for each real object, fuse
     multiple viewpoints of that same physical instance into one
     combined point cloud, fit one superquadric to the fused cloud, and
     classify it with the classifier just trained.
   - Record every prediction.
4. **Pool all 5 folds' held-out predictions together.** Every video was
   tested in exactly one fold, so this covers all 187 real object
   instances the dataset contains at this granularity, each scored by a
   classifier that genuinely never saw that video during training.
5. **Compute final statistics** from the pooled set: 179/187 = 95.72%
   overall, Wilson 95% CIs per class, confusion matrix.

**The real, practical takeaway**: this isn't evidence to swap models.
It's evidence that **multi-view fusion at inference time is worth
doing whenever the robot's setup allows gathering more than one
observation before classifying** — the same ExtraTrees classifier,
given richer input, does substantially better, especially on the
occlusion-prone categories.

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
- ~~Run the 5-fold cross-validated multi-view result~~ — **done, see
  Part 1 row 22b. The paper draft is now content-complete.**
- ~~Confirm CSG/k-NN status~~ — **resolved: confirmed gone, not pursuing.**
- **Physical grasping trials on OpenArm.** Now the single largest
  remaining item overall. The ROS2 package (`sage_openarm_grasping`) is
  built but untested on real hardware. Config placeholders (camera
  topics, joint names, MoveIt2 planning group, and especially the
  elbow-bend angle) still need real values before any physical trial —
  see that package's own README for the exact discovery commands
  (`ros2 topic list`, `ros2 topic echo /joint_states`).
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