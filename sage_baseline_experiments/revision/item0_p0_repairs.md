# P0 — repairs before anything else

Not one of the seven items, but every one of them was blocked behind
these. Commit `fc6a53d`.

**Status: done.** Nothing here needs the dataset.

---

## 1. A 36-minute run that crashed at the end

**Where:** `sbe/src/task2_pointnet.py:1189`

**What was wrong:** `run_sage_comparison()` imported
`evaluate_sage_condition` from `task4_robustness`, which stopped
existing when task 4's SAGE half was rewritten around
`evaluate_sage_robustness.run_sweep`. It is called unconditionally from
`main()`, so a plain `python task2_pointnet.py` raised `ImportError`
**after** all 15 fold trainings had completed.

**What changed:**

- New `evaluate_sage_robustness.evaluate_clouds()` evaluates a trained
  registry on in-memory clouds, which is what task 2 actually needed.
  Unlike the frame-based path it never silently drops an instance: every
  input returns a row with a `reason`, and the metrics report
  `n_evaluated`, `n_abstained` and `n_input` separately so
  `n_evaluated + n_abstained == n_input` always holds.
- New `check_sage_comparison_available()` preflights the model, the
  imports and the presence of an ML classifier **before** training
  starts, and refuses to begin a run that would fail at the end.

**Why it matters beyond the crash:** that abstention accounting is the
foundation item 3 needs. It was cheaper to build it here than to bolt it
on later.

**Verified:** 24 real clouds — 21 evaluated, 3 implausible, accounting
reconciles.

---

## 2. Two different SAGE numbers in the same repo

**Where:** `sbe/src/task1_feature_baselines.py` said `0.894`;
`sbe/src/results_io.py` said `0.784`; `task3` said `0.784`; `task5`
carried the `0.894` per-class rates.

**What was wrong:** the notebooks read `results_io`, so every notebook
figure rendered **78.4%** while the task-1 comparison table used
**89.4%**. Both numbers are real — they are different *scoring modes*
(Welford prototypes vs ExtraTrees), not a correction of one by the
other — but nothing recorded which was which.

**What changed:** new `sbe/src/sage_reference.py` is the single source.
It keeps both runs with provenance, names `ml_v2` the headline, and
**derives** the numbers from `results/sage/sage_predictions.csv` when
that file exists rather than trusting a constant. `task1`, `task3`,
`task5` and `results_io` all import from it and now agree.

Two things fell out:

- `task1`'s summary row hardcoded `balanced_accuracy = np.nan` when the
  real 0.7131 was available all along.
- `task5`'s can-vs-bottle test now uses **measured** per-instance
  predictions instead of counts reconstructed from rounded rates. The
  gap is **+31.5 pp [+23.97, +39.04], p = 6.8e-28** — not the +63.4 pp
  previously reported, which came from the superseded joint-scoring
  rates.

**To check it:** `python3 src/sage_reference.py` prints every recorded
run with its provenance.

---

## 3, 4, 5 — smaller repairs

| What | Where | Why |
|---|---|---|
| `run_all.py` forwarded no arguments | `sbe/run_all.py` | `--dataset_root` on steps 3 and 4 was unreachable through it. Now forwards only the flags each step understands, and warns when pass-through flags match no selected step. |
| Two orphaned result CSVs removed | `results/task4/sage_{point_count,noise}_robustness.csv` | No script wrote those names. They sat beside the live `sage_*_results.csv` with a different schema and different numbers (94.06% vs 92.89% at 1024 points). Recoverable from git history. |
| Stale cv2 note | `sbe/requirements.txt` | Claimed opencv was needed for task 3 only. Task 4's SAGE half, `evaluate_sage_robustness.py` and task 2's SAGE comparison all reach the same cv2-backed loader. |

---

## Reproducibility finding

**This one affects how you should use the repo, so it is worth reading.**

Re-running task 1 on the Windows machine changes the committed grouped
numbers — `knn_grouped` **0.9089 → 0.8999** — with:

- identical input (`features_val_sample.npz` unchanged since `4cebaa2`)
- identical source (the only task-1 change between `d441ed3` and `HEAD`
  is the `SAGE_REFERENCE` dict)
- identical package versions (numpy 2.5.2, sklearn 1.9.0, scipy 1.18.1,
  pandas 3.0.5)

No `GROUP_DISTANCE_THRESHOLD` in a 0.002–0.005 sweep reproduces the
committed value, so it is not parameter drift. The likely cause is
`AgglomerativeClustering` inside `data_utils.compute_object_groups`
resolving ties differently across platforms, with grouped folds
amplifying a small change in group membership. The same applies to the
PointNet checkpoints, which differ by 512 bytes between machines.

**What to do about it:**

1. **Regenerate results only on the Linux box.** Commits from this work
   are deliberately source-only; result CSVs and `.pt` files were
   restored rather than overwritten with Windows numbers.
2. The real fix is item 1: once the export carries `video_id`, the
   inferred proxy disappears and grouping becomes exact and portable.
