# Revision notes

**Running this? Start with [`RUNBOOK.md`](RUNBOOK.md)** — the exact
sequence, what to check at each step, and what to expect.

One file per pre-submission item: what changed, where, why, and what is
left to do.

Paths are relative to the repository root (`pnsg_superquadrics/`).
`sbe/` is shorthand for `sage_baseline_experiments/`.

| File | Item | Status | Blocked on |
|---|---|---|---|
| [`item0_p0_repairs.md`](item0_p0_repairs.md) | Repairs (not one of the seven) | **Done** | — |
| [`item1_unified_protocol.md`](item1_unified_protocol.md) | Unified evaluation protocol | Code complete | Dataset run |
| [`item2_fitting_constraint.md`](item2_fitting_constraint.md) | Remove label-conditioned fitting | Code complete | Dataset run + **retrain** |
| [`item3_fixed_population_noise.md`](item3_fixed_population_noise.md) | Fixed-population noise | Code complete | Dataset run |
| [`item4_ablation.md`](item4_ablation.md) | Classifier × observation 2×2 | Not started | Shared folds from item 1 |
| [`item5_grouped_statistics.md`](item5_grouped_statistics.md) | Grouped statistics | **Done** (improves further after item 1) | — |
| [`item6_failure_diagnosis.md`](item6_failure_diagnosis.md) | Geometric failure diagnosis | Code complete | Dataset run |
| [`item7_corrective_multiview.md`](item7_corrective_multiview.md) | Corrective multi-view policy | Not started | Item 6's real numbers |

## The one-line summary

Two defects were found that change how existing results must be read,
and both are now fixed in code but need a run on the dataset machine to
produce corrected numbers:

1. **The fitting strategy was chosen from the ground-truth label.**
   `a1 == a2` identified `can` with 100% accuracy. Every ~99% `can`
   figure in the repo, including SAGE's own, is partly this.
   → [item 2](item2_fitting_constraint.md)

2. **Confidence intervals treated 1109 correlated frames as independent
   draws.** Clustering by video widens SAGE's interval 5.5×, from
   ±1.8 pp to ±9.6 pp.
   → [item 5](item5_grouped_statistics.md)

## Where to run

Everything that needs `--dataset_root` runs on the **Linux box**. This
was decided because the dataset lives there, and because results do not
reproduce across machines (see
[item0](item0_p0_repairs.md#reproducibility-finding)).

```bash
cd ~/pnsg_superquadrics && git checkout task1 && git pull
cd sage_baseline_experiments
python3 -m venv experiment && experiment/bin/pip install -r requirements.txt

PY=experiment/bin/python

$PY src/run_corrected_pipeline.py --selftest                    # 30 s, no dataset
$PY src/run_corrected_pipeline.py --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --limit 50 --workers 30                                     # smoke test
$PY src/run_corrected_pipeline.py --dataset_root ~/pnsg_superquadrics/ycb_dataset \
    --splits train val_sample --workers 30                      # real run
```

That single command covers items 1, 5 and 6, and produces the export
items 3 and 4 need.

## What to expect when you run it

Numbers will move, and mostly downward. That is the point.

- **`can` recall will fall from ~100% to somewhere near 84%.** The 100%
  was the leak. Do not treat the drop as a regression.
- **Overall accuracy on the paired features went UP** in a 125-instance
  check (72.8% → 81.6%), because carrying both fits preserves geometry
  the single label-selected fit discarded. Whether that holds at full
  scale is the first thing to check.
- **Confidence intervals will be much wider** — roughly 5× on anything
  clustered by video. Some currently-significant comparisons will stop
  being significant.
- **The denominator will be visible for the first time.** Expect ~30% of
  instances to abstain; that fraction was previously dropped silently.

## Commits

| Commit | Contents |
|---|---|
| `fc6a53d` | P0 repairs, `sage_reference.py` |
| `37a1eff` | Item 2 core: `symmetry.py`, `canonicalize_pair` |
| `35f7c51` | Item 5: cluster bootstrap |
| `988aaab` | Item 6 plumbing + instrumented export |
| `7315e80` | `run_corrected_pipeline.py` |
| `8503c4f` | Export write path made testable |
