# Item 3 — fixed-population noise experiment

> *Evaluate all noise levels on the same original samples and report
> fitting failures/abstentions separately from recognition accuracy.*

**Status: code complete. Needs a dataset run.**
Commits `fc6a53d`, and the `--fixed-population` default.

---

## What is wrong

Two problems, one solved and one not.

### Solved: the denominator was invisible

The pipeline drops instances silently in five places:

| Where | Line | Dropped |
|---|---|---|
| `ycb_dataset_loader.py` | 102 | unmapped YCB class |
| `ycb_dataset_loader.py` | 74, 78, 83 | fewer than 80 valid depth pixels |
| `evaluate_on_ycbv.py` | 90, 99 → 184-186 | `__IMPLAUSIBLE__` fits |
| `evaluate_on_ycbv.py` | 118-119 | whole-frame exception, **counted nowhere** |
| `export_baseline_data.py` | 69-76, 87-88, 92-93 | same, at export time |

Every one is a `continue`. The reported `n_total` is therefore already a
filtered population before any metric is computed.

**Scale of it:** `src/sage_sample_efficiency.json` records **535
implausible against 1109 evaluated** — roughly a third of the population
removed without appearing in any number.

### Also solved: the population changed per noise level

`evaluate_sage_robustness.run_sweep()` used to re-derive the evaluated
set at **every** level. A harsher noise level makes more fits
implausible, so those instances silently left the denominator — and
accuracy was then computed over a **different, easier** population.

That makes the curve incomparable across levels in the worst possible
way: it flatters robustness exactly where the method struggles most. A
method that abstains on everything hard would look perfectly robust.

The point-count sweep has the same defect, plus a hard `len(cloud) < 50`
skip after degradation.

---

## What changed

**Where:** `sbe/src/evaluate_sage_robustness.py`

New `evaluate_clouds()` and `_process_one_cloud()` never silently drop
an instance. Every input returns a row carrying a `reason`, and the
metrics report:

```
n_evaluated, n_abstained, n_input, abstain_reasons
```

with `n_evaluated + n_abstained == n_input` guaranteed.

**Verified:** 24 real clouds → 21 evaluated, 3 `implausible_fit`,
reconciles. And on the full set: 1061 evaluated, 48 abstained, 1109 in.

`export_instrumented.py` applies the same discipline at export time —
abstentions become **rows** in the companion CSV, not skips.

---

## What is left

### ~~The `--fixed-population` flag~~ — done

`run_sweep(..., fixed_population=True)` is now the **default**:

1. `_process_one_frame_degraded` now returns `instance_id` per object
   (format `"<video>/<frame>#<n>"`, matching `evaluate_on_ycbv.py`) and
   reports unscored instances rather than skipping them.
2. The first value in `values` — the undegraded condition — fixes the
   population. Every later level scores that same set.
3. An instance that stops fitting at a harsher level counts as an
   **error**, not an absence.
4. Each row now carries `n_lost_at_level`, `n_abstained` and
   `fixed_population` alongside accuracy.

`--per-level-population` restores the old behaviour, for the comparison.
`task4_robustness.py`'s 8-positional call is unaffected.

**Still to do:** run it. See [`RUNBOOK.md`](RUNBOOK.md) step 4.

### Re-run task 4's head-to-head

`results/task4/sage_vs_pointnet_*.csv` currently compares SAGE and
PointNet on differently-filtered populations. PointNet never abstains
(it pads clouds), while SAGE drops implausible fits — so the comparison
is between a method scored on everything and a method scored on the
subset it found easy.

**What to expect:** SAGE's robustness curve will look **worse** once the
population is fixed, possibly substantially at high noise. That is the
honest number. The current `n` column already shows the population
moving (1111 at 1024 points) — evidence the problem is real, not
hypothetical.

---

## What to do

1. ~~Add the flag~~ — done; fixed population is the default and the old
   behaviour requires opting in via `--per-level-population`.
2. Re-run both sweeps on the Linux box:
   ```bash
   python3 src/evaluate_sage_robustness.py \
       --dataset_root ~/pnsg_superquadrics/ycb_dataset \
       --model ../trained_ycbv_ml_v2.json --workers 30
   ```
3. Re-run task 4 so the head-to-head uses the fixed population on both
   sides.
4. In the paper, report abstention rate alongside accuracy for every
   condition. A method that abstains on 30% of instances and is accurate
   on the rest is a different proposition from one that answers
   everything — and the reader cannot tell which they are looking at
   from an accuracy number alone.
