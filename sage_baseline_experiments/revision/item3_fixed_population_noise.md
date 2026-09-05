# Item 3 — fixed-population noise experiment

> *Evaluate all noise levels on the same original samples and report
> fitting failures/abstentions separately from recognition accuracy.*

**Status: ~40%. The accounting exists; the fixed population does not.**
Commit `fc6a53d`.

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

### Not solved: the population changes per noise level

`evaluate_sage_robustness.run_sweep()` re-derives the evaluated set at
**every** level (the loop at line ~212). A harsher noise level makes more
fits implausible, so those instances silently leave the denominator —
and accuracy is then computed over a **different, easier** population.

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

### The `--fixed-population` flag

**Where:** `sbe/src/evaluate_sage_robustness.py`, `run_sweep()`

**What to build:**

1. Run the sweep once at the baseline condition (σ = 0, full points) and
   record the set of `instance_id`s that fitted successfully.
2. Score **that same set** at every subsequent level.
3. Instances that fail at a harsher level count as **errors**, not as
   absences — they are a failure of the method, not a smaller test set.
4. Report two series per sweep: accuracy on the fixed population, and
   abstention rate as its own curve.

The accounting from `evaluate_clouds` is what makes step 3 possible; it
was built for exactly this.

**Estimated work:** small — the machinery exists, this is bookkeeping
plus a flag.

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

1. Add `--fixed-population` to `run_sweep`, defaulting to **on**. The
   old behaviour should require opting in, not the other way round.
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
