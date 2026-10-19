# Item 7 — corrective multi-view policy

> *Implement/evaluate the closed-loop policy: single view → detect
> unreliable estimate → acquire additional view → re-estimate/recognize,
> and compare against single-view recognition.*

**Status: not started. Deliberately gated on
[item 6](item6_failure_diagnosis.md)'s real numbers.**

---

## Why it is gated

The policy needs a detector that actually detects. If no geometric
indicator separates reliable from unreliable fits, there is nothing to
trigger on and the "corrective" policy degenerates into either
always-single-view or always-multi-view — neither of which is a
contribution.

The fixture signal is encouraging but not sufficient:

```
ax_nrmse  AUROC 0.683
error at 100% coverage : 25.00%
error at  50% coverage : 12.07%
```

**Decision rule: if no indicator clears ~0.65 AUROC on real data, do not
build this.** Say so in the paper instead — "we tested whether geometric
fit quality predicts recognition failure, and it does not" is a real
finding, and a defensible reason not to claim a closed loop.

---

## The experiment, when it is warranted

Three policies on the same instances:

| Policy | Views used | |
|---|---|---|
| always single-view | 1.0 per object | the baseline to beat |
| **corrective** | 1 + (fraction flagged) | the proposal |
| always multi-view | k per object | the ceiling |

**Headline plot: accuracy vs mean views acquired per object.** A cost
curve, not a bar chart. The corrective policy has to beat single-view at
a genuinely *lower* budget than always-multi-view, or it is not buying
anything.

Sweep the detector threshold to trace the curve — each threshold is a
point.

---

## The honesty constraint that shapes the design

Fusing a second view currently requires **ground-truth pose**
(`ycb_pose_aggregation.py:63`). A closed-loop policy that silently
assumes known pose is not a robotics result.

Two acceptable ways forward:

1. **Estimate relative pose from the clouds** (ICP between the two
   views), and report accuracy under estimated rather than ground-truth
   pose. Harder, and the honest version.
2. **State plainly that this is an upper bound assuming known pose**,
   and report it as such.

Option 1 is the stronger paper. Option 2 is acceptable if time is short,
but the caveat must be in the figure caption, not buried.

---

## What to do, in order

1. Run the corrected pipeline; read `results/corrected/failure_diagnosis.csv`.
2. **Gate:** does any indicator clear ~0.65 AUROC? If not, stop and
   write up the negative result.
3. If yes, add the score margin as an indicator first — it is free
   (`classify_graph` already returns the full ranked list) and is often
   the strongest error predictor. Re-check the gate with it included.
4. Build the policy against the best indicator. Reuse
   `aggregate_multiview_cloud` for the fusion step and
   `evaluate_multiview_ycbv.py` for the re-estimation path; neither
   needs to be written from scratch.
5. Decide pose handling (ICP vs stated upper bound) **before** running,
   not after seeing the numbers.

---

## What to expect

If the detector holds up, the plausible result is a policy that reaches
most of the multi-view accuracy at ~1.3–1.5 views per object. That is a
genuine contribution: it is the "when should a robot look again?"
question, which is exactly what an interpretable geometric
representation is *supposed* to be good for, and what a learned
embedding cannot answer as directly.

If the detector does not hold up, the negative result is still worth
reporting — it bounds what the representation's interpretability
actually buys, which is more useful than silence.
