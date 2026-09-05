"""
The single source of truth for SAGE's own reported numbers.

Why this module exists
----------------------
Four places held a SAGE reference number and two of them disagreed:

    task1_feature_baselines.SAGE_REFERENCE  0.894   (ExtraTrees scoring)
    results_io.SAGE                         0.784   (joint scoring)
    task3_sample_efficiency.SAGE_FULL_ACCURACY 0.784
    task5_statistics.SAGE_PER_CLASS         0.894 rates

The notebooks read `results_io`, so every notebook figure rendered 78.4%
while the task-1 comparison table used 89.4%. Both numbers are real --
they are two different *scoring modes* of the same system, not a
correction of one by the other -- but nothing recorded which was which.

This module keeps every run, each tagged with how it was produced, and
names one as the headline. Import from here; do not re-declare.

Preference order for any number reported downstream:
  1. Derived from instance-level predictions on disk (authoritative --
     it cannot drift from the artefact it is computed from).
  2. The recorded constant below, used only when that file is absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CLASSES = ["box", "can", "mug", "bottle", "bowl"]

# Per-class instance counts in val_sample (n = 1109).
#
# RECORDED FALLBACK ONLY. class_support() below derives these from the
# data whenever it can, because a hardcoded support silently goes wrong
# the moment the split changes -- and every downstream proportion test
# divides by it. Used only when neither the predictions nor the feature
# export is on disk.
CLASS_SUPPORT_RECORDED = {
    "box": 475,
    "can": 355,
    "mug": 89,
    "bottle": 146,
    "bowl": 44,
}

# CLASS_SUPPORT and its provenance are derived below, once SAGE_RUNS
# exists -- see class_support().


# ---------------------------------------------------------
# Recorded runs
# ---------------------------------------------------------

SAGE_RUNS = {
    # The original locked result quoted in co-author_guide.md.
    "joint_v1": {
        "model": "SAGE (joint scoring)",
        "scoring": "joint",
        "split": "val_sample",
        "eval_protocol": "video_level",
        "source_model": "trained_ycbv_final.json",
        "predictions": None,          # no instance-level file kept
        "overall_accuracy": 0.784,
        "balanced_accuracy": np.nan,  # never computed for this run
        "box_accuracy": 0.947,
        "can_accuracy": 0.949,
        "mug_accuracy": 0.326,
        "bottle_accuracy": 0.315,
        "bowl_accuracy": 0.182,
        "note": (
            "Welford prototype scoring. Superseded as the headline by "
            "ml_v2, but kept because the guide, the README and older "
            "figures all quote it."
        ),
    },
    # Current headline: ExtraTrees classifier baked in via
    # bake_ml_classifier.py from the per-frame train export
    # (features_train.npz, 35178 examples), evaluated on val_sample.
    "ml_v2": {
        "model": "SAGE (ML scoring)",
        "scoring": "ml",
        "split": "val_sample",
        "eval_protocol": "video_level",
        "source_model": "trained_ycbv_ml_v2.json",
        "predictions": "sage/sage_predictions.csv",
        "overall_accuracy": 0.8945,
        "balanced_accuracy": 0.7131,
        "box_accuracy": 1.000,
        "can_accuracy": 1.000,
        "mug_accuracy": 0.517,
        "bottle_accuracy": 0.685,
        "bowl_accuracy": 0.364,
        "note": (
            "Living number. box and can both sit at 100%, which is "
            "consistent with the label-conditioned fitting leak (the "
            "fitting path is chosen from the true label, so a1==a2 "
            "identifies 'can' outright). Treat per-class numbers as "
            "provisional until that is fixed."
        ),
    },
}

HEADLINE = "ml_v2"


# ---------------------------------------------------------
# Class support, derived rather than trusted
# ---------------------------------------------------------

def class_support(run=HEADLINE):
    """Per-class instance counts, derived from data where possible.

    Every downstream proportion test divides by these, so a stale
    hardcoded support silently corrupts p-values the moment the split
    changes. Preference order matches the rest of this module:

      1. the run's instance-level predictions
      2. the exported feature file
      3. CLASS_SUPPORT_RECORDED

    Returns (counts, provenance).
    """

    relative = SAGE_RUNS.get(run, {}).get("predictions")

    if relative:
        path = RESULTS_DIR / relative
        if path.exists():
            counts = pd.read_csv(path)["true_label"].value_counts().to_dict()
            return (
                {label: int(counts.get(label, 0)) for label in CLASSES},
                f"derived from results/{relative}",
            )

    try:
        from data_utils import load_feature_data

        _X, labels, _names = load_feature_data()
        labels = np.asarray(labels)

        return (
            {label: int((labels == label).sum()) for label in CLASSES},
            "derived from baseline_data/features_val_sample.npz",
        )

    except Exception:
        return (
            dict(CLASS_SUPPORT_RECORDED),
            "recorded fallback -- no data on disk",
        )


# Module-level dict, data-derived at import, same shape as before.
CLASS_SUPPORT, CLASS_SUPPORT_PROVENANCE = class_support()


# ---------------------------------------------------------
# Derivation from instance-level predictions
# ---------------------------------------------------------

def predictions_path(run=HEADLINE):
    """Path to a run's instance-level predictions, or None."""

    relative = SAGE_RUNS[run].get("predictions")

    if relative is None:
        return None

    return RESULTS_DIR / relative


def load_predictions(run=HEADLINE):
    """
    Instance-level predictions for one run, or None if not on disk.

    Adds a `video` column parsed from `instance_id` ("0029/000591#0"),
    which is what the grouped/cluster bootstrap needs.
    """

    path = predictions_path(run)

    if path is None or not path.exists():
        return None

    frame = pd.read_csv(path)

    if "instance_id" in frame.columns:
        frame["video"] = (
            frame["instance_id"].astype(str).str.split("/").str[0]
        )

    return frame


def metrics_from_predictions(frame):
    """Overall / balanced / per-class accuracy from a predictions table."""

    correct = frame["true_label"] == frame["predicted_label"]

    per_class = {}

    for label in CLASSES:
        mask = frame["true_label"] == label
        per_class[f"{label}_accuracy"] = (
            float(correct[mask].mean()) if mask.any() else np.nan
        )

    finite = [
        value for value in per_class.values() if not np.isnan(value)
    ]

    return {
        "overall_accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(finite)) if finite else np.nan,
        "n_instances": int(len(frame)),
        **per_class,
    }


def get(run=HEADLINE, derive=True):
    """
    One SAGE run as a dict, derived from predictions where possible.

    The returned dict always carries `provenance` describing how the
    numbers were obtained, so a table can state it rather than implying
    a number is more authoritative than it is.
    """

    recorded = dict(SAGE_RUNS[run])

    if derive:

        frame = load_predictions(run)

        if frame is not None:
            recorded.update(metrics_from_predictions(frame))
            recorded["provenance"] = (
                f"derived from results/{recorded['predictions']} "
                f"({recorded['scoring']} scoring, {recorded['split']})"
            )
            return recorded

    recorded["provenance"] = (
        f"recorded constant ({recorded['scoring']} scoring, "
        f"{recorded['split']}); no instance-level file on disk"
    )

    return recorded


def summary_row(run=HEADLINE, derive=True):
    """A SAGE row shaped like the task1/task2 summary tables."""

    entry = get(run, derive=derive)

    row = {
        "model": entry["model"],
        "split": entry["eval_protocol"],
        "overall_accuracy": entry["overall_accuracy"],
        "balanced_accuracy": entry["balanced_accuracy"],
    }

    for label in CLASSES:
        row[f"{label}_accuracy"] = entry[f"{label}_accuracy"]

    return row


def verify(tolerance=0.005):
    """Check the recorded constants against what the data actually says.

    A hardcoded number that has quietly gone stale is worse than no
    number at all, because it looks authoritative. This compares every
    recorded value against the derived one and returns the
    discrepancies, so drift is loud rather than silent.

    Returns a list of dicts; empty means everything agrees.
    """

    problems = []

    for name, recorded in SAGE_RUNS.items():

        frame = load_predictions(name)

        if frame is None:
            problems.append({
                "run": name,
                "field": "*",
                "recorded": None,
                "derived": None,
                "issue": "no instance-level predictions on disk; "
                         "values are unverifiable constants",
            })
            continue

        derived = metrics_from_predictions(frame)

        for field, derived_value in derived.items():

            if field not in recorded:
                continue

            recorded_value = recorded[field]

            if recorded_value is None or (
                isinstance(recorded_value, float) and np.isnan(recorded_value)
            ):
                continue

            if abs(float(recorded_value) - float(derived_value)) > tolerance:
                problems.append({
                    "run": name,
                    "field": field,
                    "recorded": float(recorded_value),
                    "derived": float(derived_value),
                    "issue": "recorded constant disagrees with the data",
                })

    # The support every proportion test divides by.
    derived_support, provenance = class_support()

    if "recorded fallback" not in provenance:
        for label, count in CLASS_SUPPORT_RECORDED.items():
            if derived_support.get(label) != count:
                problems.append({
                    "run": "class_support",
                    "field": label,
                    "recorded": count,
                    "derived": derived_support.get(label),
                    "issue": f"support disagrees ({provenance})",
                })

    return problems


def write_csv(path=None):
    """Write every SAGE run to a CSV, with provenance per row.

    The point of the provenance column is that a reader can tell, per
    row, whether a number came out of the pipeline or out of a constant
    someone typed. Those are not the same kind of evidence and a table
    that hides the difference is misleading.
    """

    path = Path(path) if path else (RESULTS_DIR / "sage" / "sage_reference.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for name in SAGE_RUNS:

        entry = get(name)

        row = {
            "run": name,
            "is_headline": name == HEADLINE,
            "model": entry["model"],
            "scoring": entry["scoring"],
            "split": entry["split"],
            "eval_protocol": entry["eval_protocol"],
            "source_model": entry["source_model"],
            "overall_accuracy": entry["overall_accuracy"],
            "balanced_accuracy": entry["balanced_accuracy"],
            "n_instances": entry.get("n_instances"),
            "derived_from_data": "derived" in entry["provenance"],
            "provenance": entry["provenance"],
        }

        for label in CLASSES:
            row[f"{label}_accuracy"] = entry[f"{label}_accuracy"]
            row[f"{label}_support"] = CLASS_SUPPORT.get(label)

        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)

    return path, frame


def describe():
    """Human-readable listing of every recorded run."""

    lines = []

    for name in SAGE_RUNS:
        entry = get(name)
        marker = "  <- headline" if name == HEADLINE else ""
        lines.append(
            f"{name:10s} {entry['overall_accuracy'] * 100:6.2f}% overall  "
            f"{entry['balanced_accuracy'] * 100:6.2f}% balanced  "
            f"[{entry['provenance']}]{marker}"
        )

    lines.append("")
    lines.append(f"class support: {CLASS_SUPPORT}")
    lines.append(f"               {CLASS_SUPPORT_PROVENANCE}")

    return "\n".join(lines)


if __name__ == "__main__":

    print(describe())

    print("\n" + "=" * 70)
    print("DRIFT CHECK -- recorded constants vs the data")
    print("=" * 70)

    problems = verify()

    if not problems:
        print("\nNo discrepancies. Every recorded value matches the data.")
    else:
        for problem in problems:
            print(
                f"\n  {problem['run']}.{problem['field']}: "
                f"{problem['issue']}"
            )
            if problem["recorded"] is not None:
                print(f"    recorded {problem['recorded']}  "
                      f"derived {problem['derived']}")

    path, frame = write_csv()
    print(f"\nWrote {len(frame)} rows -> {path}")
