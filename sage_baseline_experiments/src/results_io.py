"""
Load the saved experiment results.

The notebooks summarise and plot; they do not retrain. Every number they show
comes from the CSVs the `task*.py` scripts wrote, so a notebook figure and the
corresponding script output can never disagree.

Run the pipeline first if anything here raises FileNotFoundError:

    python run_all.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

CLASSES = ["box", "can", "mug", "bottle", "bowl"]

CLASS_ACCURACY_COLUMNS = [
    f"{label}_accuracy" for label in CLASSES
]

# SAGE's locked numbers from the co-author guide, plus the class supports.
# Kept here so notebooks can build the comparison row without re-deriving it.
SAGE = {
    "model": "SAGE",
    "split": "video_level",
    "overall_accuracy": 0.784,
    "balanced_accuracy": np.nan,
    "box_accuracy": 0.947,
    "can_accuracy": 0.949,
    "mug_accuracy": 0.326,
    "bottle_accuracy": 0.315,
    "bowl_accuracy": 0.182,
}

CLASS_SUPPORT = {
    "box": 475,
    "can": 355,
    "mug": 89,
    "bottle": 146,
    "bowl": 44,
}

# The three cross-validation regimes, in reporting order.
# Result files are suffixed with the regime, except the
# random one which is unsuffixed for backwards compatibility.
SPLIT_SUFFIXES = {
    "random": "",
    "grouped": "_grouped",
    "groupkfold": "_groupkfold",
}

SPLIT_ORDER = ["video_level", "random", "grouped", "groupkfold"]

# What each regime actually is, for axis and legend labels.
SPLIT_LABELS = {
    "video_level": "video-level (SAGE's own)",
    "random": "StratifiedKFold (leaky)",
    "grouped": "StratifiedGroupKFold",
    "groupkfold": "GroupKFold",
}

# Short forms, for tight legends and tick labels.
SPLIT_SHORT = {
    "video_level": "video-level",
    "random": "random",
    "grouped": "stratified-group",
    "groupkfold": "group",
}


def split_of(model_name):
    """
    Recover which cross-validation regime produced a result.
    """

    if model_name.startswith("SAGE"):
        return "video_level"

    if model_name.endswith("_groupkfold"):
        return "groupkfold"

    if model_name.endswith("_grouped"):
        return "grouped"

    return "random"


def base_model_of(model_name):
    """
    Strip the regime suffix, leaving the model identity.
    """

    for suffix in ("_groupkfold", "_grouped"):
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]

    return model_name


def _require(path):
    """
    Read a results CSV, with a message that says how to produce it.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing results file:\n{path}\n\n"
            f"Run the pipeline first:\n"
            f"    python run_all.py\n"
        )

    return pd.read_csv(path)


# ---------------------------------------------------------
# Summary tables
# ---------------------------------------------------------

def load_summary():
    """
    The full model comparison: SAGE, the feature baselines and PointNet,
    under all three cross-validation regimes.

    Returns
    -------
    pd.DataFrame
        One row per (model, split), with a `base_model` column that strips
        the regime suffix so the regimes of one model can be paired.
    """

    summary = _require(
        RESULTS_DIR / "task2" / "task1_task2_summary.csv"
    )

    summary["base_model"] = summary["model"].map(base_model_of)

    return summary


def load_task1_summary():
    """
    Task 1 only: SAGE plus the three same-feature baselines.
    """

    summary = _require(
        RESULTS_DIR / "task1" / "task1_summary.csv"
    )

    summary["base_model"] = summary["model"].map(base_model_of)

    return summary


# ---------------------------------------------------------
# Instance-level predictions
# ---------------------------------------------------------

def load_predictions(model, task="task1"):
    """
    One model's out-of-fold predictions.

    Parameters
    ----------
    model : str
        Result name, e.g. "knn" or "pointnet_grouped".

    task : str
        Which results directory to read from.
    """

    return _require(
        RESULTS_DIR / task / f"{model}_predictions.csv"
    )


def load_confusion(model, task="task1", normalize=False):
    """
    One model's confusion matrix, as a labelled DataFrame.

    normalize=True divides each row by its true-class support, giving the
    per-class recall breakdown in percent.
    """

    matrix = _require(
        RESULTS_DIR / task / f"{model}_confusion_matrix.csv"
    )

    # The row labels were written as an unnamed index column.
    matrix = matrix.set_index(matrix.columns[0])

    matrix.index = [
        name.replace("true_", "") for name in matrix.index
    ]

    matrix.columns = [
        name.replace("pred_", "") for name in matrix.columns
    ]

    matrix.index.name = "true"
    matrix.columns.name = "predicted"

    if normalize:
        row_totals = matrix.sum(axis=1)
        matrix = matrix.div(row_totals, axis=0) * 100.0

    return matrix


# ---------------------------------------------------------
# Per-task results
# ---------------------------------------------------------

def load_sample_efficiency():
    """
    Task 3: the aggregated n-shot curve.
    """

    return _require(
        RESULTS_DIR / "task3" / "sample_efficiency_summary.csv"
    )


def load_robustness(kind, split_mode):
    """
    Task 4 degradation results.

    Parameters
    ----------
    kind : {"point_count", "noise"}
    split_mode : {"random", "grouped", "groupkfold"}
    """

    return _require(
        RESULTS_DIR
        / "task4"
        / f"{kind}_robustness_{split_mode}.csv"
    )


def load_bootstrap_ci():
    """
    Task 5: bootstrap confidence intervals for every model.
    """

    return _require(
        RESULTS_DIR / "task5" / "bootstrap_ci_summary.csv"
    )


def load_can_vs_bottle():
    """
    Task 5: the can-vs-bottle fitting-strategy comparison.
    """

    return _require(
        RESULTS_DIR / "task5" / "can_vs_bottle_tests.csv"
    )


def load_mcnemar():
    """
    Task 5: paired classifier comparisons.
    """

    return _require(
        RESULTS_DIR / "task5" / "mcnemar_model_comparisons.csv"
    )


# ---------------------------------------------------------
# Presentation
# ---------------------------------------------------------

def as_percent(frame, columns=None, decimals=1):
    """
    Return a copy with accuracy columns scaled to percentages.

    Notebooks show the table beside every chart, so this keeps the two
    reading in the same units.
    """

    frame = frame.copy()

    if columns is None:
        columns = [
            column
            for column in frame.columns
            if column.endswith("accuracy")
            or column in {"accuracy", "ci_lower", "ci_upper"}
        ]

    for column in columns:
        if column in frame.columns:
            frame[column] = (frame[column] * 100).round(decimals)

    return frame


PRETTY_NAMES = {
    "knn": "k-NN",
    "svm_linear": "SVM (linear)",
    "svm_rbf": "SVM (RBF)",
    "pointnet": "PointNet",
    "SAGE": "SAGE",
}


def pretty(model_name):
    """
    Human-readable model name, regime suffix stripped.
    """

    base = base_model_of(model_name)

    return PRETTY_NAMES.get(base, base)


def pretty_with_split(model_name):
    """
    Model name plus its regime, e.g. "SVM (RBF) (group)".
    """

    if model_name.startswith("SAGE"):
        return "SAGE"

    return (
        f"{pretty(model_name)} "
        f"({SPLIT_SHORT[split_of(model_name)]})"
    )
