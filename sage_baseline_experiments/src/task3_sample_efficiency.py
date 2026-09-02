import sys
from pathlib import Path

# Allow these scripts to be launched from any working
# directory (repo root or src/), not only from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from data_utils import (
    compute_object_groups,
    load_feature_data,
    load_pointcloud_data,
)
from evaluation import (
    LABEL_ORDER,
    calculate_overall_metrics,
    calculate_per_class_accuracy,
)


# =========================================================
# WHAT THIS SCRIPT DOES, AND WHAT IT DELIBERATELY DOES NOT
# =========================================================
#
# The co-author guide describes Task 3 as: retrain the SAGE
# registry with 1, 2, 5, 10 and 20 confirmed examples per
# category, evaluate each version, and plot accuracy vs n.
# The point being made is that a prototype-based vocabulary
# can pick up a usable category from a handful of real
# examples, which a learned model structurally cannot.
#
# That experiment needs `train_registry_multiview.py` plus a
# local copy of YCB-Video. Neither is available here: this
# checkout has the training script but `image_sets/train.txt`
# and `image_sets/val.txt` are empty stubs and there is no
# dataset root. So the SAGE half of the curve cannot be run
# from the exported .npz files, and is not attempted.
#
# What IS runnable, and what this script produces, is the
# other half of exactly that comparison: the sample-efficiency
# curve for the learned baselines, over the same n values, on
# the same held-out instances. That is the reference curve the
# SAGE registry curve gets plotted against. Hand the SAGE
# numbers to `plot_combined_curve` (or drop them into
# SAGE_REFERENCE_CURVE below) and the two appear on one axis.
#
# Design decision that matters for interpretation
# -----------------------------------------------
# "n examples per category" is counted in *distinct physical
# objects*, not rows. The export emits one row per object per
# video frame, so drawing n rows at random would usually draw
# n views of the same object and would not be "n confirmed
# examples" in the sense the guide means. Training objects and
# evaluation objects are also kept strictly disjoint, so the
# curve measures generalisation to unseen objects rather than
# recall of memorised frames.


# =========================================================
# 1. CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# Examples (distinct objects) per category, as in the guide.
SAMPLE_SIZES = [1, 2, 5, 10, 20]

# Each (n, model) point is averaged over this many draws,
# because a single 1-shot draw is almost pure luck.
N_REPEATS = 20

# Fraction of object groups reserved for evaluation.
TEST_GROUP_FRACTION = 0.30

GROUP_DISTANCE_THRESHOLD = 0.003

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "task3"
FIGURES_DIR = RESULTS_DIR / "figures"


# SAGE's own sample-efficiency numbers, once the authors run
# the registry side of this experiment. Fill in as
# {n: accuracy} and the combined plot picks them up.
SAGE_REFERENCE_CURVE = {}

# SAGE's locked full-training result, for a horizontal
# reference line.
SAGE_FULL_ACCURACY = 0.784


# =========================================================
# 2. MODELS
# =========================================================

def create_models(n_per_class):
    """
    Build the baseline classifiers for one budget.

    k-NN's k has to shrink with the training set: with one
    example per category there are only five training rows
    in total, so the default k=5 would average over every
    class at once and make the 1-shot point meaningless.

    Parameters
    ----------
    n_per_class : int
        Training objects per category.

    Returns
    -------
    dict
        Mapping from model name to sklearn Pipeline.
    """

    n_neighbors = max(
        1,
        min(5, n_per_class),
    )

    return {
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights="uniform",
                    ),
                ),
            ]
        ),
        "svm_linear": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=1.0,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                    ),
                ),
            ]
        ),
    }


# =========================================================
# 3. OBJECT-LEVEL TRAIN / TEST POOLS
# =========================================================

def split_group_pools(
    y,
    groups,
    rng,
):
    """
    Split object groups into a training pool and a
    disjoint evaluation pool.

    Splitting on groups rather than rows is what keeps the
    curve honest: no physical object contributes to both
    training and evaluation.

    Returns
    -------
    train_groups : np.ndarray
        Group ids available for sampling training examples.

    test_indices : np.ndarray
        Row indices making up the fixed evaluation set.
    """

    unique_groups = np.unique(groups)

    shuffled = rng.permutation(unique_groups)

    n_test = int(
        round(
            len(shuffled)
            * TEST_GROUP_FRACTION
        )
    )

    test_groups = shuffled[:n_test]
    train_groups = shuffled[n_test:]

    test_mask = np.isin(groups, test_groups)

    test_indices = np.flatnonzero(test_mask)

    # An evaluation set that is missing a class makes the
    # per-class columns meaningless, so require all five.
    missing = (
        set(LABEL_ORDER)
        - set(np.asarray(y)[test_indices])
    )

    if missing:
        raise ValueError(
            f"Evaluation pool is missing classes: "
            f"{sorted(missing)}"
        )

    return train_groups, test_indices


def sample_training_indices(
    y,
    groups,
    train_groups,
    n_per_class,
    rng,
):
    """
    Draw n distinct objects per category, then take one
    row from each drawn object.

    Returns None when some category does not have
    n distinct objects available, so the caller can skip
    that budget rather than silently report a curve point
    built from fewer examples than it claims.
    """

    y = np.asarray(y)

    train_mask = np.isin(groups, train_groups)

    selected = []

    for label in LABEL_ORDER:

        label_mask = train_mask & (y == label)

        available_groups = np.unique(
            groups[label_mask]
        )

        if len(available_groups) < n_per_class:
            return None

        chosen_groups = rng.choice(
            available_groups,
            size=n_per_class,
            replace=False,
        )

        for group_id in chosen_groups:

            candidates = np.flatnonzero(
                label_mask
                & (groups == group_id)
            )

            selected.append(
                rng.choice(candidates)
            )

    return np.asarray(selected, dtype=int)


# =========================================================
# 4. ONE (n, repeat) TRIAL
# =========================================================

def run_trial(
    X,
    y,
    groups,
    n_per_class,
    seed,
):
    """
    Train every baseline on one n-shot draw and evaluate
    on the disjoint object pool.

    Returns
    -------
    list of dict
        One row per model, or an empty list when this
        budget cannot be satisfied.
    """

    rng = np.random.default_rng(seed)

    train_groups, test_indices = split_group_pools(
        y,
        groups,
        rng,
    )

    train_indices = sample_training_indices(
        y,
        groups,
        train_groups,
        n_per_class,
        rng,
    )

    if train_indices is None:
        return []

    X_train = X[train_indices]
    y_train = np.asarray(y)[train_indices]

    X_test = X[test_indices]
    y_test = np.asarray(y)[test_indices]

    rows = []

    for model_name, model in create_models(
        n_per_class
    ).items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        overall = calculate_overall_metrics(
            y_test,
            y_pred,
        )

        per_class = calculate_per_class_accuracy(
            y_test,
            y_pred,
        )

        row = {
            "model": model_name,
            "n_per_class": n_per_class,
            "seed": seed,
            "n_train_rows": len(train_indices),
            "n_test_rows": len(test_indices),
            "overall_accuracy": overall["accuracy"],
            "balanced_accuracy": overall[
                "balanced_accuracy"
            ],
        }

        for label in LABEL_ORDER:
            row[f"{label}_accuracy"] = per_class[label]

        rows.append(row)

    return rows


# =========================================================
# 5. FULL CURVE
# =========================================================

def run_sample_efficiency(
    X,
    y,
    groups,
):
    """
    Sweep every budget and every repeat.

    Returns
    -------
    pd.DataFrame
        One row per (model, n, seed) trial.
    """

    all_rows = []

    for n_per_class in SAMPLE_SIZES:

        print("\n" + "=" * 70)
        print(
            f"BUDGET: {n_per_class} "
            f"object(s) per category"
        )
        print("=" * 70)

        budget_rows = []

        for repeat in range(N_REPEATS):

            seed = (
                RANDOM_STATE
                + 1000 * n_per_class
                + repeat
            )

            budget_rows.extend(
                run_trial(
                    X,
                    y,
                    groups,
                    n_per_class,
                    seed,
                )
            )

        if not budget_rows:
            print(
                "Skipped: not enough distinct objects "
                "in every category for this budget."
            )
            continue

        budget_df = pd.DataFrame(budget_rows)

        for model_name, model_df in budget_df.groupby(
            "model"
        ):
            print(
                f"{model_name:12s}: "
                f"{model_df['overall_accuracy'].mean() * 100:6.2f}% "
                f"+/- {model_df['overall_accuracy'].std() * 100:5.2f} "
                f"(n={len(model_df)} draws)"
            )

        all_rows.extend(budget_rows)

    return pd.DataFrame(all_rows)


# =========================================================
# 6. AGGREGATION
# =========================================================

def summarise(trials_df):
    """
    Collapse the per-trial table into mean / std /
    percentile bands per (model, n).
    """

    metric_columns = [
        "overall_accuracy",
        "balanced_accuracy",
    ] + [
        f"{label}_accuracy"
        for label in LABEL_ORDER
    ]

    grouped = trials_df.groupby(
        ["model", "n_per_class"]
    )

    summary = grouped[metric_columns].agg(
        ["mean", "std"]
    )

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    # Percentile band over draws. This is spread across
    # random n-shot draws, not a bootstrap confidence
    # interval on a fixed test set (that is Task 5).
    bands = grouped["overall_accuracy"].agg(
        p05=lambda values: np.percentile(values, 5),
        p95=lambda values: np.percentile(values, 95),
        n_trials="size",
    )

    return summary.join(bands).reset_index()


# =========================================================
# 7. PLOT
# =========================================================

def plot_combined_curve(
    summary_df,
    sage_curve=None,
    output_path=None,
):
    """
    Plot accuracy vs number of examples per category.

    sage_curve, when supplied as {n: accuracy}, is drawn
    alongside the baselines so the two sample-efficiency
    profiles can be compared directly.
    """

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model_name, model_df in summary_df.groupby(
        "model"
    ):

        model_df = model_df.sort_values(
            "n_per_class"
        )

        ax.plot(
            model_df["n_per_class"],
            model_df["overall_accuracy_mean"] * 100,
            marker="o",
            label=model_name,
        )

        ax.fill_between(
            model_df["n_per_class"],
            model_df["p05"] * 100,
            model_df["p95"] * 100,
            alpha=0.15,
        )

    if sage_curve:

        sizes = sorted(sage_curve)

        ax.plot(
            sizes,
            [
                sage_curve[size] * 100
                for size in sizes
            ],
            marker="s",
            linewidth=2.5,
            color="black",
            label="SAGE registry",
        )

    ax.axhline(
        SAGE_FULL_ACCURACY * 100,
        linestyle="--",
        linewidth=1.2,
        color="grey",
        label=(
            f"SAGE, full training "
            f"({SAGE_FULL_ACCURACY * 100:.1f}%)"
        ),
    )

    # Tick only the budgets that actually produced a point.
    # A budget is skipped when some category does not have
    # that many distinct objects, and a tick with nothing
    # above it reads as a missing result rather than an
    # unreachable one.
    plotted_sizes = sorted(
        set(summary_df["n_per_class"])
        | set(sage_curve or {})
    )

    ax.set_xscale("log")
    ax.set_xticks(plotted_sizes)
    ax.set_xticklabels(
        [str(size) for size in plotted_sizes]
    )

    ax.set_xlabel(
        "Distinct objects per category used for training"
    )

    ax.set_ylabel("Accuracy on held-out objects (%)")

    ax.set_title(
        "Task 3 - Sample efficiency\n"
        "(shaded band = 5th-95th percentile over "
        f"{N_REPEATS} random draws)"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)

    return fig


# =========================================================
# 8. MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)
    print("TASK 3 - SAMPLE-EFFICIENCY CURVE")
    print("=" * 70)

    print(
        "\nNote: this runs the learned-baseline half of "
        "the comparison only."
    )

    print(
        "The SAGE registry half needs "
        "train_registry_multiview.py plus a local "
        "YCB-Video copy, which is not available in this "
        "checkout (image_sets/*.txt are empty stubs)."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    X, y, _ = load_feature_data()

    clouds, cloud_labels = load_pointcloud_data()

    if not np.array_equal(
        np.asarray(cloud_labels),
        np.asarray(y),
    ):
        raise ValueError(
            "Feature and point-cloud exports are not in "
            "the same instance order."
        )

    print(f"\nInstances: {len(y)}")

    # -----------------------------------------------------
    # Object identity
    # -----------------------------------------------------

    print(
        "\nInferring object-identity groups "
        "from point-cloud extents..."
    )

    groups = compute_object_groups(
        clouds,
        distance_threshold=(
            GROUP_DISTANCE_THRESHOLD
        ),
    )

    print(
        f"Distinct object groups: "
        f"{len(np.unique(groups))}"
    )

    print("\nDistinct objects available per category:")

    y_array = np.asarray(y)

    for label in LABEL_ORDER:

        n_groups = len(
            np.unique(groups[y_array == label])
        )

        print(
            f"{label:10s}: {n_groups:3d} objects "
            f"({int((y_array == label).sum()):4d} rows)"
        )

    # -----------------------------------------------------
    # Sweep
    # -----------------------------------------------------

    trials_df = run_sample_efficiency(X, y, groups)

    if trials_df.empty:
        print(
            "\nNo budget could be satisfied. "
            "Nothing written."
        )
        return

    trials_path = (
        RESULTS_DIR
        / "sample_efficiency_trials.csv"
    )

    trials_df.to_csv(trials_path, index=False)

    # -----------------------------------------------------
    # Summarise
    # -----------------------------------------------------

    summary_df = summarise(trials_df)

    summary_path = (
        RESULTS_DIR
        / "sample_efficiency_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("SAMPLE-EFFICIENCY SUMMARY")
    print("=" * 70)

    display_df = summary_df[
        [
            "model",
            "n_per_class",
            "overall_accuracy_mean",
            "overall_accuracy_std",
            "balanced_accuracy_mean",
            "n_trials",
        ]
    ].copy()

    for column in (
        "overall_accuracy_mean",
        "overall_accuracy_std",
        "balanced_accuracy_mean",
    ):
        display_df[column] = display_df[column] * 100

    print(
        display_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    figure_path = (
        FIGURES_DIR
        / "sample_efficiency_curve.png"
    )

    plot_combined_curve(
        summary_df,
        sage_curve=SAGE_REFERENCE_CURVE,
        output_path=figure_path,
    )

    print("\nSaved files:")
    print(f"trials  : {trials_path}")
    print(f"summary : {summary_path}")
    print(f"figure  : {figure_path}")

    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE (baseline half)")
    print("=" * 70)


if __name__ == "__main__":
    main()
