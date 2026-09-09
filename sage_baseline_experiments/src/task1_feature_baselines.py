import sys
from pathlib import Path

# Allow these scripts to be launched from any working
# directory (repo root or src/), not only from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import sage_reference
from data_utils import (
    compute_object_groups,
    load_feature_data,
    load_pointcloud_data,
)
from evaluation import (
    LABEL_ORDER,
    print_evaluation_summary,
    save_evaluation_results,
)


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5

# Size tolerance, in meters, used to group instances that
# are almost certainly the same physical object seen in a
# different video frame. See data_utils.compute_object_groups.
GROUP_DISTANCE_THRESHOLD = 0.003

# Three cross-validation regimes are run for every model.
#
# "random"     -- plain StratifiedKFold. The naive split, and
#                 it leaks: the export contains one row per
#                 object *per frame*, so near-identical views
#                 of one physical object land in both the
#                 train and the test fold.
#
# "grouped"    -- StratifiedGroupKFold over inferred object
#                 identity. Every view of one object stays on
#                 one side of the split, while fold class
#                 balance is still enforced.
#
# "groupkfold" -- plain GroupKFold. Also keeps objects whole,
#                 but does NOT stratify: folds are balanced by
#                 size only, and whatever class mix falls out
#                 is what you get.
#
# The third regime matters because stratifying uses the label
# distribution to build the folds, which a real video-level
# split cannot do -- you hold out whole videos and accept the
# class mix that results. GroupKFold is therefore the closer
# analogue of how SAGE's own 78.4% was measured, and the
# honest headline number. StratifiedGroupKFold is kept as the
# lower-variance estimate.
SPLIT_MODES = [
    "random",
    "grouped",
    "groupkfold",
]

# Regimes that need object identity passed to the splitter.
GROUP_AWARE_MODES = {
    "grouped",
    "groupkfold",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task1"
)


# ---------------------------------------------------------
# SAGE reference results
# ---------------------------------------------------------

# These are SAGE's own results on the SAME val_sample split used for
# every other model in this file, kept here as a fixed reference point
# (not re-derived by this script itself -- update the dict below when
# a new SAGE number is available).
#
# We do NOT have instance-level SAGE predictions here,
# so they are not part of the cross-validation procedure.

# SAGE's own numbers come from src/sage_reference.py, which is the
# single source of truth and derives them from the instance-level
# predictions on disk where those exist. Do not re-declare them here --
# this file and results_io.py previously disagreed (0.894 vs 0.784),
# which is why every notebook figure rendered the wrong one.
SAGE_REFERENCE = sage_reference.summary_row()


# ---------------------------------------------------------
# Classifier definitions
# ---------------------------------------------------------

def create_models():
    """
    Create the feature-based baseline classifiers.

    StandardScaler is included inside each pipeline so
    scaling is fitted only on the training portion of
    each cross-validation fold.

    Returns
    -------
    dict
        Mapping from model name to sklearn Pipeline.
    """

    models = {
        # -------------------------------------------------
        # k-Nearest Neighbors
        # -------------------------------------------------
        "knn": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="uniform",
                    ),
                ),
            ]
        ),

        # -------------------------------------------------
        # Linear SVM
        # -------------------------------------------------
        "svm_linear": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=1.0,
                    ),
                ),
            ]
        ),

        # -------------------------------------------------
        # RBF SVM
        # -------------------------------------------------
        "svm_rbf": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
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

    return models


# ---------------------------------------------------------
# Cross-validation setup
# ---------------------------------------------------------

def create_cross_validation(
    split_mode="random",
):
    """
    Create the shared 5-fold splitter for one regime.

    Parameters
    ----------
    split_mode : {"random", "grouped", "groupkfold"}
        "random" uses StratifiedKFold, which splits rows
        independently and therefore leaks repeated views
        of the same physical object across folds.

        "grouped" uses StratifiedGroupKFold: objects stay
        whole and fold class balance is enforced.

        "groupkfold" uses GroupKFold: objects stay whole
        but folds are balanced by size only, so the class
        mix per fold is whatever falls out.

    Shuffle is enabled so folds do not depend on the
    original instance ordering. random_state makes the
    split reproducible.
    """

    if split_mode == "random":

        return StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    if split_mode == "grouped":

        return StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    if split_mode == "groupkfold":

        return GroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unknown split_mode: {split_mode}"
    )


# ---------------------------------------------------------
# Result naming
# ---------------------------------------------------------

def result_name(
    model_name,
    split_mode,
):
    """
    Build the file/report name for one model under one
    cross-validation regime.

    The random-split names are left unsuffixed so that
    files already consumed by Task 5 keep working.
    """

    if split_mode == "random":
        return model_name

    return f"{model_name}_{split_mode}"


# ---------------------------------------------------------
# Run one model
# ---------------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X,
    y,
    cv,
    groups=None,
):
    """
    Generate out-of-fold predictions for one classifier.

    Every instance is predicted only when it belongs to
    the test portion of a fold.

    Parameters
    ----------
    model_name : str
        Name used for reporting and saved files.

    model : sklearn estimator
        Classification pipeline.

    X : np.ndarray
        Feature matrix.

    y : np.ndarray
        Ground-truth labels.

    cv : cross-validation splitter
        Shared fold generator.

    groups : np.ndarray, optional
        Object-identity group per instance. Required by
        StratifiedGroupKFold, ignored by StratifiedKFold.

    Returns
    -------
    np.ndarray
        Out-of-fold predicted labels.
    """

    print("\n" + "=" * 70)
    print(f"RUNNING: {model_name}")
    print("=" * 70)

    y_pred = cross_val_predict(
        estimator=model,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )

    return y_pred


# ---------------------------------------------------------
# Create combined Task 1 summary
# ---------------------------------------------------------

def create_combined_summary():
    """
    Combine all model metric files with the original
    SAGE reference results.

    The output is saved as:

        results/task1/task1_summary.csv
    """

    rows = []

    # -----------------------------------------------------
    # Original SAGE result
    # -----------------------------------------------------

    # summary_row() already carries every column in the right shape,
    # including balanced_accuracy -- which used to be hardcoded to NaN
    # here even when the underlying run had a real value.
    sage_row = dict(SAGE_REFERENCE)

    rows.append(sage_row)

    # -----------------------------------------------------
    # Learned classifiers
    # -----------------------------------------------------

    model_names = [
        "knn",
        "svm_linear",
        "svm_rbf",
    ]

    for split_mode in SPLIT_MODES:

        for model_name in model_names:

            metric_file = (
                RESULTS_DIR
                / f"{result_name(model_name, split_mode)}"
                  f"_metrics.csv"
            )

            if not metric_file.exists():
                continue

            df = pd.read_csv(metric_file)

            if len(df) > 0:
                row = df.iloc[0].to_dict()
                row["split"] = split_mode
                rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary_path = (
        RESULTS_DIR
        / "task1_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    return summary_df


# ---------------------------------------------------------
# Pretty console comparison
# ---------------------------------------------------------

def print_comparison_table(summary_df):
    """
    Print SAGE and learned classifier results as
    percentages for easy comparison.
    """

    display_df = summary_df.copy()

    metric_columns = [
        "overall_accuracy",
        "balanced_accuracy",
        "box_accuracy",
        "can_accuracy",
        "mug_accuracy",
        "bottle_accuracy",
        "bowl_accuracy",
    ]

    for column in metric_columns:

        if column in display_df.columns:

            display_df[column] = (
                display_df[column] * 100
            )

    print("\n" + "=" * 90)
    print("TASK 1 - SAME-FEATURE BASELINE COMPARISON")
    print("=" * 90)

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("TASK 1 - SAME-FEATURE BASELINES")
    print("=" * 70)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    X, y, feature_names = load_feature_data()

    print("\nDataset loaded successfully.")

    print(f"Instances: {len(y)}")
    print(f"Features:  {X.shape[1]}")

    print("\nFeatures used:")

    for index, name in enumerate(feature_names):
        print(f"{index:2d}: {name}")

    # -----------------------------------------------------
    # Dataset sanity check
    # -----------------------------------------------------

    if X.shape[1] != 13:
        raise ValueError(
            f"Expected 13 features, "
            f"but found {X.shape[1]}."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y contain different numbers "
            "of instances."
        )

    # -----------------------------------------------------
    # Create results directory
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Object-identity groups
    # -----------------------------------------------------

    # The exported rows are per object *per frame*, so the
    # same physical object appears many times. Grouping is
    # what keeps a random split from leaking those repeats
    # across folds.

    print(
        "\nInferring object-identity groups "
        "from point-cloud extents..."
    )

    clouds, cloud_labels = load_pointcloud_data()

    if not np.array_equal(
        np.asarray(cloud_labels),
        np.asarray(y),
    ):
        raise ValueError(
            "Feature and point-cloud exports are not in "
            "the same instance order, so groups cannot "
            "be aligned to the feature matrix."
        )

    groups = compute_object_groups(
        clouds,
        distance_threshold=(
            GROUP_DISTANCE_THRESHOLD
        ),
    )

    print(
        f"Distinct object groups: "
        f"{len(np.unique(groups))} "
        f"(from {len(y)} instances)"
    )

    # -----------------------------------------------------
    # Create classifiers
    # -----------------------------------------------------

    models = create_models()

    # -----------------------------------------------------
    # Run all models under both CV regimes
    # -----------------------------------------------------

    for split_mode in SPLIT_MODES:

        cv = create_cross_validation(
            split_mode
        )

        print("\n" + "#" * 70)

        print(
            f"CROSS-VALIDATION REGIME: "
            f"{split_mode.upper()}"
        )

        print("#" * 70)

        print(
            f"\n{N_SPLITS}-fold "
            f"{type(cv).__name__}"
        )

        print(
            f"Random state: {RANDOM_STATE}"
        )

        for model_name, model in models.items():

            report_name = result_name(
                model_name,
                split_mode,
            )

            y_pred = evaluate_model(
                model_name=report_name,
                model=model,
                X=X,
                y=y,
                cv=cv,
                groups=(
                    groups
                    if split_mode in GROUP_AWARE_MODES
                    else None
                ),
            )

            # -------------------------------------------
            # Print results
            # -------------------------------------------

            print_evaluation_summary(
                report_name,
                y,
                y_pred,
            )

            # -------------------------------------------
            # Save results
            # -------------------------------------------

            saved_files = save_evaluation_results(
                model_name=report_name,
                y_true=y,
                y_pred=y_pred,
                output_dir=RESULTS_DIR,
            )

            print("\nSaved files:")

            for result_type, path in saved_files.items():
                print(
                    f"{result_type:25s}: {path}"
                )

    # -----------------------------------------------------
    # SAGE vs baselines
    # -----------------------------------------------------

    summary_df = create_combined_summary()

    print_comparison_table(
        summary_df
    )

    print(
        "\nSummary saved to:"
    )

    print(
        RESULTS_DIR
        / "task1_summary.csv"
    )

    print("\n" + "=" * 70)
    print("TASK 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()