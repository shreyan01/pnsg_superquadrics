from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from data_utils import load_feature_data
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task1"
)


# ---------------------------------------------------------
# SAGE reference results
# ---------------------------------------------------------

# These are the locked results supplied by the SAGE authors.
# They are included only for comparison.
#
# We do NOT have instance-level SAGE predictions here,
# so they are not part of the cross-validation procedure.

SAGE_REFERENCE = {
    "model": "SAGE",
    "overall_accuracy": 0.784,
    "box_accuracy": 0.947,
    "can_accuracy": 0.949,
    "mug_accuracy": 0.326,
    "bottle_accuracy": 0.315,
    "bowl_accuracy": 0.182,
}


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

def create_cross_validation():
    """
    Create one shared stratified 5-fold split.

    Shuffle is enabled so that folds are not dependent
    on the original instance ordering.

    random_state makes the split reproducible.
    """

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------
# Run one model
# ---------------------------------------------------------

def evaluate_model(
    model_name,
    model,
    X,
    y,
    cv,
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
        Shared StratifiedKFold object.

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

    sage_row = {
        "model": SAGE_REFERENCE["model"],
        "overall_accuracy": (
            SAGE_REFERENCE["overall_accuracy"]
        ),
        "balanced_accuracy": np.nan,
        "box_accuracy": (
            SAGE_REFERENCE["box_accuracy"]
        ),
        "can_accuracy": (
            SAGE_REFERENCE["can_accuracy"]
        ),
        "mug_accuracy": (
            SAGE_REFERENCE["mug_accuracy"]
        ),
        "bottle_accuracy": (
            SAGE_REFERENCE["bottle_accuracy"]
        ),
        "bowl_accuracy": (
            SAGE_REFERENCE["bowl_accuracy"]
        ),
    }

    rows.append(sage_row)

    # -----------------------------------------------------
    # Learned classifiers
    # -----------------------------------------------------

    model_names = [
        "knn",
        "svm_linear",
        "svm_rbf",
    ]

    for model_name in model_names:

        metric_file = (
            RESULTS_DIR
            / f"{model_name}_metrics.csv"
        )

        if not metric_file.exists():
            continue

        df = pd.read_csv(metric_file)

        if len(df) > 0:
            rows.append(
                df.iloc[0].to_dict()
            )

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
    print("TASK 1 — SAME-FEATURE BASELINE COMPARISON")
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
    print("TASK 1 — SAME-FEATURE BASELINES")
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
    # Shared cross-validation
    # -----------------------------------------------------

    cv = create_cross_validation()

    print(
        f"\nCross-validation: "
        f"{N_SPLITS}-fold StratifiedKFold"
    )

    print(
        f"Random state: {RANDOM_STATE}"
    )

    # -----------------------------------------------------
    # Create classifiers
    # -----------------------------------------------------

    models = create_models()

    # -----------------------------------------------------
    # Run all models
    # -----------------------------------------------------

    for model_name, model in models.items():

        y_pred = evaluate_model(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            cv=cv,
        )

        # -----------------------------------------------
        # Print results
        # -----------------------------------------------

        print_evaluation_summary(
            model_name,
            y,
            y_pred,
        )

        # -----------------------------------------------
        # Save results
        # -----------------------------------------------

        saved_files = save_evaluation_results(
            model_name=model_name,
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