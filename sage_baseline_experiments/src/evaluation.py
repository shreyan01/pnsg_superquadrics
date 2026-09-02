from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------
# Class order
# ---------------------------------------------------------

# Keep a fixed class order so that every table,
# confusion matrix, and later statistical analysis
# uses the same ordering.
LABEL_ORDER = [
    "box",
    "can",
    "mug",
    "bottle",
    "bowl",
]


# ---------------------------------------------------------
# Overall metrics
# ---------------------------------------------------------

def calculate_overall_metrics(y_true, y_pred):
    """
    Calculate overall classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing overall accuracy and
        balanced accuracy.
    """

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


# ---------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------

def calculate_per_class_accuracy(
    y_true,
    y_pred,
    labels=LABEL_ORDER,
):
    """
    Calculate accuracy separately for each object class.

    Per-class accuracy here is equivalent to recall:

        correctly classified examples of class C
        -----------------------------------------
        total true examples of class C

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.

    y_pred : array-like
        Predicted labels.

    labels : list
        Class labels in the desired reporting order.

    Returns
    -------
    dict
        Mapping from class name to class accuracy.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {}

    for label in labels:

        mask = y_true == label

        total = np.sum(mask)

        if total == 0:
            results[label] = np.nan
            continue

        correct = np.sum(
            y_pred[mask] == y_true[mask]
        )

        results[label] = correct / total

    return results


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

def calculate_confusion_matrix(
    y_true,
    y_pred,
    labels=LABEL_ORDER,
):
    """
    Calculate the confusion matrix using a fixed
    class ordering.
    """

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

def create_classification_report(
    y_true,
    y_pred,
    labels=LABEL_ORDER,
):
    """
    Generate precision, recall, F1-score, and support
    for each class.

    This is supplementary information. The main Task 1
    results remain overall accuracy, per-class accuracy,
    and confusion matrix.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(report).transpose()


# ---------------------------------------------------------
# Prediction table
# ---------------------------------------------------------

def create_prediction_dataframe(
    y_true,
    y_pred,
):
    """
    Create an instance-level prediction table.

    Saving individual predictions is important because
    Task 5 will later use them for bootstrap confidence
    intervals and statistical comparisons.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    prediction_df = pd.DataFrame(
        {
            "instance_id": np.arange(len(y_true)),
            "true_label": y_true,
            "predicted_label": y_pred,
        }
    )

    prediction_df["correct"] = (
        prediction_df["true_label"]
        == prediction_df["predicted_label"]
    ).astype(int)

    return prediction_df


# ---------------------------------------------------------
# Metric summary table
# ---------------------------------------------------------

def create_metrics_dataframe(
    model_name,
    y_true,
    y_pred,
    labels=LABEL_ORDER,
):
    """
    Create a one-row summary table containing
    overall and per-class classification accuracy.
    """

    overall = calculate_overall_metrics(
        y_true,
        y_pred,
    )

    per_class = calculate_per_class_accuracy(
        y_true,
        y_pred,
        labels=labels,
    )

    row = {
        "model": model_name,
        "overall_accuracy": overall["accuracy"],
        "balanced_accuracy": overall[
            "balanced_accuracy"
        ],
    }

    for label in labels:
        row[f"{label}_accuracy"] = per_class[label]

    return pd.DataFrame([row])


# ---------------------------------------------------------
# Confusion-matrix DataFrame
# ---------------------------------------------------------

def create_confusion_matrix_dataframe(
    y_true,
    y_pred,
    labels=LABEL_ORDER,
):
    """
    Return a labeled confusion matrix as a DataFrame.
    """

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[
            f"pred_{label}"
            for label in labels
        ],
    )


# ---------------------------------------------------------
# Save complete evaluation
# ---------------------------------------------------------

def save_evaluation_results(
    model_name,
    y_true,
    y_pred,
    output_dir,
):
    """
    Save the main numerical results for one classifier.

    Files created
    -------------
    <model>_metrics.csv
    <model>_predictions.csv
    <model>_confusion_matrix.csv
    <model>_classification_report.csv
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Main metrics
    # -------------------------

    metrics_df = create_metrics_dataframe(
        model_name,
        y_true,
        y_pred,
    )

    metrics_path = (
        output_dir
        / f"{model_name}_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------
    # Instance predictions
    # -------------------------

    prediction_df = create_prediction_dataframe(
        y_true,
        y_pred,
    )

    predictions_path = (
        output_dir
        / f"{model_name}_predictions.csv"
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    # -------------------------
    # Confusion matrix
    # -------------------------

    confusion_df = (
        create_confusion_matrix_dataframe(
            y_true,
            y_pred,
        )
    )

    confusion_path = (
        output_dir
        / f"{model_name}_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_path
    )

    # -------------------------
    # Classification report
    # -------------------------

    report_df = create_classification_report(
        y_true,
        y_pred,
    )

    report_path = (
        output_dir
        / f"{model_name}_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion_matrix": confusion_path,
        "classification_report": report_path,
    }


# ---------------------------------------------------------
# Console summary
# ---------------------------------------------------------

def print_evaluation_summary(
    model_name,
    y_true,
    y_pred,
):
    """
    Print a concise classification summary
    to the terminal.
    """

    overall = calculate_overall_metrics(
        y_true,
        y_pred,
    )

    per_class = calculate_per_class_accuracy(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(
        f"\nOverall accuracy: "
        f"{overall['accuracy'] * 100:.2f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{overall['balanced_accuracy'] * 100:.2f}%"
    )

    print("\nPer-class accuracy")
    print("-" * 70)

    for label in LABEL_ORDER:
        value = per_class[label]

        print(
            f"{label:10s}: "
            f"{value * 100:6.2f}%"
        )

    print("\nConfusion matrix")
    print("-" * 70)

    confusion_df = (
        create_confusion_matrix_dataframe(
            y_true,
            y_pred,
        )
    )

    print(confusion_df)