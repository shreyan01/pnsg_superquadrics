
from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar

from evaluation import LABEL_ORDER


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TASK1_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task1"
)

TASK5_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task5"
)


# ---------------------------------------------------------
# Basic accuracy
# ---------------------------------------------------------

def accuracy_score_manual(
    y_true,
    y_pred,
):
    """
    Calculate classification accuracy.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(
        y_true == y_pred
    )


# ---------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------

def bootstrap_accuracy_ci(
    y_true,
    y_pred,
    n_bootstrap=N_BOOTSTRAP,
    confidence_level=CONFIDENCE_LEVEL,
    random_state=RANDOM_STATE,
):
    """
    Estimate a percentile bootstrap confidence interval
    for classification accuracy.

    The same instance indices are resampled for y_true
    and y_pred so the prediction-label pairing is
    preserved.

    Returns
    -------
    accuracy : float
        Accuracy on the original sample.

    lower_ci : float
        Lower confidence bound.

    upper_ci : float
        Upper confidence bound.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    n_samples = len(y_true)

    if n_samples == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_scores = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for iteration in range(
        n_bootstrap
    ):

        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        sampled_true = y_true[
            indices
        ]

        sampled_pred = y_pred[
            indices
        ]

        bootstrap_scores[
            iteration
        ] = accuracy_score_manual(
            sampled_true,
            sampled_pred,
        )

    alpha = (
        1.0 - confidence_level
    )

    lower_percentile = (
        100 * alpha / 2
    )

    upper_percentile = (
        100 * (1 - alpha / 2)
    )

    lower_ci = np.percentile(
        bootstrap_scores,
        lower_percentile,
    )

    upper_ci = np.percentile(
        bootstrap_scores,
        upper_percentile,
    )

    observed_accuracy = (
        accuracy_score_manual(
            y_true,
            y_pred,
        )
    )

    return (
        observed_accuracy,
        lower_ci,
        upper_ci,
    )


# ---------------------------------------------------------
# Per-class bootstrap CI
# ---------------------------------------------------------

def bootstrap_per_class_ci(
    y_true,
    y_pred,
    labels=LABEL_ORDER,
    n_bootstrap=N_BOOTSTRAP,
):
    """
    Calculate bootstrap confidence intervals separately
    for each true class.

    For each class, only instances belonging to that
    true class are resampled.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = []

    for label in labels:

        mask = (
            y_true == label
        )

        class_true = y_true[
            mask
        ]

        class_pred = y_pred[
            mask
        ]

        (
            accuracy,
            lower_ci,
            upper_ci,
        ) = bootstrap_accuracy_ci(
            class_true,
            class_pred,
            n_bootstrap=n_bootstrap,
            random_state=RANDOM_STATE,
        )

        results.append(
            {
                "class": label,
                "n_instances": len(
                    class_true
                ),
                "accuracy": accuracy,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci,
            }
        )

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------
# Complete bootstrap summary for one model
# ---------------------------------------------------------

def calculate_model_bootstrap_results(
    model_name,
    prediction_file,
):
    """
    Calculate overall and per-class bootstrap
    confidence intervals for one classifier.
    """

    predictions = pd.read_csv(
        prediction_file
    )

    y_true = predictions[
        "true_label"
    ].to_numpy()

    y_pred = predictions[
        "predicted_label"
    ].to_numpy()

    # -----------------------------------------------------
    # Overall CI
    # -----------------------------------------------------

    (
        overall_accuracy,
        overall_lower,
        overall_upper,
    ) = bootstrap_accuracy_ci(
        y_true,
        y_pred,
    )

    overall_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "metric": "overall_accuracy",
                "class": "all",
                "n_instances": len(y_true),
                "accuracy": overall_accuracy,
                "ci_lower": overall_lower,
                "ci_upper": overall_upper,
            }
        ]
    )

    # -----------------------------------------------------
    # Per-class CIs
    # -----------------------------------------------------

    class_df = bootstrap_per_class_ci(
        y_true,
        y_pred,
    )

    class_df.insert(
        0,
        "model",
        model_name,
    )

    class_df.insert(
        1,
        "metric",
        "per_class_accuracy",
    )

    return pd.concat(
        [
            overall_df,
            class_df,
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------
# McNemar test
# ---------------------------------------------------------

def run_mcnemar_test(
    y_true,
    y_pred_a,
    y_pred_b,
    exact=True,
):
    """
    Compare two classifiers evaluated on the SAME
    instances using McNemar's test.

    Important
    ---------
    McNemar's test requires paired predictions.

    It is appropriate for comparisons such as:

        classifier A vs classifier B

    when both classifiers predict every same object.

    It should NOT be used simply to compare accuracy
    for 'can' objects against accuracy for 'bottle'
    objects because those are different instances.

    Returns
    -------
    dict
        Contingency counts, test statistic, and p-value.
    """

    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(
        y_pred_a
    )
    y_pred_b = np.asarray(
        y_pred_b
    )

    if not (
        len(y_true)
        == len(y_pred_a)
        == len(y_pred_b)
    ):
        raise ValueError(
            "McNemar's test requires predictions "
            "for the same instances."
        )

    correct_a = (
        y_pred_a == y_true
    )

    correct_b = (
        y_pred_b == y_true
    )

    # Rows: classifier A
    # Columns: classifier B
    #
    #              B correct   B wrong
    # A correct       n11        n10
    # A wrong         n01        n00

    n11 = np.sum(
        correct_a
        & correct_b
    )

    n10 = np.sum(
        correct_a
        & ~correct_b
    )

    n01 = np.sum(
        ~correct_a
        & correct_b
    )

    n00 = np.sum(
        ~correct_a
        & ~correct_b
    )

    table = [
        [
            int(n11),
            int(n10),
        ],
        [
            int(n01),
            int(n00),
        ],
    ]

    result = mcnemar(
        table,
        exact=exact,
        correction=not exact,
    )

    return {
        "both_correct": int(n11),
        "a_correct_b_wrong": int(
            n10
        ),
        "a_wrong_b_correct": int(
            n01
        ),
        "both_wrong": int(n00),
        "statistic": float(
            result.statistic
        ),
        "p_value": float(
            result.pvalue
        ),
    }


# ---------------------------------------------------------
# Pairwise classifier comparison
# ---------------------------------------------------------

def compare_saved_models(
    model_a,
    model_b,
):
    """
    Run McNemar's test between two learned baseline
    classifiers using their saved Task 1 predictions.
    """

    file_a = (
        TASK1_RESULTS_DIR
        / f"{model_a}_predictions.csv"
    )

    file_b = (
        TASK1_RESULTS_DIR
        / f"{model_b}_predictions.csv"
    )

    predictions_a = pd.read_csv(
        file_a
    )

    predictions_b = pd.read_csv(
        file_b
    )

    # Ensure the two files refer to
    # the same instances and ground truth.
    if not np.array_equal(
        predictions_a[
            "instance_id"
        ].to_numpy(),
        predictions_b[
            "instance_id"
        ].to_numpy(),
    ):
        raise ValueError(
            "Prediction files contain different "
            "instance ordering."
        )

    if not np.array_equal(
        predictions_a[
            "true_label"
        ].to_numpy(),
        predictions_b[
            "true_label"
        ].to_numpy(),
    ):
        raise ValueError(
            "Prediction files contain different "
            "ground-truth labels."
        )

    y_true = predictions_a[
        "true_label"
    ].to_numpy()

    y_pred_a = predictions_a[
        "predicted_label"
    ].to_numpy()

    y_pred_b = predictions_b[
        "predicted_label"
    ].to_numpy()

    result = run_mcnemar_test(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        exact=True,
    )

    result["model_a"] = model_a
    result["model_b"] = model_b

    return result


# ---------------------------------------------------------
# Format confidence interval
# ---------------------------------------------------------

def format_ci(
    accuracy,
    lower,
    upper,
):
    """
    Format an accuracy and confidence interval
    as percentages.
    """

    return (
        f"{accuracy * 100:.2f}% "
        f"[{lower * 100:.2f}%, "
        f"{upper * 100:.2f}%]"
    )


# ---------------------------------------------------------
# Main Task 5 workflow
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("TASK 5 — STATISTICAL RIGOR")
    print("=" * 70)

    TASK5_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = [
        "knn",
        "svm_linear",
        "svm_rbf",
    ]

    all_bootstrap_results = []

    # -----------------------------------------------------
    # Bootstrap confidence intervals
    # -----------------------------------------------------

    for model_name in models:

        prediction_file = (
            TASK1_RESULTS_DIR
            / f"{model_name}_predictions.csv"
        )

        if not prediction_file.exists():

            print(
                f"\nSkipping {model_name}: "
                "prediction file not found."
            )

            continue

        print(
            f"\nBootstrapping: {model_name}"
        )

        model_results = (
            calculate_model_bootstrap_results(
                model_name,
                prediction_file,
            )
        )

        all_bootstrap_results.append(
            model_results
        )

        # Individual model file
        output_file = (
            TASK5_RESULTS_DIR
            / (
                f"{model_name}_"
                f"bootstrap_ci.csv"
            )
        )

        model_results.to_csv(
            output_file,
            index=False,
        )

        print("\nOverall accuracy:")

        overall_row = (
            model_results[
                model_results["class"]
                == "all"
            ]
            .iloc[0]
        )

        print(
            format_ci(
                overall_row["accuracy"],
                overall_row["ci_lower"],
                overall_row["ci_upper"],
            )
        )

        print(
            "\nPer-class accuracy:"
        )

        class_rows = model_results[
            model_results["class"]
            != "all"
        ]

        for _, row in class_rows.iterrows():

            formatted_result = format_ci(
                row["accuracy"],
                row["ci_lower"],
                row["ci_upper"],
                    )

            print(
                f"{row['class']:10s}: "
                f"{formatted_result}"
                )
    # -----------------------------------------------------
    # Save combined bootstrap table
    # -----------------------------------------------------

    if all_bootstrap_results:

        combined_df = pd.concat(
            all_bootstrap_results,
            ignore_index=True,
        )

        combined_path = (
            TASK5_RESULTS_DIR
            / "bootstrap_ci_summary.csv"
        )

        combined_df.to_csv(
            combined_path,
            index=False,
        )

        print(
            "\nCombined bootstrap results saved to:"
        )

        print(
            combined_path
        )

    # -----------------------------------------------------
    # Pairwise McNemar tests between learned models
    # -----------------------------------------------------

    comparisons = [
        (
            "knn",
            "svm_linear",
        ),
        (
            "knn",
            "svm_rbf",
        ),
        (
            "svm_linear",
            "svm_rbf",
        ),
    ]

    mcnemar_results = []

    print(
        "\n" + "=" * 70
    )
    print(
        "PAIRED CLASSIFIER COMPARISONS"
    )
    print(
        "=" * 70
    )

    for model_a, model_b in comparisons:

        result = compare_saved_models(
            model_a,
            model_b,
        )

        mcnemar_results.append(
            result
        )

        print(
            f"\n{model_a} vs {model_b}"
        )

        print(
            f"A correct / B wrong: "
            f"{result['a_correct_b_wrong']}"
        )

        print(
            f"A wrong / B correct: "
            f"{result['a_wrong_b_correct']}"
        )

        print(
            f"McNemar statistic: "
            f"{result['statistic']:.4f}"
        )

        print(
            f"p-value: "
            f"{result['p_value']:.6g}"
        )

    mcnemar_df = pd.DataFrame(
        mcnemar_results
    )

    mcnemar_path = (
        TASK5_RESULTS_DIR
        / "mcnemar_model_comparisons.csv"
    )

    mcnemar_df.to_csv(
        mcnemar_path,
        index=False,
    )

    print(
        "\nMcNemar results saved to:"
    )

    print(
        mcnemar_path
    )

    # -----------------------------------------------------
    # Methodological warning
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "CAN VS BOTTLE SIGNIFICANCE TEST"
    )
    print(
        "=" * 70
    )

    print(
        "\nNot run yet."
    )

    print(
        "McNemar's test requires paired predictions "
        "from two methods on the same instances."
    )

    print(
        "Can and bottle class accuracies alone are "
        "based on different object instances."
    )

    print(
        "We therefore need the intended paired SAGE "
        "fitting-strategy predictions before running "
        "that requested test."
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "TASK 5 COMPLETE"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()

