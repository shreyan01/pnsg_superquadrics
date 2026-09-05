import sys
from pathlib import Path

# Allow these scripts to be launched from any working
# directory (repo root or src/), not only from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact

from statsmodels.stats.contingency_tables import mcnemar

import sage_reference
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

TASK2_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task2"
)

# Real per-instance SAGE predictions, if you've run
# evaluate_on_ycbv.py --predictions_out <this path>. If this file
# doesn't exist, SAGE is simply skipped from the bootstrap/McNemar
# sections below (same graceful skip every other model gets) --
# the can-vs-bottle Fisher's exact test further down still works either
# way via the separate SAGE_PER_CLASS reconstructed-rate path, since
# that's a different, weaker requirement (rates only, not full
# per-instance data).
SAGE_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "sage"
)

TASK5_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task5"
)


# ---------------------------------------------------------
# Which prediction files to analyse
# ---------------------------------------------------------

# Every headline number in Tasks 1 and 2 gets a confidence
# interval, under all three cross-validation regimes, so the
# leakage gap is visible with uncertainty attached rather
# than as bare point estimates.
#
#   <name>             -- StratifiedKFold      (random, leaky)
#   <name>_grouped     -- StratifiedGroupKFold (objects whole)
#   <name>_groupkfold  -- GroupKFold           (objects whole,
#                                               unstratified)
MODEL_SPECS = [
    ("sage", SAGE_RESULTS_DIR),
    ("knn", TASK1_RESULTS_DIR),
    ("svm_linear", TASK1_RESULTS_DIR),
    ("svm_rbf", TASK1_RESULTS_DIR),
    ("knn_grouped", TASK1_RESULTS_DIR),
    ("svm_linear_grouped", TASK1_RESULTS_DIR),
    ("svm_rbf_grouped", TASK1_RESULTS_DIR),
    ("knn_groupkfold", TASK1_RESULTS_DIR),
    ("svm_linear_groupkfold", TASK1_RESULTS_DIR),
    ("svm_rbf_groupkfold", TASK1_RESULTS_DIR),
    ("pointnet", TASK2_RESULTS_DIR),
    ("pointnet_grouped", TASK2_RESULTS_DIR),
    ("pointnet_groupkfold", TASK2_RESULTS_DIR),
]


# ---------------------------------------------------------
# SAGE reported per-class results
# ---------------------------------------------------------

# SAGE's own numbers on val_sample. We have no instance-level
# SAGE predictions loaded into THIS script, only these rates
# and the class supports, so the correct/incorrect counts
# below are reconstructed by rounding rate * support to the
# nearest integer. That is exact enough for a test whose
# p-value lands many orders of magnitude from 0.05, but the
# counts are reconstructed, not measured -- worth restating
# if this ends up in a paper.
#
# NOTE: this no longer has to be true going forward. Task 3's SAGE
# half (task3_sample_efficiency.py) now runs with --scoring ml by
# default and saves REAL per-instance predictions to
# results/task3/sage_predictions_n<N>_seed<S>.csv for every n it
# evaluates. Once you have a full-size (n=max, i.e. not sample-
# efficiency-limited) run of that, load its predictions.csv here
# instead of reconstructing from a summary rate -- it'll give this
# script's paired tests real per-instance power instead of an
# approximation.
SAGE_PER_CLASS = {
    "box": 1.000,
    "can": 1.000,
    "mug": 0.517,
    "bottle": 0.685,
    "bowl": 0.364,
}

CLASS_SUPPORT = {
    "box": 475,
    "can": 355,
    "mug": 89,
    "bottle": 146,
    "bowl": 44,
}


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
# Can vs bottle: two independent proportions
# ---------------------------------------------------------

def run_two_proportion_test(
    n_correct_a,
    n_total_a,
    n_correct_b,
    n_total_b,
    label_a="can",
    label_b="bottle",
    rng=None,
):
    """
    Test whether two *independent* groups are classified
    at different rates.

    Why not McNemar
    ---------------
    McNemar's test is the right tool for two classifiers
    scored on the same instances: it conditions on the
    discordant pairs, and pairing is what gives it its
    power. Cans and bottles are different objects, so
    there are no pairs to condition on and McNemar simply
    does not apply.

    The can-vs-bottle question in the guide is really
    "does the axisymmetric fitting path succeed at a
    higher rate than the flexible one?", which is a
    comparison of two independent binomial proportions.
    Fisher's exact test answers exactly that, without the
    large-sample approximation a chi-square would need for
    the smaller bottle group.

    A bootstrap interval on the difference in rates is
    reported alongside the p-value, because the effect
    size is the part that actually matters here.

    Returns
    -------
    dict
        Counts, rates, difference with CI, odds ratio and
        two-sided p-value.
    """

    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    n_wrong_a = n_total_a - n_correct_a
    n_wrong_b = n_total_b - n_correct_b

    #            correct      wrong
    # group A    n_correct_a  n_wrong_a
    # group B    n_correct_b  n_wrong_b
    table = [
        [int(n_correct_a), int(n_wrong_a)],
        [int(n_correct_b), int(n_wrong_b)],
    ]

    odds_ratio, p_value = fisher_exact(
        table,
        alternative="two-sided",
    )

    rate_a = n_correct_a / n_total_a
    rate_b = n_correct_b / n_total_b

    # Bootstrap the difference by resampling each group
    # independently, which is what "independent groups"
    # means operationally.
    outcomes_a = np.zeros(n_total_a, dtype=int)
    outcomes_a[: int(n_correct_a)] = 1

    outcomes_b = np.zeros(n_total_b, dtype=int)
    outcomes_b[: int(n_correct_b)] = 1

    differences = np.empty(N_BOOTSTRAP)

    for index in range(N_BOOTSTRAP):

        sample_a = rng.choice(
            outcomes_a,
            size=n_total_a,
            replace=True,
        )

        sample_b = rng.choice(
            outcomes_b,
            size=n_total_b,
            replace=True,
        )

        differences[index] = (
            sample_a.mean()
            - sample_b.mean()
        )

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower, upper = np.percentile(
        differences,
        [
            100 * (alpha / 2),
            100 * (1 - alpha / 2),
        ],
    )

    return {
        "group_a": label_a,
        "group_b": label_b,
        "n_total_a": int(n_total_a),
        "n_correct_a": int(n_correct_a),
        "accuracy_a": rate_a,
        "n_total_b": int(n_total_b),
        "n_correct_b": int(n_correct_b),
        "accuracy_b": rate_b,
        "difference": rate_a - rate_b,
        "difference_ci_lower": lower,
        "difference_ci_upper": upper,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "test": "fisher_exact_two_sided",
    }


def can_vs_bottle_from_predictions(
    model_name,
    prediction_file,
    rng=None,
):
    """
    Run the can-vs-bottle comparison from one model's
    saved instance-level predictions.
    """

    predictions = pd.read_csv(prediction_file)

    y_true = predictions["true_label"].to_numpy()
    y_pred = predictions["predicted_label"].to_numpy()

    correct = y_pred == y_true

    can_mask = y_true == "can"
    bottle_mask = y_true == "bottle"

    result = run_two_proportion_test(
        n_correct_a=int(correct[can_mask].sum()),
        n_total_a=int(can_mask.sum()),
        n_correct_b=int(correct[bottle_mask].sum()),
        n_total_b=int(bottle_mask.sum()),
        rng=rng,
    )

    result["model"] = model_name

    return result


def can_vs_bottle_for_sage(rng=None):
    """
    Run the same comparison on SAGE's own result.

    This is the version the guide actually asks about: it tests whether
    SAGE's axisymmetric-fitting result on cans really is better than its
    flexible-fitting result on bottles, rather than a fluke of a modest
    held-out set.

    Uses the real per-instance predictions when they are on disk, and
    only falls back to counts reconstructed from reported rates when
    they are not. The returned `model` label says which was used, so a
    table can never silently present a reconstruction as measured.

    IMPORTANT caveat on interpretation: the fitting strategy is chosen
    from the ground-truth label (see sage_reference), so 'can' is fitted
    axisymmetrically *because it is a can*. Part of any can-vs-bottle
    gap is therefore that leak rather than a property of the fitting
    strategy. Re-run this after the item-2 fix before quoting it.
    """

    frame = sage_reference.load_predictions()

    if frame is not None:

        correct = frame["true_label"] == frame["predicted_label"]

        can = frame["true_label"] == "can"
        bottle = frame["true_label"] == "bottle"

        result = run_two_proportion_test(
            n_correct_a=int(correct[can].sum()),
            n_total_a=int(can.sum()),
            n_correct_b=int(correct[bottle].sum()),
            n_total_b=int(bottle.sum()),
            rng=rng,
        )

        result["model"] = "SAGE (measured)"

        return result

    result = run_two_proportion_test(
        n_correct_a=round(
            SAGE_PER_CLASS["can"]
            * CLASS_SUPPORT["can"]
        ),
        n_total_a=CLASS_SUPPORT["can"],
        n_correct_b=round(
            SAGE_PER_CLASS["bottle"]
            * CLASS_SUPPORT["bottle"]
        ),
        n_total_b=CLASS_SUPPORT["bottle"],
        rng=rng,
    )

    result["model"] = "SAGE (reconstructed counts)"

    return result


# ---------------------------------------------------------
# Pairwise classifier comparison
# ---------------------------------------------------------

def compare_saved_models(
    model_a,
    model_b,
    dir_a=TASK1_RESULTS_DIR,
    dir_b=TASK1_RESULTS_DIR,
):
    """
    Run McNemar's test between two learned baseline
    classifiers using their saved instance-level
    predictions.

    Both models must have been evaluated on the same
    instances in the same order, which every out-of-fold
    prediction file in this project satisfies.
    """

    file_a = (
        dir_a
        / f"{model_a}_predictions.csv"
    )

    file_b = (
        dir_b
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
    print("TASK 5 - STATISTICAL RIGOR")
    print("=" * 70)

    TASK5_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    all_bootstrap_results = []

    available_models = []

    # -----------------------------------------------------
    # Bootstrap confidence intervals
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    print(
        f"\n{N_BOOTSTRAP} resamples, "
        f"{CONFIDENCE_LEVEL * 100:.0f}% percentile "
        f"intervals."
    )

    for model_name, results_dir in MODEL_SPECS:

        prediction_file = (
            results_dir
            / f"{model_name}_predictions.csv"
        )

        if not prediction_file.exists():

            print(
                f"\nSkipping {model_name}: "
                "prediction file not found."
            )

            continue

        available_models.append(
            (model_name, results_dir)
        )

        print(f"\nBootstrapping: {model_name}")

        model_results = (
            calculate_model_bootstrap_results(
                model_name,
                prediction_file,
            )
        )

        all_bootstrap_results.append(model_results)

        model_results.to_csv(
            TASK5_RESULTS_DIR
            / f"{model_name}_bootstrap_ci.csv",
            index=False,
        )

        overall_row = model_results[
            model_results["class"] == "all"
        ].iloc[0]

        print("\nOverall accuracy:")

        print(
            format_ci(
                overall_row["accuracy"],
                overall_row["ci_lower"],
                overall_row["ci_upper"],
            )
        )

        print("\nPer-class accuracy:")

        class_rows = model_results[
            model_results["class"] != "all"
        ]

        for _, row in class_rows.iterrows():

            print(
                f"{row['class']:10s}: "
                f"{format_ci(row['accuracy'], row['ci_lower'], row['ci_upper'])}"
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

        print(combined_path)

    # -----------------------------------------------------
    # Paired classifier comparisons (McNemar)
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIRED CLASSIFIER COMPARISONS (McNEMAR)")
    print("=" * 70)

    print(
        "\nMcNemar is valid here because both members of "
        "each pair predict the same instances."
    )

    model_dirs = dict(available_models)

    # Within-regime comparisons only. Comparing a random-split
    # model against a grouped-split one would confound the
    # model difference with the split difference.
    candidate_comparisons = [
        ("knn", "svm_linear"),
        ("knn", "svm_rbf"),
        ("svm_linear", "svm_rbf"),
        ("knn", "pointnet"),
        ("svm_rbf", "pointnet"),
        ("knn_grouped", "svm_linear_grouped"),
        ("knn_grouped", "svm_rbf_grouped"),
        ("svm_linear_grouped", "svm_rbf_grouped"),
        ("knn_grouped", "pointnet_grouped"),
        ("svm_rbf_grouped", "pointnet_grouped"),
        ("knn_groupkfold", "svm_linear_groupkfold"),
        ("knn_groupkfold", "svm_rbf_groupkfold"),
        ("svm_linear_groupkfold", "svm_rbf_groupkfold"),
        ("knn_groupkfold", "pointnet_groupkfold"),
        ("svm_rbf_groupkfold", "pointnet_groupkfold"),
        # The two group-aware regimes score the same
        # instances, so they are directly comparable: this
        # asks whether stratifying the folds changes the
        # result at all.
        ("svm_rbf_grouped", "svm_rbf_groupkfold"),
        ("pointnet_grouped", "pointnet_groupkfold"),
    ]

    mcnemar_results = []

    for model_a, model_b in candidate_comparisons:

        if (
            model_a not in model_dirs
            or model_b not in model_dirs
        ):
            continue

        result = compare_saved_models(
            model_a,
            model_b,
            dir_a=model_dirs[model_a],
            dir_b=model_dirs[model_b],
        )

        mcnemar_results.append(result)

        print(f"\n{model_a} vs {model_b}")

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
            f"p-value: {result['p_value']:.6g}"
        )

    if mcnemar_results:

        mcnemar_path = (
            TASK5_RESULTS_DIR
            / "mcnemar_model_comparisons.csv"
        )

        pd.DataFrame(mcnemar_results).to_csv(
            mcnemar_path,
            index=False,
        )

        print(
            f"\nMcNemar results saved to:\n{mcnemar_path}"
        )

    # -----------------------------------------------------
    # Can vs bottle fitting-strategy comparison
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CAN VS BOTTLE FITTING-STRATEGY COMPARISON")
    print("=" * 70)

    print(
        "\nCans and bottles are different objects, so this "
        "is a comparison of two independent proportions, "
        "not a paired one."
    )

    print(
        "McNemar's test does not apply; Fisher's exact "
        "test does. See run_two_proportion_test."
    )

    can_bottle_results = [
        can_vs_bottle_for_sage(rng=rng)
    ]

    for model_name, results_dir in available_models:

        can_bottle_results.append(
            can_vs_bottle_from_predictions(
                model_name,
                results_dir
                / f"{model_name}_predictions.csv",
                rng=rng,
            )
        )

    can_bottle_df = pd.DataFrame(can_bottle_results)

    column_order = [
        "model",
        "group_a",
        "n_correct_a",
        "n_total_a",
        "accuracy_a",
        "group_b",
        "n_correct_b",
        "n_total_b",
        "accuracy_b",
        "difference",
        "difference_ci_lower",
        "difference_ci_upper",
        "odds_ratio",
        "p_value",
        "test",
    ]

    can_bottle_df = can_bottle_df[column_order]

    can_bottle_path = (
        TASK5_RESULTS_DIR
        / "can_vs_bottle_tests.csv"
    )

    can_bottle_df.to_csv(
        can_bottle_path,
        index=False,
    )

    for _, row in can_bottle_df.iterrows():

        print(f"\n{row['model']}")

        print(
            f"  can    : {row['n_correct_a']:4d}/"
            f"{row['n_total_a']:4d} = "
            f"{row['accuracy_a'] * 100:6.2f}%"
        )

        print(
            f"  bottle : {row['n_correct_b']:4d}/"
            f"{row['n_total_b']:4d} = "
            f"{row['accuracy_b'] * 100:6.2f}%"
        )

        print(
            f"  difference: "
            f"{row['difference'] * 100:+.2f} pp "
            f"[{row['difference_ci_lower'] * 100:+.2f}, "
            f"{row['difference_ci_upper'] * 100:+.2f}]"
        )

        print(
            f"  Fisher exact p = {row['p_value']:.4g}"
            f"  (odds ratio {row['odds_ratio']:.2f})"
        )

    print(
        f"\nCan/bottle results saved to:\n"
        f"{can_bottle_path}"
    )

    print("\n" + "=" * 70)
    print("TASK 5 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()