"""
Task 3 - sample-efficiency curve.

From the co-author guide:

    Retrain the registry with only 1, 2, 5, 10, and 20 confirmed examples
    per category, then evaluate each version, and plot accuracy vs n. This
    is meant to demonstrate something a neural baseline structurally can't
    match - a usable category from a handful of real examples.

This script runs BOTH halves of that comparison:

  * The SAGE half - retrains the real registry at each n and evaluates it
    on YCB-Video. Needs `--dataset_root` pointing at a YCB-Video copy.
    Drives the author's own `train_registry_multiview.py` and
    `evaluate_on_ycbv.py` code; nothing in the parent repo is modified.

  * The baseline half - the same curve for learned classifiers on the
    exported 13D features. Needs only `baseline_data/`, so it always runs.

Usage
-----
Both halves (what you want if you have the dataset):

    python src/task3_sample_efficiency.py --dataset_root ~/ycb_dataset

Baseline half only (no dataset needed):

    python src/task3_sample_efficiency.py

Useful while checking it works, before committing to a full run:

    python src/task3_sample_efficiency.py --dataset_root ~/ycb_dataset \\
        --sizes 1 2 --max-frames 200 --workers 8

Cost
----
The expensive step is aggregating and fitting multi-view clouds. This
script fits each (video, class) pair ONCE per repeat and reuses those
fits across every n, since the n=1 registry is a prefix of the n=20 one.
That makes the whole sweep cost about one n=20 training run rather than
five separate runs. Evaluation then runs once per n; use `--max-frames`
to subsample the validation split while testing.

Requires `opencv-python` for the SAGE half (the YCB-Video loader reads
depth/label PNGs through cv2).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

# Allow these scripts to be launched from any working
# directory (repo root or src/), not only from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
# MUST be set before numpy/scipy are imported -- see registry.py's
# module docstring for the real incident this fix is for (a load average
# of 935 on a nominally 30-process run). This is very likely the actual
# root cause behind this script's SAGE half being dramatically slower
# than expected (repeats=1 default should finish in ~15-20 minutes, not
# multiple hours) -- worker processes spawned later (by run_sage_half's
# calls into train_registry_multiview.py / evaluate_on_ycbv.py) inherit,
# via fork, whatever BLAS threading state THIS process already latched
# onto at its own `import numpy as np` below, if that happens before
# these env vars are set.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
# MUST come before importing pyplot -- see plotting.py for why (headless
# Qt/xcb crash otherwise).
import matplotlib
matplotlib.use('Agg')
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
import sage_pipeline
import sage_reference


# =========================================================
# 1. CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# Examples (distinct objects) per category, as in the guide.
SAMPLE_SIZES = [1, 2, 5, 10, 20]

# Each (n, model) point in the BASELINE curve is averaged over this many
# draws, because a single 1-shot draw is almost pure luck. The SAGE half
# uses --repeats instead, since each of its runs is far more expensive.
N_REPEATS = 20

# Fraction of object groups reserved for evaluation, baseline half only.
TEST_GROUP_FRACTION = 0.30

GROUP_DISTANCE_THRESHOLD = 0.003

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "task3"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "task3"

# SAGE's full-training result, for a horizontal reference line.
# Single source of truth; see src/sage_reference.py.
SAGE_FULL_ACCURACY = sage_reference.get()["overall_accuracy"]


# =========================================================
# 2. BASELINE HALF - MODELS
# =========================================================

def create_models(n_per_class):
    """
    Build the baseline classifiers for one budget.

    k-NN's k has to shrink with the training set: with one example per
    category there are only five training rows in total, so the default
    k=5 would average over every class at once and make the 1-shot point
    meaningless.
    """

    n_neighbors = max(1, min(5, n_per_class))

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
                ("classifier", SVC(kernel="linear", C=1.0)),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", C=1.0, gamma="scale"),
                ),
            ]
        ),
    }


# =========================================================
# 3. BASELINE HALF - OBJECT-LEVEL POOLS
# =========================================================

def split_group_pools(y, groups, rng):
    """
    Split object groups into a training pool and a disjoint evaluation
    pool.

    Splitting on groups rather than rows is what keeps the curve honest:
    no physical object contributes to both training and evaluation.
    """

    unique_groups = np.unique(groups)

    shuffled = rng.permutation(unique_groups)

    n_test = int(round(len(shuffled) * TEST_GROUP_FRACTION))

    test_groups = shuffled[:n_test]
    train_groups = shuffled[n_test:]

    test_indices = np.flatnonzero(np.isin(groups, test_groups))

    missing = (
        set(LABEL_ORDER)
        - set(np.asarray(y)[test_indices])
    )

    if missing:
        raise ValueError(
            f"Evaluation pool is missing classes: {sorted(missing)}"
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
    Draw n distinct objects per category, then take one row from each.

    Returns None when some category does not have n distinct objects
    available, so the caller can skip that budget rather than silently
    report a curve point built from fewer examples than it claims.
    """

    y = np.asarray(y)

    train_mask = np.isin(groups, train_groups)

    selected = []

    for label in LABEL_ORDER:

        label_mask = train_mask & (y == label)

        available_groups = np.unique(groups[label_mask])

        if len(available_groups) < n_per_class:
            return None

        chosen_groups = rng.choice(
            available_groups,
            size=n_per_class,
            replace=False,
        )

        for group_id in chosen_groups:

            candidates = np.flatnonzero(
                label_mask & (groups == group_id)
            )

            selected.append(rng.choice(candidates))

    return np.asarray(selected, dtype=int)


def run_baseline_trial(X, y, groups, n_per_class, seed):
    """
    Train every baseline on one n-shot draw and evaluate on the disjoint
    object pool.
    """

    rng = np.random.default_rng(seed)

    train_groups, test_indices = split_group_pools(y, groups, rng)

    train_indices = sample_training_indices(
        y, groups, train_groups, n_per_class, rng
    )

    if train_indices is None:
        return []

    X_train, y_train = X[train_indices], np.asarray(y)[train_indices]
    X_test, y_test = X[test_indices], np.asarray(y)[test_indices]

    rows = []

    for model_name, model in create_models(n_per_class).items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        overall = calculate_overall_metrics(y_test, y_pred)
        per_class = calculate_per_class_accuracy(y_test, y_pred)

        row = {
            "model": model_name,
            "n_per_class": n_per_class,
            "seed": seed,
            "n_train_rows": len(train_indices),
            "n_test_rows": len(test_indices),
            "overall_accuracy": overall["accuracy"],
            "balanced_accuracy": overall["balanced_accuracy"],
        }

        for label in LABEL_ORDER:
            row[f"{label}_accuracy"] = per_class[label]

        rows.append(row)

    return rows


def run_baseline_half(X, y, groups, sizes):
    """
    Sweep every budget and every repeat for the learned baselines.
    """

    all_rows = []

    for n_per_class in sizes:

        print("\n" + "=" * 70)
        print(f"BASELINE BUDGET: {n_per_class} object(s) per category")
        print("=" * 70)

        budget_rows = []

        for repeat in range(N_REPEATS):

            seed = RANDOM_STATE + 1000 * n_per_class + repeat

            budget_rows.extend(
                run_baseline_trial(X, y, groups, n_per_class, seed)
            )

        if not budget_rows:
            print(
                "Skipped: not enough distinct objects in every "
                "category for this budget."
            )
            continue

        budget_df = pd.DataFrame(budget_rows)

        for model_name, model_df in budget_df.groupby("model"):
            print(
                f"{model_name:12s}: "
                f"{model_df['overall_accuracy'].mean() * 100:6.2f}% "
                f"+/- {model_df['overall_accuracy'].std() * 100:5.2f} "
                f"(n={len(model_df)} draws)"
            )

        all_rows.extend(budget_rows)

    return pd.DataFrame(all_rows)


# =========================================================
# 4. SAGE HALF - COLLECT CONFIRMED EXAMPLES
# =========================================================

def collect_training_graphs(
    training_module,
    dataset_root,
    split,
    max_per_class,
    frame_stride,
    max_nfev,
    workers,
    seed,
):
    """
    Fit multi-view clouds until every category has `max_per_class`
    confirmed examples.

    One "confirmed example" is one (video, class) pair - the same unit
    `train_registry_multiview.py` counts in `per_class_counts`, so
    "n examples per category" means here exactly what it means there.

    The fits are the expensive part of the whole task, so they are done
    once and reused for every n: the n=1 registry is a prefix of the
    n=20 one. Pairs are shuffled with `seed` first, so which examples a
    category gets is reproducible but not an artefact of video ordering.

    Returns
    -------
    dict
        vocab_word -> list of graphs, in confirmation order.
    """

    video_ids = sorted(
        {
            line.split("/")[0]
            for line in open(
                Path(dataset_root) / "image_sets" / f"{split}.txt"
            )
            if line.strip()
        }
    )

    print(f"Videos in split '{split}': {len(video_ids)}")

    work_items = []

    for video_id in video_ids:
        for class_id in training_module.discover_video_classes(
            dataset_root, video_id
        ):
            work_items.append(
                (
                    dataset_root,
                    video_id,
                    class_id,
                    frame_stride,
                    max_nfev,
                )
            )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(work_items))
    work_items = [work_items[index] for index in order]

    print(
        f"{len(work_items)} (video, class) pairs available; "
        f"fitting until every category has {max_per_class} "
        f"(workers={workers})"
    )

    graphs = defaultdict(list)
    n_skipped = 0
    started = time.time()

    def enough():
        return all(
            len(graphs[word]) >= max_per_class
            for word in LABEL_ORDER
        )

    # Submit in chunks so we can stop early once every category is
    # satisfied, instead of fitting all several-hundred pairs.
    chunk_size = max(workers * 4, 32)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=training_module._init_worker,
    ) as executor:

        for start in range(0, len(work_items), chunk_size):

            if enough():
                break

            chunk = work_items[start : start + chunk_size]

            for (
                video_id,
                class_id,
                vocab_word,
                graph,
                error,
            ) in executor.map(
                training_module._aggregate_and_fit, chunk
            ):

                if error or graph is None:
                    n_skipped += 1
                    continue

                if vocab_word not in LABEL_ORDER:
                    continue

                if len(graphs[vocab_word]) >= max_per_class:
                    continue

                graphs[vocab_word].append(graph)

            counts = {
                word: len(graphs[word]) for word in LABEL_ORDER
            }

            print(
                f"  fitted {min(start + chunk_size, len(work_items))}"
                f"/{len(work_items)} pairs | {counts} | "
                f"{time.time() - started:.0f}s"
            )

    print(f"Skipped {n_skipped} pairs (failed or implausible fits).")

    short = {
        word: len(graphs[word])
        for word in LABEL_ORDER
        if len(graphs[word]) < max_per_class
    }

    if short:
        print(
            f"WARNING: could not reach {max_per_class} examples for: "
            f"{short}. Budgets above those counts will be skipped."
        )

    return dict(graphs)


def build_registry(
    Registry,
    training_module,
    graphs,
    n_per_class,
):
    """
    Build a fresh registry from the first n confirmed examples per
    category.

    From-scratch each time, not incremental - the guide asks for five
    clean runs at each n, not a continuation of the previous one.
    """

    registry = Registry()
    registry.axisymmetric_words = training_module.AXISYMMETRIC_WORDS

    counts = {}

    for word in LABEL_ORDER:

        available = graphs.get(word, [])

        if len(available) < n_per_class:
            return None, None

        for graph in available[:n_per_class]:
            registry.confirm_graph(graph, word, F=1)

        counts[word] = n_per_class

    return registry, counts


# =========================================================
# 5. SAGE HALF - EVALUATE ONE REGISTRY
# =========================================================

def evaluate_registry(
    evaluation_module,
    model_path,
    dataset_root,
    split,
    max_frames,
    max_nfev,
    workers,
    scoring="joint",
):
    """
    Evaluate a saved registry on a YCB-Video split.

    Mirrors `evaluate_on_ycbv.main()`, reusing that module's own worker
    functions, but returns the numbers instead of printing them.
    """

    read_split_file = evaluation_module.read_split_file

    frame_keys = read_split_file(dataset_root, split)

    if max_frames:
        frame_keys = frame_keys[:max_frames]

    work_items = [
        (dataset_root, frame_key, max_nfev, scoring)
        for frame_key in frame_keys
    ]

    n_total = 0
    n_correct = 0
    n_implausible = 0

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    predictions = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=evaluation_module._init_worker,
        initargs=(str(model_path),),
    ) as executor:

        iterator = executor.map(evaluation_module._eval_one_frame, work_items)
        if HAVE_TQDM:
            iterator = tqdm(iterator, total=len(work_items), desc='  evaluating', unit='frame')

        for frame_key, results, error in iterator:

            if error:
                if HAVE_TQDM:
                    iterator.write(f'  [frame {frame_key}] skipped: {error}')
                continue

            for (
                instance_id,
                true_word,
                pred_word,
                confidence,
                _score_dict,
            ) in results:

                if pred_word == "__IMPLAUSIBLE__":
                    n_implausible += 1
                    continue

                n_total += 1
                per_class_total[true_word] += 1

                if pred_word == true_word:
                    n_correct += 1
                    per_class_correct[true_word] += 1

                predictions.append(
                    {
                        "instance_id": instance_id,
                        "frame": frame_key,
                        "true_label": true_word,
                        "predicted_label": pred_word,
                        "confidence": confidence,
                        "correct": int(pred_word == true_word),
                    }
                )

    per_class_accuracy = {
        word: (
            per_class_correct[word] / per_class_total[word]
            if per_class_total[word]
            else np.nan
        )
        for word in LABEL_ORDER
    }

    finite = [
        value
        for value in per_class_accuracy.values()
        if not np.isnan(value)
    ]

    return {
        "n_evaluated": n_total,
        "n_correct": n_correct,
        "n_implausible": n_implausible,
        "overall_accuracy": (
            n_correct / n_total if n_total else np.nan
        ),
        "balanced_accuracy": (
            float(np.mean(finite)) if finite else np.nan
        ),
        **{
            f"{word}_accuracy": per_class_accuracy[word]
            for word in LABEL_ORDER
        },
    }, pd.DataFrame(predictions)


# =========================================================
# 6. SAGE HALF - FULL SWEEP
# =========================================================

def run_sage_half(
    dataset_root,
    train_split,
    val_split,
    sizes,
    repeats,
    frame_stride,
    max_nfev,
    workers,
    max_frames,
    scoring="ml",
):
    """
    Retrain and evaluate the real registry at each n.
    """

    training, evaluation, Registry = (
        sage_pipeline.load_sage_modules()
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for repeat in range(repeats):

        seed = RANDOM_STATE + repeat

        print("\n" + "#" * 70)
        print(
            f"SAGE HALF - repeat {repeat + 1}/{repeats} (seed {seed})"
        )
        print("#" * 70)

        graphs = collect_training_graphs(
            training_module=training,
            dataset_root=dataset_root,
            split=train_split,
            max_per_class=max(sizes),
            frame_stride=frame_stride,
            max_nfev=max_nfev,
            workers=workers,
            seed=seed,
        )

        for n_per_class in sizes:

            registry, counts = build_registry(
                Registry, training, graphs, n_per_class
            )

            if registry is None:
                print(
                    f"\nn={n_per_class}: skipped, not enough "
                    f"confirmed examples in every category."
                )
                continue

            model_path = (
                MODEL_DIR
                / f"registry_n{n_per_class}_seed{seed}.json"
            )

            registry.save(str(model_path))

            print(
                f"\nn={n_per_class}: trained on {counts}, "
                f"saved -> {model_path.name}"
            )
            print("  evaluating...", flush=True)

            started = time.time()

            metrics, predictions = evaluate_registry(
                evaluation_module=evaluation,
                model_path=model_path,
                dataset_root=dataset_root,
                split=val_split,
                max_frames=max_frames,
                max_nfev=max_nfev,
                workers=workers,
                scoring=scoring,
            )

            metrics.update(
                {
                    "n_per_class": n_per_class,
                    "seed": seed,
                    "model": "SAGE registry",
                    "model_path": str(model_path),
                }
            )

            rows.append(metrics)

            predictions.to_csv(
                RESULTS_DIR
                / f"sage_predictions_n{n_per_class}_seed{seed}.csv",
                index=False,
            )

            print(
                f"  overall {metrics['overall_accuracy'] * 100:.2f}% | "
                f"balanced {metrics['balanced_accuracy'] * 100:.2f}% | "
                f"{metrics['n_evaluated']} instances | "
                f"{time.time() - started:.0f}s"
            )

    return pd.DataFrame(rows)


# =========================================================
# 7. AGGREGATION
# =========================================================

def summarise_baseline(trials_df):
    """
    Collapse the per-trial baseline table into mean / std / percentile
    bands per (model, n).
    """

    metric_columns = [
        "overall_accuracy",
        "balanced_accuracy",
    ] + [f"{label}_accuracy" for label in LABEL_ORDER]

    grouped = trials_df.groupby(["model", "n_per_class"])

    summary = grouped[metric_columns].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    bands = grouped["overall_accuracy"].agg(
        p05=lambda values: np.percentile(values, 5),
        p95=lambda values: np.percentile(values, 95),
        n_trials="size",
    )

    return summary.join(bands).reset_index()


def summarise_sage(sage_df):
    """
    Average the SAGE runs per n, when more than one repeat was run.
    """

    if sage_df.empty:
        return sage_df

    metric_columns = [
        "overall_accuracy",
        "balanced_accuracy",
    ] + [f"{label}_accuracy" for label in LABEL_ORDER]

    grouped = sage_df.groupby("n_per_class")

    summary = grouped[metric_columns].agg(["mean", "std"])

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    summary["n_runs"] = grouped.size()

    return summary.reset_index()


# =========================================================
# 8. PLOT
# =========================================================

def plot_curve(
    baseline_summary,
    sage_summary=None,
    output_path=None,
):
    """
    Accuracy vs number of confirmed examples per category.

    The SAGE registry curve, when present, is the point of the figure and
    is drawn in the emphasis colour with the baselines as context.
    """

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    have_sage = (
        sage_summary is not None and not sage_summary.empty
    )

    baseline_color = "#c3c2b7" if have_sage else None

    for index, (model_name, model_df) in enumerate(
        baseline_summary.groupby("model")
    ):

        model_df = model_df.sort_values("n_per_class")

        color = baseline_color or ["#2a78d6", "#eb6834", "#1baf7a"][index % 3]

        ax.plot(
            model_df["n_per_class"],
            model_df["overall_accuracy_mean"] * 100,
            marker="o",
            color=color,
            linewidth=1.8,
            markeredgecolor="#fcfcfb",
            markeredgewidth=1.5,
            label=model_name,
            zorder=2,
        )

        if not have_sage:
            ax.fill_between(
                model_df["n_per_class"],
                model_df["p05"] * 100,
                model_df["p95"] * 100,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    if have_sage:

        sage_sorted = sage_summary.sort_values("n_per_class")

        ax.plot(
            sage_sorted["n_per_class"],
            sage_sorted["overall_accuracy_mean"] * 100,
            marker="s",
            linewidth=2.6,
            color="#2a78d6",
            markeredgecolor="#fcfcfb",
            markeredgewidth=2,
            label="SAGE registry",
            zorder=3,
        )

        if sage_sorted["overall_accuracy_std"].notna().any():
            ax.fill_between(
                sage_sorted["n_per_class"],
                (
                    sage_sorted["overall_accuracy_mean"]
                    - sage_sorted["overall_accuracy_std"]
                )
                * 100,
                (
                    sage_sorted["overall_accuracy_mean"]
                    + sage_sorted["overall_accuracy_std"]
                )
                * 100,
                color="#2a78d6",
                alpha=0.13,
                linewidth=0,
            )

    ax.axhline(
        SAGE_FULL_ACCURACY * 100,
        linewidth=1.5,
        color="#52514e",
        zorder=1,
    )

    sizes = sorted(baseline_summary["n_per_class"].unique())

    if have_sage:
        sizes = sorted(
            set(sizes) | set(sage_summary["n_per_class"])
        )

    ax.annotate(
        f"SAGE, full training ({SAGE_FULL_ACCURACY * 100:.1f}%)",
        xy=(min(sizes), SAGE_FULL_ACCURACY * 100),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9,
        color="#52514e",
    )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])

    ax.set_xlabel(
        "Confirmed examples per category used for training"
    )
    ax.set_ylabel("Accuracy on held-out data (%)")
    ax.set_title("Task 3 - sample efficiency")

    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    return fig


# =========================================================
# 9. CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Task 3 sample-efficiency curve. Runs the learned-baseline "
            "half always, and the SAGE registry half when "
            "--dataset_root is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/task3_sample_efficiency.py\n"
            "  python src/task3_sample_efficiency.py "
            "--dataset_root ~/ycb_dataset\n"
            "  python src/task3_sample_efficiency.py "
            "--dataset_root ~/ycb_dataset --sizes 1 2 --max-frames 200\n"
        ),
    )

    parser.add_argument(
        "--dataset_root",
        default=None,
        help=(
            "Path to a YCB-Video dataset root (the directory holding "
            "image_sets/ and data/). Enables the SAGE registry half."
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Split to train the registry on (default: train).",
    )

    parser.add_argument(
        "--val-split",
        default="val",
        help="Split to evaluate on (default: val).",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=SAMPLE_SIZES,
        help=(
            f"Examples per category to sweep "
            f"(default: {' '.join(map(str, SAMPLE_SIZES))})."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Independent SAGE runs per n, each with a different draw of "
            "which examples get confirmed (default: 1). Costly."
        ),
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="Frame stride for multi-view aggregation (default: 10).",
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=2000,
        help="Least-squares iteration cap for fitting (default: 2000).",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: os.cpu_count()).",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Evaluate on only the first N validation frames. Useful for "
            "a quick check; omit for the real number."
        ),
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the learned-baseline half.",
    )

    parser.add_argument(
        "--scoring",
        choices=["joint", "ensembled", "ml"],
        default="ml",
        help=(
            "Which registry scoring path to evaluate. Was hardcoded to "
            "'joint' (the original single-Gaussian-per-mode scorer) -- "
            "changed the default to 'ml' (ExtraTreesClassifier) since "
            "that's the current best-performing classifier, and every "
            "other option in this file existed already. Falls back to "
            "'ensembled' scoring per-instance if the registry has no "
            "trained ML classifier (e.g. too few raw examples at very "
            "small n)."
        ),
    )

    return parser.parse_args()


# =========================================================
# 10. MAIN
# =========================================================

def main():

    import os

    args = parse_args()

    workers = args.workers or os.cpu_count()

    print("\n" + "=" * 70)
    print("TASK 3 - SAMPLE-EFFICIENCY CURVE")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Is the SAGE half runnable?
    # -----------------------------------------------------

    run_sage = args.dataset_root is not None

    if run_sage:

        available, reason = sage_pipeline.check_available()

        if not available:
            print("\nCannot run the SAGE half:\n")
            print(reason)
            raise SystemExit(1)

        from registry import assert_thread_limits_ok
        assert_thread_limits_ok()

        dataset_root = sage_pipeline.resolve_dataset_root(
            args.dataset_root
        )

        print(f"\nDataset root: {dataset_root}")
        print(f"Train split : {args.split}")
        print(f"Val split   : {args.val_split}")
        print(f"Sizes       : {args.sizes}")
        print(f"Repeats     : {args.repeats}")
        print(f"Workers     : {workers}")
        print(f"Scoring     : {args.scoring}")

    else:
        print(
            "\nNo --dataset_root given, so the SAGE registry half is "
            "not run."
        )
        print(
            "Pass --dataset_root <path-to-ycb_dataset> to produce the "
            "full comparison."
        )

        available, reason = sage_pipeline.check_available()

        print(f"\n(SAGE pipeline importable: {available})")

        if not available:
            print(f"  {reason}")

    # -----------------------------------------------------
    # Baseline half
    # -----------------------------------------------------

    baseline_summary = None

    if not args.skip_baseline:

        X, y, _ = load_feature_data()

        clouds, cloud_labels = load_pointcloud_data()

        if not np.array_equal(
            np.asarray(cloud_labels), np.asarray(y)
        ):
            raise ValueError(
                "Feature and point-cloud exports are not in the same "
                "instance order."
            )

        print(f"\nBaseline instances: {len(y)}")

        print(
            "Inferring object-identity groups from point-cloud "
            "extents..."
        )

        groups = compute_object_groups(
            clouds, distance_threshold=GROUP_DISTANCE_THRESHOLD
        )

        print(
            f"Distinct object groups: {len(np.unique(groups))}"
        )

        y_array = np.asarray(y)

        print("\nDistinct objects available per category:")

        for label in LABEL_ORDER:
            n_groups = len(np.unique(groups[y_array == label]))
            print(
                f"{label:10s}: {n_groups:3d} objects "
                f"({int((y_array == label).sum()):4d} rows)"
            )

        trials_df = run_baseline_half(X, y, groups, args.sizes)

        if not trials_df.empty:

            trials_df.to_csv(
                RESULTS_DIR / "sample_efficiency_trials.csv",
                index=False,
            )

            baseline_summary = summarise_baseline(trials_df)

            baseline_summary.to_csv(
                RESULTS_DIR / "sample_efficiency_summary.csv",
                index=False,
            )

    # -----------------------------------------------------
    # SAGE half
    # -----------------------------------------------------

    sage_summary = None

    if run_sage:

        sage_df = run_sage_half(
            dataset_root=dataset_root,
            train_split=args.split,
            val_split=args.val_split,
            sizes=args.sizes,
            repeats=args.repeats,
            frame_stride=args.frame_stride,
            max_nfev=args.max_nfev,
            workers=workers,
            max_frames=args.max_frames,
            scoring=args.scoring,
        )

        if not sage_df.empty:

            sage_df.to_csv(
                RESULTS_DIR / "sage_sample_efficiency_runs.csv",
                index=False,
            )

            sage_summary = summarise_sage(sage_df)

            sage_summary.to_csv(
                RESULTS_DIR / "sage_sample_efficiency_summary.csv",
                index=False,
            )

            print("\n" + "=" * 70)
            print("SAGE REGISTRY SAMPLE EFFICIENCY")
            print("=" * 70)

            display = sage_summary[
                [
                    "n_per_class",
                    "overall_accuracy_mean",
                    "balanced_accuracy_mean",
                    "n_runs",
                ]
            ].copy()

            for column in (
                "overall_accuracy_mean",
                "balanced_accuracy_mean",
            ):
                display[column] = display[column] * 100

            print(
                display.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.2f}",
                )
            )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    if baseline_summary is not None:

        print("\n" + "=" * 70)
        print("BASELINE SAMPLE EFFICIENCY")
        print("=" * 70)

        display = baseline_summary[
            [
                "model",
                "n_per_class",
                "overall_accuracy_mean",
                "overall_accuracy_std",
                "n_trials",
            ]
        ].copy()

        for column in (
            "overall_accuracy_mean",
            "overall_accuracy_std",
        ):
            display[column] = display[column] * 100

        print(
            display.to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )

    if baseline_summary is not None:

        figure_path = FIGURES_DIR / "sample_efficiency_curve.png"

        plot_curve(
            baseline_summary,
            sage_summary=sage_summary,
            output_path=figure_path,
        )

        print(f"\nFigure: {figure_path}")

    print(f"Results: {RESULTS_DIR}")

    print("\n" + "=" * 70)

    if run_sage and sage_summary is not None:
        print("TASK 3 COMPLETE (both halves)")
    else:
        print("TASK 3 COMPLETE (baseline half only)")

    print("=" * 70)


if __name__ == "__main__":
    main()