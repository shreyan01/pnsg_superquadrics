import sys
from pathlib import Path

# Allow these scripts to be launched from any working
# directory (repo root or src/), not only from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from data_utils import (
    compute_object_groups,
    load_pointcloud_data,
)

from evaluation import (
    LABEL_ORDER,
    calculate_overall_metrics,
    calculate_per_class_accuracy,
)

from task2_pointnet import (
    SmallPointNet,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    GROUP_DISTANCE_THRESHOLD,
    MODEL_NAME,
    SPLIT_MODES,
    result_name,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

RANDOM_STATE = 42

N_SPLITS = 5

BASE_NUM_POINTS = 1024

# Task 4 reuses the fold models Task 2 trained, so it has to
# reuse Task 2's fold assignment too: each object is only ever
# scored by the model that held it out.
#
# Both CV regimes are swept for the same reason as in Tasks 1
# and 2. The random-split curve is the one to be careful with:
# those models saw other frames of every test object during
# training, so what it measures is partly how well degradation
# preserves a memorised object, not how well the model
# generalises under degradation. The grouped-split curve is
# the one to quote.


# ---------------------------------------------------------
# Point-count degradation conditions
# ---------------------------------------------------------

POINT_COUNTS = [
    1024,
    512,
    256,
    128,
    64,
]


# ---------------------------------------------------------
# Gaussian noise conditions
#
# Units are meters.
#
# 0.0015 m = 1.5 mm
# 0.0040 m = 4.0 mm
# ---------------------------------------------------------

NOISE_SIGMAS = [
    0.0000,
    0.0015,
    0.0020,
    0.0025,
    0.0030,
    0.0035,
    0.0040,
]


# ---------------------------------------------------------
# Evaluation batch size
# ---------------------------------------------------------

BATCH_SIZE = 64


# =========================================================
# 2. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "task2"
)

TASK2_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task2"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task4"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)


# =========================================================
# 3. DEVICE
# =========================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# 4. LOAD TRAINED POINTNET FOLD MODEL
# =========================================================

def load_fold_model(
    fold_number,
    device,
    split_mode="random",
):
    """
    Load one PointNet model produced by Task 2.

    split_mode selects which family of checkpoints to
    read, so the robustness sweep is always run against
    models trained under the matching CV regime.
    """

    model_path = (
        MODEL_DIR
        / f"{result_name(MODEL_NAME, split_mode)}"
          f"_fold_{fold_number}.pt"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nTask 2 model not found:\n"
            f"{model_path}\n\n"
            f"Run Task 2 successfully before Task 4."
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model = SmallPointNet(
        num_classes=len(LABEL_ORDER)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    model.eval()

    return model


# =========================================================
# 5. DEGRADE ONE POINT CLOUD
# =========================================================

def prepare_degraded_cloud(
    cloud,
    instance_id,
    num_points=BASE_NUM_POINTS,
    noise_sigma=0.0,
):
    """
    Apply controlled sensor degradation.

    Processing order
    ----------------
    1. Start with the original raw depth-camera cloud.
    2. Add Gaussian noise if requested.
    3. Sample the requested number of points.
    4. Center the sampled object.

    Physical object scale is NOT normalized.

    Parameters
    ----------
    cloud : np.ndarray
        Original point cloud with shape (N, 3).

    instance_id : int
        Original dataset index.

    num_points : int
        Number of points retained.

    noise_sigma : float
        Standard deviation of Gaussian noise in meters.

    Returns
    -------
    np.ndarray
        Processed point cloud with shape
        (num_points, 3).
    """

    cloud = np.asarray(
        cloud,
        dtype=np.float32,
    ).copy()

    if (
        cloud.ndim != 2
        or cloud.shape[1] != 3
    ):

        raise ValueError(
            f"Invalid point cloud shape: "
            f"{cloud.shape}"
        )

    # -----------------------------------------------------
    # Gaussian sensor noise
    # -----------------------------------------------------

    if noise_sigma > 0:

        # Separate deterministic noise generator.
        #
        # Each object gets the same underlying random
        # noise pattern across sigma levels; only the
        # magnitude changes.

        noise_rng = (
            np.random.default_rng(
                RANDOM_STATE
                + 100000
                + instance_id
            )
        )

        standard_noise = (
            noise_rng.normal(
                loc=0.0,
                scale=1.0,
                size=cloud.shape,
            )
        )

        cloud = (
            cloud
            + standard_noise
            * noise_sigma
        )

    # -----------------------------------------------------
    # Point sampling
    # -----------------------------------------------------

    sampling_rng = (
        np.random.default_rng(
            RANDOM_STATE
            + instance_id
        )
    )

    available_points = (
        cloud.shape[0]
    )

    replace = (
        available_points
        < num_points
    )

    selected_indices = (
        sampling_rng.choice(
            available_points,
            size=num_points,
            replace=replace,
        )
    )

    cloud = cloud[
        selected_indices
    ]

    # -----------------------------------------------------
    # Center object
    # -----------------------------------------------------

    centroid = cloud.mean(
        axis=0,
        keepdims=True,
    )

    cloud = (
        cloud
        - centroid
    )

    return cloud.astype(
        np.float32
    )


# =========================================================
# 6. EVALUATE ONE FOLD / CONDITION
# =========================================================

def evaluate_fold_condition(
    model,
    clouds,
    labels,
    test_indices,
    device,
    num_points,
    noise_sigma,
):
    """
    Evaluate one trained fold model under one
    degradation condition.
    """

    all_instance_ids = []
    all_predictions = []

    # -----------------------------------------------------
    # Process in batches
    # -----------------------------------------------------

    for start in range(
        0,
        len(test_indices),
        BATCH_SIZE,
    ):

        batch_indices = (
            test_indices[
                start:
                start + BATCH_SIZE
            ]
        )

        batch_clouds = []

        for instance_id in (
            batch_indices
        ):

            degraded_cloud = (
                prepare_degraded_cloud(
                    cloud=clouds[
                        instance_id
                    ],
                    instance_id=int(
                        instance_id
                    ),
                    num_points=num_points,
                    noise_sigma=noise_sigma,
                )
            )

            batch_clouds.append(
                degraded_cloud
            )

        batch_array = np.stack(
            batch_clouds
        )

        points = torch.from_numpy(
            batch_array
        ).to(
            device
        )

        with torch.no_grad():

            logits = model(
                points
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

        predictions = (
            predictions
            .cpu()
            .numpy()
        )

        for (
            instance_id,
            prediction,
        ) in zip(
            batch_indices,
            predictions,
        ):

            all_instance_ids.append(
                int(instance_id)
            )

            all_predictions.append(
                INDEX_TO_LABEL[
                    int(prediction)
                ]
            )

    return (
        np.asarray(
            all_instance_ids,
            dtype=int,
        ),
        np.asarray(
            all_predictions,
            dtype=object,
        ),
    )


# =========================================================
# 7. EVALUATE COMPLETE 5-FOLD CONDITION
# =========================================================

def evaluate_condition(
    clouds,
    labels,
    device,
    num_points,
    noise_sigma,
    split_mode="random",
    groups=None,
):
    """
    Evaluate all five PointNet fold models under
    one degradation condition.

    Each object is evaluated only by the model whose
    fold held that object out during Task 2 training,
    so the splitter here must be constructed exactly as
    Task 2 constructed it -- same class, same seed, same
    groups -- or objects get scored by a model that was
    trained on them.
    """

    number_instances = len(
        labels
    )

    predictions = np.empty(
        number_instances,
        dtype=object,
    )

    predictions[:] = None

    fold_assignments = np.full(
        number_instances,
        -1,
        dtype=int,
    )

    if split_mode == "random":

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        split_groups = None

    elif split_mode in ("grouped", "groupkfold"):

        if groups is None:
            raise ValueError(
                f"Split mode {split_mode!r} needs "
                f"object groups."
            )

        splitter = (
            StratifiedGroupKFold
            if split_mode == "grouped"
            else GroupKFold
        )

        cv = splitter(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        split_groups = groups

    else:

        raise ValueError(
            f"Unknown split_mode: {split_mode}"
        )

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(
            np.zeros(
                number_instances
            ),
            labels,
            groups=split_groups,
        ),
        start=1,
    ):

        print(
            f"  Fold {fold_number}/{N_SPLITS}",
            end="",
        )

        model = load_fold_model(
            fold_number,
            device,
            split_mode=split_mode,
        )

        (
            instance_ids,
            fold_predictions,
        ) = evaluate_fold_condition(
            model=model,
            clouds=clouds,
            labels=labels,
            test_indices=test_indices,
            device=device,
            num_points=num_points,
            noise_sigma=noise_sigma,
        )

        predictions[
            instance_ids
        ] = fold_predictions

        fold_assignments[
            instance_ids
        ] = fold_number

        print(" ok")

        # Free GPU memory between folds
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Coverage check
    # -----------------------------------------------------

    if any(
        prediction is None
        for prediction in predictions
    ):

        raise RuntimeError(
            "Some instances did not receive "
            "predictions."
        )

    return (
        predictions,
        fold_assignments,
    )


# =========================================================
# 8. CALCULATE CONDITION METRICS
# =========================================================

def calculate_condition_metrics(
    labels,
    predictions,
):
    """
    Calculate overall, balanced, and per-class
    accuracy for one degradation condition.
    """

    overall = (
        calculate_overall_metrics(
            labels,
            predictions,
        )
    )

    per_class = (
        calculate_per_class_accuracy(
            labels,
            predictions,
        )
    )

    results = {
        "overall_accuracy":
            overall[
                "accuracy"
            ],

        "balanced_accuracy":
            overall[
                "balanced_accuracy"
            ],
    }

    for label in LABEL_ORDER:

        results[
            f"{label}_accuracy"
        ] = per_class[
            label
        ]

    return results


# =========================================================
# 9. POINT-COUNT ROBUSTNESS
# =========================================================

def run_point_count_experiment(
    clouds,
    labels,
    device,
    split_mode="random",
    groups=None,
):
    """
    Evaluate PointNet as the number of input points
    progressively decreases.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK 4A - POINT-COUNT ROBUSTNESS"
    )

    print(
        "=" * 70
    )

    metric_rows = []

    prediction_rows = []

    for point_count in (
        POINT_COUNTS
    ):

        print(
            f"\nTesting "
            f"{point_count} points"
        )

        (
            predictions,
            folds,
        ) = evaluate_condition(
            clouds=clouds,
            labels=labels,
            device=device,
            num_points=point_count,
            noise_sigma=0.0,
            split_mode=split_mode,
            groups=groups,
        )

        metrics = (
            calculate_condition_metrics(
                labels,
                predictions,
            )
        )

        metrics[
            "point_count"
        ] = point_count

        metric_rows.append(
            metrics
        )

        print(
            f"Overall accuracy: "
            f"{metrics['overall_accuracy'] * 100:.2f}%"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy'] * 100:.2f}%"
        )

        # -------------------------------------------------
        # Save instance-level predictions
        # -------------------------------------------------

        for instance_id in range(
            len(labels)
        ):

            prediction_rows.append(
                {
                    "instance_id":
                        instance_id,

                    "true_label":
                        labels[
                            instance_id
                        ],

                    "predicted_label":
                        predictions[
                            instance_id
                        ],

                    "correct":
                        int(
                            predictions[
                                instance_id
                            ]
                            == labels[
                                instance_id
                            ]
                        ),

                    "fold":
                        folds[
                            instance_id
                        ],

                    "point_count":
                        point_count,
                }
            )

    metric_df = pd.DataFrame(
        metric_rows
    )

    metric_df = metric_df.sort_values(
        "point_count",
        ascending=False,
    )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    metric_path = (
        RESULTS_DIR
        / f"point_count_robustness_{split_mode}.csv"
    )

    prediction_path = (
        RESULTS_DIR
        / f"point_count_predictions_{split_mode}.csv"
    )

    metric_df.to_csv(
        metric_path,
        index=False,
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    return metric_df


# =========================================================
# 10. GAUSSIAN NOISE ROBUSTNESS
# =========================================================

def run_noise_experiment(
    clouds,
    labels,
    device,
    split_mode="random",
    groups=None,
):
    """
    Evaluate PointNet under increasing Gaussian
    sensor noise.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK 4B - GAUSSIAN-NOISE ROBUSTNESS"
    )

    print(
        "=" * 70
    )

    metric_rows = []

    prediction_rows = []

    for sigma in (
        NOISE_SIGMAS
    ):

        print(
            f"\nTesting sigma = "
            f"{sigma:.4f} m "
            f"({sigma * 1000:.1f} mm)"
        )

        (
            predictions,
            folds,
        ) = evaluate_condition(
            clouds=clouds,
            labels=labels,
            device=device,
            num_points=BASE_NUM_POINTS,
            noise_sigma=sigma,
            split_mode=split_mode,
            groups=groups,
        )

        metrics = (
            calculate_condition_metrics(
                labels,
                predictions,
            )
        )

        metrics[
            "noise_sigma_m"
        ] = sigma

        metrics[
            "noise_sigma_mm"
        ] = (
            sigma * 1000
        )

        metric_rows.append(
            metrics
        )

        print(
            f"Overall accuracy: "
            f"{metrics['overall_accuracy'] * 100:.2f}%"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy'] * 100:.2f}%"
        )

        # -------------------------------------------------
        # Instance-level predictions
        # -------------------------------------------------

        for instance_id in range(
            len(labels)
        ):

            prediction_rows.append(
                {
                    "instance_id":
                        instance_id,

                    "true_label":
                        labels[
                            instance_id
                        ],

                    "predicted_label":
                        predictions[
                            instance_id
                        ],

                    "correct":
                        int(
                            predictions[
                                instance_id
                            ]
                            == labels[
                                instance_id
                            ]
                        ),

                    "fold":
                        folds[
                            instance_id
                        ],

                    "noise_sigma_m":
                        sigma,

                    "noise_sigma_mm":
                        sigma
                        * 1000,
                }
            )

    metric_df = pd.DataFrame(
        metric_rows
    )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    metric_path = (
        RESULTS_DIR
        / f"noise_robustness_{split_mode}.csv"
    )

    prediction_path = (
        RESULTS_DIR
        / f"noise_predictions_{split_mode}.csv"
    )

    metric_df.to_csv(
        metric_path,
        index=False,
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    return metric_df


# =========================================================
# 11. DEGRADATION CURVES
# =========================================================

def plot_degradation_curve(
    curves,
    x_column,
    x_label,
    title,
    output_path,
    invert_x=False,
):
    """
    Plot one accuracy-vs-degradation curve per CV regime
    on a shared axis.

    Drawing both regimes together is the point: the gap
    between them is the part of the "robustness" that is
    really just the random-split models recognising
    objects they were trained on.

    Parameters
    ----------
    curves : dict
        Mapping from split_mode to its metrics DataFrame.

    x_column : str
        Column holding the degradation level.

    x_label, title : str
        Axis and figure labels.

    output_path : Path
        Where to save the figure.

    invert_x : bool
        Draw the x axis high-to-low, for point counts
        where "worse" means fewer points.
    """

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for split_mode, dataframe in curves.items():

        if dataframe is None or dataframe.empty:
            continue

        dataframe = dataframe.sort_values(x_column)

        ax.plot(
            dataframe[x_column],
            dataframe["overall_accuracy"] * 100,
            marker="o",
            label=f"{split_mode} split",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)

    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()

    if invert_x:
        ax.invert_xaxis()

    fig.tight_layout()

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


# =========================================================
# 12. BASELINE CONSISTENCY CHECK
# =========================================================

def verify_task2_baseline(
    point_count_df,
    noise_df,
    split_mode,
):
    """
    Compare the undegraded Task 4 condition against the
    matching Task 2 PointNet result.

    The 1024-point / zero-noise condition has to reproduce
    Task 2 exactly. If it does not, Task 4 is pairing
    objects with the wrong fold models, which would make
    every curve above it meaningless.
    """

    task2_metric_file = (
        TASK2_RESULTS_DIR
        / f"{result_name(MODEL_NAME, split_mode)}"
          f"_metrics.csv"
    )

    if not task2_metric_file.exists():

        print(
            "\nTask 2 metrics not found; "
            "baseline consistency check skipped."
        )

        return

    task2_metrics = pd.read_csv(task2_metric_file)

    original_accuracy = float(
        task2_metrics.iloc[0]["overall_accuracy"]
    )

    point_baseline = float(
        point_count_df[
            point_count_df["point_count"]
            == BASE_NUM_POINTS
        ]["overall_accuracy"].iloc[0]
    )

    noise_baseline = float(
        noise_df[
            noise_df["noise_sigma_m"] == 0.0
        ]["overall_accuracy"].iloc[0]
    )

    print("\n" + "=" * 70)
    print(
        f"BASELINE CONSISTENCY CHECK ({split_mode})"
    )
    print("=" * 70)

    print(
        f"Task 2 PointNet: "
        f"{original_accuracy * 100:.2f}%"
    )

    print(
        f"Task 4 / 1024 points: "
        f"{point_baseline * 100:.2f}%"
    )

    print(
        f"Task 4 / sigma=0: "
        f"{noise_baseline * 100:.2f}%"
    )

    tolerance = 1e-12

    if (
        abs(original_accuracy - point_baseline)
        < tolerance
        and abs(original_accuracy - noise_baseline)
        < tolerance
    ):

        print(
            "\nSuccess: undegraded Task 4 "
            "results reproduce Task 2."
        )

    else:

        print(
            "\nWARNING: Task 4 baseline differs "
            "from Task 2."
        )

        print(
            "Check that Task 2 was rerun with the "
            "same final fold models."
        )


# =========================================================
# 13. PRINT ONE REGIME
# =========================================================

def print_regime_results(
    split_mode,
    point_count_df,
    noise_df,
):
    """
    Print both degradation tables for one CV regime.
    """

    print("\n" + "=" * 70)
    print(
        f"POINT-COUNT RESULTS ({split_mode})"
    )
    print("=" * 70)

    display_point_df = point_count_df.copy()

    display_point_df["overall_accuracy"] *= 100
    display_point_df["balanced_accuracy"] *= 100

    print(
        display_point_df[
            [
                "point_count",
                "overall_accuracy",
                "balanced_accuracy",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print(f"NOISE RESULTS ({split_mode})")
    print("=" * 70)

    display_noise_df = noise_df.copy()

    display_noise_df["overall_accuracy"] *= 100
    display_noise_df["balanced_accuracy"] *= 100

    print(
        display_noise_df[
            [
                "noise_sigma_mm",
                "overall_accuracy",
                "balanced_accuracy",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


# =========================================================
# 14. MAIN
# =========================================================

def main():

    print("\n" + "=" * 70)
    print(
        "TASK 4 - ROBUSTNESS TO SENSOR DEGRADATION"
    )
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Scope note
    # -----------------------------------------------------

    print(
        "\nScope: this degrades the point clouds and "
        "re-runs the Task 2 PointNet baseline."
    )

    print(
        "It does NOT re-evaluate SAGE itself. Noise "
        "changes the superquadric *fit*, not just the "
        "stored numbers, so re-scoring SAGE under noise "
        "needs the fitting pipeline and the dataset -- "
        "see the guide's note on Task 4."
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = get_device()

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -----------------------------------------------------
    # Load exported raw point clouds
    # -----------------------------------------------------

    clouds, labels = load_pointcloud_data()

    labels = np.asarray(labels)

    print(f"\nInstances: {len(labels)}")

    # -----------------------------------------------------
    # Object-identity groups
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

    # -----------------------------------------------------
    # Sweep every regime whose models exist
    # -----------------------------------------------------

    point_count_curves = {}
    noise_curves = {}

    for split_mode in SPLIT_MODES:

        missing = [
            fold_number
            for fold_number in range(
                1,
                N_SPLITS + 1,
            )
            if not (
                MODEL_DIR
                / f"{result_name(MODEL_NAME, split_mode)}"
                  f"_fold_{fold_number}.pt"
            ).exists()
        ]

        if missing:

            print(
                f"\nSkipping {split_mode}: missing "
                f"Task 2 fold models {missing}. "
                f"Run Task 2 first."
            )

            continue

        print("\n" + "#" * 70)
        print(
            f"CROSS-VALIDATION REGIME: "
            f"{split_mode.upper()}"
        )
        print("#" * 70)

        point_count_df = run_point_count_experiment(
            clouds,
            labels,
            device,
            split_mode=split_mode,
            groups=groups,
        )

        noise_df = run_noise_experiment(
            clouds,
            labels,
            device,
            split_mode=split_mode,
            groups=groups,
        )

        point_count_curves[split_mode] = point_count_df
        noise_curves[split_mode] = noise_df

        verify_task2_baseline(
            point_count_df,
            noise_df,
            split_mode,
        )

        print_regime_results(
            split_mode,
            point_count_df,
            noise_df,
        )

    if not point_count_curves:

        raise FileNotFoundError(
            "No Task 2 fold models found for any "
            "cross-validation regime. Run Task 2 first."
        )

    # -----------------------------------------------------
    # Figures
    # -----------------------------------------------------

    plot_degradation_curve(
        curves=point_count_curves,
        x_column="point_count",
        x_label="Number of input points",
        title=(
            "PointNet robustness to point-cloud "
            "downsampling"
        ),
        output_path=(
            FIGURES_DIR
            / "accuracy_vs_point_count.png"
        ),
        invert_x=True,
    )

    plot_degradation_curve(
        curves=noise_curves,
        x_column="noise_sigma_mm",
        x_label="Gaussian noise sigma (mm)",
        title=(
            "PointNet robustness to Gaussian "
            "sensor noise"
        ),
        output_path=(
            FIGURES_DIR
            / "accuracy_vs_noise.png"
        ),
    )

    print(f"\nResults saved to:\n{RESULTS_DIR}")

    print("\n" + "=" * 70)
    print("TASK 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
