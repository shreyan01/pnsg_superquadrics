from pathlib import Path
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)

from sklearn.model_selection import StratifiedKFold

from data_utils import load_pointcloud_data

from evaluation import (
    LABEL_ORDER,
    print_evaluation_summary,
    save_evaluation_results,
)

from plotting import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
)


# =========================================================
# 1. EXPERIMENT CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# Number of points given to PointNet for each object
NUM_POINTS = 1024

# Same outer CV structure as Task 1
N_SPLITS = 5

# Training configuration
BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Keep 0 on Windows for maximum compatibility
NUM_WORKERS = 0

# Model name used in result files
MODEL_NAME = "pointnet"


# =========================================================
# 2. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "task2"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "task2"
)


# =========================================================
# 3. LABEL MAPPING
# =========================================================

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABEL_ORDER)
}

INDEX_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_INDEX.items()
}

NUM_CLASSES = len(
    LABEL_ORDER
)


# =========================================================
# 4. REPRODUCIBILITY
# =========================================================

def seed_everything(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# 5. DEVICE
# =========================================================

def get_device():
    """
    Select GPU when available.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# 6. POINT-CLOUD PREPROCESSING
# =========================================================

def prepare_point_cloud(
    cloud,
    num_points=NUM_POINTS,
    random_state=RANDOM_STATE,
):
    """
    Convert one variable-length point cloud into a
    fixed-size PointNet input.

    Steps
    -----
    1. Validate the cloud.
    2. Sample exactly num_points.
    3. Center the sampled cloud around its centroid.

    Important
    ---------
    We DO NOT normalize the object to unit size.

    Physical scale is therefore preserved.
    """

    cloud = np.asarray(
        cloud,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if cloud.ndim != 2:
        raise ValueError(
            f"Expected 2D point cloud, "
            f"received shape {cloud.shape}."
        )

    if cloud.shape[1] != 3:
        raise ValueError(
            f"Expected shape (N, 3), "
            f"received {cloud.shape}."
        )

    if cloud.shape[0] == 0:
        raise ValueError(
            "Point cloud contains zero points."
        )

    if not np.all(
        np.isfinite(cloud)
    ):
        raise ValueError(
            "Point cloud contains NaN "
            "or infinite values."
        )

    # -----------------------------------------------------
    # Deterministic random sampling
    # -----------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    number_available = (
        cloud.shape[0]
    )

    # Sample with replacement only when necessary
    replace = (
        number_available
        < num_points
    )

    indices = rng.choice(
        number_available,
        size=num_points,
        replace=replace,
    )

    sampled_cloud = cloud[
        indices
    ].copy()

    # -----------------------------------------------------
    # Remove camera-space translation
    # -----------------------------------------------------

    centroid = sampled_cloud.mean(
        axis=0,
        keepdims=True,
    )

    sampled_cloud = (
        sampled_cloud
        - centroid
    )

    return sampled_cloud.astype(
        np.float32
    )


# =========================================================
# 7. PYTORCH DATASET
# =========================================================

class PointCloudDataset(Dataset):
    """
    Dataset containing all 1109 point clouds.

    Returns
    -------
    points
        Tensor with shape (NUM_POINTS, 3).

    label
        Integer class label.

    instance_id
        Original exported-data index.
    """

    def __init__(
        self,
        clouds,
        labels,
        num_points=NUM_POINTS,
        random_state=RANDOM_STATE,
    ):

        self.clouds = clouds

        self.labels = np.asarray(
            labels
        )

        self.num_points = (
            num_points
        )

        self.random_state = (
            random_state
        )

        if len(self.clouds) != len(
            self.labels
        ):
            raise ValueError(
                "Point-cloud count and label count "
                "do not match."
            )

        # -------------------------------------------------
        # Validate labels
        # -------------------------------------------------

        unknown_labels = (
            set(self.labels)
            - set(LABEL_TO_INDEX.keys())
        )

        if unknown_labels:
            raise ValueError(
                f"Unknown labels found: "
                f"{unknown_labels}"
            )

        # -------------------------------------------------
        # Preprocess everything once
        # -------------------------------------------------

        processed_clouds = []
        encoded_labels = []

        print(
            "\nPreprocessing point clouds..."
        )

        for instance_id, (
            cloud,
            label,
        ) in enumerate(
            zip(
                self.clouds,
                self.labels,
            )
        ):

            instance_seed = (
                self.random_state
                + instance_id
            )

            processed = (
                prepare_point_cloud(
                    cloud=cloud,
                    num_points=self.num_points,
                    random_state=instance_seed,
                )
            )

            processed_clouds.append(
                processed
            )

            encoded_labels.append(
                LABEL_TO_INDEX[
                    label
                ]
            )

        self.processed_clouds = (
            np.stack(
                processed_clouds
            ).astype(
                np.float32
            )
        )

        self.encoded_labels = (
            np.asarray(
                encoded_labels,
                dtype=np.int64,
            )
        )

        print(
            "Point-cloud preprocessing complete."
        )

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index,
    ):

        points = torch.from_numpy(
            self.processed_clouds[
                index
            ]
        )

        label = torch.tensor(
            self.encoded_labels[
                index
            ],
            dtype=torch.long,
        )

        instance_id = int(
            index
        )

        return (
            points,
            label,
            instance_id,
        )


# =========================================================
# 8. SMALL POINTNET MODEL
# =========================================================

class SmallPointNet(nn.Module):
    """
    Small PointNet-style classifier.

    This is intentionally simpler than a full original
    PointNet implementation.

    Raw point cloud
        ↓
    shared point-wise MLPs
        ↓
    global max pooling
        ↓
    fully connected classifier
        ↓
    5 object categories
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        # -------------------------------------------------
        # Shared point-wise feature extraction
        # -------------------------------------------------

        self.conv1 = nn.Conv1d(
            3,
            64,
            kernel_size=1,
        )

        self.bn1 = nn.BatchNorm1d(
            64
        )

        self.conv2 = nn.Conv1d(
            64,
            64,
            kernel_size=1,
        )

        self.bn2 = nn.BatchNorm1d(
            64
        )

        self.conv3 = nn.Conv1d(
            64,
            128,
            kernel_size=1,
        )

        self.bn3 = nn.BatchNorm1d(
            128
        )

        self.conv4 = nn.Conv1d(
            128,
            256,
            kernel_size=1,
        )

        self.bn4 = nn.BatchNorm1d(
            256
        )

        self.relu = nn.ReLU()

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------

        self.fc1 = nn.Linear(
            256,
            128,
        )

        self.fc2 = nn.Linear(
            128,
            64,
        )

        self.fc3 = nn.Linear(
            64,
            num_classes,
        )

        self.dropout1 = nn.Dropout(
            p=0.30
        )

        self.dropout2 = nn.Dropout(
            p=0.20
        )

    def forward(
        self,
        points,
    ):
        """
        Parameters
        ----------
        points
            Shape:
            (batch_size, num_points, 3)
        """

        # PointNet Conv1d expects:
        #
        # batch × channels × points
        #
        # therefore:
        #
        # B × N × 3
        # becomes
        # B × 3 × N

        x = points.transpose(
            1,
            2,
        ).contiguous()

        # -------------------------------------------------
        # Point-wise feature learning
        # -------------------------------------------------

        x = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = self.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = self.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = self.relu(
            self.bn4(
                self.conv4(x)
            )
        )

        # -------------------------------------------------
        # Global symmetric aggregation
        # -------------------------------------------------

        x = torch.max(
            x,
            dim=2,
        ).values

        # Shape now:
        #
        # batch × 256

        # -------------------------------------------------
        # Classification head
        # -------------------------------------------------

        x = self.relu(
            self.fc1(x)
        )

        x = self.dropout1(x)

        x = self.relu(
            self.fc2(x)
        )

        x = self.dropout2(x)

        logits = self.fc3(x)

        return logits


# =========================================================
# 9. TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    """
    Train PointNet for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for (
        points,
        labels,
        _,
    ) in loader:

        points = points.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            points
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_correct += (
            predictions
            == labels
        ).sum().item()

        total_samples += (
            batch_size
        )

    average_loss = (
        total_loss
        / total_samples
    )

    training_accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        training_accuracy,
    )


# =========================================================
# 10. EVALUATE MODEL
# =========================================================

def evaluate_model(
    model,
    loader,
    device,
):
    """
    Evaluate the model without updating weights.
    """

    model.eval()

    all_instance_ids = []
    all_true_labels = []
    all_predictions = []

    with torch.no_grad():

        for (
            points,
            labels,
            instance_ids,
        ) in loader:

            points = points.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(
                points
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_instance_ids.extend(
                instance_ids
                .cpu()
                .numpy()
                .tolist()
            )

            all_true_labels.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            all_instance_ids,
            dtype=int,
        ),
        np.asarray(
            all_true_labels,
            dtype=int,
        ),
        np.asarray(
            all_predictions,
            dtype=int,
        ),
    )


# =========================================================
# 11. DATA LOADER CREATION
# =========================================================

def create_data_loader(
    dataset,
    indices,
    batch_size,
    shuffle,
    seed,
    device,
):
    """
    Create a DataLoader for a fold subset.
    """

    subset = Subset(
        dataset,
        indices,
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        generator=generator,
    )

    return loader


# =========================================================
# 12. TRAIN ONE CROSS-VALIDATION FOLD
# =========================================================

def run_fold(
    fold_number,
    train_indices,
    test_indices,
    dataset,
    device,
):
    """
    Train a fresh PointNet model for one outer
    cross-validation fold.
    """

    fold_seed = (
        RANDOM_STATE
        + fold_number
    )

    seed_everything(
        fold_seed
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"POINTNET FOLD "
        f"{fold_number}/{N_SPLITS}"
    )

    print(
        "=" * 70
    )

    print(
        f"Training instances: "
        f"{len(train_indices)}"
    )

    print(
        f"Test instances:     "
        f"{len(test_indices)}"
    )

    # -----------------------------------------------------
    # Loaders
    # -----------------------------------------------------

    train_loader = (
        create_data_loader(
            dataset=dataset,
            indices=train_indices,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=fold_seed,
            device=device,
        )
    )

    test_loader = (
        create_data_loader(
            dataset=dataset,
            indices=test_indices,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            seed=fold_seed,
            device=device,
        )
    )

    # -----------------------------------------------------
    # New model for every fold
    # -----------------------------------------------------

    model = SmallPointNet(
        num_classes=NUM_CLASSES
    ).to(
        device
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    # No class weighting in the main baseline.
    #
    # This keeps the baseline simple and makes it
    # conceptually consistent with Task 1.

    criterion = (
        nn.CrossEntropyLoss()
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Learning-rate scheduler
    # -----------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5,
        )
    )

    # -----------------------------------------------------
    # Training history
    # -----------------------------------------------------

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history.append(
            {
                "fold": fold_number,
                "epoch": epoch,
                "train_loss": (
                    train_loss
                ),
                "train_accuracy": (
                    train_accuracy
                ),
                "learning_rate": (
                    current_lr
                ),
            }
        )

        # Print progress
        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:3d}/{EPOCHS} | "
                f"Loss: "
                f"{train_loss:.4f} | "
                f"Train Acc: "
                f"{train_accuracy * 100:6.2f}% | "
                f"LR: "
                f"{current_lr:.6f}"
            )

        scheduler.step()

    # -----------------------------------------------------
    # Evaluate only after training
    # -----------------------------------------------------

    (
        instance_ids,
        true_indices,
        predicted_indices,
    ) = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
    )

    fold_accuracy = np.mean(
        true_indices
        == predicted_indices
    )

    print(
        f"\nFold {fold_number} "
        f"test accuracy: "
        f"{fold_accuracy * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Save trained fold model
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"pointnet_fold_{fold_number}.pt"
    )

    torch.save(
        {
            "fold": fold_number,
            "model_state_dict": (
                model.state_dict()
            ),
            "num_points": NUM_POINTS,
            "num_classes": NUM_CLASSES,
            "label_to_index": (
                LABEL_TO_INDEX
            ),
            "random_state": (
                fold_seed
            ),
        },
        model_path,
    )

    print(
        f"Model saved: "
        f"{model_path}"
    )

    return (
        instance_ids,
        true_indices,
        predicted_indices,
        history,
        fold_accuracy,
    )


# =========================================================
# 13. TASK 1 + TASK 2 SUMMARY
# =========================================================

def create_combined_summary():
    """
    Combine SAGE, Task 1 learned baselines,
    and PointNet into one table.
    """

    task1_summary_path = (
        PROJECT_ROOT
        / "results"
        / "task1"
        / "task1_summary.csv"
    )

    pointnet_metrics_path = (
        RESULTS_DIR
        / "pointnet_metrics.csv"
    )

    if not (
        task1_summary_path.exists()
        and pointnet_metrics_path.exists()
    ):
        return None

    task1_df = pd.read_csv(
        task1_summary_path
    )

    pointnet_df = pd.read_csv(
        pointnet_metrics_path
    )

    combined = pd.concat(
        [
            task1_df,
            pointnet_df,
        ],
        ignore_index=True,
    )

    output_path = (
        RESULTS_DIR
        / "task1_task2_summary.csv"
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    return combined


# =========================================================
# 14. PRINT COMBINED COMPARISON
# =========================================================

def print_combined_summary(
    dataframe,
):
    """
    Print model comparison as percentages.
    """

    if dataframe is None:
        return

    display_df = (
        dataframe.copy()
    )

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

        if column in (
            display_df.columns
        ):

            display_df[column] = (
                display_df[column]
                * 100
            )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SAGE / SAME-FEATURE / "
        "RAW POINT-CLOUD COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )


# =========================================================
# 15. MAIN EXPERIMENT
# =========================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK 2 — RAW POINT-CLOUD "
        "POINTNET BASELINE"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    seed_everything(
        RANDOM_STATE
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # -----------------------------------------------------
    # Load raw data
    # -----------------------------------------------------

    clouds, labels = (
        load_pointcloud_data()
    )

    labels = np.asarray(
        labels
    )

    print(
        f"\nInstances: "
        f"{len(labels)}"
    )

    print(
        f"Classes: "
        f"{LABEL_ORDER}"
    )

    print(
        f"Points per PointNet input: "
        f"{NUM_POINTS}"
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = PointCloudDataset(
        clouds=clouds,
        labels=labels,
        num_points=NUM_POINTS,
        random_state=RANDOM_STATE,
    )

    print(
        "\nProcessed data shape:"
    )

    print(
        dataset.processed_clouds.shape
    )

    print(
        f"NaN values: "
        f"{np.isnan(dataset.processed_clouds).sum()}"
    )

    print(
        f"Infinite values: "
        f"{np.isinf(dataset.processed_clouds).sum()}"
    )

    # -----------------------------------------------------
    # Verify centering
    # -----------------------------------------------------

    example_points = (
        dataset
        .processed_clouds[0]
    )

    example_centroid = (
        example_points.mean(
            axis=0
        )
    )

    print(
        "\nExample processed centroid:"
    )

    print(
        example_centroid
    )

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Stratified 5-fold CV
    # -----------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nCross-validation: "
        f"{N_SPLITS}-fold StratifiedKFold"
    )

    # -----------------------------------------------------
    # Storage for out-of-fold predictions
    # -----------------------------------------------------

    number_instances = (
        len(labels)
    )

    out_of_fold_predictions = (
        np.empty(
            number_instances,
            dtype=object,
        )
    )

    out_of_fold_predictions[:] = (
        None
    )

    out_of_fold_folds = (
        np.full(
            number_instances,
            -1,
            dtype=int,
        )
    )

    all_history = []
    fold_results = []

    # -----------------------------------------------------
    # Run folds
    # -----------------------------------------------------

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(
            np.zeros(
                number_instances
            ),
            labels,
        ),
        start=1,
    ):

        (
            instance_ids,
            true_indices,
            predicted_indices,
            history,
            fold_accuracy,
        ) = run_fold(
            fold_number=fold_number,
            train_indices=train_indices,
            test_indices=test_indices,
            dataset=dataset,
            device=device,
        )

        # -------------------------------------------------
        # Store predictions in original instance order
        # -------------------------------------------------

        for (
            instance_id,
            predicted_index,
        ) in zip(
            instance_ids,
            predicted_indices,
        ):

            out_of_fold_predictions[
                instance_id
            ] = INDEX_TO_LABEL[
                int(
                    predicted_index
                )
            ]

            out_of_fold_folds[
                instance_id
            ] = fold_number

        all_history.extend(
            history
        )

        fold_results.append(
            {
                "fold": fold_number,
                "train_size": (
                    len(train_indices)
                ),
                "test_size": (
                    len(test_indices)
                ),
                "accuracy": (
                    fold_accuracy
                ),
            }
        )

    # -----------------------------------------------------
    # Check complete coverage
    # -----------------------------------------------------

    if any(
        prediction is None
        for prediction
        in out_of_fold_predictions
    ):
        raise RuntimeError(
            "Some instances did not receive "
            "an out-of-fold prediction."
        )

    if np.any(
        out_of_fold_folds < 1
    ):
        raise RuntimeError(
            "Some instances were not assigned "
            "to a fold."
        )

    # -----------------------------------------------------
    # Save fold training history
    # -----------------------------------------------------

    history_df = pd.DataFrame(
        all_history
    )

    history_path = (
        RESULTS_DIR
        / "pointnet_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # -----------------------------------------------------
    # Save fold-level accuracy
    # -----------------------------------------------------

    fold_results_df = (
        pd.DataFrame(
            fold_results
        )
    )

    fold_results_path = (
        RESULTS_DIR
        / "pointnet_fold_results.csv"
    )

    fold_results_df.to_csv(
        fold_results_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FOLD RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        fold_results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # -----------------------------------------------------
    # Final aggregated out-of-fold evaluation
    # -----------------------------------------------------

    print_evaluation_summary(
        model_name=MODEL_NAME,
        y_true=labels,
        y_pred=(
            out_of_fold_predictions
        ),
    )

    # -----------------------------------------------------
    # Save standard evaluation files
    # -----------------------------------------------------

    saved_files = (
        save_evaluation_results(
            model_name=MODEL_NAME,
            y_true=labels,
            y_pred=(
                out_of_fold_predictions
            ),
            output_dir=RESULTS_DIR,
        )
    )

    print(
        "\nSaved evaluation files:"
    )

    for (
        result_type,
        path,
    ) in saved_files.items():

        print(
            f"{result_type:25s}: "
            f"{path}"
        )

    # -----------------------------------------------------
    # Add fold number to predictions file
    # -----------------------------------------------------

    prediction_path = (
        RESULTS_DIR
        / "pointnet_predictions.csv"
    )

    prediction_df = pd.read_csv(
        prediction_path
    )

    prediction_df[
        "fold"
    ] = (
        out_of_fold_folds
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # -----------------------------------------------------
    # Confusion-matrix figures
    # -----------------------------------------------------

    plot_confusion_matrix(
        y_true=labels,
        y_pred=(
            out_of_fold_predictions
        ),
        model_name="PointNet",
        output_path=(
            FIGURES_DIR
            / "pointnet_confusion_matrix.png"
        ),
    )

    plot_normalized_confusion_matrix(
        y_true=labels,
        y_pred=(
            out_of_fold_predictions
        ),
        model_name="PointNet",
        output_path=(
            FIGURES_DIR
            / (
                "pointnet_normalized_"
                "confusion_matrix.png"
            )
        ),
    )

    # -----------------------------------------------------
    # Combined Task 1 + Task 2 summary
    # -----------------------------------------------------

    combined_df = (
        create_combined_summary()
    )

    print_combined_summary(
        combined_df
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK 2 COMPLETE"
    )

    print(
        "=" * 70
    )


# =========================================================
# 16. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()