
import numpy as np

from data_utils import (
    load_feature_data,
    load_pointcloud_data,
    print_data_paths,
)


def inspect_feature_data():
    """
    Inspect the exported 13D SAGE feature dataset.
    """

    print("\n" + "=" * 70)
    print("FEATURE DATASET INSPECTION")
    print("=" * 70)

    X, y, feature_names = load_feature_data()

    print("\nDataset shape")
    print("-" * 70)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of features: {len(feature_names)}")

    print("\nFeature names")
    print("-" * 70)

    for index, feature_name in enumerate(feature_names):
        print(f"{index:2d}: {feature_name}")

    print("\nClass distribution")
    print("-" * 70)

    labels, counts = np.unique(y, return_counts=True)

    for label, count in zip(labels, counts):
        percentage = (count / len(y)) * 100
        print(
            f"{label:10s}: "
            f"{count:4d} instances "
            f"({percentage:5.2f}%)"
        )

    print(f"\nTotal instances: {len(y)}")

    print("\nData-quality checks")
    print("-" * 70)

    print(f"NaN values: {np.isnan(X).sum()}")
    print(f"Infinite values: {np.isinf(X).sum()}")

    print("\nFeature ranges")
    print("-" * 70)

    for index, feature_name in enumerate(feature_names):
        column = X[:, index]

        print(
            f"{feature_name:15s} "
            f"min={np.min(column):10.5f} "
            f"max={np.max(column):10.5f} "
            f"mean={np.mean(column):10.5f} "
            f"std={np.std(column):10.5f}"
        )

    print("\nHue and saturation checks")
    print("-" * 70)

    hue_values = np.unique(X[:, 5])
    saturation_values = np.unique(X[:, 6])

    print(f"Unique hue values: {hue_values}")
    print(f"Unique saturation values: {saturation_values}")

    print("\nRadial-profile features by class")
    print("-" * 70)

    radial_indices = [7, 8, 9, 10, 11]

    for label in labels:
        class_rows = X[y == label]
        radial_values = class_rows[:, radial_indices]

        nonzero_count = np.count_nonzero(radial_values)
        total_values = radial_values.size

        print(
            f"{label:10s}: "
            f"{nonzero_count}/{total_values} "
            f"radial-profile values are non-zero"
        )


def inspect_pointcloud_data():
    """
    Inspect the raw point-cloud dataset.
    """

    print("\n" + "=" * 70)
    print("POINT-CLOUD DATASET INSPECTION")
    print("=" * 70)

    clouds, y = load_pointcloud_data()

    print("\nDataset information")
    print("-" * 70)

    print(f"Number of point clouds: {len(clouds)}")
    print(f"Number of labels: {len(y)}")

    labels, counts = np.unique(y, return_counts=True)

    print("\nClass distribution")
    print("-" * 70)

    for label, count in zip(labels, counts):
        percentage = (count / len(y)) * 100

        print(
            f"{label:10s}: "
            f"{count:4d} instances "
            f"({percentage:5.2f}%)"
        )

    print("\nPoint-count statistics")
    print("-" * 70)

    point_counts = np.array(
        [cloud.shape[0] for cloud in clouds]
    )

    print(f"Minimum points: {point_counts.min()}")
    print(f"Maximum points: {point_counts.max()}")
    print(f"Mean points:    {point_counts.mean():.2f}")
    print(f"Median points:  {np.median(point_counts):.2f}")

    print("\nPoint-cloud shape check")
    print("-" * 70)

    invalid_clouds = []

    for index, cloud in enumerate(clouds):
        if (
            not isinstance(cloud, np.ndarray)
            or cloud.ndim != 2
            or cloud.shape[1] != 3
        ):
            invalid_clouds.append(index)

    if len(invalid_clouds) == 0:
        print("All point clouds have valid shape (Mi, 3).")
    else:
        print(
            f"Warning: {len(invalid_clouds)} "
            "point clouds have unexpected shapes."
        )
        print("Indices:", invalid_clouds[:20])

    print("\nExample point cloud")
    print("-" * 70)

    print(f"Label: {y[0]}")
    print(f"Shape: {clouds[0].shape}")
    print("First 5 points:")
    print(clouds[0][:5])


def check_dataset_alignment():
    """
    Confirm that the feature dataset and point-cloud dataset
    have the same labels in the same instance order.
    """

    print("\n" + "=" * 70)
    print("DATASET ALIGNMENT CHECK")
    print("=" * 70)

    _, feature_labels, _ = load_feature_data()
    _, pointcloud_labels = load_pointcloud_data()

    if len(feature_labels) != len(pointcloud_labels):
        print("WARNING: Dataset lengths do not match.")
        print(f"Feature labels: {len(feature_labels)}")
        print(f"Point-cloud labels: {len(pointcloud_labels)}")
        return

    labels_match = np.array_equal(
        feature_labels,
        pointcloud_labels,
    )

    if labels_match:
        print(
            "\nSuccess: feature and point-cloud labels "
            "match in the same instance order."
        )
    else:
        mismatch_indices = np.where(
            feature_labels != pointcloud_labels
        )[0]

        print("\nWARNING: Dataset labels do not match.")
        print(f"Number of mismatches: {len(mismatch_indices)}")
        print(
            "First mismatch indices:",
            mismatch_indices[:20],
        )


if __name__ == "__main__":

    print_data_paths()

    inspect_feature_data()

    inspect_pointcloud_data()

    check_dataset_alignment()

    print("\n" + "=" * 70)
    print("DATA INSPECTION COMPLETE")
    print("=" * 70)
