
from pathlib import Path
import numpy as np


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# Path to:
# sage_baseline_experiments/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the directory containing both:
# baseline_data/
# sage_baseline_experiments/
PARENT_DIR = PROJECT_ROOT.parent

# External baseline dataset folder
DATA_DIR = PARENT_DIR / "baseline_data"


# ---------------------------------------------------------
# Feature dataset loader
# ---------------------------------------------------------

def load_feature_data():
    """
    Load the exported 13-dimensional SAGE feature dataset.

    Returns
    -------
    X : np.ndarray
        Feature matrix with shape (N, 13).

    y : np.ndarray
        Ground-truth class labels with shape (N,).

    feature_names : np.ndarray
        Names of the 13 features in column order.
    """

    file_path = DATA_DIR / "features_val_sample.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nFeature dataset was not found.\n"
            f"Expected location:\n{file_path}\n"
        )

    data = np.load(file_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]

    return X, y, feature_names


# ---------------------------------------------------------
# Point-cloud dataset loader
# ---------------------------------------------------------

def load_pointcloud_data():
    """
    Load the exported raw point-cloud dataset.

    Returns
    -------
    clouds : np.ndarray
        Object array where clouds[i] contains an
        (Mi, 3) point cloud for object i.

    y : np.ndarray
        Ground-truth class labels with shape (N,).
    """

    file_path = DATA_DIR / "pointclouds_val_sample.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nPoint-cloud dataset was not found.\n"
            f"Expected location:\n{file_path}\n"
        )

    data = np.load(file_path, allow_pickle=True)

    clouds = data["clouds"]
    y = data["y"]

    return clouds, y



def print_data_paths():
    """
    Print the paths being used by the experiment project.
    Useful for debugging folder-location problems.
    """

    print("Project root:")
    print(PROJECT_ROOT)

    print("\nBaseline data directory:")
    print(DATA_DIR)

    print("\nFeature dataset:")
    print(DATA_DIR / "features_val_sample.npz")

    print("\nPoint-cloud dataset:")
    print(DATA_DIR / "pointclouds_val_sample.npz")
