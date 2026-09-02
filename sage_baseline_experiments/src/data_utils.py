
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


# ---------------------------------------------------------
# Object-identity groups (leakage control)
# ---------------------------------------------------------

def compute_object_groups(
    clouds,
    distance_threshold=0.003,
):
    """
    Assign each instance a group id that approximates
    *physical object identity*.

    Why this is needed
    ------------------
    `export_baseline_data.py` iterates over YCB-Video
    *frames* and emits one row per object per frame.
    The 1109 rows are therefore repeated observations of
    a much smaller set of physical objects, seen across
    the frames of ~12 validation videos.

    A plain StratifiedKFold splits those rows at random,
    so near-identical views of the SAME physical object
    land in both the training and the test fold. A
    classifier can then retrieve a memorised neighbour
    instead of generalising, which inflates accuracy.

    SAGE's own reported 78.4% used a *video-level* split,
    so a random-split baseline is not comparable to it.

    How the proxy works
    -------------------
    The export does not carry `video_id` / `frame_id`, so
    true object identity cannot be recovered from the
    `.npz` files alone. We approximate it with a
    rotation-invariant physical size signature: the three
    principal extents of the point cloud (singular values
    of the centred cloud, normalised by point count).

    Two observations of the same physical object have
    nearly identical principal extents regardless of
    viewpoint, so complete-linkage agglomerative
    clustering with a millimetre-scale threshold groups
    them together.

    Limitations
    -----------
    This is a proxy, not ground truth. It over-merges
    genuinely distinct objects of similar size, and it
    can split one object whose visible portion changes a
    lot under occlusion. Grouped results should be read
    as "leakage substantially reduced", not "leakage
    eliminated". The clean fix is a re-export carrying
    `video_id` - see README.

    Parameters
    ----------
    clouds : sequence
        Object array of (Mi, 3) point clouds.

    distance_threshold : float
        Maximum difference in principal extents, in
        meters, for two instances to share a group.
        Default 0.003 (3 mm).

    Returns
    -------
    np.ndarray
        Integer group id per instance, shape (N,).
    """

    from sklearn.cluster import AgglomerativeClustering

    signatures = []

    for cloud in clouds:

        cloud = np.asarray(
            cloud,
            dtype=np.float64,
        )

        centred = cloud - cloud.mean(axis=0)

        # Singular values of the centred cloud are the
        # principal extents; dividing by sqrt(M) makes the
        # value independent of how many points were kept.
        extents = np.linalg.svd(
            centred,
            compute_uv=False,
        ) / np.sqrt(len(cloud))

        signatures.append(extents)

    signatures = np.asarray(signatures)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="complete",
    )

    return clustering.fit_predict(signatures)
