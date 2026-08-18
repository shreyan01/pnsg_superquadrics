"""
YCB-Video dataset loader. Built against the well-documented standard
format (confirmed via the official toolbox and BOP benchmark page):

  <dataset_root>/data/<video_id>/<frame_id>-color.png   RGB
  <dataset_root>/data/<video_id>/<frame_id>-depth.png   16-bit depth (mm-ish, scaled by factor_depth)
  <dataset_root>/data/<video_id>/<frame_id>-label.png   per-pixel class-id mask
  <dataset_root>/data/<video_id>/<frame_id>-meta.mat    cls_indexes, poses, intrinsic_matrix, factor_depth
  <dataset_root>/image_sets/train.txt                    list of "<video_id>/<frame_id>" for training
  <dataset_root>/image_sets/val.txt (or keyframe.txt)     held-out list (name varies by distribution)

HONEST FLAG: this loader is written against the documented/standard
format and has NOT been run against real files (no dataset download
access in the sandbox this was built in). Before trusting it on a real
run, do the sanity check in `inspect_one_frame()` first -- it prints
every field it finds so you can confirm names/shapes match your
specific copy of the dataset before running the full training script.
"""
import os
import numpy as np
import cv2
from scipy.io import loadmat

# YCB-Video was intentionally kept split across its original three
# download archives (data1.zip -> data/, data2.zip -> data2/,
# data3.zip -> data3/) rather than merged into one folder. Every video
# ID is unique across the three (no overlaps), so resolving a video's
# location just means checking each in order.
DATA_SUBDIRS = ['data', 'data2', 'data3']


def resolve_video_dir(dataset_root, video_id):
    """Searches data/, data2/, data3/ (in that order) for this video's
    folder. Returns the full path to the video folder. Raises
    FileNotFoundError with a clear message (listing what WAS searched)
    if the video isn't in any of them -- better than a confusing
    downstream cv2/loadmat error pointing at a path that was never
    going to exist."""
    for subdir in DATA_SUBDIRS:
        candidate = os.path.join(dataset_root, subdir, video_id)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Video '{video_id}' not found in any of {DATA_SUBDIRS} under {dataset_root}")


def read_split_file(dataset_root, split_name):
    """split_name: 'train' or 'val' (or whatever your distribution calls
    the held-out list -- check image_sets/ and adjust if needed)."""
    path = os.path.join(dataset_root, 'image_sets', f'{split_name}.txt')
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def inspect_one_frame(dataset_root, frame_key):
    """Run this FIRST on a real download, before trusting the loader
    below. Prints every field found in the meta.mat file and basic
    image stats, so field-name mismatches surface immediately instead
    of silently producing garbage point clouds."""
    video_id, frame_id = frame_key.split('/')
    video_dir = resolve_video_dir(dataset_root, video_id)
    frame_path = os.path.join(video_dir, frame_id)

    meta = loadmat(frame_path + '-meta.mat')
    print(f'meta.mat fields: {[k for k in meta.keys() if not k.startswith("__")]}')
    for k in meta.keys():
        if not k.startswith('__'):
            v = meta[k]
            print(f'  {k}: shape={getattr(v, "shape", None)} dtype={getattr(v, "dtype", None)}')

    depth = cv2.imread(frame_path + '-depth.png', cv2.IMREAD_UNCHANGED)
    label = cv2.imread(frame_path + '-label.png', cv2.IMREAD_UNCHANGED)
    print(f'depth: shape={depth.shape} dtype={depth.dtype} min={depth.min()} max={depth.max()}')
    print(f'label: shape={label.shape} dtype={label.dtype} unique_ids={np.unique(label)}')


def load_frame(dataset_root, frame_key):
    """Returns (depth_meters, label_mask, intrinsics_3x3, cls_indexes).
    frame_key is 'video_id/frame_id' -- the video's actual location
    (data/, data2/, or data3/) is resolved automatically."""
    video_id, frame_id = frame_key.split('/')
    video_dir = resolve_video_dir(dataset_root, video_id)
    frame_path = os.path.join(video_dir, frame_id)

    meta = loadmat(frame_path + '-meta.mat')

    factor_depth = float(meta['factor_depth'].squeeze())
    intrinsic = meta['intrinsic_matrix'].astype(np.float64)
    cls_indexes = meta['cls_indexes'].squeeze().astype(int)
    if cls_indexes.ndim == 0:
        cls_indexes = np.array([int(cls_indexes)])

    depth_raw = cv2.imread(frame_path + '-depth.png', cv2.IMREAD_UNCHANGED).astype(np.float64)
    depth_m = depth_raw / factor_depth

    label = cv2.imread(frame_path + '-label.png', cv2.IMREAD_UNCHANGED)

    return depth_m, label, intrinsic, cls_indexes


def extract_object_cloud(depth_m, label, intrinsic, class_id, max_points=2000,
                          min_points=80, rng=None, outlier_mad_threshold=3.5):
    """Real pinhole backprojection using REAL sensor depth -- this is
    the step that was stubbed/fake in the RGB-only bridge; here it's
    correct by construction, since depth_m comes from an actual depth
    sensor, not a guessed formula. Returns None if the mask is too
    small (heavily occluded/truncated instance).

    outlier_mad_threshold: real segmentation masks aren't pixel-perfect
    -- a handful of boundary pixels commonly pick up background/adjacent-
    object depth (or invalid sensor readings), and even a few such
    points are enough to make a superquadric fit degenerate (found via
    a real first training run: a small number of contaminated frames
    produced fits with heights in the METERS instead of centimeters,
    with shape exponents pinned at their optimizer bound -- the same
    degenerate-solution signature characterized earlier when stress-
    testing the fitter on synthetic data, here caused by real sensor
    noise instead of a synthetic edge case). Points whose depth is more
    than `outlier_mad_threshold` median-absolute-deviations from the
    masked region's median depth are dropped before backprojection.
    """
    rng = rng or np.random.default_rng(0)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    mask = (label == class_id)
    ys, xs = np.where(mask)
    if len(xs) < min_points:
        return None

    Z_all = depth_m[ys, xs]
    valid = Z_all > 0
    xs, ys, Z_all = xs[valid], ys[valid], Z_all[valid]
    if len(xs) < min_points:
        return None

    # robust outlier rejection: median + MAD, not mean/std (mean/std are
    # themselves corrupted by the very outliers we're trying to remove)
    median_z = np.median(Z_all)
    mad = np.median(np.abs(Z_all - median_z)) + 1e-9
    keep = np.abs(Z_all - median_z) / mad < outlier_mad_threshold
    xs, ys, Z_all = xs[keep], ys[keep], Z_all[keep]
    if len(xs) < min_points:
        return None

    if len(xs) > max_points:
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys, Z_all = xs[idx], ys[idx], Z_all[idx]

    X = (xs - cx) * Z_all / fx
    Y = (ys - cy) * Z_all / fy
    return np.stack([X, Y, Z_all], axis=1)


def iter_frame_objects(dataset_root, frame_key, vocab_mapper, max_points=2000, min_points=80):
    """Generator: yields (vocab_word, point_cloud) for every mapped,
    sufficiently-visible object instance in one frame."""
    depth_m, label, intrinsic, cls_indexes = load_frame(dataset_root, frame_key)
    for class_id in cls_indexes:
        vocab_word = vocab_mapper(int(class_id))
        if vocab_word is None:
            continue
        cloud = extract_object_cloud(depth_m, label, intrinsic, int(class_id),
                                      max_points=max_points, min_points=min_points)
        if cloud is not None:
            yield vocab_word, cloud