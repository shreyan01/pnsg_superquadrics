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
    frame_path = os.path.join(dataset_root, 'data', frame_key)
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
    """Returns (depth_meters, label_mask, intrinsics_3x3, cls_indexes)."""
    frame_path = os.path.join(dataset_root, 'data', frame_key)
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
                          min_points=80, rng=None):
    """Real pinhole backprojection using REAL sensor depth -- this is
    the step that was stubbed/fake in the RGB-only bridge; here it's
    correct by construction, since depth_m comes from an actual depth
    sensor, not a guessed formula. Returns None if the mask is too
    small (heavily occluded/truncated instance)."""
    rng = rng or np.random.default_rng(0)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    mask = (label == class_id)
    ys, xs = np.where(mask)
    if len(xs) < min_points:
        return None

    if len(xs) > max_points:
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys = xs[idx], ys[idx]

    Z = depth_m[ys, xs]
    valid = Z > 0   # zero depth = sensor dropout, must be excluded
    xs, ys, Z = xs[valid], ys[valid], Z[valid]
    if len(xs) < min_points:
        return None

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=1)


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
