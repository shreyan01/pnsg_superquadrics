import os
import numpy as np
import cv2
from scipy.io import loadmat

DATA_SUBDIRS = ['data', 'data2', 'data3']

def resolve_video_dir(dataset_root, video_id):
    for subdir in DATA_SUBDIRS:
        direct = os.path.join(dataset_root, subdir, video_id)
        if os.path.isdir(direct):
            return direct
        nested = os.path.join(dataset_root, subdir, 'data', video_id)
        if os.path.isdir(nested):
            return nested
    raise FileNotFoundError(f"Video '{video_id}' not found in any of {DATA_SUBDIRS} (flat or nested) under {dataset_root}")

def read_split_file(dataset_root, split_name):
    path = os.path.join(dataset_root, 'image_sets', f'{split_name}.txt')
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def inspect_one_frame(dataset_root, frame_key):
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
    video_id, frame_id = frame_key.split('/')
    video_dir = resolve_video_dir(dataset_root, video_id)
    frame_path = os.path.join(video_dir, frame_id)
    meta = loadmat(frame_path + '-meta.mat')
    factor_depth = float(meta['factor_depth'].squeeze())
    intrinsic = meta['intrinsic_matrix'].astype(np.float64)
    cls_indexes = meta['cls_indexes'].squeeze().astype(int)
    if cls_indexes.ndim == 0:
        cls_indexes = np.array([int(cls_indexes)])
    depth_img = cv2.imread(frame_path + '-depth.png', cv2.IMREAD_UNCHANGED)
    label = cv2.imread(frame_path + '-label.png', cv2.IMREAD_UNCHANGED)
    if depth_img is None or label is None:
        raise FileNotFoundError(f"Failed to read depth/label image for '{frame_path}' (file missing or corrupted)")
    depth_m = depth_img.astype(np.float64) / factor_depth
    return depth_m, label, intrinsic, cls_indexes

def load_color_image(dataset_root, frame_key):
    """Loads the -color.jpg frame, never touched by the pipeline until
    now (depth+label only). Returns None if missing/corrupted rather
    than raising -- color is a genuine enhancement, not something that
    should crash a frame that's otherwise perfectly usable for shape
    fitting; callers should treat a None color image as 'extract shape
    only, skip color for this instance' rather than a hard failure."""
    video_id, frame_id = frame_key.split('/')
    video_dir = resolve_video_dir(dataset_root, video_id)
    frame_path = os.path.join(video_dir, frame_id)
    color_img = cv2.imread(frame_path + '-color.jpg', cv2.IMREAD_COLOR)
    return color_img  # BGR, uint8; None if missing

def extract_object_cloud(depth_m, label, intrinsic, class_id, max_points=2000, min_points=80, rng=None, outlier_mad_threshold=3.5):
    rng = rng or np.random.default_rng(0)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    mask = (label == class_id)
    ys, xs = np.where(mask)
    if len(xs) < min_points: return None
    Z_all = depth_m[ys, xs]
    valid = Z_all > 0
    xs, ys, Z_all = xs[valid], ys[valid], Z_all[valid]
    if len(xs) < min_points: return None
    median_z = np.median(Z_all)
    mad = np.median(np.abs(Z_all - median_z)) + 1e-9
    keep = np.abs(Z_all - median_z) / mad < outlier_mad_threshold
    xs, ys, Z_all = xs[keep], ys[keep], Z_all[keep]
    if len(xs) < min_points: return None
    if len(xs) > max_points:
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys, Z_all = xs[idx], ys[idx], Z_all[idx]
    X = (xs - cx) * Z_all / fx
    Y = (ys - cy) * Z_all / fy
    return np.stack([X, Y, Z_all], axis=1)

def iter_frame_objects(dataset_root, frame_key, vocab_mapper, max_points=2000, min_points=80,
                        with_color=True):
    """Yields (vocab_word, cloud, color_features) per object, where
    color_features is (hue_degrees, saturation) or None if the color
    image is missing/failed or with_color=False -- callers should
    handle None gracefully (shape-only fallback), not crash."""
    from color_features import extract_object_color_features
    depth_m, label, intrinsic, cls_indexes = load_frame(dataset_root, frame_key)
    color_img = load_color_image(dataset_root, frame_key) if with_color else None
    for class_id in cls_indexes:
        vocab_word = vocab_mapper(int(class_id))
        if vocab_word is None: continue
        cloud = extract_object_cloud(depth_m, label, intrinsic, int(class_id), max_points=max_points, min_points=min_points)
        if cloud is None: continue
        color_features = None
        if color_img is not None:
            color_features = extract_object_color_features(color_img, label, int(class_id))
        yield vocab_word, cloud, color_features