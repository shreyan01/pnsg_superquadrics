"""
Multi-view aggregation: the real fix for the partial-view bias found
tonight (cans fitting as elongated/boxy instead of round, because a
single camera frame only ever sees one side of a cylinder).

Mechanism: YCB-Video's meta.mat gives each object's pose (R, t) mapping
OBJECT-frame coordinates into CAMERA-frame coordinates, per frame:
    P_camera = R @ P_object + t
Since the object is physically static within one video (only the camera
moves), inverting this per-frame transform:
    P_object = R^T @ (P_camera - t)
maps every frame's partial view into ONE SHARED object-centered frame.
Union several frames spread across a video's camera motion, and you
approximate a real multi-angle scan -- instead of fitting one shape per
frame (many biased, partial examples), you fit ONE shape per (video,
object) pair, from much more complete geometry.

HONEST FLAG: pose indexing verified against the documented format
(cls_indexes and poses are parallel arrays -- poses[:,:,k] corresponds
to cls_indexes[k]) but not yet run against real files in this sandbox.
Run validate_pose_aggregation() on one real (video, class) pair FIRST,
same practice as ycb_dataset_loader.py's inspect_one_frame(), before
trusting a full run.
"""
import os
import glob
import numpy as np
from scipy.io import loadmat

from ycbv_training.ycb_dataset_loader import extract_object_cloud, resolve_video_dir


def get_pose_for_class(cls_indexes, poses, class_id):
    """cls_indexes and poses are parallel arrays: poses[:,:,k] is the
    pose for cls_indexes[k]. Returns (R, t) or None if this class isn't
    in this frame."""
    cls_indexes = np.atleast_1d(cls_indexes.squeeze())
    matches = np.where(cls_indexes == class_id)[0]
    if len(matches) == 0:
        return None
    k = matches[0]
    pose = poses[:, :, k]  # 3x4
    R, t = pose[:, :3], pose[:, 3]
    return R, t


def discover_frames(dataset_root, video_id):
    video_dir = resolve_video_dir(dataset_root, video_id)
    pattern = os.path.join(video_dir, '*-meta.mat')
    files = glob.glob(pattern)
    return sorted(os.path.basename(f).replace('-meta.mat', '') for f in files)


def aggregate_multiview_cloud(dataset_root, video_id, class_id, frame_ids=None,
                                frame_stride=10, max_points_per_frame=500,
                                max_total_points=4000, rng=None):
    """Aggregates one object's point cloud across multiple frames of one
    video into a single OBJECT-CENTERED cloud. Returns None if the class
    never appears in the sampled frames.

    frame_stride: don't use every frame (consecutive frames are near-
    identical camera angles, adding redundant points, not new coverage)
    -- sample every Nth frame instead, for real angular diversity per
    point spent.
    """
    rng = rng or np.random.default_rng(0)
    if frame_ids is None:
        frame_ids = discover_frames(dataset_root, video_id)[::frame_stride]

    all_points = []
    for frame_id in frame_ids:
        video_dir = resolve_video_dir(dataset_root, video_id)
        frame_path = os.path.join(video_dir, frame_id)
        try:
            meta = loadmat(frame_path + '-meta.mat')
        except Exception:
            continue

        cls_indexes = meta['cls_indexes']
        pose_info = get_pose_for_class(cls_indexes, meta['poses'], class_id)
        if pose_info is None:
            continue
        R, t = pose_info

        import cv2
        depth_raw = cv2.imread(frame_path + '-depth.png', cv2.IMREAD_UNCHANGED)
        label = cv2.imread(frame_path + '-label.png', cv2.IMREAD_UNCHANGED)
        if depth_raw is None or label is None:
            continue
        factor_depth = float(meta['factor_depth'].squeeze())
        depth_m = depth_raw.astype(np.float64) / factor_depth
        intrinsic = meta['intrinsic_matrix'].astype(np.float64)

        cloud_camera = extract_object_cloud(depth_m, label, intrinsic, class_id,
                                             max_points=max_points_per_frame, rng=rng)
        if cloud_camera is None:
            continue

        # the real fix: transform this frame's partial view into the
        # SHARED object-centered frame using this frame's real pose
        cloud_object = (cloud_camera - t) @ R   # R^T @ (P - t), via right-multiply by R
        all_points.append(cloud_object)

    if not all_points:
        return None

    merged = np.vstack(all_points)
    if len(merged) > max_total_points:
        idx = rng.choice(len(merged), size=max_total_points, replace=False)
        merged = merged[idx]
    return merged


def validate_pose_aggregation(dataset_root, video_id, class_id, verbose=True):
    """Run this FIRST on one real (video, class) pair before trusting a
    full run. Compares single-frame vs multi-view point spread -- if
    aggregation is working, the multi-view cloud should span a visibly
    wider range (more of the object's true extent covered) than any one
    frame alone, since different frames see different sides."""
    frame_ids = discover_frames(dataset_root, video_id)
    single_cloud = None
    for fid in frame_ids[:20]:
        video_dir = resolve_video_dir(dataset_root, video_id)
        frame_path = os.path.join(video_dir, fid)
        meta = loadmat(frame_path + '-meta.mat')
        if class_id not in np.atleast_1d(meta['cls_indexes'].squeeze()):
            continue
        import cv2
        depth_raw = cv2.imread(frame_path + '-depth.png', cv2.IMREAD_UNCHANGED)
        label = cv2.imread(frame_path + '-label.png', cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float64) / float(meta['factor_depth'].squeeze())
        single_cloud = extract_object_cloud(depth_m, label, meta['intrinsic_matrix'], class_id)
        break

    multi_cloud = aggregate_multiview_cloud(dataset_root, video_id, class_id)

    if verbose:
        if single_cloud is not None:
            print(f'Single frame ({fid}): {len(single_cloud)} points, '
                  f'camera-frame extent: {(single_cloud.max(0)-single_cloud.min(0)).round(3)}')
        if multi_cloud is not None:
            print(f'Multi-view aggregated: {len(multi_cloud)} points, '
                  f'object-frame extent: {(multi_cloud.max(0)-multi_cloud.min(0)).round(3)}')
            print('(object-frame extent along the AXIS THE OBJECT IS ROTATIONALLY '
                  'SYMMETRIC ABOUT should now look much more like a full-coverage '
                  'blob than a thin one-sided slice)')
        else:
            print('No multi-view cloud produced -- check class_id/video_id are correct.')

    return single_cloud, multi_cloud