import os
import glob
import numpy as np
from scipy.io import loadmat
from ycbv_training.ycb_dataset_loader import extract_object_cloud, resolve_video_dir

def get_pose_for_class(cls_indexes, poses, class_id):
    cls_indexes = np.atleast_1d(cls_indexes.squeeze())
    matches = np.where(cls_indexes == class_id)[0]
    if len(matches) == 0: return None
    k = matches[0]
    pose = poses[:, :, k]
    return pose[:, :3], pose[:, 3]

def discover_frames(dataset_root, video_id):
    video_dir = resolve_video_dir(dataset_root, video_id)
    pattern = os.path.join(video_dir, '*-meta.mat')
    files = glob.glob(pattern)
    return sorted(os.path.basename(f).replace('-meta.mat', '') for f in files)

def aggregate_multiview_cloud(dataset_root, video_id, class_id, frame_ids=None,
                                frame_stride=10, max_points_per_frame=500,
                                max_total_points=4000, rng=None, with_color=True):
    """Returns (point_cloud, color_features) where color_features is the
    circular-mean hue + mean saturation averaged across the SAME window
    of frames used for the point cloud -- more robust than any single
    frame's lighting/reflection, same principle as multi-view shape
    aggregation. Returns (None, None) if nothing usable was found;
    color_features is None (shape-only) if with_color=False or every
    frame's color image was missing."""
    rng = rng or np.random.default_rng(0)
    if frame_ids is None:
        frame_ids = discover_frames(dataset_root, video_id)[::frame_stride]

    all_points = []
    hue_sin_sum, hue_cos_sum, sat_sum, n_color = 0.0, 0.0, 0.0, 0

    for frame_id in frame_ids:
        video_dir = resolve_video_dir(dataset_root, video_id)
        frame_path = os.path.join(video_dir, frame_id)
        try:
            meta = loadmat(frame_path + '-meta.mat')
        except Exception:
            continue

        cls_indexes = meta['cls_indexes']
        pose_info = get_pose_for_class(cls_indexes, meta['poses'], class_id)
        if pose_info is None: continue
        R, t = pose_info

        import cv2
        depth_raw = cv2.imread(frame_path + '-depth.png', cv2.IMREAD_UNCHANGED)
        label = cv2.imread(frame_path + '-label.png', cv2.IMREAD_UNCHANGED)
        if depth_raw is None or label is None: continue
        factor_depth = float(meta['factor_depth'].squeeze())
        depth_m = depth_raw.astype(np.float64) / factor_depth
        intrinsic = meta['intrinsic_matrix'].astype(np.float64)

        cloud_camera = extract_object_cloud(depth_m, label, intrinsic, class_id,
                                             max_points=max_points_per_frame, rng=rng)
        if cloud_camera is None: continue

        cloud_object = (cloud_camera - t) @ R
        all_points.append(cloud_object)

        if with_color:
            from color_features import extract_object_color_features
            color_img = cv2.imread(frame_path + '-color.jpg', cv2.IMREAD_COLOR)
            if color_img is not None:
                cf = extract_object_color_features(color_img, label, class_id)
                if cf is not None:
                    hue, sat = cf
                    hue_rad = np.deg2rad(hue)
                    hue_sin_sum += np.sin(hue_rad); hue_cos_sum += np.cos(hue_rad)
                    sat_sum += sat; n_color += 1

    if not all_points: return None, None
    merged = np.vstack(all_points)
    if len(merged) > max_total_points:
        idx = rng.choice(len(merged), size=max_total_points, replace=False)
        merged = merged[idx]

    color_features = None
    if n_color > 0:
        mean_hue = np.rad2deg(np.arctan2(hue_sin_sum / n_color, hue_cos_sum / n_color)) % 360.0
        mean_sat = sat_sum / n_color
        color_features = (float(mean_hue), float(mean_sat))

    return merged, color_features