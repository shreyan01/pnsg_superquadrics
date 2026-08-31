"""
Classical (non-neural) color feature extraction, aligned to the same
pixel mask already used for depth-based point extraction. Adds mean
hue and mean saturation as two new, fully interpretable dimensions to
each learned mode -- "this word's dominant color is yellow (hue=45deg)"
is just as explainable as "a1=51mm", not a departure from the project's
thesis.

Targets a specific, already-diagnosed failure: mustard_bottle (yellow)
vs bleach_cleanser (white/blue) are being confused by shape alone
because a single partial view under-constrains geometry -- but they are
trivially separable by color, which we've never used until now.
"""
import numpy as np
import cv2


def extract_object_color_features(color_image_bgr, label, class_id):
    """color_image_bgr: the raw -color.jpg frame, already loaded via
    cv2.imread (BGR, uint8). label: same per-pixel class mask used for
    depth extraction (guarantees pixel alignment -- color.jpg and
    depth/label share the same resolution and pixel grid in YCB-Video).
    Returns (mean_hue_degrees, mean_saturation) or None if the mask is
    empty. Hue is circular (0=360), so a naive mean is wrong near the
    wraparound -- averaged via its circular (sin/cos) representation
    instead of linearly."""
    mask = (label == class_id)
    if mask.sum() < 10:
        return None

    hsv = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0][mask].astype(np.float64) * 2.0   # OpenCV hue is 0-179, scale to 0-360 degrees
    s = hsv[:, :, 1][mask].astype(np.float64) / 255.0  # normalize to 0-1

    # circular mean for hue (a simple arithmetic mean is wrong for a
    # wraparound quantity -- e.g. mean of 350deg and 10deg should be
    # 0deg/360deg, not 180deg)
    angles_rad = np.deg2rad(h)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    mean_hue = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0

    mean_sat = float(np.mean(s))
    return mean_hue, mean_sat