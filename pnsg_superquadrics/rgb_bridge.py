"""
The RGB -> our-existing-3D-pipeline bridge. This is the architecture
needed to run our system on a flat photo instead of RGB-D sensor data.
Four stages, and it's important to be precise about which are REAL and
which are STUBBED:

  1. propose_regions_2d()   -- REAL. Classical CV (OpenCV contour
     detection on saliency), no neural network, no downloaded weights.
     Crude compared to a real detector (Grounded-SAM/YOLO-World), but
     genuinely functioning object proposal, not a placeholder.

  2. estimate_depth_STUB()  -- STUBBED, clearly marked as such. Real
     monocular depth estimation requires a trained neural network
     (MiDaS, Depth Anything, etc.) with downloaded weights. This sandbox
     has no internet access to model-hosting services, so this function
     uses a crude photographic heuristic (lower-in-frame + larger-in-
     pixels = closer) instead of an actual learned depth map. This is
     the one stage that CANNOT be made real without either internet
     access to fetch model weights, or a real depth sensor (which is
     what OpenArm + RealSense will provide after Aug 21 -- at which
     point this entire stub is deleted, not improved, because real
     depth sensing makes it unnecessary).

  3. lift_region_to_pointcloud() -- REAL. Standard pinhole camera
     back-projection math (pixel + depth + intrinsics -> 3D point).
     Legitimate regardless of whether the depth INPUT is real or
     stubbed -- this is just geometry.

  4. reproject_bbox_to_image()   -- REAL. Standard camera projection,
     inverse of #3. Takes our pipeline's fitted 3D bounding box and
     projects it back into 2D pixel coordinates for drawing.

Stages 1, 3, 4 are genuine, correct, and would work unchanged once stage
2 is swapped for a real depth model or real sensor. Stage 2 is the
single honest gap, isolated on purpose so it's a one-function swap, not
a rewrite.
"""
import numpy as np
import cv2


# --- Stage 1: 2D region proposal (REAL, classical CV) ---------------------

def propose_regions_2d(image_bgr, min_area=1500, max_regions=10):
    """Classical CV object proposal: no neural network. Lighter dilation
    (3x3, 1 iteration) than the original version -- the aggressive 9x9/2x
    version was merging separate objects' edges into single giant blobs,
    found by inspecting the intermediate edge map directly."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        regions.append({'bbox_px': (x, y, w, h), 'mask': mask, 'area': area})

    regions.sort(key=lambda r: -r['area'])
    return regions[:max_regions]


# --- Stage 2: depth estimation (STUB -- see module docstring) -------------

def estimate_depth_STUB(image_shape, region, focal_length_px=800.0,
                          assumed_real_width_m=0.08):
    """
    NOT a real depth estimate. Placeholder using a crude photographic
    heuristic: assumes the object's real-world width is roughly
    `assumed_real_width_m` (a single global guess -- genuinely wrong for
    a vase vs. a candle, listed here as exactly the kind of error a real
    depth model or real sensor would not make), and derives an implied
    depth from the pinhole relationship (pixel_width = focal_length *
    real_width / depth). This exists ONLY so the rest of the bridge
    architecture (stages 3+4, which are real) has something to operate
    on for a demonstration; the resulting 3D reconstruction's ABSOLUTE
    scale and relative depth ordering should not be trusted.
    """
    x, y, w, h = region['bbox_px']
    pixel_width = max(w, 1)
    depth_m = (focal_length_px * assumed_real_width_m) / pixel_width
    return depth_m


# Per-category priors refined by inspecting several real reference photos
# per category (via image search + vision) rather than one memorized
# guess. Honest caveat: aspect ratio / proportions are directly visible
# in any photo and are well-grounded by this inspection; absolute scale
# (meters) still relies on general knowledge of typical object sizes,
# since no photo carries a metric reference. Still not a measured
# dataset -- an informed estimate range, now checked against real images
# instead of asserted from memory alone.
DIMENSION_PRIORS = {
    'mug':        0.08,   # ~7-9cm diameter, confirmed roughly-cylindrical via reference photos
    'bowl':       0.12,   # ~10-14cm diameter, confirmed shallow/wide proportions via reference photos
    'candle_jar': 0.065,  # ~5.5-7.5cm diameter, confirmed squat-jar proportions via reference photos
    'vase':       0.10,
    'plate':      0.20,
    'bottle':     0.07,
}
DEFAULT_WIDTH = 0.08


def rough_2d_preclassify(region, image_shape):
    """Cheap, classical (non-3D) guess at rough category from 2D box
    shape alone -- aspect ratio and relative size -- used ONLY to pick a
    better real-world-size prior for depth estimation, not as a final
    label. A tall/thin box suggests bottle/candle; a wide/flat box
    suggests bowl/plate; roughly square suggests mug/vase. Crude by
    design -- this is a cheap pre-pass, not a classifier."""
    x, y, w, h = region['bbox_px']
    aspect = h / max(w, 1)
    img_h, img_w = image_shape[:2]
    rel_area = (w * h) / (img_w * img_h)

    if aspect > 2.2:
        return 'bottle' if rel_area < 0.15 else 'candle_jar'
    elif aspect < 0.6:
        return 'plate' if rel_area > 0.1 else 'bowl'
    else:
        return 'mug' if rel_area < 0.2 else 'vase'


def estimate_depth_with_prior(image_shape, region, focal_length_px=800.0):
    """Two-pass depth estimation: use the cheap 2D pre-classifier to pick
    a category-appropriate real-world width from DIMENSION_PRIORS instead
    of one global guess, then apply the same pinhole formula as
    estimate_depth_STUB. Still a stub (still not a real depth model --
    still no verified per-object measurement) but a meaningfully better-
    calibrated one, addressing the single biggest source of error in the
    original version."""
    guessed_category = rough_2d_preclassify(region, image_shape)
    real_width = DIMENSION_PRIORS.get(guessed_category, DEFAULT_WIDTH)
    x, y, w, h = region['bbox_px']
    pixel_width = max(w, 1)
    depth_m = (focal_length_px * real_width) / pixel_width
    return depth_m, guessed_category


# --- Stage 3: 2D region + depth -> 3D point cloud (REAL geometry) ---------

def lift_region_to_pointcloud(region, depth_m, image_shape,
                                focal_length_px=800.0, n_samples=800, rng=None):
    """Standard pinhole back-projection: for each sampled pixel in the
    region's mask, compute its 3D position given an assumed/estimated
    depth. This is correct, real camera geometry -- it just inherits
    whatever error exists in the depth INPUT (stub or real)."""
    rng = rng or np.random.default_rng(0)
    h_img, w_img = image_shape[:2]
    cx_img, cy_img = w_img / 2.0, h_img / 2.0

    ys, xs = np.where(region['mask'] > 0)
    if len(xs) == 0:
        return np.zeros((0, 3))
    idx = rng.choice(len(xs), size=min(n_samples, len(xs)), replace=False)
    xs, ys = xs[idx], ys[idx]

    # small synthetic depth variation across the region so it isn't a
    # perfectly flat plane (real objects have some depth extent) --
    # explicitly another stand-in, proportional to the region's own pixel
    # size, not a real measurement
    x, y, w, h = region['bbox_px']
    depth_jitter = rng.normal(0, 0.15 * (depth_m * max(w, h) / focal_length_px), len(xs))

    Z = depth_m + depth_jitter
    X = (xs - cx_img) * Z / focal_length_px
    Y = (ys - cy_img) * Z / focal_length_px
    return np.stack([X, Y, Z], axis=1)


# --- Stage 4: 3D bbox -> 2D pixel projection (REAL geometry) --------------

def reproject_bbox_to_image(bbox_params, image_shape, focal_length_px=800.0):
    """Inverse of stage 3: projects the pipeline's fitted 3D bounding box
    corners back into image pixel coordinates, for drawing on the
    original photo. Real math, correct regardless of whether the 3D
    points it's summarizing came from a real or stubbed depth stage."""
    from scipy.spatial.transform import Rotation as R
    h_img, w_img = image_shape[:2]
    cx_img, cy_img = w_img / 2.0, h_img / 2.0

    cx, cy, cz = bbox_params['center']
    ex, ey, ez = bbox_params['half_extents']
    roll, pitch, yaw = bbox_params['rotation_rpy']
    rot = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()

    local_corners = np.array([
        [-ex, -ey, -ez], [ex, -ey, -ez], [ex, ey, -ez], [-ex, ey, -ez],
        [-ex, -ey, ez], [ex, -ey, ez], [ex, ey, ez], [-ex, ey, ez],
    ])
    world_corners = local_corners @ rot.T + np.array([cx, cy, cz])

    Z = np.clip(world_corners[:, 2], 0.05, None)  # avoid divide-by-zero/behind-camera
    px = world_corners[:, 0] * focal_length_px / Z + cx_img
    py = world_corners[:, 1] * focal_length_px / Z + cy_img

    x_min, x_max = int(px.min()), int(px.max())
    y_min, y_max = int(py.min()), int(py.max())
    return (max(x_min, 0), max(y_min, 0), min(x_max, w_img), min(y_max, h_img))
