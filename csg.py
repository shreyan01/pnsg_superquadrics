"""
CSG union of two superquadrics (each optionally Barr-bent), fit JOINTLY
from one raw, unlabeled point cloud. This is structurally different from
iterative_segment.py's approach: that method ASSIGNS each point to one
primitive or the other (E-step) then refits each SEPARATELY (M-step) --
which is exactly why the registry ends up with two separately-scored
parts multiplied together, the mechanism behind tonight's mug->bottle
bug (one noisy part's bad score drags down an otherwise-good match).

This module instead treats "the point cloud is the union of two
primitives" as ONE joint optimization: every point only needs to be
well-explained by WHICHEVER primitive is closer, and both primitives'
parameters are solved for simultaneously against the whole cloud. There
is no hard assignment step, no separate per-part score to multiply --
just one compound residual and one fit.

SCOPE, stated explicitly: only UNION is implemented. Subtraction (for a
genuinely hollow interior -- a real genus-1 hole, still an open
limitation from way earlier tonight) needs a complement operation with
different, trickier math, and is NOT implemented here -- shipping that
without careful validation felt like the wrong tradeoff at this hour.
Intersection is defined for completeness but similarly untested.

HONEST STATUS: validated only against synthetic ground truth (see
test_csg_union.py). NOT yet run against real YCB-Video point clouds.
"""
import numpy as np
from scipy.optimize import least_squares

from superquadric import world_to_local, PARAM_ORDER, BOUNDS_LOW, BOUNDS_HIGH
from fit_bent import BENT_PARAM_ORDER, BENT_BOUNDS_LOW, BENT_BOUNDS_HIGH, bent_radial_residual

N_PARAMS = len(BENT_PARAM_ORDER)  # 12 per primitive (11 base + bend_k)


def per_primitive_residual(points, params):
    """Per-point residual for ONE (possibly bent) primitive -- reuses
    bent_radial_residual, which already correctly falls back to the
    plain (unbent) case when bend_k is near zero."""
    return bent_radial_residual(points, params)


def union_residual(points, params_A, params_B):
    """The core CSG union residual: each point only needs to be well-
    explained by WHICHEVER primitive is closer -- no assignment, no
    multiplication of separate scores, just an elementwise minimum over
    two smooth residual fields. This is what lets a well-fit body
    'cover' for a noisier handle fit (or vice versa) instead of one bad
    part dragging down a combined score, which is the actual mechanism
    behind tonight's mug->bottle confusion."""
    resid_A = per_primitive_residual(points, params_A)
    resid_B = per_primitive_residual(points, params_B)
    return np.minimum(np.abs(resid_A), np.abs(resid_B))


def _pack(params_A, params_B):
    return np.concatenate([
        np.array([params_A[k] for k in BENT_PARAM_ORDER]),
        np.array([params_B[k] for k in BENT_PARAM_ORDER]),
    ])


def _unpack(vec):
    a_vec, b_vec = vec[:N_PARAMS], vec[N_PARAMS:]
    return (dict(zip(BENT_PARAM_ORDER, a_vec)), dict(zip(BENT_PARAM_ORDER, b_vec)))


def initial_guess_two_primitives(points, offset_hint=None):
    """Cheap initialization: both primitives start centered on the whole
    cloud's centroid/extent (a crude but workable starting point for
    joint optimization to refine from), offset apart slightly if a hint
    is given, with a nonzero bend_k for each (same fix validated earlier
    tonight -- k=0 is a flat-gradient trap for the optimizer)."""
    centroid = points.mean(axis=0)
    extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
    extents = np.clip(extents, 1e-3, None)

    base = {
        'a1': extents[0] * 0.7, 'a2': extents[1] * 0.7, 'a3': extents[2] * 0.7,
        'eps1': 1.0, 'eps2': 1.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'bend_k': 8.0,
    }
    offset = offset_hint if offset_hint is not None else np.array([extents[0] * 0.5, 0, 0])

    params_A = dict(base, cx=centroid[0] - offset[0] / 2, cy=centroid[1] - offset[1] / 2,
                     cz=centroid[2] - offset[2] / 2)
    params_B = dict(base, a1=extents[0] * 0.25, a2=extents[1] * 0.25, a3=extents[2] * 0.25,
                     cx=centroid[0] + offset[0] / 2, cy=centroid[1] + offset[1] / 2,
                     cz=centroid[2] + offset[2] / 2)
    return params_A, params_B


def fit_csg_union(points, init_A=None, init_B=None, max_nfev=8000, verbose=0):
    """Jointly fits TWO primitives to explain the WHOLE point cloud as
    their union -- one optimization, one compound residual, no separate
    segmentation step. Returns (params_A, params_B, info)."""
    if init_A is None or init_B is None:
        init_A, init_B = initial_guess_two_primitives(points)

    x0 = _pack(init_A, init_B)
    lo = np.concatenate([[BENT_BOUNDS_LOW[k] for k in BENT_PARAM_ORDER]] * 2)
    hi = np.concatenate([[BENT_BOUNDS_HIGH[k] for k in BENT_PARAM_ORDER]] * 2)

    def resid_fn(vec):
        pa, pb = _unpack(vec)
        return union_residual(points, pa, pb)

    result = least_squares(
        resid_fn, x0, bounds=(lo, hi),
        method='trf', loss='soft_l1', f_scale=0.05,
        max_nfev=max_nfev, verbose=verbose,
    )

    fitted_A, fitted_B = _unpack(result.x)
    final_resid = resid_fn(result.x)
    info = {
        'success': result.success,
        'rmse': float(np.sqrt(np.mean(final_resid ** 2))),
        'nfev': result.nfev,
    }
    return fitted_A, fitted_B, info


def csg_fit_two_part(cloud, min_relative_size=0.08, min_points=50):
    """Pipeline-compatible bridge: same (params_a, params_b, assignment)
    call shape as iterative_two_part_segment(), so it's a drop-in
    replacement anywhere that function is used -- this is what makes
    'try CSG, revert to adjacency-graph segmentation if it's not better'
    a one-line swap rather than two divergent pipelines.

    Falls back to single-part (params_b=None) if the fitted secondary
    primitive is physically implausible or negligibly small relative to
    the dominant one (i.e. the object was probably single-part to begin
    with, and CSG found a spurious tiny second primitive fitting noise).
    assignment is always returned as None -- CSG's joint fit doesn't
    produce a per-point hard assignment the way iterative reassignment
    does, and nothing downstream in the real pipeline uses it."""
    from superquadric import is_physically_plausible

    if len(cloud) < min_points:
        return None, None, None

    init_A, init_B = initial_guess_two_primitives(cloud)
    fitted_A, fitted_B, info = fit_csg_union(cloud, init_A, init_B)

    vol_A = fitted_A['a1'] * fitted_A['a2'] * fitted_A['a3']
    vol_B = fitted_B['a1'] * fitted_B['a2'] * fitted_B['a3']
    if vol_B > vol_A:
        fitted_A, fitted_B = fitted_B, fitted_A
        vol_A, vol_B = vol_B, vol_A

    if not is_physically_plausible(fitted_A):
        return None, None, None  # dominant fit itself failed -- nothing usable

    if not is_physically_plausible(fitted_B) or vol_B < min_relative_size * vol_A:
        return fitted_A, None, None

    return fitted_A, fitted_B, None