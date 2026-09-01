import numpy as np
from scipy.optimize import least_squares

from superquadric import (
    world_to_local, local_to_world, radial_residual, vec_to_params,
    add_noise, PARAM_ORDER, BOUNDS_LOW, BOUNDS_HIGH, _fexp,
)
from deformation import bend, inverse_bend

BENT_PARAM_ORDER = PARAM_ORDER + ['bend_k']
BENT_BOUNDS_LOW = dict(BOUNDS_LOW, bend_k=-60.0)
BENT_BOUNDS_HIGH = dict(BOUNDS_HIGH, bend_k=60.0)


def sample_bent_rod_surface(rod_params, bend_k, pose, n_points=1500, rng=None):
    rng = rng or np.random.default_rng(0)
    eta = rng.uniform(-np.pi / 2, np.pi / 2, n_points)
    omega = rng.uniform(-np.pi, np.pi, n_points)

    def sc(a, e): return _fexp(np.cos(a), e)
    def ss(a, e): return _fexp(np.sin(a), e)

    e1, e2 = rod_params['eps1'], rod_params['eps2']
    a1, a2, a3 = rod_params['a1'], rod_params['a2'], rod_params['a3']
    x = a1 * sc(eta, e1) * sc(omega, e2)
    y = a2 * sc(eta, e1) * ss(omega, e2)
    z = a3 * ss(eta, e1)
    local = np.stack([x, y, z], axis=1)

    bent_local = bend(local, k=bend_k, axis='z')
    return local_to_world(bent_local, pose['cx'], pose['cy'], pose['cz'],
                           pose['roll'], pose['pitch'], pose['yaw'])


def bent_radial_residual(points_world, params_with_k):
    k = params_with_k['bend_k']
    local = world_to_local(points_world, params_with_k['cx'], params_with_k['cy'],
                            params_with_k['cz'], params_with_k['roll'],
                            params_with_k['pitch'], params_with_k['yaw'])
    unbent_local = inverse_bend(local, k, axis='z')
    identity_pose_params = dict(params_with_k, cx=0.0, cy=0.0, cz=0.0,
                                 roll=0.0, pitch=0.0, yaw=0.0)
    return radial_residual(unbent_local, identity_pose_params)


def initial_guess_bent(points, bend_k_init=8.0):
    """bend_k_init defaults to a NONZERO value (8.0), not 0.0. Found
    empirically: starting the optimizer exactly at k=0 traps it in a
    flat-gradient region (the bend function's own special-case at
    k=0 has near-zero sensitivity there), so it never discovers that
    increasing curvature improves the fit -- it distorts eps1/eps2
    instead and reports 'straight' every time, even on a known-curved
    test case. Any nonzero starting point (tested at +5, -5, +10, all
    converging to the same ~13.5 on an 18.0-ground-truth case) escapes
    this trap; the specific value 8.0 isn't special, just reliably
    nonzero. NOTE: this has only been validated on synthetic data --
    NOT yet tested against real YCB-Video handle point clouds, so its
    practical value on real sensor data is still unverified."""
    centroid = points.mean(axis=0)
    extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
    extents = np.clip(extents, 1e-3, None)
    return {
        'a1': extents[0], 'a2': extents[1], 'a3': extents[2],
        'eps1': 1.0, 'eps2': 1.0,
        'cx': centroid[0], 'cy': centroid[1], 'cz': centroid[2],
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'bend_k': bend_k_init,
    }


def params_to_vec_bent(params):
    return np.array([params[k] for k in BENT_PARAM_ORDER])


def vec_to_params_bent(vec):
    return dict(zip(BENT_PARAM_ORDER, vec))


def fit_bent_superquadric(points, init=None, verbose=0):
    if init is None:
        init = initial_guess_bent(points)

    x0 = params_to_vec_bent(init)
    lo = np.array([BENT_BOUNDS_LOW[k] for k in BENT_PARAM_ORDER])
    hi = np.array([BENT_BOUNDS_HIGH[k] for k in BENT_PARAM_ORDER])

    def resid_fn(vec):
        p = vec_to_params_bent(vec)
        return bent_radial_residual(points, p)

    result = least_squares(
        resid_fn, x0, bounds=(lo, hi),
        method='trf', loss='soft_l1', f_scale=0.05,
        max_nfev=5000, verbose=verbose,
    )

    fitted = vec_to_params_bent(result.x)
    final_resid = resid_fn(result.x)
    info = {
        'success': result.success,
        'rmse': float(np.sqrt(np.mean(final_resid ** 2))),
        'nfev': result.nfev,
    }
    return fitted, info