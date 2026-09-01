import numpy as np


def bend(points_local, k, axis='z'):
    if abs(k) < 1e-8:
        return points_local.copy()
    x, y, z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
    if axis != 'z':
        raise NotImplementedError('only z-axis bending implemented for now')
    r = 1.0 / k
    theta = k * z
    x_new = x + r * (np.cos(theta) - 1.0)
    z_new = r * np.sin(theta)
    y_new = y
    return np.stack([x_new, y_new, z_new], axis=1)


def inverse_bend(points_local, k, axis='z'):
    if abs(k) < 1e-8:
        return points_local.copy()
    x, y, z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
    r = 1.0 / k
    ratio = np.clip(z / r, -0.999, 0.999)
    theta = np.arcsin(ratio)
    x_orig = x - r * (np.cos(theta) - 1.0)
    z_orig = theta / k
    y_orig = y
    return np.stack([x_orig, y_orig, z_orig], axis=1)


def taper(points_local, tx, ty, axis='z'):
    x, y, z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
    if axis != 'z':
        raise NotImplementedError('only z-axis tapering implemented for now')
    fx = 1.0 + tx * z
    fy = 1.0 + ty * z
    return np.stack([x * fx, y * fy, z], axis=1)