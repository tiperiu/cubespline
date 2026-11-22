"""Small ellipsoid: ellipsoid with semi-axes (0.4, 0.25, 0.18) centred at (0.1, 0.0, 0.05), rotated 20 degrees about z."""
import numpy as np

NAME = "ellipsoid small"
CENTER = np.array([0.1, 0.0, 0.05])
AXES = np.array([0.4, 0.25, 0.18])


def _rot(axis, deg):
    """Rotation matrix about the x, y or z axis."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R

R = _rot("z", 20)


def g(X):
    """Approximate signed distance (exact on the surface, negative inside). X: (..., 3).
    k0 (k0 - 1) / k1 with k0 = |x / a|, k1 = |x / a^2|: zero on the ellipsoid, with
    gradient of length about 1 near it."""
    L = (np.asarray(X) - CENTER) @ R            # into the ellipsoid's own frame
    k0 = np.linalg.norm(L / AXES, axis=-1)
    k1 = np.linalg.norm(L / AXES ** 2, axis=-1)
    # at the centre itself k0 = k1 = 0 and the ratio is 0 / 0: use the inside value
    return np.where(k1 > 1e-12, k0 * (k0 - 1.0) / np.maximum(k1, 1e-12), -AXES.min())
