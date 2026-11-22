"""Large thin torus: torus with major radius 0.75 and tube radius 0.12 centred at (0.0, 0.0, 0.0), axis y rotated 90 degrees about z."""
import numpy as np

NAME = "torus large thin"
CENTER = np.array([0.0, 0.0, 0.0])
MAJOR = 0.75
MINOR = 0.12


def _rot(axis, deg):
    """Rotation matrix about the x, y or z axis."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R

R = _rot("z", 90)


def g(X):
    """Signed distance to the torus (negative inside the tube). X: (..., 3).
    In the torus frame the axis is y: distance to the circle of radius MAJOR in the
    xz-plane, minus MINOR."""
    L = (np.asarray(X) - CENTER) @ R
    ring = np.sqrt(L[..., 0] ** 2 + L[..., 2] ** 2) - MAJOR
    return np.sqrt(ring ** 2 + L[..., 1] ** 2) - MINOR
