"""Small sphere: sphere of radius 0.3 centred at (0.1, -0.05, 0.05)."""
import numpy as np

NAME = "sphere small"
CENTER = np.array([0.1, -0.05, 0.05])
RADIUS = 0.3


def g(X):
    """Signed distance to the sphere (negative inside). X: (..., 3)."""
    return np.linalg.norm(np.asarray(X) - CENTER, axis=-1) - RADIUS
