"""Large sphere: sphere of radius 0.8 centred at (0.0, 0.0, 0.0)."""
import numpy as np

NAME = "sphere large"
CENTER = np.array([0.0, 0.0, 0.0])
RADIUS = 0.8


def g(X):
    """Signed distance to the sphere (negative inside). X: (..., 3)."""
    return np.linalg.norm(np.asarray(X) - CENTER, axis=-1) - RADIUS
