"""
warpcube.py -- a cube with 16 values: an 8-value offset field and an 8-value scalar field.

    values = [ offset c1 (8) | field c2 (8) ]       corners numbered n = i + 2j + 4k (as cube.py)

Evaluation at a point x of the cube:

  1. Offset. Corner n is moved along its cube diagonal e_n = (2i-1, 2j-1, 2k-1)
     by c1[n] half-widths (outward if positive): by h * c1[n] in each of x, y, z,
     with h = size / 2. For the default cube [-1, 1]^3, h = 1 and c1 is in world units.
     Trilinear interpolation of the moved corners gives a new point

         x' = sum_n w_n(x) (p_n + h c1[n] e_n) = x + D(x),   D(x) = h sum_n w_n(x) c1[n] e_n

     (w_n are the trilinear weights of x, p_n the original corners). Because the
     offsets are relative to the cube, the same 16 values give the same shape,
     moved and scaled, in a cube of any position and size (see box.py).
  2. Field. f(x) = T2(x'), the trilinear interpolation of c2, evaluated at x'.
     If x' leaves the cube the same trilinear polynomial is used (extrapolation).

With c1 = 0, x' = x and f is exactly the 8-value Cube of cube.py with values c2.

The offset rule is the single function `offset_local`; change it there to try a
different warp. Everything is PyTorch and differentiable in all 16 values; the
numpy versions at the end are used for the discrete steps of the mesher (signs,
pruning, root bracketing), where reproducible float64 arithmetic matters.

Each component of x' is trilinear in x, so f is a polynomial of degree <= 3 in
each coordinate. Unlike the 8-value cube, f restricted to an edge is no longer
linear, and its level set can have bumps and dips inside the cube (the
8-value cube's level set is monotone in every axis-aligned slice).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cube import CORNER_IJK, CUBE_EDGES, Cube, PRESETS

# diagonal direction of each corner, (8, 3), entries +-1
CORNER_DIR = (2 * CORNER_IJK - 1).astype(np.float64)


# ----------------------------------------------------------------------------
# Basis helpers (local coordinates u in [0,1]^3)
# ----------------------------------------------------------------------------
def basis_local(u: torch.Tensor) -> torch.Tensor:
    """Trilinear weights (..., 8)."""
    return Cube.basis_local(u)


def basis_grad_local(u: torch.Tensor) -> torch.Tensor:
    """d w_n / d u_j  ->  (..., 3, 8)."""
    X, Y, Z = u[..., 0], u[..., 1], u[..., 2]
    one = torch.ones_like(X)
    wx, wy, wz = (torch.stack([1 - t, t], -1) for t in (X, Y, Z))
    dw = torch.stack([-one, one], -1)

    def outer(a, b, c):
        return (c[..., :, None, None] * b[..., None, :, None] * a[..., None, None, :]).reshape(*X.shape, 8)
    return torch.stack([outer(dw, wy, wz), outer(wx, dw, wz), outer(wx, wy, dw)], dim=-2)


def offset_local(u: torch.Tensor, c1: torch.Tensor, size: float = 2.0) -> torch.Tensor:
    """The offset D, in local units, at local points u (..., 3) -> (..., 3).
    Offsets are in half-widths, so in local units (width 1) corner n moves by
    c1[n] / 2 along e_n whatever the cube's size (`size` is kept for the API)."""
    E = torch.as_tensor(CORNER_DIR, dtype=u.dtype, device=u.device)
    return (basis_local(u) * c1) @ E / 2


def offset_jacobian_local(u: torch.Tensor, c1: torch.Tensor, size: float = 2.0) -> torch.Tensor:
    """dD_k / du_j (local units) -> (..., 3[k], 3[j])."""
    E = torch.as_tensor(CORNER_DIR, dtype=u.dtype, device=u.device)
    G = basis_grad_local(u) * c1                       # (..., 3[j], 8)
    return torch.einsum("...jn,nk->...kj", G, E) / 2


# ----------------------------------------------------------------------------
# Functional forms (used by the module and by the mesher's differentiable step)
# ----------------------------------------------------------------------------
def warp_world(values, x, lo, size):
    """x' = x + D(x) in world coordinates."""
    u = (x - lo) / size
    return lo + (u + offset_local(u, values[:8], size)) * size


def field_world(values, x, lo, size):
    """f(x) = T2(x + D(x)) at world points x (..., 3) -> (...)."""
    u = (x - lo) / size
    up = u + offset_local(u, values[:8], size)
    return basis_local(up) @ values[8:]


def gradient_world(values, x, lo, size):
    """Analytic spatial gradient of f at world points x -> (..., 3)."""
    u = (x - lo) / size
    c1, c2 = values[:8], values[8:]
    up = u + offset_local(u, c1, size)
    g_up = (basis_grad_local(up) @ c2)                 # dT2/du'  (..., 3)
    J = torch.eye(3, dtype=u.dtype, device=u.device) + offset_jacobian_local(u, c1, size)
    return torch.einsum("...k,...kj->...j", g_up, J) / size


# ----------------------------------------------------------------------------
class WarpCube(nn.Module):
    """16-value cube on [lo, lo + size]^3 (default [-1, 1]^3).

    Args:
        values: 16 numbers [offset(8), field(8)], or 8 numbers (field only, offset 0).
    """

    def __init__(self, values=None, lo=-1.0, size=2.0, requires_grad=True, dtype=torch.float64):
        super().__init__()
        if values is None:
            values = [0.0] * 8 + list(PRESETS["bowl"])
        v = torch.as_tensor(values, dtype=dtype).reshape(-1).clone()
        if len(v) == 8:
            v = torch.cat([torch.zeros(8, dtype=dtype), v])
        assert len(v) == 16, "WarpCube takes 16 values (or 8 field values)"
        self.values = nn.Parameter(v, requires_grad=requires_grad)
        self.register_buffer("lo", torch.as_tensor(lo, dtype=dtype).expand(3).clone())
        self.size = float(size)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def offset(self) -> torch.Tensor:
        return self.values[:8]

    @property
    def field(self) -> torch.Tensor:
        return self.values[8:]

    def set_values(self, values):
        v = torch.as_tensor(values, dtype=self.dtype).reshape(-1)
        if len(v) == 8:
            v = torch.cat([torch.zeros(8, dtype=self.dtype), v])
        with torch.no_grad():
            self.values.copy_(v)

    def corner_positions(self) -> torch.Tensor:
        ijk = torch.as_tensor(CORNER_IJK, dtype=self.dtype, device=self.lo.device)
        return self.lo + ijk * self.size

    def moved_corners(self) -> torch.Tensor:
        """Corners after the offset: p_n + (size/2) c1[n] e_n."""
        E = torch.as_tensor(CORNER_DIR, dtype=self.dtype, device=self.lo.device)
        return self.corner_positions() + self.offset[:, None] * E * (self.size / 2)

    def _x(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.lo.device)

    def warp(self, x, values=None):
        return warp_world(self.values if values is None else values, self._x(x), self.lo, self.size)

    def evaluate(self, x, values=None):
        return field_world(self.values if values is None else values, self._x(x), self.lo, self.size)

    forward = evaluate

    def gradient(self, x, values=None):
        return gradient_world(self.values if values is None else values, self._x(x), self.lo, self.size)

    def __repr__(self):
        v = self.values.detach().cpu().numpy()
        f = lambda a: ", ".join(f"{t:+.3f}" for t in a)  # noqa: E731
        return f"WarpCube(lo={self.lo.tolist()}, size={self.size}, offset=[{f(v[:8])}], field=[{f(v[8:])}])"


# ----------------------------------------------------------------------------
# numpy float64 versions (discrete steps of the mesher)
# ----------------------------------------------------------------------------
def _basis_np(u):
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    wx = np.stack([1 - x, x], -1)
    wy = np.stack([1 - y, y], -1)
    wz = np.stack([1 - z, z], -1)
    return (wz[..., :, None, None] * wy[..., None, :, None] * wx[..., None, None, :]).reshape(*u.shape[:-1], 8)


def warp_local_np(values, u, size=2.0):
    """Local u (..., 3) -> local u' = u + D(u) (offsets in half-widths, see offset_local)."""
    c1 = np.asarray(values, np.float64)[:8]
    return u + (_basis_np(u) * c1) @ CORNER_DIR / 2


def trilinear_np(c, u):
    """Trilinear polynomial of 8 values at local points (works outside [0,1] too)."""
    c = np.asarray(c, np.float64).reshape(2, 2, 2)
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    c00 = c[0, 0, 0] + (c[0, 0, 1] - c[0, 0, 0]) * x
    c10 = c[0, 1, 0] + (c[0, 1, 1] - c[0, 1, 0]) * x
    c01 = c[1, 0, 0] + (c[1, 0, 1] - c[1, 0, 0]) * x
    c11 = c[1, 1, 0] + (c[1, 1, 1] - c[1, 1, 0]) * x
    c0 = c00 + (c10 - c00) * y
    c1 = c01 + (c11 - c01) * y
    return c0 + (c1 - c0) * z


def field_local_np(values, u, size):
    values = np.asarray(values, np.float64)
    return trilinear_np(values[8:], warp_local_np(values, u, size))


# ----------------------------------------------------------------------------
# presets: (offset, field). Offset 0 = the 8-value cube.
# ----------------------------------------------------------------------------
WARP_PRESETS = {f"{k} (no offset)": [0.0] * 8 + list(v) for k, v in PRESETS.items()}
WARP_PRESETS.update({
    # fitted to the "sphere cap" target (a bump inside the cube; the 8-value cube cannot)
    "sphere cap (fitted)": [-0.2358, 1.1161, -1.0014, -0.8395, 1.2652, -0.3875, -0.8828, -0.9575, -0.5644, -0.0390, 0.6538, 1.7513, 0.0011, -0.8066, 1.8340, 0.4136],
    # fitted to the "torus inner" target (saddle with its crest inside the cube)
    "torus inner (fitted)": [-0.9984, 1.8444, -0.8802, -0.8893, -1.0104, -0.8377, 2.2775, -0.9186, -1.4444, -0.0031, 0.0490, 0.5556, -1.6580, 0.8216, -0.0530, 1.4750],
    # pinched corner: the 'corner' preset with its corner pulled in
    "pinched corner":    [-0.4, 0.2, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0] + list(PRESETS["corner"]),
    # twisted plane: opposite offsets on the two x-faces
    "twisted plane":     [0.3, -0.3, -0.3, 0.3, -0.3, 0.3, 0.3, -0.3] + list(PRESETS["plane x"]),
    # saddle with skewed offsets
    "warped saddle":     [0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.25, -0.15] + list(PRESETS["saddle"]),
})
