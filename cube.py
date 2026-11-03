"""
cube.py -- a single trilinear cube.

The cube stores 8 corner values. Inside the cube the field is the trilinear
interpolation of those values; its zero level set (or any other level) is the
surface we extract with marching.py and, later, fit with optimize.py.

Corner numbering
----------------
Corner n has integer coordinates (i, j, k) in {0,1}^3 with

    n = i + 2*j + 4*k          (i -> x, j -> y, k -> z)

    n : 0      1      2      3      4      5      6      7
  ijk : 000    100    010    110    001    101    011    111

Corner (i,j,k) sits at  lo + (i,j,k) * size.

Everything is PyTorch: `values` is an nn.Parameter, so any quantity computed
from `evaluate` / `basis` is differentiable with respect to the 8 values.

The key fact for the rest of the project: the field is LINEAR in the values,

    f(x) = basis(x) @ values,     basis(x) in R^8, sum = 1,

and along any axis-aligned segment it is also linear in position. So marching
cubes on any sub-grid places its vertices *exactly* on the true level set, and a
vertex position is a smooth function of `values` (see marching.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# (8, 3) integer corner coordinates, row n = (i, j, k) with n = i + 2j + 4k
CORNER_IJK = np.array([[n & 1, (n >> 1) & 1, (n >> 2) & 1] for n in range(8)], dtype=np.int64)

# 12 cube edges as (corner a, corner b) pairs: 4 along x, 4 along y, 4 along z
CUBE_EDGES = np.array([
    [0, 1], [2, 3], [4, 5], [6, 7],      # x
    [0, 2], [1, 3], [4, 6], [5, 7],      # y
    [0, 4], [1, 5], [2, 6], [3, 7],      # z
], dtype=np.int64)


class Cube(nn.Module):
    """Trilinear field on an axis-aligned cube [lo, lo + size]^3.

    Args:
        values: 8 corner values (any array-like / tensor, ordered n = i + 2j + 4k).
                Defaults to a single-corner configuration (corner 0 inside).
        lo:     minimum corner of the cube (scalar or 3-vector). Default -1.
        size:   edge length. Default 2, i.e. the cube [-1, 1]^3.
        requires_grad: make `values` a trainable parameter (default True).
    """

    def __init__(self, values=None, lo=-1.0, size=2.0, requires_grad=True,
                 dtype=torch.float64):
        super().__init__()
        if values is None:
            values = [-1.0, 1, 1, 1, 1, 1, 1, 1]
        v = torch.as_tensor(values, dtype=dtype).reshape(8).clone()
        self.values = nn.Parameter(v, requires_grad=requires_grad)
        lo = torch.as_tensor(lo, dtype=dtype).expand(3).clone()
        self.register_buffer("lo", lo)
        self.size = float(size)

    # ---------------------------------------------------------------- helpers
    @property
    def hi(self) -> torch.Tensor:
        return self.lo + self.size

    @property
    def dtype(self):
        return self.values.dtype

    def corner_positions(self) -> torch.Tensor:
        """(8, 3) world positions of the corners, in corner order."""
        ijk = torch.as_tensor(CORNER_IJK, dtype=self.dtype, device=self.lo.device)
        return self.lo + ijk * self.size

    def to_local(self, x: torch.Tensor) -> torch.Tensor:
        """World -> local coordinates u in [0,1]^3."""
        return (x - self.lo) / self.size

    def to_world(self, u: torch.Tensor) -> torch.Tensor:
        return self.lo + u * self.size

    def set_values(self, values):
        """Overwrite the 8 values in place (keeps the Parameter object)."""
        with torch.no_grad():
            self.values.copy_(torch.as_tensor(values, dtype=self.dtype).reshape(8))

    # ------------------------------------------------------------ evaluation
    @staticmethod
    def basis_local(u: torch.Tensor) -> torch.Tensor:
        """Trilinear basis at local coordinates u (..., 3) -> (..., 8)."""
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        wx = torch.stack([1 - x, x], dim=-1)      # (..., 2)  index i
        wy = torch.stack([1 - y, y], dim=-1)      # index j
        wz = torch.stack([1 - z, z], dim=-1)      # index k
        # w[k, j, i] flattened row-major gives n = i + 2j + 4k
        w = wz[..., :, None, None] * wy[..., None, :, None] * wx[..., None, None, :]
        return w.reshape(*u.shape[:-1], 8)

    def basis(self, x) -> torch.Tensor:
        """Trilinear basis weights at world points x (..., 3) -> (..., 8).
        Does not depend on `values`, so it can be precomputed and reused."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.lo.device)
        return self.basis_local(self.to_local(x))

    def evaluate(self, x, values: torch.Tensor | None = None) -> torch.Tensor:
        """Field value at world points x (..., 3) -> (...).
        `values` overrides the stored ones (e.g. a batch of candidates (B, 8)
        gives (..., B)); by default the module's parameter is used."""
        v = self.values if values is None else values
        return self.basis(x) @ v.T if v.dim() == 2 else self.basis(x) @ v

    forward = evaluate

    def gradient(self, x) -> torch.Tensor:
        """Analytic spatial gradient df/dx at world points x (..., 3) -> (..., 3)."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.lo.device)
        u = self.to_local(x)
        c = self.values.reshape(2, 2, 2)          # c[k, j, i]
        X, Y, Z = u[..., 0], u[..., 1], u[..., 2]

        def lerp(a, b, t):
            return a + (b - a) * t
        # d/dx: difference along i, bilinear in (y, z)
        dx = lerp(lerp(c[0, 0, 1] - c[0, 0, 0], c[0, 1, 1] - c[0, 1, 0], Y),
                  lerp(c[1, 0, 1] - c[1, 0, 0], c[1, 1, 1] - c[1, 1, 0], Y), Z)
        dy = lerp(lerp(c[0, 1, 0] - c[0, 0, 0], c[0, 1, 1] - c[0, 0, 1], X),
                  lerp(c[1, 1, 0] - c[1, 0, 0], c[1, 1, 1] - c[1, 0, 1], X), Z)
        dz = lerp(lerp(c[1, 0, 0] - c[0, 0, 0], c[1, 0, 1] - c[0, 0, 1], X),
                  lerp(c[1, 1, 0] - c[0, 1, 0], c[1, 1, 1] - c[0, 1, 1], X), Y)
        return torch.stack([dx, dy, dz], dim=-1) / self.size

    def grid_values(self, n: int) -> torch.Tensor:
        """Field sampled on the (n+1)^3 lattice of an n-subdivision, indexed [i, j, k]."""
        t = torch.linspace(0, 1, n + 1, dtype=self.dtype, device=self.lo.device)
        u = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1)
        return self.basis_local(u) @ self.values

    def __repr__(self):
        vals = ", ".join(f"{v:+.3f}" for v in self.values.detach().cpu().tolist())
        lo = self.lo.cpu().tolist()
        return f"Cube(lo={lo}, size={self.size}, values=[{vals}])"


# ----------------------------------------------------------------- presets
PRESETS = {
    # one corner inside: a small corner patch (MC case 1)
    "corner":      [-1.0, 1, 1, 1, 1, 1, 1, 1],
    # x-plane through the middle
    "plane x":     [-1.0, 1, -1, 1, -1, 1, -1, 1],
    # tilted plane x + y + z = 1.5 (in local coords)
    "tilted plane": [-1.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 1.5],
    # two opposite corners inside: separated or tunnel depending on saddle
    "diagonal pair": [-1.0, 1, 1, 1, 1, 1, 1, -1],
    # ambiguous face: face saddle decides if the two patches join
    "face saddle": [-1.0, 1, 1, -1, 0.6, 0.6, 0.6, 0.6],
    # hyperbolic paraboloid / saddle surface (x - 1/2)(y - 1/2) = 0 style
    "saddle":      [1.0, -1, -1, 1, 1, -1, -1, 1],
    # x*y*z-type: alternating signs (checkerboard) -- most complex case
    "checkerboard": [1.0, -1, -1, 1, -1, 1, 1, -1],
    # sphere-like cap: centre corner very negative
    "bowl":        [-2.0, 0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 2.5],
}
