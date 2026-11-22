"""
grid.py -- a regular grid of cubes ("cube spline") with numbers stored at the grid vertices.

The grid
    nx x ny x nz cubic cells of edge h, starting at lo:  domain = lo + h * [0,nx] x [0,ny] x [0,nz].
    Vertex (a, b, c) has id  a + (nx+1) * (b + (ny+1) * c)  and position lo + h * (a, b, c).
    A point x lies in the cell c = floor((x - lo) / h) (clamped to the grid), at local
    coordinates t = (x - lo) / h - c in [0, 1]^3; the cell's 8 corners are ordered
    n = i + 2j + 4k as everywhere in this project.

Two models (the parameter vector p of each):
    scalar   p = G, one value per vertex.
             f(x) = sum_n w_n(t) G[corner n of the cell of x]          (trilinear in each cell)
             Neighbouring cells share the values on their common corners, so f is
             continuous across cell faces. On a 1 x 1 x 1 grid this is the 8-value cube.
    warp     p = [G, D], a value and a 3D displacement per vertex (D in cell units).
             x' = x + h * sum_n w_n(t) D[corner n]                    (trilinear displacement)
             f(x) = G(x')                                             (the scalar grid at x')
             Both the displacement and G are continuous over the grid, so f is continuous.
             x' may land in another cell; it is clamped to the grid domain (G is only
             defined on the grid: extending a boundary cell's polynomial outwards could
             make spurious surface pieces at the domain faces). D = 0 gives the scalar model.
             On one cell: 8 values + 8 displacements = 32 numbers.

Every field and gradient here is differentiable in p (torch autograd), and the
gradients with respect to x are analytic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DT = torch.float64
CORNER_IJK = np.array([[n & 1, (n >> 1) & 1, (n >> 2) & 1] for n in range(8)], dtype=np.int64)


def basis(t: torch.Tensor) -> torch.Tensor:
    """Trilinear weights (..., 8) at local coordinates t (..., 3), corner n = i + 2j + 4k."""
    x, y, z = t[..., 0], t[..., 1], t[..., 2]
    X = torch.stack([1 - x, x], -1)
    Y = torch.stack([1 - y, y], -1)
    Z = torch.stack([1 - z, z], -1)
    return (Z[..., :, None, None] * Y[..., None, :, None] * X[..., None, None, :]).reshape(*t.shape[:-1], 8)


def basis_grad(t: torch.Tensor) -> torch.Tensor:
    """d w_n / d t_j  -> (..., 3[j], 8)."""
    x, y, z = t[..., 0], t[..., 1], t[..., 2]
    one = torch.ones_like(x)
    X, dX = torch.stack([1 - x, x], -1), torch.stack([-one, one], -1)
    Y, dY = torch.stack([1 - y, y], -1), torch.stack([-one, one], -1)
    Z, dZ = torch.stack([1 - z, z], -1), torch.stack([-one, one], -1)

    def outer(a, b, c):
        return (c[..., :, None, None] * b[..., None, :, None] * a[..., None, None, :]).reshape(*t.shape[:-1], 8)
    return torch.stack([outer(dX, Y, Z), outer(X, dY, Z), outer(X, Y, dZ)], -2)


@dataclass(frozen=True)
class Grid:
    lo: tuple            # (3,) minimum corner of the domain
    h: float             # cell edge length
    dims: tuple          # (nx, ny, nz) cells per axis

    def __post_init__(self):
        object.__setattr__(self, "lo", tuple(float(v) for v in np.broadcast_to(np.asarray(self.lo, float), (3,))))
        object.__setattr__(self, "h", float(self.h))
        object.__setattr__(self, "dims", tuple(int(d) for d in np.broadcast_to(np.asarray(self.dims), (3,))))
        if self.h <= 0 or min(self.dims) < 1:
            raise ValueError(f"bad grid: h={self.h}, dims={self.dims}")

    # ------------------------------------------------------------ construction
    @classmethod
    def covering(cls, lo, hi, dims) -> "Grid":
        """Cubic cells, `dims` cells per axis, covering the box [lo, hi] (centred on it;
        the cell size is set by the axis that needs the largest cells)."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        dims = np.broadcast_to(np.asarray(dims, int), (3,))
        h = float(np.max((hi - lo) / dims))
        centre = (lo + hi) / 2
        return cls(tuple(centre - h * dims / 2), h, tuple(dims))

    @classmethod
    def cube(cls, dims, lo=(-1.0, -1.0, -1.0), size=2.0) -> "Grid":
        """`dims` cells per axis over the cube [lo, lo + size]^3 (cells are cubic only
        when all dims are equal; otherwise covering() of that cube)."""
        lo = np.broadcast_to(np.asarray(lo, float), (3,))
        return cls.covering(lo, lo + size, dims)

    def to_dict(self) -> dict:
        return {"lo": list(self.lo), "h": self.h, "dims": list(self.dims)}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(tuple(d["lo"]), d["h"], tuple(d["dims"]))

    # ------------------------------------------------------------ geometry
    @property
    def lo_np(self) -> np.ndarray:
        return np.array(self.lo)

    @property
    def hi_np(self) -> np.ndarray:
        return self.lo_np + self.h * np.array(self.dims)

    @property
    def vdims(self) -> tuple:
        return tuple(d + 1 for d in self.dims)

    @property
    def num_vertices(self) -> int:
        nx, ny, nz = self.vdims
        return nx * ny * nz

    @property
    def num_cells(self) -> int:
        return int(np.prod(self.dims))

    def vertex_ijk(self) -> np.ndarray:
        """(V, 3) integer coordinates of the vertices, in id order."""
        a, b, c = np.meshgrid(*[np.arange(n) for n in self.vdims], indexing="ij")
        ijk = np.stack([a, b, c], -1).reshape(-1, 3)
        order = np.argsort(self.vertex_id(ijk))
        return ijk[order]

    def vertex_positions(self) -> np.ndarray:
        return self.lo_np + self.h * self.vertex_ijk()

    def vertex_id(self, ijk) -> np.ndarray:
        ijk = np.asarray(ijk, np.int64)
        nx, ny, _ = self.vdims
        return ijk[..., 0] + nx * (ijk[..., 1] + ny * ijk[..., 2])

    def cell_corner_ids(self, cells) -> np.ndarray:
        """(m, 8) vertex ids of the corners of cells (m, 3), order n = i + 2j + 4k."""
        return self.vertex_id(np.asarray(cells, np.int64)[:, None, :] + CORNER_IJK[None])

    def contains(self, X, eps=1e-9) -> np.ndarray:
        X = np.asarray(X)
        return np.all((X >= self.lo_np - eps) & (X <= self.hi_np + eps), axis=-1)

    def clamp(self, X: torch.Tensor) -> torch.Tensor:
        lo = torch.as_tensor(self.lo_np, dtype=X.dtype)
        hi = torch.as_tensor(self.hi_np, dtype=X.dtype)
        return torch.maximum(torch.minimum(X, hi), lo)

    def uniform(self, n, gen: torch.Generator, dtype=DT) -> torch.Tensor:
        U = torch.rand(n, 3, dtype=dtype, generator=gen)
        return torch.as_tensor(self.lo_np, dtype=dtype) + U * torch.as_tensor(self.hi_np - self.lo_np, dtype=dtype)

    def lattice(self, k: int, faces=False) -> np.ndarray:
        """A k x k x k lattice of points over the whole domain (centres of a k^3
        subdivision, or i/(k-1) along each axis with faces=True)."""
        t = np.arange(k) / (k - 1) if (faces and k > 1) else (np.arange(k) + 0.5) / k
        U = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
        return self.lo_np + U * (self.hi_np - self.lo_np)

    def scale(self, length: float) -> float:
        """A length given for a domain of width 2, scaled to this grid's largest extent."""
        return length * float(np.max(self.hi_np - self.lo_np)) / 2

    def edges(self) -> tuple:
        """All cell edges as (positions, (E, 2) index pairs), for drawing the lattice."""
        P = self.vertex_positions()
        ijk = self.vertex_ijk()
        E = []
        for ax in range(3):
            ok = ijk[:, ax] < self.dims[ax]
            a = np.nonzero(ok)[0]
            step = np.eye(3, dtype=np.int64)[ax]
            E.append(np.stack([self.vertex_id(ijk[a]), self.vertex_id(ijk[a] + step)], 1))
        return P, np.concatenate(E, 0)

    def box_edges(self) -> tuple:
        """The 8 domain corners and the 12 edges of the domain box."""
        lo, hi = self.lo_np, self.hi_np
        C = np.array([[hi[0] if i else lo[0], hi[1] if j else lo[1], hi[2] if k else lo[2]]
                      for i, j, k in CORNER_IJK])
        from core.cube import CUBE_EDGES
        return C, CUBE_EDGES

    # ------------------------------------------------------------ evaluation
    def locate(self, X: torch.Tensor):
        """Cell of each point (clamped to the grid) and the local coordinates there.
        Returns (corner ids (..., 8) long, t (..., 3)); t is outside [0, 1] for points
        outside the grid (the boundary cell's polynomial is extended)."""
        lo = torch.as_tensor(self.lo_np, dtype=X.dtype, device=X.device)
        u = (X - lo) / self.h
        with torch.no_grad():
            hi = torch.as_tensor(self.dims, device=X.device) - 1
            c = torch.minimum(torch.clamp(torch.floor(u), min=0).long(), hi)
            nx, ny, _ = self.vdims
            base = c[..., 0] + nx * (c[..., 1] + ny * c[..., 2])
            off = torch.as_tensor(self.vertex_id(CORNER_IJK), device=X.device)
            ids = base[..., None] + off
        return ids, u - c.to(X.dtype)

    def interp(self, vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Piecewise-trilinear interpolation of per-vertex values (V,) or (V, C) at X (..., 3)."""
        ids, t = self.locate(X)
        w = basis(t)
        if vals.dim() == 1:
            return (w * vals[ids]).sum(-1)
        return (w[..., None] * vals[ids]).sum(-2)

    def interp_grad(self, vals: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """World gradient of the scalar interpolation of vals (V,) at X -> (..., 3)."""
        ids, t = self.locate(X)
        return (basis_grad(t) * vals[ids][..., None, :]).sum(-1) / self.h


class GridModel:
    """A scalar or warp field on a Grid, with a flat parameter vector p (see the module
    docstring). Same interface as the single cube's model in classify_optimize.py
    (n, field, field_batch, normalised, field_slice, offset_slice)."""

    def __init__(self, grid: Grid, kind: str = "scalar"):
        if kind not in ("scalar", "warp"):
            raise ValueError(f"unknown grid model '{kind}' (scalar or warp)")
        self.grid, self.kind = grid, kind
        self.nv = grid.num_vertices
        self.n = self.nv if kind == "scalar" else 4 * self.nv

    # ------------------------------------------------------------ parameters
    @property
    def field_slice(self):
        return slice(0, self.nv)

    @property
    def offset_slice(self):
        """The displacement entries (penalised by the classification optimizers)."""
        return None if self.kind == "scalar" else slice(self.nv, self.n)

    def split(self, p: torch.Tensor):
        """(G (V,), D (V, 3) or None)."""
        G = p[..., :self.nv]
        if self.kind == "scalar":
            return G, None
        return G, p[..., self.nv:].reshape(*p.shape[:-1], self.nv, 3)

    def join(self, G, D=None) -> torch.Tensor:
        G = torch.as_tensor(G, dtype=DT).reshape(-1)
        if self.kind == "scalar":
            return G.clone()
        D = torch.zeros(self.nv, 3, dtype=DT) if D is None else torch.as_tensor(D, dtype=DT).reshape(self.nv, 3)
        return torch.cat([G, D.reshape(-1)])

    def normalised(self, p: torch.Tensor, norm: float | None = None) -> torch.Tensor:
        """G scaled to RMS 1 (or to the given norm); the surface does not depend on the
        scale of G. D is unchanged."""
        p = p.detach().clone()
        G = p[..., self.field_slice]
        nrm = G.norm()
        if nrm > 0:
            p[..., self.field_slice] = G * ((np.sqrt(self.nv) if norm is None else norm) / nrm)
        return p

    # ------------------------------------------------------------ evaluation
    def warp(self, p: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """x' = x + h * D(x), clamped to the grid domain (x itself for the scalar model)."""
        G, D = self.split(p)
        if D is None:
            return X
        return self.grid.clamp(X + self.grid.h * self.grid.interp(D, X))

    def field(self, p: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        G, _ = self.split(p)
        return self.grid.interp(G, self.warp(p, X))

    def gradient(self, p: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """World gradient of f at X -> (..., 3); differentiable in p."""
        G, D = self.split(p)
        if D is None:
            return self.grid.interp_grad(G, X)
        ids, t = self.grid.locate(X)
        Dn = D[ids]                                                # (..., 8, 3)
        J = torch.eye(3, dtype=X.dtype) + torch.einsum("...nk,...jn->...kj", Dn, basis_grad(t))
        Xw = X + self.grid.h * (basis(t)[..., None] * Dn).sum(-2)
        Xc = self.grid.clamp(Xw)
        gG = self.grid.interp_grad(G, Xc) * (Xc == Xw)            # clamped axes: no change
        return torch.einsum("...kj,...k->...j", J, gG)

    def field_batch(self, P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """f for a population of parameter vectors P (pop, n) at X (m, 3) -> (m, pop)."""
        ids, t = self.grid.locate(X)
        w = basis(t)                                               # (m, 8)
        G = P[:, :self.nv]                                         # (pop, V)
        if self.kind == "scalar":
            return (w[None] * G[:, ids]).sum(-1).T
        D = P[:, self.nv:].reshape(len(P), self.nv, 3)
        Xw = self.grid.clamp(X[None] + self.grid.h * (w[None, :, :, None] * D[:, ids]).sum(-2))  # (pop, m, 3)
        ids2, t2 = self.grid.locate(Xw)
        pop = torch.arange(len(P))[:, None, None]
        return (basis(t2) * G[pop, ids2]).sum(-1).T

    def displacement_at_vertices(self, p) -> np.ndarray | None:
        """World displacement h * D at the vertices (V, 3), or None for the scalar model."""
        _, D = self.split(torch.as_tensor(p, dtype=DT))
        return None if D is None else (self.grid.h * D).detach().numpy()

    def describe(self) -> str:
        nx, ny, nz = self.grid.dims
        per = "1 value" if self.kind == "scalar" else "1 value + 3D displacement"
        return (f"{self.kind} grid {nx}x{ny}x{nz} cells of {self.grid.h:g}, {self.nv} vertices "
                f"({per} each, {self.n} numbers)")
