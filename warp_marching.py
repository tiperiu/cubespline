"""
warp_marching.py -- iso-surface of a 16-value WarpCube by subdivided marching cubes.

Same pipeline as marching.py (octree descent, marching cubes with the asymptotic
decider on the leaves, vertices welded per lattice edge), with three changes
because f = T2(x + D(x)) is no longer trilinear:

1. Pruning uses a rigorous bound instead of the corner signs. On a sub-cube,
   each component of x' = x + D(x) is trilinear, so its range is spanned by its
   values at the 8 corners: x' stays in the box [min, max]^3 of those values.
   T2 is trilinear too, so its range over that box is spanned by its values at
   the box's 8 corners. If that range excludes the level, the sub-cube contains
   no surface and is discarded with all its children. The bound is conservative:
   it never discards surface, it may keep some empty cells.

2. Vertex placement. f is not linear along an edge, so the crossing is found
   by bisection (float64, to round-off), between the two lattice points whose
   signs differ. Each lattice edge is solved once, so neighbours share it.

3. Differentiability. The crossing parameter t* is found without gradients;
   the vertex position is then rebuilt as

       t = t* - f(p(t*); c) / [d f(p(t); c)/dt at t*]      (derivative detached)

   which equals t* in value and has exactly the implicit-function-theorem
   gradient dt/dc = -f_c / f_t. So `Topology.vertices(values)` is differentiable
   in all 16 values, as in the 8-value mesher.

As in any marching cubes, a surface piece entirely inside one leaf cell with all
8 corners on the same side is not captured; the subdivision depth controls that.
Conventions as in marching.py: inside is f < level, normals point towards +f.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from cube import CORNER_IJK
from marching import SNAP_REL, _mesh_cells, mesh_check, save_obj, to_numpy  # noqa: F401
from warpcube import WarpCube, field_world, gradient_world, trilinear_np, warp_local_np


@dataclass
class WarpTopology:
    """Connectivity plus what is needed to rebuild vertex positions.

    F       (nf, 3)  triangles
    pa, pb  (nv, 3)  world endpoints of each vertex's lattice edge
    t       (nv,)    crossing parameter along the edge (found by bisection)
    cells   (nc, 3)  leaf cells with a sign change (integer coordinates at resolution R)
    """
    F: np.ndarray
    pa: torch.Tensor
    pb: torch.Tensor
    t: torch.Tensor
    cells: np.ndarray
    depth: int
    level: float
    lo: torch.Tensor
    size: float
    stats: dict = field(default_factory=dict)

    @property
    def resolution(self) -> int:
        return 2 ** self.depth

    def vertices(self, values: torch.Tensor, level=None) -> torch.Tensor:
        """Vertex positions (nv, 3), differentiable w.r.t. the 16 values."""
        lv = self.level if level is None else level
        if not len(self.t):
            return self.pa
        d = self.pb - self.pa
        p0 = self.pa + self.t[:, None] * d
        f = field_world(values, p0, self.lo, self.size) - lv
        ft = (gradient_world(values, p0, self.lo, self.size) * d).sum(-1).detach()
        eps = 1e-12 * (d.norm(dim=-1) + 1e-30)
        ft = torch.where(ft.abs() < eps, torch.where(ft < 0, -eps, eps), ft)
        t = (self.t - f / ft).clamp(0.0, 1.0)
        return self.pa + t[:, None] * d

    def cell_bounds(self):
        h = self.size / self.resolution
        return self.lo.detach().cpu().numpy() + self.cells * h, h


# ----------------------------------------------------------------------------
def _snap_tol(values, level):
    c2 = np.asarray(values, np.float64)[8:]
    return SNAP_REL * max(float(np.abs(c2 - level).max()), 1e-300)


def _unique_lattice(cells, R):
    """Unique lattice points of the cells' corners: (points (m,3), inverse (nc,8))."""
    corners = (cells[:, None, :] + CORNER_IJK[None]).reshape(-1, 3)
    key = corners[:, 0] + (R + 1) * (corners[:, 1] + (R + 1) * corners[:, 2])
    ukey, inv = np.unique(key, return_inverse=True)
    pts = np.stack([ukey % (R + 1), (ukey // (R + 1)) % (R + 1), ukey // (R + 1) ** 2], 1)
    return pts, inv.reshape(len(cells), 8)


def lattice_values(values, pts_ijk, R, size, level, tol):
    """g = f - level at integer lattice points, snapped to 0 within tol (0 = outside)."""
    g = trilinear_np(np.asarray(values, np.float64)[8:],
                     warp_local_np(values, np.asarray(pts_ijk, np.float64) / R, size)) - level
    g[np.abs(g) < tol] = 0.0
    return g


def cell_bounds(values, cells, R, size, level):
    """Rigorous [min, max] of f - level over each cell, plus the corner values
    of f - level. Returns (fmin, fmax, g_corners) each (nc,) / (nc, 8)."""
    values = np.asarray(values, np.float64)
    pts, inv = _unique_lattice(cells, R)
    u = pts.astype(np.float64) / R
    up = warp_local_np(values, u, size)                     # x' at lattice points (local)
    g_pts = trilinear_np(values[8:], up) - level
    up_c = up[inv]                                          # (nc, 8, 3)
    blo, bhi = up_c.min(1), up_c.max(1)                     # box containing x'(cell)
    box = blo[:, None, :] + CORNER_IJK[None].astype(np.float64) * (bhi - blo)[:, None, :]
    t2 = trilinear_np(values[8:], box) - level              # (nc, 8)
    return t2.min(1), t2.max(1), g_pts[inv]


def extract_topology(cube: WarpCube, depth: int = 4, level: float = 0.0, bisect_iters: int = 60):
    """Subdivide 2^depth per axis with rigorous pruning, then marching cubes."""
    depth, level = int(depth), float(level)
    values = cube.values.detach().cpu().numpy().astype(np.float64)
    size = cube.size
    tol = _snap_tol(values, level)
    margin = 4 * tol

    cells = np.zeros((1, 3), dtype=np.int64)
    per_level = []
    for lvl in range(depth):
        R = 2 ** lvl
        fmin, fmax, _ = cell_bounds(values, cells, R, size, level)
        keep = ~((fmin > margin) | (fmax < -margin))
        per_level.append((lvl, len(cells), int(keep.sum())))
        cells = cells[keep]
        if not len(cells):
            break
        cells = (2 * cells[:, None, :] + CORNER_IJK[None]).reshape(-1, 3)

    R = 2 ** depth
    stats = {"levels": per_level, "total_cells": R ** 3, "resolution": R}
    dt, dev = cube.dtype, cube.lo.device
    lo = cube.lo.detach().clone()

    def empty():
        z3 = torch.zeros(0, 3, dtype=dt, device=dev)
        stats.update({"active_cells": 0, "vertices": 0, "faces": 0})
        return WarpTopology(np.zeros((0, 3), np.int64), z3, z3, torch.zeros(0, dtype=dt, device=dev),
                            np.zeros((0, 3), np.int64), depth, level, lo, size, stats)

    if not len(cells) or len(per_level) < depth:
        return empty()

    # leaf level: candidate cells from the bound, then the actual corner signs
    fmin, fmax, _ = cell_bounds(values, cells, R, size, level)
    cand = ~((fmin > margin) | (fmax < -margin))
    cells = cells[cand]
    pts, inv = _unique_lattice(cells, R) if len(cells) else (np.zeros((0, 3), np.int64), None)
    if not len(cells):
        per_level.append((depth, len(cand), 0))
        return empty()
    g = lattice_values(values, pts, R, size, level, tol)[inv]       # (nc, 8)
    n_in = (g < 0).sum(1)
    mixed = (n_in > 0) & (n_in < 8)
    per_level.append((depth, len(cand), int(mixed.sum())))
    stats["candidate_leaves_without_sign_change"] = int((~mixed).sum())
    cells, g = cells[mixed], g[mixed]
    stats["active_cells"] = len(cells)
    meshed = _mesh_cells(cells, g, R) if len(cells) else None
    if meshed is None:
        return empty()
    F, a_ijk, b_ijk, n_cases, n_amb = meshed

    # ---- crossing on each lattice edge: bisection in float64
    ua, ub = a_ijk.astype(np.float64) / R, b_ijk.astype(np.float64) / R
    ga = lattice_values(values, a_ijk, R, size, level, tol)
    a_in = ga < 0
    t_lo, t_hi = np.zeros(len(ua)), np.ones(len(ua))      # f(t_lo) on a's side
    for _ in range(bisect_iters):
        tm = 0.5 * (t_lo + t_hi)
        gm = trilinear_np(values[8:], warp_local_np(values, ua + tm[:, None] * (ub - ua), size)) - level
        gm[np.abs(gm) < tol] = 0.0
        same = (gm < 0) == a_in
        t_lo = np.where(same, tm, t_lo)
        t_hi = np.where(same, t_hi, tm)
    t = 0.5 * (t_lo + t_hi)

    pa = lo + torch.as_tensor(ua, dtype=dt, device=dev) * size
    pb = lo + torch.as_tensor(ub, dtype=dt, device=dev) * size
    stats.update({"vertices": len(t), "faces": len(F), "cases": n_cases, "ambiguous_faces": n_amb})
    return WarpTopology(F, pa, pb, torch.as_tensor(t, dtype=dt, device=dev), cells, depth, level, lo, size, stats)


def marching_cubes(cube: WarpCube, depth: int = 4, level: float = 0.0):
    """(V, F, topology); V is a torch tensor attached to cube.values' graph."""
    topo = extract_topology(cube, depth, level)
    return topo.vertices(cube.values), topo.F, topo


def vertex_normals(cube: WarpCube, V) -> np.ndarray:
    with torch.no_grad():
        n = cube.gradient(torch.as_tensor(to_numpy(V), dtype=cube.dtype))
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return n.cpu().numpy()
