"""
grid_marching.py -- marching cubes over a whole Grid (scalar or warp model).

Each grid cell is split into 2^depth sub-cells per axis, so the lattice has
dims * 2^depth sub-cells per axis. The field is evaluated once per lattice point
(so neighbouring cells always agree), the sub-cells whose corners change sign are
triangulated with the same case table, face decider and welding as the single cube
(core.marching._mesh_cells), and one vertex is placed on each crossed lattice edge.
The mesh is watertight wherever the surface is inside the grid.

Vertex positions, differentiable in the parameters p:
  scalar  the field is linear along a lattice edge (it lies inside one cell), so
          the crossing is exact: t = f(a) / (f(a) - f(b)).
  warp    the crossing is found by bisection (no gradient), then written as
          t = t* - f(x*) / f_t  with f_t = df/dt detached: the value is t*, and the
          gradient is the implicit-function derivative dt/dp = -(df/dp) / f_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from core.marching import CORNER_IJK, SNAP_REL, _mesh_cells
from grid.grid import DT, GridModel


@dataclass
class GridTopology:
    F: np.ndarray                 # (nf, 3) triangles
    pa: torch.Tensor              # (nv, 3) world endpoints of each vertex's lattice edge
    pb: torch.Tensor
    t_star: torch.Tensor          # (nv,) crossing parameter found at extraction (warp)
    model: GridModel
    depth: int
    level: float
    stats: dict = field(default_factory=dict)

    @property
    def num_vertices(self) -> int:
        return len(self.pa)

    def vertices(self, p: torch.Tensor) -> torch.Tensor:
        """Vertex positions (nv, 3) for parameters p with this connectivity."""
        if not len(self.pa):
            return self.pa
        lv = self.level
        d = self.pb - self.pa
        if self.model.kind == "scalar":
            ga = self.model.field(p, self.pa) - lv
            gb = self.model.field(p, self.pb) - lv
            den = ga - gb
            eps = torch.finfo(DT).eps
            den = torch.where(den.abs() < eps, torch.full_like(den, eps), den)
            t = (ga / den).clamp(0.0, 1.0)
            return self.pa + t[:, None] * d
        xs = (self.pa + self.t_star[:, None] * d).detach()
        f = self.model.field(p, xs) - lv
        with torch.no_grad():
            ft = (self.model.gradient(p.detach(), xs) * d).sum(-1)
            ft = torch.where(ft.abs() < 1e-12, torch.full_like(ft, 1e-12), ft)
        t = self.t_star - f / ft
        return self.pa + t[:, None] * d


def lattice_values(model: GridModel, p: torch.Tensor, depth: int, level: float = 0.0) -> np.ndarray:
    """f - level at every lattice point, shape (Lx+1, Ly+1, Lz+1), snapped near 0."""
    g = model.grid
    R = 2 ** depth
    L = np.array(g.dims) * R
    axes = [np.arange(n + 1) for n in L]
    ijk = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    X = torch.as_tensor(g.lo_np + ijk * (g.h / R), dtype=DT)
    with torch.no_grad():
        vals = []
        for i in range(0, len(X), 200000):
            vals.append(model.field(p.detach(), X[i:i + 200000]).numpy())
        G = np.concatenate(vals) - level
    tol = SNAP_REL * max(float(np.abs(G).max()), 1e-300)
    G[np.abs(G) < tol] = 0.0
    return G.reshape(*(L + 1))


def extract_topology(model: GridModel, p: torch.Tensor, depth: int = 2, level: float = 0.0,
                     bisect_iters: int = 60) -> GridTopology:
    p = torch.as_tensor(p, dtype=DT)
    g = model.grid
    R = 2 ** depth
    G = lattice_values(model, p, depth, level)
    L = np.array(G.shape) - 1
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in L], indexing="ij"), -1).reshape(-1, 3)
    corners = ijk[:, None, :] + CORNER_IJK[None]
    gc = G[corners[..., 0], corners[..., 1], corners[..., 2]]           # (ncells, 8)
    n_in = (gc < 0).sum(1)
    mixed = (n_in > 0) & (n_in < 8)
    stats = {"lattice": tuple(int(v) for v in L), "active_subcells": int(mixed.sum()), "depth": depth}
    empty = GridTopology(np.zeros((0, 3), np.int64), torch.zeros(0, 3, dtype=DT), torch.zeros(0, 3, dtype=DT),
                         torch.zeros(0, dtype=DT), model, depth, level, stats)
    if not mixed.any():
        return empty
    meshed = _mesh_cells(ijk[mixed], gc[mixed], int(L.max()))
    if meshed is None:
        return empty
    F, a_ijk, b_ijk, n_cases, n_amb = meshed
    pa = torch.as_tensor(g.lo_np + a_ijk * (g.h / R), dtype=DT)
    pb = torch.as_tensor(g.lo_np + b_ijk * (g.h / R), dtype=DT)
    ga = G[a_ijk[:, 0], a_ijk[:, 1], a_ijk[:, 2]]
    gb = G[b_ijk[:, 0], b_ijk[:, 1], b_ijk[:, 2]]
    if model.kind == "scalar":
        t_star = torch.as_tensor(np.clip(ga / (ga - gb), 0, 1), dtype=DT)
    else:                                                               # bisection on the true field
        a_in = torch.as_tensor(ga < 0)
        lo_t = torch.zeros(len(pa), dtype=DT)
        hi_t = torch.ones(len(pa), dtype=DT)
        d = pb - pa
        with torch.no_grad():
            for _ in range(bisect_iters):
                mid = (lo_t + hi_t) / 2
                fm = model.field(p, pa + mid[:, None] * d) - level
                same = (fm < 0) == a_in
                lo_t = torch.where(same, mid, lo_t)
                hi_t = torch.where(same, hi_t, mid)
        t_star = (lo_t + hi_t) / 2
    stats.update({"vertices": len(pa), "faces": len(F), "cases": n_cases, "ambiguous_faces": n_amb})
    return GridTopology(F, pa, pb, t_star, model, depth, level, stats)


def marching_cubes(model: GridModel, p, depth: int = 2, level: float = 0.0):
    """(V (nv, 3) tensor, attached to p's graph; F (nf, 3) int64; topology)."""
    p = torch.as_tensor(p, dtype=DT)
    topo = extract_topology(model, p, depth, level)
    return topo.vertices(p), topo.F, topo


def mesh_check(V, F, grid, tol=1e-9) -> dict:
    """Every edge in at most two triangles, consistently oriented; edges used once
    (the mesh boundary) must lie on the faces of the grid domain."""
    V = V.detach().numpy() if torch.is_tensor(V) else np.asarray(V)
    F = np.asarray(F)
    if not len(F):
        return {"faces": 0, "ok": True, "boundary_edges": 0}
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    und = np.sort(he, 1)
    ue, ucnt = np.unique(und, axis=0, return_counts=True)
    _, dcnt = np.unique(he, axis=0, return_counts=True)
    bnd = ue[ucnt == 1]
    lo, hi = grid.lo_np, grid.hi_np
    on_face = np.zeros(len(bnd), bool)
    for ax in range(3):
        for val in (lo[ax], hi[ax]):
            on_face |= (np.abs(V[bnd[:, 0], ax] - val) < tol) & (np.abs(V[bnd[:, 1], ax] - val) < tol)
    return {"faces": len(F), "nonmanifold_edges": int((ucnt > 2).sum()),
            "duplicate_directed_edges": int((dcnt > 1).sum()), "boundary_edges": int(len(bnd)),
            "boundary_edges_off_domain_faces": int((~on_face).sum()),
            "ok": bool((ucnt <= 2).all() and (dcnt == 1).all() and on_face.all())}
