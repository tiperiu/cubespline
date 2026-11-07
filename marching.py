"""
marching.py -- iso-surface of a trilinear Cube by subdivided marching cubes.

Pipeline
--------
1. Octree descent with exact pruning. The cube is split into 2^d x 2^d x 2^d
   sub-cubes, one level at a time. A trilinear function restricted to any
   axis-aligned box is again trilinear (multilinear), so its min and max over the
   box are attained at the box corners: a sub-cube whose 8 corners are all on one
   side of the level contains NO surface and is discarded together with all its
   descendants. The pruning is exact, not a heuristic.

2. Marching cubes on the surviving leaves (all at the same depth, so the leaves
   form a conforming grid and the mesh has no cracks or T-junctions).
   * Ambiguous faces (diagonal corners of equal sign) are resolved with the
     asymptotic decider. On a sub-cube face the field is exactly bilinear, so the
     decider gives the TRUE topology of the level set on that face, and two
     neighbours sharing the face always make the same decision -> watertight.
   * The triangle table is generated programmatically from those face rules
     (no hand-typed 256-entry table), see `_case_triangles`.
   * Vertices are welded by global lattice-edge id.

3. Vertex placement. Along an axis-aligned edge the trilinear field is LINEAR in
   position, so the usual linear interpolation t = g_a / (g_a - g_b) gives the
   exact crossing: every mesh vertex lies exactly on the true level set (up to
   round-off), at every depth.

Differentiability (needed for the corner-value optimization in part 3)
-----------------------------------------------------------------------
The discrete part (which sub-cubes, which edges, the connectivity F) is computed
without gradients. Each vertex stores the two endpoints p_a, p_b of its lattice
edge and their trilinear basis rows W_a, W_b (fixed, independent of the values),
so for any corner values c

    g_a = W_a @ c - level,   g_b = W_b @ c - level,
    t   = g_a / (g_a - g_b),  V = p_a + t (p_b - p_a)

is a smooth torch expression in c. `Topology.vertices(c)` rebuilds V with the
connectivity frozen, which is exactly what an optimizer needs between topology
updates.

Conventions: "inside" means f < level, and triangles are oriented so that their
normals point towards increasing f (out of the inside region).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from cube import CORNER_IJK, CUBE_EDGES, Cube

# ----------------------------------------------------------------------------
# Cube combinatorics
# ----------------------------------------------------------------------------
# 6 faces: 4 corners in cyclic order, and the outward normal.
# For each axis the two opposite faces list matching corners in the same order
# (corner n on the low face <-> corner n + offset on the high face), so a face
# shared by two neighbouring sub-cubes is evaluated with identical arithmetic.
FACES = [
    ((0, 2, 6, 4), (-1, 0, 0)),   # x = 0
    ((1, 3, 7, 5), (+1, 0, 0)),   # x = 1
    ((0, 1, 5, 4), (0, -1, 0)),   # y = 0
    ((2, 3, 7, 6), (0, +1, 0)),   # y = 1
    ((0, 1, 3, 2), (0, 0, -1)),   # z = 0
    ((4, 5, 7, 6), (0, 0, +1)),   # z = 1
]
FACE_CORNERS = np.array([f[0] for f in FACES], dtype=np.int64)    # (6, 4)

_EDGE_OF = {}
for _e, (_a, _b) in enumerate(CUBE_EDGES):
    _EDGE_OF[(int(_a), int(_b))] = _e
    _EDGE_OF[(int(_b), int(_a))] = _e
EDGE_AXIS = np.array([e // 4 for e in range(12)], dtype=np.int64)  # 0:x 1:y 2:z


@lru_cache(maxsize=None)
def _case_triangles(config: int, decisions: int) -> tuple:
    """Triangles (tuple of (e0, e1, e2) local edge ids) for one cube case.

    config:    8-bit mask, bit n set <=> corner n is inside (f < level).
    decisions: 6-bit mask, bit f set <=> on ambiguous face f the INSIDE corners
               are connected through the face (asymptotic decider). Ignored for
               non-ambiguous faces.

    Construction: on each face, draw the segments separating inside from
    outside corners, oriented so the inside lies to the left when the face is
    seen from outside the cube. Every crossing edge then has exactly one
    outgoing and one incoming segment; chaining them gives closed polygons on
    the cube boundary, which are fan-triangulated.
    """
    inside = [(config >> n) & 1 for n in range(8)]
    P = CORNER_IJK.astype(float)
    mid = {e: 0.5 * (P[a] + P[b]) for e, (a, b) in enumerate(CUBE_EDGES)}

    nxt = {}
    for f, (cyc, normal) in enumerate(FACES):
        n = np.array(normal, dtype=float)
        s = [inside[c] for c in cyc]
        crossing = [_EDGE_OF[(cyc[i], cyc[(i + 1) % 4])] for i in range(4)
                    if s[i] != s[(i + 1) % 4]]
        if not crossing:
            continue
        if len(crossing) == 2:
            segs = [tuple(crossing)]
            # reference corner: the one shared by both edges if they are adjacent
            ea, eb = CUBE_EDGES[crossing[0]], CUBE_EDGES[crossing[1]]
            shared = set(ea.tolist()) & set(eb.tolist())
            refs = [shared.pop() if shared else cyc[0]]
        else:  # 4 crossings: ambiguous face, cut off the corners NOT connected
            connected = bool((decisions >> f) & 1)          # inside corners joined?
            cut_inside = not connected                      # then isolate inside ones
            segs, refs = [], []
            for i, c in enumerate(cyc):
                if bool(inside[c]) == cut_inside:
                    e_prev = _EDGE_OF[(cyc[i - 1], c)]
                    e_next = _EDGE_OF[(c, cyc[(i + 1) % 4])]
                    segs.append((e_prev, e_next))
                    refs.append(c)
        for (p, q), r in zip(segs, refs):
            side = np.dot(np.cross(mid[q] - mid[p], P[r] - mid[p]), n)
            want_left = inside[r] == 1
            if (side > 0) != want_left:
                p, q = q, p
            assert p not in nxt, "inconsistent segment orientation"
            nxt[p] = q

    # chain into loops
    tris, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, e = [], start
        while e not in seen:
            seen.add(e)
            loop.append(e)
            e = nxt[e]
        assert e == start, "open loop in case table"
        for i in range(1, len(loop) - 1):
            # loop runs with inside on the left seen from outside the cube;
            # reversing makes the normal point from inside to outside (+grad f)
            tris.append((loop[0], loop[i + 1], loop[i]))
    return tuple(tris)


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class Topology:
    """Discrete output of the mesher plus what is needed to rebuild positions.

    F        (nf, 3) int64   triangles, indices into the vertex array
    pa, pb   (nv, 3) tensor  world endpoints of each vertex's lattice edge
    Wa, Wb   (nv, 8) tensor  trilinear basis at pa / pb (independent of values)
    cells    (nc, 3) int64   active leaf sub-cubes, integer coords at resolution R
    """
    F: np.ndarray
    pa: torch.Tensor
    pb: torch.Tensor
    Wa: torch.Tensor
    Wb: torch.Tensor
    cells: np.ndarray
    depth: int
    level: float
    lo: torch.Tensor
    size: float
    stats: dict = field(default_factory=dict)

    @property
    def resolution(self) -> int:
        return 2 ** self.depth

    @property
    def num_vertices(self) -> int:
        return len(self.pa)

    def vertices(self, values: torch.Tensor, level: float | None = None) -> torch.Tensor:
        """Vertex positions (nv, 3) for corner `values` with this connectivity.
        Differentiable w.r.t. `values` (and `level`, if a tensor)."""
        lv = self.level if level is None else level
        ga = self.Wa @ values - lv
        gb = self.Wb @ values - lv
        denom = ga - gb
        # keep the division safe if the values drifted away from the topology
        eps = torch.finfo(values.dtype).eps
        denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
        t = (ga / denom).clamp(0.0, 1.0)
        return self.pa + t[:, None] * (self.pb - self.pa)

    def cell_bounds(self):
        """(nc, 3) min corners and the sub-cube edge length, in world coords."""
        h = self.size / self.resolution
        lo = self.lo.detach().cpu().numpy()
        return lo + self.cells * h, h


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
SNAP_REL = 1e-12   # |f - level| below SNAP_REL * scale is treated as exactly 0


def _snap_tolerance(cube: Cube, level: float) -> float:
    c = cube.values.detach().cpu().numpy().astype(np.float64)
    return SNAP_REL * max(float(np.abs(c - level).max()), 1e-300)


def _lattice_values(cube: Cube, pts_ijk: np.ndarray, R: int, level: float) -> np.ndarray:
    """g = f - level at integer lattice points (m, 3) of resolution R (no grad).

    The discrete decisions (inside/outside, pruning, face saddles) must give the
    same answer for the same lattice point every time it is evaluated, at every
    level of the descent and in every neighbouring cell. So this is computed
    element-wise in numpy float64 (no BLAS matmul, whose rounding can depend on
    the batch), and values within round-off of the level are snapped to exactly
    0, which counts as outside.
    """
    c = cube.values.detach().cpu().numpy().astype(np.float64)
    u = np.asarray(pts_ijk, dtype=np.float64) / R            # exact: dyadic coordinates
    g = _trilinear_np(c, u) - level
    g[np.abs(g) < _snap_tolerance(cube, level)] = 0.0
    return g


def _trilinear_np(c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of 8 values c (n = i + 2j + 4k) at local points u (m, 3)."""
    c = c.reshape(2, 2, 2)                                    # [k, j, i]
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    c00 = c[0, 0, 0] + (c[0, 0, 1] - c[0, 0, 0]) * x
    c10 = c[0, 1, 0] + (c[0, 1, 1] - c[0, 1, 0]) * x
    c01 = c[1, 0, 0] + (c[1, 0, 1] - c[1, 0, 0]) * x
    c11 = c[1, 1, 0] + (c[1, 1, 1] - c[1, 1, 0]) * x
    c0 = c00 + (c10 - c00) * y
    c1 = c01 + (c11 - c01) * y
    return c0 + (c1 - c0) * z


def _corner_values(cube, cells, R, level):
    """(nc, 8) values at the corners of integer cells, each lattice point evaluated once."""
    corners = cells[:, None, :] + CORNER_IJK[None, :, :]                # (nc, 8, 3)
    flat = corners.reshape(-1, 3)
    key = flat[:, 0] + (R + 1) * (flat[:, 1] + (R + 1) * flat[:, 2])
    ukey, inv = np.unique(key, return_inverse=True)
    upts = np.stack([ukey % (R + 1), (ukey // (R + 1)) % (R + 1), ukey // (R + 1) ** 2], 1)
    g = _lattice_values(cube, upts, R, level)
    return g[inv].reshape(len(cells), 8)


def _mesh_cells(cells: np.ndarray, g: np.ndarray, R: int):
    """Marching cubes on integer cells (nc, 3) of a resolution-R lattice with corner
    values g (nc, 8) (already relative to the level; < 0 is inside).

    Returns (F, a_ijk, b_ijk, n_cases, n_ambiguous_faces), or None if no triangles.
    Vertex v lies on the lattice edge a_ijk[v] -> b_ijk[v] (integer lattice points);
    the caller decides how to place it along the edge.
    """
    # ---- case per cell: corner config + asymptotic decider on ambiguous faces
    inside = g < 0                                                      # (nc, 8)
    config = (inside * (1 << np.arange(8))).sum(1)
    fv = g[:, FACE_CORNERS]                                             # (nc, 6, 4)
    fin = inside[:, FACE_CORNERS]
    amb = (fin[..., 0] == fin[..., 2]) & (fin[..., 1] == fin[..., 3]) & (fin[..., 0] != fin[..., 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        saddle = (fv[..., 0] * fv[..., 2] - fv[..., 1] * fv[..., 3]) / \
                 (fv[..., 0] + fv[..., 2] - fv[..., 1] - fv[..., 3])
    joined = amb & (saddle < 0)                                         # inside corners connected
    decisions = (joined * (1 << np.arange(6))).sum(1)
    case_key = config * 64 + decisions

    # ---- global lattice-edge id for the 12 edges of every cell
    a_pts = cells[:, None, :] + CORNER_IJK[CUBE_EDGES[:, 0]][None]      # (nc, 12, 3)
    a_idx = a_pts[..., 0] + (R + 1) * (a_pts[..., 1] + (R + 1) * a_pts[..., 2])
    edge_gid = a_idx * 3 + EDGE_AXIS[None, :]                           # (nc, 12)

    tri_list = []
    ukeys, inv = np.unique(case_key, return_inverse=True)
    for u, key in enumerate(ukeys):
        tris = _case_triangles(int(key) // 64, int(key) % 64)
        if not tris:
            continue
        sel = np.nonzero(inv == u)[0]
        tri_list.append(edge_gid[sel][:, np.asarray(tris)].reshape(-1, 3))
    if not tri_list:
        return None
    tri_gid = np.concatenate(tri_list, 0)

    # ---- weld: one vertex per lattice edge
    ugid, F = np.unique(tri_gid, return_inverse=True)
    F = F.reshape(-1, 3).astype(np.int64)
    axis = ugid % 3
    a_lin = ugid // 3
    a_ijk = np.stack([a_lin % (R + 1), (a_lin // (R + 1)) % (R + 1), a_lin // (R + 1) ** 2], 1)
    b_ijk = a_ijk + np.eye(3, dtype=np.int64)[axis]

    return F, a_ijk, b_ijk, len(ukeys), int(amb.sum())


def extract_topology(cube: Cube, depth: int = 3, level: float = 0.0) -> Topology:
    """Subdivide `cube` `depth` times (2^depth cells per axis), prune sub-cubes
    the surface cannot touch, and run marching cubes on the rest."""
    depth = int(depth)
    level = float(level)
    child_offsets = CORNER_IJK                                          # (8, 3)

    # Coarse cells are pruned only when all corners are clearly on one side
    # (beyond a margin of a few round-off tolerances), so a point that sits on
    # the level up to round-off can never be "outside" at a coarse level and
    # "inside" at a fine one. Cells kept by the margin but empty just produce
    # no triangles. The final level uses the plain sign test.
    margin = 4 * _snap_tolerance(cube, level)
    cells = np.zeros((1, 3), dtype=np.int64)
    per_level = []
    for lvl in range(depth + 1):
        R = 2 ** lvl
        g = _corner_values(cube, cells, R, level)
        if lvl < depth:
            active = ~((g > margin).all(1) | (g < -margin).all(1))
        else:
            n_in = (g < 0).sum(1)
            active = (n_in > 0) & (n_in < 8)
        per_level.append((lvl, len(cells), int(active.sum())))
        cells, g = cells[active], g[active]
        if lvl < depth and len(cells):
            cells = (2 * cells[:, None, :] + child_offsets[None]).reshape(-1, 3)
        if not len(cells):
            break

    R = 2 ** depth
    stats = {"levels": per_level, "active_cells": len(cells),
             "total_cells": R ** 3, "resolution": R}
    dev, dt = cube.lo.device, cube.dtype
    empty = Topology(np.zeros((0, 3), np.int64), torch.zeros(0, 3, dtype=dt, device=dev),
                     torch.zeros(0, 3, dtype=dt, device=dev), torch.zeros(0, 8, dtype=dt, device=dev),
                     torch.zeros(0, 8, dtype=dt, device=dev), np.zeros((0, 3), np.int64),
                     depth, level, cube.lo.detach().clone(), cube.size, stats)
    if not len(cells) or len(per_level) <= depth:
        return empty

    meshed = _mesh_cells(cells, g, R)
    if meshed is None:
        return empty
    F, a_ijk, b_ijk, n_cases, n_amb = meshed

    ua = torch.as_tensor(a_ijk, dtype=dt, device=dev) / R
    ub = torch.as_tensor(b_ijk, dtype=dt, device=dev) / R
    Wa, Wb = Cube.basis_local(ua), Cube.basis_local(ub)
    lo = cube.lo.detach()
    pa, pb = lo + ua * cube.size, lo + ub * cube.size

    stats.update({"vertices": len(a_ijk), "faces": len(F),
                  "cases": n_cases, "ambiguous_faces": n_amb})
    return Topology(F, pa, pb, Wa, Wb, cells, depth, level, lo.clone(), cube.size, stats)


def marching_cubes(cube: Cube, depth: int = 3, level: float = 0.0):
    """Convenience: (V, F, topology).
    V is a torch tensor (nv, 3) attached to cube.values' graph; F is int64 (nf, 3)."""
    topo = extract_topology(cube, depth, level)
    return topo.vertices(cube.values), topo.F, topo


def mesh_from_grid(G: np.ndarray, lo=-1.0, size=2.0, level: float = 0.0):
    """Iso-surface of an arbitrary function sampled on a regular grid.

    G: (R+1, R+1, R+1) values indexed [i, j, k] on the lattice of [lo, lo+size]^3.
    Same case table, face decider and welding as extract_topology, so the mesh is
    watertight inside the box and oriented towards increasing G. Vertices are
    placed by linear interpolation along lattice edges (exact for trilinear data,
    an approximation otherwise). Returns numpy (V, F).
    """
    G = np.asarray(G, dtype=np.float64) - level
    R = G.shape[0] - 1
    tol = SNAP_REL * max(float(np.abs(G).max()), 1e-300)
    G = np.where(np.abs(G) < tol, 0.0, G)
    ijk = np.stack(np.meshgrid(*[np.arange(R)] * 3, indexing="ij"), -1).reshape(-1, 3)
    corners = ijk[:, None, :] + CORNER_IJK[None]
    g = G[corners[..., 0], corners[..., 1], corners[..., 2]]             # (R^3, 8)
    n_in = (g < 0).sum(1)
    mixed = (n_in > 0) & (n_in < 8)
    meshed = _mesh_cells(ijk[mixed], g[mixed], R)
    if meshed is None:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    F, a_ijk, b_ijk, _, _ = meshed
    ga = G[a_ijk[:, 0], a_ijk[:, 1], a_ijk[:, 2]]
    gb = G[b_ijk[:, 0], b_ijk[:, 1], b_ijk[:, 2]]
    t = np.clip(ga / (ga - gb), 0.0, 1.0)
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (3,))
    V = lo + (a_ijk + t[:, None] * (b_ijk - a_ijk)) * (size / R)
    return V, F


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def to_numpy(V):
    return V.detach().cpu().numpy() if torch.is_tensor(V) else np.asarray(V)


def vertex_normals(cube: Cube, V) -> np.ndarray:
    """Unit normals from the analytic field gradient (point towards +f)."""
    with torch.no_grad():
        n = cube.gradient(torch.as_tensor(to_numpy(V), dtype=cube.dtype))
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return n.cpu().numpy()


def save_obj(path, V, F):
    V, F = to_numpy(V), np.asarray(F)
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for f in F + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")


def mesh_check(V, F, lo, size, tol=1e-9) -> dict:
    """Combinatorial sanity checks. A trilinear level set has no closed
    components inside the cube (no interior extrema), so the mesh boundary must
    lie on the cube faces; every other edge must be shared by exactly two
    consistently oriented triangles."""
    V, F = to_numpy(V), np.asarray(F)
    lo = np.broadcast_to(np.asarray(to_numpy(lo), dtype=float), (3,))
    hi = lo + size
    if not len(F):
        return {"faces": 0, "ok": True}
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)   # directed
    und = np.sort(he, 1)
    _, cnt = np.unique(und, axis=0, return_counts=True)
    _, dcnt = np.unique(he, axis=0, return_counts=True)
    ue, ucnt = np.unique(und, axis=0, return_counts=True)
    bnd = ue[ucnt == 1]
    on_face = np.zeros(len(bnd), bool)
    for ax in range(3):
        for val in (lo[ax], hi[ax]):
            on_face |= (np.abs(V[bnd[:, 0], ax] - val) < tol) & (np.abs(V[bnd[:, 1], ax] - val) < tol)
    return {
        "faces": len(F),
        "nonmanifold_edges": int((cnt > 2).sum()),
        "duplicate_directed_edges": int((dcnt > 1).sum()),
        "boundary_edges": int(len(bnd)),
        "boundary_edges_off_cube_faces": int((~on_face).sum()),
        "ok": bool((cnt <= 2).all() and (dcnt == 1).all() and on_face.all()),
    }
