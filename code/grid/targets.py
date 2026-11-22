"""
targets.py -- what a grid is fitted to: an implicit function from a Python file, or a
point cloud file, clipped to the grid's domain.

Function file (e.g. examples/functions/sphere_medium.py):
    a Python file that defines   g(X) -> values
    where X is a numpy array of shape (..., 3) and the result has shape (...),
    negative inside, positive outside, zero on the surface. Optional: NAME = "...".

Point cloud file:
    .xyz / .txt / .csv / .pts   one point per line: x y z [nx ny nz]  (space or comma separated)
    .npy                        array (N, 3) or (N, 6)
    .ply                        vertices with x y z [nx ny nz] (ascii or binary little endian)
    .obj                        "v" lines, and "vn" lines if there is one per vertex
    Normals are optional. Without them they are estimated (PCA of the 16 nearest
    neighbours, oriented consistently by propagation over the neighbour graph, and
    outward at the extreme point of each connected piece).

Both kinds provide the same interface to the optimizers:
    points, normals   (n, 3) samples on the surface (torch), unit outward normals
    label(X)          +1 outside, -1 inside: sign of g, or for a point cloud the side of
                      the nearest point's tangent plane, sign((x - p) . n)
    distance(X)       |g| / |grad g| (first-order distance); for a point cloud the distance
                      to the tangent plane of the nearest point (no sampling floor), or to
                      the point itself when that is more than two point spacings away
    dense(n)          many surface points for the final error evaluation
"""

from __future__ import annotations

import heapq
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import optimization.optimize as O
from core.box import Box
from core.paths import EXAMPLES
from grid.grid import DT, Grid

FUNCTION_DIR = EXAMPLES / "functions"
POINTS_DIR = EXAMPLES / "point_clouds"
POINT_EXTS = (".xyz", ".txt", ".csv", ".pts", ".npy", ".ply", ".obj")


# ============================================================================
# The target
# ============================================================================
@dataclass
class GridTarget:
    name: str
    kind: str                         # "function" | "points"
    points: torch.Tensor              # (n, 3) surface samples used by the optimizers
    normals: torch.Tensor             # (n, 3) unit outward normals
    cloud: np.ndarray                 # (N, 3) all surface points in the domain (display, evaluation)
    cloud_normals: np.ndarray
    fn: object = None                 # g(X) for function targets
    V: np.ndarray | None = None       # display mesh of a function target (clipped to the domain)
    F: np.ndarray | None = None
    source: str | None = None         # file it came from
    normals_estimated: bool = False
    grid: Grid | None = None
    _cloud_t: torch.Tensor | None = field(default=None, repr=False)

    def label(self, X: torch.Tensor) -> torch.Tensor:
        """+1 outside, -1 inside (0 counts as outside)."""
        if self.fn is not None:
            g = self.fn(X.detach().numpy())
            return torch.as_tensor(np.where(g > 0, 1.0, -1.0), dtype=DT)
        if self._cloud_t is None:
            self._cloud_t = torch.as_tensor(self.cloud, dtype=DT)
        idx = O.nearest(X, self._cloud_t)
        side = ((X - self._cloud_t[idx]) * torch.as_tensor(self.cloud_normals, dtype=DT)[idx]).sum(-1)
        return torch.where(side >= 0, 1.0, -1.0).to(DT)

    def distance(self, X: torch.Tensor) -> np.ndarray:
        if self.fn is not None:
            Xn = X.detach().numpy()
            G = O._fn_gradient(self.fn, Xn)
            return np.abs(self.fn(Xn)) / np.maximum(np.linalg.norm(G, axis=-1), 1e-12)
        if self._cloud_t is None:
            self._cloud_t = torch.as_tensor(self.cloud, dtype=DT)
        # distance to the nearest point's tangent plane, unless x is far from that point
        # along the plane (beyond the edge of an open cloud): then to the point itself.
        # The limit is 4 point spacings, because random samples leave gaps: with 2
        # spacings, 6 % of the points of a perfect surface fell in a gap and doubled the RMS.
        idx = O.nearest(X, self._cloud_t)
        diff = (X - self._cloud_t[idx]).detach()
        d_point = diff.norm(dim=-1).numpy()
        d_plane = np.abs((diff * torch.as_tensor(self.cloud_normals, dtype=DT)[idx]).sum(-1).numpy())
        along = np.sqrt(np.maximum(d_point ** 2 - d_plane ** 2, 0))
        return np.where(along < 4 * self.spacing(), d_plane, d_point)

    def spacing(self) -> float:
        """Median distance between neighbouring cloud points."""
        if not hasattr(self, "_spacing"):
            sub = self.cloud[np.random.default_rng(0).choice(len(self.cloud), min(2000, len(self.cloud)), False)]
            self._spacing = float(np.median(np.linalg.norm(sub - self.cloud[knn_to(sub, self.cloud)], axis=1)))
        return self._spacing

    def dense(self, n=20000, seed=123) -> np.ndarray:
        """Surface points for the final error evaluation (fresh samples for a function;
        the cloud itself, subsampled to n, for a point cloud)."""
        if self.fn is not None:
            return function_samples(self.fn, self.grid, n, seed)[0]
        if len(self.cloud) <= n:
            return self.cloud
        return self.cloud[np.random.default_rng(seed).choice(len(self.cloud), n, replace=False)]

    def describe(self) -> str:
        if self.kind == "function":
            return f"function '{self.name}'" + (f" ({self.source})" if self.source else "")
        est = ", normals estimated" if self.normals_estimated else ""
        return f"point cloud '{self.name}', {len(self.cloud)} points in the grid{est}"


# ============================================================================
# Functions
# ============================================================================
def load_function(path) -> tuple:
    """(g, name) from a Python file that defines g(X)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"cubespline_target_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "g", None)):
        raise ValueError(f"{path} must define a function g(X)")
    return mod.g, getattr(mod, "NAME", path.stem)


def _bounding_box(grid: Grid) -> Box:
    """The smallest cube containing the grid domain (the target mesher works on cubes)."""
    return Box(tuple(grid.lo_np), float(np.max(grid.hi_np - grid.lo_np)))


def function_samples(fn, grid: Grid, n: int, seed: int = 0):
    """n area-uniform points on g = 0 inside the grid domain, projected onto the exact
    zero set, with outward unit normals; plus the display mesh (V, F) clipped to the domain."""
    box = _bounding_box(grid)
    inside_frac = 1.0
    for attempt in range(4):
        want = int(n / max(inside_frac, 0.05) * 1.2) + 10
        t = O.target_from_function(fn, "f", box=box, n=want, seed=seed)
        P, N = t.points.numpy(), t.normals.numpy()
        keep = grid.contains(P)
        inside_frac = max(keep.mean(), 1e-3)
        if keep.sum() >= n or attempt == 3:
            break
    P, N = P[keep][:n], N[keep][:n]
    Fd = t.F[grid.contains(t.V[t.F].mean(1))]
    return P, N, t.V, Fd


def target_from_function(fn, name, grid: Grid, n=4000, seed=0, source=None) -> GridTarget:
    P, N, V, F = function_samples(fn, grid, n, seed)
    if len(P) < 50:
        raise ValueError(f"target '{name}' has (almost) no surface inside the grid")
    cloud, cloud_n = function_samples(fn, grid, 20000, seed + 1)[:2]
    return GridTarget(name, "function", torch.as_tensor(P, dtype=DT), torch.as_tensor(N, dtype=DT),
                      cloud, cloud_n, fn=fn, V=V, F=F, source=source, grid=grid)


# ============================================================================
# Point clouds
# ============================================================================
def _read_ply(path):
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header" or not line:
                break
        fmt, count, props, in_vertex = "ascii", 0, [], False
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex and parts[1] != "list":
                props.append((parts[2], parts[1]))
        names = [p[0] for p in props]
        if fmt == "ascii":
            rows = [fh.readline().split() for _ in range(count)]
            A = np.array(rows, float)[:, :len(names)]
        else:
            tmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
                    "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
                    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}
            end = "<" if "little" in fmt else ">"
            dt = np.dtype([(nm, end + tmap[tp]) for nm, tp in props])
            rec = np.frombuffer(fh.read(dt.itemsize * count), dtype=dt, count=count)
            A = np.stack([rec[nm].astype(float) for nm in names], 1)
    col = {nm: i for i, nm in enumerate(names)}
    P = A[:, [col["x"], col["y"], col["z"]]]
    N = A[:, [col["nx"], col["ny"], col["nz"]]] if all(k in col for k in ("nx", "ny", "nz")) else None
    return P, N


def load_point_cloud(path) -> tuple:
    """(points (N, 3), normals (N, 3) or None)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        A = np.load(path)
    elif ext == ".ply":
        return _read_ply(path)
    elif ext == ".obj":
        V, VN = [], []
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if parts[:1] == ["v"]:
                    V.append([float(x) for x in parts[1:4]])
                elif parts[:1] == ["vn"]:
                    VN.append([float(x) for x in parts[1:4]])
        P = np.array(V, float)
        return P, (np.array(VN, float) if len(VN) == len(V) and len(V) else None)
    else:
        with open(path) as fh:
            text = fh.read()
        delim = "," if "," in text.split("\n", 3)[-1] else None
        A = np.genfromtxt(path, comments="#", delimiter=delim)
        A = A[~np.isnan(A).any(1)] if A.ndim == 2 else A[None]
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[1] < 3:
        raise ValueError(f"{path}: expected N x 3 or N x 6 numbers")
    return A[:, :3], (A[:, 3:6] if A.shape[1] >= 6 else None)


def knn(P: np.ndarray, k: int) -> np.ndarray:
    """Indices (N, k) of the k nearest other points."""
    T = torch.as_tensor(P, dtype=torch.float32)
    out = []
    for i in range(0, len(T), 4096):
        out.append(torch.cdist(T[i:i + 4096], T).topk(k + 1, largest=False).indices[:, 1:])
    return torch.cat(out).numpy()


def knn_to(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Index of the nearest point of P to each X that is not X itself (X subset of P)."""
    d = torch.cdist(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(P, dtype=torch.float32))
    return d.topk(2, largest=False).indices[:, 1].numpy()


def estimate_normals(P: np.ndarray, k: int = 16) -> np.ndarray:
    """Unit normals by PCA of the k nearest neighbours, oriented consistently (Hoppe et
    al. 1992: propagate along the neighbour graph, most parallel neighbours first) and
    outward at the extreme (largest x) point of each connected piece."""
    P = np.asarray(P, float)
    nb = knn(P, k)
    Q = P[nb] - P[nb].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    _, vec = np.linalg.eigh(C)
    N = vec[:, :, 0].copy()
    done = np.zeros(len(P), bool)
    order = np.argsort(-P[:, 0])                     # seeds: largest x first
    for seed in order:
        if done[seed]:
            continue
        if N[seed, 0] < 0:                            # outward at the extreme point
            N[seed] *= -1
        done[seed] = True
        heap = [(1 - abs(float(N[seed] @ N[j])), int(j), int(seed)) for j in nb[seed] if not done[j]]
        heapq.heapify(heap)
        while heap:
            _, j, i = heapq.heappop(heap)
            if done[j]:
                continue
            if N[j] @ N[i] < 0:
                N[j] *= -1
            done[j] = True
            for m in nb[j]:
                if not done[m]:
                    heapq.heappush(heap, (1 - abs(float(N[j] @ N[m])), int(m), int(j)))
    return N


def target_from_points(P, N, name, grid: Grid, n=4000, seed=0, source=None) -> GridTarget:
    P = np.asarray(P, float)
    estimated = N is None
    N = estimate_normals(P) if estimated else np.asarray(N, float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    keep = grid.contains(P)
    if keep.sum() < 50:
        raise ValueError(f"point cloud '{name}' has (almost) no points inside the grid")
    P, N = P[keep], N[keep]
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(P), min(n, len(P)), replace=False)
    return GridTarget(name, "points", torch.as_tensor(P[sel], dtype=DT), torch.as_tensor(N[sel], dtype=DT),
                      P, N, source=source, normals_estimated=estimated, grid=grid)


# ============================================================================
# Finding files and examples
# ============================================================================
def list_examples() -> dict:
    """{'functions': [names], 'point_clouds': [file names]} in examples/."""
    fns = sorted(p.stem for p in FUNCTION_DIR.glob("*.py")) if FUNCTION_DIR.is_dir() else []
    pcs = sorted(p.name for p in POINTS_DIR.iterdir() if p.suffix.lower() in POINT_EXTS) if POINTS_DIR.is_dir() else []
    return {"functions": fns, "point_clouds": pcs}


def resolve_function(spec: str) -> Path:
    """A function file: a path, or the name of an example in examples/functions."""
    p = Path(spec).expanduser()
    for cand in (p, FUNCTION_DIR / p, FUNCTION_DIR / f"{spec}.py"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no function file '{spec}' (examples: {', '.join(list_examples()['functions'])})")


def resolve_points(spec: str) -> Path:
    """A point cloud file: a path, or the name of an example in examples/point_clouds."""
    p = Path(spec).expanduser()
    cands = [p, POINTS_DIR / p] + [POINTS_DIR / f"{spec}{e}" for e in POINT_EXTS]
    for cand in cands:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no point cloud '{spec}' (examples: {', '.join(list_examples()['point_clouds'])})")


def load_target(grid: Grid, function: str | None = None, points: str | None = None, n=4000, seed=0) -> GridTarget:
    """The target from a function file or a point cloud file (paths or example names)."""
    if (function is None) == (points is None):
        raise ValueError("give exactly one of a function file or a point cloud file")
    if function is not None:
        path = resolve_function(function)
        fn, name = load_function(path)
        return target_from_function(fn, name, grid, n=n, seed=seed, source=str(path))
    path = resolve_points(points)
    P, N = load_point_cloud(path)
    return target_from_points(P, N, path.stem, grid, n=n, seed=seed, source=str(path))


def default_grid(dims, target_bounds=None, lo=None, size=None) -> Grid:
    """The grid for `dims` cells: over the cube [lo, lo + size]^3 if given, else over
    the target's bounding box (plus one cell of margin) if known, else [-1, 1]^3."""
    if lo is not None or size is not None:
        lo = np.broadcast_to(np.asarray(-1.0 if lo is None else lo, float), (3,))
        return Grid.cube(dims, lo, 2.0 if size is None else size)
    if target_bounds is None:
        return Grid.cube(dims)
    blo, bhi = (np.asarray(b, float) for b in target_bounds)
    dims = np.broadcast_to(np.asarray(dims, int), (3,))
    h = float(np.max((bhi - blo) / np.maximum(dims - 2, 1)))     # one free cell on each side
    return Grid.covering(blo - h, bhi + h, dims)
