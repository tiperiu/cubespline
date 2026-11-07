#!/usr/bin/env python3
"""
optimize.py -- fit the 8 corner values so the marching-cubes surface matches a target.

    target (function g(x) or mesh), clipped to the cube
        -> samples: points + outward normals
    init:   closed-form algebraic fit (Taubin), linear in the 8 values
    refine: for each iteration
              topology = extract_topology(cube, depth)        discrete, no gradients
              V        = topology.vertices(values)            differentiable
              loss     = sum_k w_k * LOSSES[k](...)           default: chamfer
              Adam step on the values, then renormalise |values|

Losses live in the LOSSES registry: add a function with the signature
`fn(ctx) -> scalar tensor` and select it with --loss name=weight.

Conventions as in marching.py: inside is f < 0, normals point towards +f, so a
target mesh should have outward-facing normals (f > 0 outside the target).

Usage:
    python optimize.py --target sphere --depth 5 --refine-iters 300
    python optimize.py --target "sphere flank" --no-refine     # closed-form start only
    python optimize.py --target trilinear --view          # watch it converge
    python optimize.py --mesh bunny.obj --loss chamfer=1 --loss algebraic=0.1
    python optimize.py --list
The fitted values are written to fit.json; open them with  python viewer.py --load fit.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from cube import Cube, CUBE_EDGES
from marching import extract_topology, mesh_from_grid, to_numpy

DT = torch.float64


# ============================================================================
# Targets
# ============================================================================
@dataclass
class Target:
    """A target surface restricted to the cube.
    V, F           target mesh (inside the cube) -- for display
    points, normals  (N, 3) torch samples on the target, area-uniform, unit normals
    """
    name: str
    V: np.ndarray
    F: np.ndarray
    points: torch.Tensor
    normals: torch.Tensor
    values: np.ndarray | None = None   # ground-truth 8 values, if the target is trilinear
    fn: object = None                  # g(x) for function targets (None for meshes)


def sample_surface(V, F, n, rng):
    """n area-uniform samples on a triangle mesh -> (points, unit face normals)."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = np.linalg.norm(cr, axis=1)
    keep = area > 0
    F, a, b, c, cr, area = F[keep], a[keep], b[keep], c[keep], cr[keep], area[keep]
    idx = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    N = cr[idx] / area[idx, None]
    return P, N


def _inside_box(P, lo, size, eps=1e-9):
    return np.all((P >= lo - eps) & (P <= lo + size + eps), axis=1)


def _fn_gradient(fn, P, h=1e-6):
    """Central-difference gradient of a numpy function at points P (m, 3)."""
    E = np.eye(3) * h
    return np.stack([(fn(P + E[a]) - fn(P - E[a])) / (2 * h) for a in range(3)], -1)


def project_to_zero_set(fn, P, steps=4):
    """Newton steps x <- x - g grad g / |grad g|^2 onto g = 0."""
    for _ in range(steps):
        g, G = fn(P), _fn_gradient(fn, P)
        P = P - (g / np.maximum((G * G).sum(-1), 1e-30))[:, None] * G
    return P


def target_from_function(fn, name="function", lo=-1.0, size=2.0, res=64, n=4000, seed=0):
    """fn: numpy (..., 3) -> (...), negative inside. Its zero set inside the cube is
    meshed with the same marching cubes (mesh_from_grid), sampled area-uniformly,
    and the samples are projected onto the true zero set (Newton); normals are
    the normalised gradient of fn (pointing towards fn > 0)."""
    t = np.linspace(lo, lo + size, res + 1)
    X = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1)
    V, F = mesh_from_grid(fn(X), lo, size)
    if not len(F):
        raise ValueError(f"target '{name}' has no surface inside the cube")
    rng = np.random.default_rng(seed)
    P, _ = sample_surface(V, F, int(1.2 * n), rng)
    P = project_to_zero_set(fn, P)
    P = P[_inside_box(P, lo, size)][:n]
    N = _fn_gradient(fn, P)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return Target(name, V, F, torch.as_tensor(P, dtype=DT), torch.as_tensor(N, dtype=DT), fn=fn)


def target_from_mesh(V, F, name="mesh", lo=-1.0, size=2.0, n=4000, seed=0):
    """Sample a triangle mesh and keep only the samples inside the cube (clipping).
    For display, keep the triangles whose centroid is inside the cube."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    rng = np.random.default_rng(seed)
    P, N = sample_surface(V, F, 8 * n, rng)
    m = _inside_box(P, lo, size)
    if m.sum() < 50:
        raise ValueError(f"mesh '{name}' has almost no area inside the cube [{lo}, {lo + size}]^3")
    P, N = P[m], N[m]
    if len(P) > n:
        sel = rng.choice(len(P), n, replace=False)
        P, N = P[sel], N[sel]
    Fd = F[_inside_box(V[F].mean(1), lo, size)]
    return Target(name, V, Fd, torch.as_tensor(P, dtype=DT), torch.as_tensor(N, dtype=DT))


def load_obj(path):
    """Minimal OBJ reader (v / f lines, polygons fan-triangulated)."""
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                V.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                F += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(V, float), np.array(F, np.int64)


def _trilinear_target(seed=7):
    vals = np.random.default_rng(seed).normal(size=8)
    cube = Cube(vals, requires_grad=False)

    def fn(X):
        with torch.no_grad():
            return cube.evaluate(torch.as_tensor(X.reshape(-1, 3))).numpy().reshape(X.shape[:-1])
    return fn, vals


# Built-in function targets on the cube [-1, 1]^3 (negative inside).
FUNCTIONS = {
    "sphere":      lambda X: np.linalg.norm(X - [0.3, 0.2, 0.1], axis=-1) - 0.8,
    "small sphere": lambda X: np.linalg.norm(X - [-0.2, 0.1, 0.0], axis=-1) - 0.45,
    "cylinder":    lambda X: np.hypot(X[..., 0] - 0.1, X[..., 2] + 0.2) - 0.6,
    "plane":       lambda X: X @ np.array([0.3, 0.8, 0.52]) - 0.2,
    "hyperboloid": lambda X: X[..., 0] ** 2 + X[..., 1] ** 2 - X[..., 2] ** 2 - 0.25,
    "paraboloid":  lambda X: X[..., 1] - 0.8 * (X[..., 0] ** 2 + X[..., 2] ** 2) + 0.5,
    "torus":       lambda X: np.hypot(np.hypot(X[..., 0], X[..., 2]) - 0.6, X[..., 1]) - 0.25,
}


# ----------------------------------------------------------------------------
# Slices: large analytic shapes placed so that only a patch of their surface
# crosses the cube. A single trilinear cube cannot close a surface, but it can
# follow a patch.
#
# There is a sharper limit: on every plane z = const the field is bilinear in
# (x, y), so the fitted curve there is y(x) = -(A + Bx)/(C + Dx), a Moebius
# function, which is monotone. The same holds for the other two plane families.
# So the fitted surface can have no bump, dip or crest inside the cube: in every
# axis-aligned slice it rises or falls monotonically. Equivalently, no component
# of its normal changes sign inside the cube.
#
#   "... cap", "torus outer/inner": the apex (sphere) or the crest (torus) is
#       inside the cube. Some normal component changes sign, which one cube
#       cannot reproduce, so the fit stalls at a fixed error.
#   "... flank": the same kind of shape, placed so the sphere centre (or the
#       tube centre and the torus axis) lies outside the cube's range in x, y
#       and z. The normal keeps its signs over the patch, and one cube can
#       follow it closely.
# ----------------------------------------------------------------------------
def sphere_slice(center, radius):
    c = np.asarray(center, float)
    return lambda X: np.linalg.norm(X - c, axis=-1) - radius


def torus_slice(axis_x, axis_z, R, r, tube_y=0.0):
    """Torus whose symmetry axis is parallel to y through (axis_x, axis_z),
    major radius R, tube radius r, tube centre at height tube_y; negative inside."""
    return lambda X: np.hypot(np.hypot(X[..., 0] - axis_x, X[..., 2] - axis_z) - R,
                              X[..., 1] - tube_y) - r


def _sphere_through_origin(normal, radius):
    n = np.asarray(normal, float)
    return sphere_slice(-radius * n / np.linalg.norm(n), radius)


def _torus_flank(R, r, tube_y, axis_z, outer):
    """Torus slice through the origin: outer (rho > R) or inner (rho < R) side."""
    d = np.sqrt(r ** 2 - tube_y ** 2)                 # |rho - R| at the origin
    rho0 = R + d if outer else R - d
    axis_x = (-1 if outer else 1) * np.sqrt(rho0 ** 2 - axis_z ** 2)
    return torus_slice(axis_x, axis_z, R, r, tube_y)


SLICES = {
    # --- apex / crest inside the cube: the limit of a single cube ---------
    # sphere of radius 2.5 below the cube; the cap's top (y = 0.2) is in the middle
    "sphere cap":         sphere_slice([0.0, -2.3, 0.0], 2.5),
    # flatter cap of a radius-5 sphere, centre off to the side in x only
    "large sphere cap":   sphere_slice([-1.5, -4.6, 0.8], 5.0),
    # outer equator of a torus (R 3, r 1.5): x is extremal at y = 0, z = 0
    "torus outer":        torus_slice(-4.5, 0.0, 3.0, 1.5),
    # inner equator (R 3.5, r 1.5), facing the hole: a saddle
    "torus inner":        torus_slice(2.0, 0.0, 3.5, 1.5),
    # --- flanks: same shapes, apex / crest outside the cube ---------------
    # radius 2.5, centre at (-1.55, -1.55, -1.21): the cube sees a flank of the ball
    "sphere flank":       _sphere_through_origin([1.0, 1.0, 0.78], 2.5),
    # radius 5, centre at (-2.33, -3.73, -2.33)
    "large sphere flank": _sphere_through_origin([0.5, 0.8, 0.5], 5.0),
    # torus R 3, tube r 2.6, tube centre at y = -1.3, axis at z = -1.6:
    # tube crest (y = 1.3) above the cube, equator (y = -1.3) below it, outer side
    "torus outer flank":  _torus_flank(3.0, 2.6, -1.3, -1.6, outer=True),
    # same tube on the inner (saddle) side of a torus with R 4.4, hole towards +x
    "torus inner flank":  _torus_flank(4.4, 2.6, -1.3, -1.6, outer=False),
}
FUNCTIONS.update(SLICES)


def builtin_target(name, n=4000, seed=0):
    if name == "trilinear":
        fn, vals = _trilinear_target()
        t = target_from_function(fn, name, n=n, seed=seed)
        t.values = vals
        return t
    return target_from_function(FUNCTIONS[name], name, n=n, seed=seed)


# ============================================================================
# Closed-form initialisation
# ============================================================================
def fit_taubin(cube: Cube, points: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
    """Trilinear field vanishing on the points, in Taubin's sense:

        minimise   sum_i f(x_i)^2 / sum_i |grad f(x_i)|^2     over the 8 values c.

    Both sums are quadratic forms in c, so this is a generalised symmetric
    eigenproblem. Exact when the points lie on a trilinear surface. It ignores
    orientation, so for curved targets it can return a saddle-like field whose
    zero set passes near the points without following the target's sides; the
    sign is chosen so grad f agrees with the normals on average.
    """
    with torch.no_grad():
        B = cube.basis(points)                                  # (N, 8)
        G = cube.basis_gradient(points)                         # (N, 3, 8)
        A = B.T @ B / len(B)
        M = torch.einsum("nki,nkj->ij", G, G) / len(B)
        # the constant field has zero gradient: M is singular along (1,..,1);
        # a tiny ridge keeps the Cholesky well defined and rules that direction out
        M = M + 1e-9 * torch.trace(M) / 8 * torch.eye(8, dtype=M.dtype)
        Li = torch.linalg.inv(torch.linalg.cholesky(M))
        _, evecs = torch.linalg.eigh(Li @ A @ Li.T)
        c = Li.T @ evecs[:, 0]
        if normals is not None and ((G @ c) * normals).sum() < 0:
            c = -c
        return c / c.norm() * np.sqrt(8)


def fit_with_normals(cube: Cube, points: torch.Tensor, normals: torch.Tensor, mu: float) -> torch.Tensor:
    """Linear least squares   sum_i f(x_i)^2 + mu * sum_i |grad f(x_i) - n_i|^2.
    The normal term fixes the sign and the scale and makes the field follow the
    target's inside/outside; small mu stays close to interpolating the points."""
    with torch.no_grad():
        B = cube.basis(points)
        G = cube.basis_gradient(points).reshape(-1, 8)
        A = torch.cat([B, np.sqrt(mu) * G], 0)
        b = torch.cat([torch.zeros(len(B), dtype=B.dtype), np.sqrt(mu) * normals.reshape(-1)])
        c = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
        return c / c.norm() * np.sqrt(8)


def fit_algebraic(cube: Cube, target: "Target", mus=(1e-3, 1e-2, 1e-1, 1.0), depth=4, verbose=False):
    """Closed-form initialisation: try Taubin and the normal-constrained fits for
    several mu, keep the candidate with the smallest true surface distance."""
    cands = {"taubin": fit_taubin(cube, target.points, target.normals)}
    for mu in mus:
        cands[f"normals mu={mu:g}"] = fit_with_normals(cube, target.points, target.normals, mu)
    scores = {}
    for k, c in cands.items():
        m = evaluate(c.numpy(), target, depth=depth, n=4000)
        scores[k] = m.get("rms", np.inf)
    best = min(scores, key=scores.get)
    if verbose:
        print("  init candidates:", ", ".join(f"{k} {v:.3e}" for k, v in scores.items()), f" -> {best}")
    return cands[best], best


# ============================================================================
# Losses (registry)
# ============================================================================
@dataclass
class LossContext:
    """Everything a loss term may use. Add fields here when a new term needs them."""
    cube: Cube
    values: torch.Tensor          # the 8 values (requires grad)
    V: torch.Tensor               # (nv, 3) fitted mesh vertices, differentiable
    F: np.ndarray                 # (nf, 3)
    mesh_points: torch.Tensor     # (M, 3) area-uniform samples on the fitted mesh, differentiable
    target: Target


def surface_samples(V: torch.Tensor, F: np.ndarray, n: int, gen: torch.Generator) -> torch.Tensor:
    """Area-uniform samples on a mesh; the choice of triangles / barycentrics is
    discrete (no gradient), the sample positions are differentiable in V."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    with torch.no_grad():
        area = torch.linalg.cross(b - a, c - a).norm(dim=1)
        if area.sum() <= 0:
            return V
        idx = torch.multinomial(area, n, replacement=True, generator=gen)
        r1 = torch.rand(n, dtype=V.dtype, generator=gen)
        r2 = torch.rand(n, dtype=V.dtype, generator=gen)
        s = r1.sqrt()
    return (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]


def nearest(X: torch.Tensor, Y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Index of the nearest Y for every X (no gradient; float32 search)."""
    with torch.no_grad():
        Yf = Y.detach().float()
        return torch.cat([torch.cdist(X[i:i + chunk].detach().float(), Yf).argmin(1)
                          for i in range(0, len(X), chunk)])


def chamfer(P: torch.Tensor, Q: torch.Tensor):
    """Two-sided chamfer: mean squared distance P->Q plus Q->P.
    Nearest neighbours are found without gradients; the loss is then formed from
    the matched pairs only, which gives the same gradient as differentiating the
    full distance matrix (the min is piecewise) at a fraction of the cost."""
    return (P - Q[nearest(P, Q)]).square().sum(-1).mean() + \
           (P[nearest(Q, P)] - Q).square().sum(-1).mean()


def loss_chamfer(ctx: LossContext):
    return chamfer(ctx.mesh_points, ctx.target.points)


def loss_algebraic(ctx: LossContext):
    """First-order (Sampson) distance of the target points to the level set,
    f / |grad f|; scale invariant and defined even when the mesh is empty."""
    f = ctx.cube.basis(ctx.target.points) @ ctx.values
    g = ctx.cube.basis_gradient(ctx.target.points) @ ctx.values
    return (f.square() / g.square().sum(-1).clamp_min(1e-12)).mean()


def loss_normal(ctx: LossContext):
    """1 - cos between the field gradient and the target normal at the target points."""
    g = ctx.cube.basis_gradient(ctx.target.points) @ ctx.values
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (1 - (g * ctx.target.normals).sum(-1)).mean()


LOSSES = {
    "chamfer": loss_chamfer,
    "algebraic": loss_algebraic,
    "normal": loss_normal,
}


# ============================================================================
# Optimiser
# ============================================================================
@dataclass
class FitConfig:
    depth: int = 5
    iters: int = 300
    lr: float = 0.05
    lr_final: float = 5e-4            # cosine decay from lr to lr_final over `iters`
    mesh_samples: int = 3000
    refresh_every: int = 1            # re-extract the topology every k iterations
    losses: dict = field(default_factory=lambda: {"chamfer": 1.0})
    init: str = "algebraic"           # "algebraic" | "random" | "given"
    seed: int = 0
    keep_best: bool = True            # return the best values seen (deterministic check)
    check_every: int = 10             # how often the deterministic check runs


class Fitter:
    def __init__(self, target: Target, config: FitConfig, values=None, verbose=False):
        self.target, self.cfg = target, config
        self.verbose = verbose
        self.init_choice = "given"
        self.cube = Cube(requires_grad=True)
        self.gen = torch.Generator().manual_seed(config.seed)
        self.reset(values)

    def reset(self, values=None):
        cfg = self.cfg
        if values is not None or cfg.init == "given":
            v = torch.as_tensor(values, dtype=DT)
        elif cfg.init == "random":
            v = torch.randn(8, dtype=DT, generator=self.gen)
        else:
            v, self.init_choice = fit_algebraic(self.cube, self.target, verbose=self.verbose)
        self.cube.set_values(v / v.norm() * np.sqrt(8))
        self.opt = torch.optim.Adam([self.cube.values], lr=cfg.lr)
        self.it = 0
        self.history = []            # (iteration, total, {term: value})
        self.topo = None
        self.V = torch.zeros(0, 3, dtype=DT)
        self.F = np.zeros((0, 3), np.int64)
        self.best_values, self.best_score, self.best_it = self.values, self.score(), 0
        self.init_values = self.values            # starting values (for the final report)
        self.last_score = self.best_score         # true RMS at the latest check

    def objective(self, values=None):
        """The configured loss terms at `values` (default: current), with a fixed
        set of mesh samples so the number is reproducible. Returns (total, {term: value})."""
        cube = self.cube if values is None else Cube(values, requires_grad=False)
        with torch.no_grad():
            topo = extract_topology(cube, self.cfg.depth)
            if not len(topo.F):
                return float("inf"), {}
            V = topo.vertices(cube.values)
            pts = surface_samples(V, topo.F, self.cfg.mesh_samples, torch.Generator().manual_seed(12345))
            ctx = LossContext(cube, cube.values, V, topo.F, pts, self.target)
            parts = {k: float(LOSSES[k](ctx)) for k, w in self.cfg.losses.items() if w}
        return float(sum(self.cfg.losses[k] * v for k, v in parts.items())), parts

    def score(self, values=None) -> float:
        """Deterministic check used to keep the best iterate: the true RMS surface
        distance (see `evaluate`) at the fitting depth. The training loss is not
        used here because chamfer between finite sample sets has a slightly biased
        minimum (it can prefer a surface that is a little off the target)."""
        v = self.values if values is None else values
        return float(evaluate(v, self.target, depth=self.cfg.depth, n=self.cfg.mesh_samples).get("rms", np.inf))

    def step(self):
        cfg, cube = self.cfg, self.cube
        if self.topo is None or self.it % cfg.refresh_every == 0:
            self.topo = extract_topology(cube, cfg.depth)
        V = self.topo.vertices(cube.values)
        F = self.topo.F
        terms = dict(cfg.losses)
        if not len(F):
            # The surface left the cube (or never entered it): the mesh losses have
            # nothing to act on, and for a near-constant field no gradient step can
            # bring it back (the step is parallel to the values and renormalising
            # undoes it). Restart from the closed-form fit, plus a small random
            # perturbation if even that is empty.
            v, _ = fit_algebraic(cube, self.target)
            if not len(extract_topology(Cube(v, requires_grad=False), cfg.depth).F):
                v = v + 0.3 * torch.randn(8, dtype=DT, generator=self.gen)
            cube.set_values(v / v.norm() * np.sqrt(8))
            self.opt = torch.optim.Adam([cube.values], lr=self.opt.param_groups[0]["lr"])
            self.topo = extract_topology(cube, cfg.depth)
            V, F = self.topo.vertices(cube.values), self.topo.F
            if self.verbose:
                print(f"  iter {self.it}: surface left the cube, restarted from the closed-form fit")
            if not len(F):
                self.history.append((self.it, float("nan"), {}))
                self.it += 1
                return float("nan")
        pts = surface_samples(V, F, cfg.mesh_samples, self.gen)
        ctx = LossContext(cube, cube.values, V, F, pts, self.target)
        parts = {k: LOSSES[k](ctx) for k, w in terms.items() if w}
        total = sum(terms[k] * v for k, v in parts.items())
        self.opt.zero_grad()
        total.backward()
        if self.it < cfg.iters:   # cosine schedule; constant at lr_final afterwards
            w = 0.5 * (1 + np.cos(np.pi * self.it / max(cfg.iters, 1)))
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr_final + (cfg.lr - cfg.lr_final) * w
        self.opt.step()
        with torch.no_grad():                     # the surface is scale invariant: fix |c|
            cube.values.mul_(np.sqrt(8) / cube.values.norm())
        self.V, self.F = V.detach(), F
        self.history.append((self.it, float(total.detach()), {k: float(v.detach()) for k, v in parts.items()}))
        self.it += 1
        if self.cfg.keep_best and self.it % self.cfg.check_every == 0:
            s = self.last_score = self.score()
            if s < self.best_score:
                self.best_values, self.best_score, self.best_it = self.values, s, self.it
        return float(total.detach())

    def run(self, iters=None, log_every=25, verbose=True):
        iters = self.cfg.iters if iters is None else iters
        t0 = time.time()
        for _ in range(iters):
            loss = self.step()
            if verbose and (self.it == 1 or self.it % log_every == 0):
                parts = "  ".join(f"{k} {v:.3e}" for k, v in self.history[-1][2].items())
                print(f"  iter {self.it:4d}  loss {loss:.4e}   {parts}   ({time.time() - t0:.1f}s)")
        if self.cfg.keep_best:
            s = self.score()
            if s < self.best_score:
                self.best_values, self.best_score, self.best_it = self.values, s, self.it
            self.cube.set_values(self.best_values)
            if verbose:
                print(f"  best iterate: {self.best_it} (score {self.best_score:.4e})")
        return self.values

    @property
    def values(self) -> np.ndarray:
        return self.cube.values.detach().numpy().copy()

    def mesh(self):
        """Current fitted mesh (numpy V, F), freshly extracted."""
        topo = extract_topology(self.cube, self.cfg.depth)
        return to_numpy(topo.vertices(self.cube.values)), topo.F


def evaluate(values, target: Target, depth=6, n=20000, seed=1):
    """Distances between the fitted surface and the target, in world units.

    target -> fit : first-order distance |f(q)| / |grad f(q)| of each target sample
                    to the fitted level set (accurate once the fit is close).
    fit -> target : for function targets the same first-order distance to g = 0
                    at area-uniform samples of the fitted mesh; for mesh targets the
                    nearest of 50k dense target samples.
    """
    cube = Cube(values, requires_grad=False)
    topo = extract_topology(cube, depth)
    if not len(topo.F):
        return {"empty": True}
    with torch.no_grad():
        V = topo.vertices(cube.values)
        P = surface_samples(V, topo.F, n, torch.Generator().manual_seed(seed)).numpy()
        Q = target.points
        f = cube.basis(Q) @ cube.values
        g = (cube.basis_gradient(Q) @ cube.values).norm(dim=-1)
        d_qp = (f.abs() / g.clamp_min(1e-12)).numpy()
    if target.fn is not None:
        gv, G = target.fn(P), _fn_gradient(target.fn, P)
        d_pq = np.abs(gv) / np.maximum(np.linalg.norm(G, axis=-1), 1e-12)
    else:
        dense = getattr(target, "_dense", None)
        if dense is None:
            dense = torch.as_tensor(sample_surface(target.V, target.F, 50000, np.random.default_rng(seed))[0])
            target._dense = dense
        d_pq = torch.cat([torch.cdist(torch.as_tensor(P[i:i + 4096]), dense).min(1).values
                          for i in range(0, len(P), 4096)]).numpy()
    return {
        "rms": float(np.sqrt((np.mean(d_pq ** 2) + np.mean(d_qp ** 2)) / 2)),
        "fit_to_target_max": float(d_pq.max()),
        "target_to_fit_max": float(d_qp.max()),
        "faces": int(len(topo.F)),
    }


def values_close_up_to_scale(a, b):
    """Angle (degrees) between two value vectors, sign included."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# ============================================================================
# Live view (same look as viewer.py / libigl playground)
# ============================================================================
def view(fitter: Fitter):
    import polyscope as ps
    import polyscope.imgui as psim
    from viewer import BASE_COLOR, BACK_COLOR, EDGE_COLOR, CUBE_COLOR

    TARGET_COLOR = (0.95, 0.55, 0.30)
    st = {"running": False, "steps_per_frame": 2, "show_target": True, "wire": True,
          "target_points": False}

    ps.init()
    ps.set_program_name("cubespline -- optimize")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    P = to_numpy(Cube().corner_positions())
    ps.register_curve_network("cube", P, CUBE_EDGES, color=CUBE_COLOR).set_radius(0.004, relative=True)
    t = fitter.target
    tm = ps.register_surface_mesh("target", t.V, t.F, color=TARGET_COLOR, transparency=0.45)
    tm.set_back_face_policy("identical")
    tp = ps.register_point_cloud("target samples", to_numpy(t.points), radius=0.002, color=TARGET_COLOR)
    tp.set_enabled(False)

    def draw():
        V, F = fitter.mesh()
        if ps.has_surface_mesh("fit"):
            ps.remove_surface_mesh("fit")
        if len(F):
            m = ps.register_surface_mesh("fit", V, F, color=BASE_COLOR)
            m.set_edge_color(EDGE_COLOR)
            m.set_edge_width(1.0 if st["wire"] else 0.0)
            m.set_back_face_policy("custom")
            m.set_back_face_color(BACK_COLOR)
        pc = ps.register_point_cloud("corners", P, radius=0.012)
        pc.set_point_render_mode("sphere")
        v = fitter.values
        pc.add_scalar_quantity("value", v, cmap="coolwarm", vminmax=(-abs(v).max(), abs(v).max()), enabled=True)

    def ui():
        psim.PushItemWidth(160)
        if psim.CollapsingHeader("Optimize", psim.ImGuiTreeNodeFlags_DefaultOpen):
            if psim.Button("Pause" if st["running"] else "Run"):
                st["running"] = not st["running"]
            psim.SameLine()
            if psim.Button("Step"):
                fitter.step(); draw()
            psim.SameLine()
            if psim.Button("Reset (algebraic)"):
                fitter.cfg.init = "algebraic"; fitter.reset(); draw()
            psim.SameLine()
            if psim.Button("Random start"):
                fitter.cfg.init = "random"; fitter.reset(); draw()
            c, d = psim.SliderInt("depth", fitter.cfg.depth, 1, 7)
            if c:
                fitter.cfg.depth = d; fitter.topo = None; draw()
            c, lr = psim.SliderFloat("learning rate", fitter.cfg.lr, 1e-4, 0.2, "%.4f")
            if c:
                fitter.cfg.lr = lr
                for g in fitter.opt.param_groups:
                    g["lr"] = lr
            _, st["steps_per_frame"] = psim.SliderInt("steps / frame", st["steps_per_frame"], 1, 20)
            for name in LOSSES:
                w = fitter.cfg.losses.get(name, 0.0)
                c, w = psim.InputFloat(f"weight {name}", w)
                if c:
                    fitter.cfg.losses[name] = max(0.0, w)
            if psim.Button("Save fit.json"):
                save_fit("fit.json", fitter)
                print_final_error(final_error(fitter))
                print("saved fit.json")
        if psim.CollapsingHeader("Display", psim.ImGuiTreeNodeFlags_DefaultOpen):
            c, st["show_target"] = psim.Checkbox("Target surface", st["show_target"])
            if c:
                tm.set_enabled(st["show_target"])
            c, st["target_points"] = psim.Checkbox("Target samples", st["target_points"])
            if c:
                tp.set_enabled(st["target_points"])
            c, st["wire"] = psim.Checkbox("Wireframe", st["wire"])
            if c and ps.has_surface_mesh("fit"):
                ps.get_surface_mesh("fit").set_edge_width(1.0 if st["wire"] else 0.0)
        if psim.CollapsingHeader("Progress", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.TextUnformatted(f"iteration  {fitter.it}")
            if fitter.history:
                _, tot, parts = fitter.history[-1]
                psim.TextUnformatted(f"loss       {tot:.4e}")
                psim.TextUnformatted(f"RMS dist.  {fitter.last_score:.4e}  (checked every {fitter.cfg.check_every} it.)")
                psim.TextUnformatted(f"best RMS   {fitter.best_score:.4e}  at iteration {fitter.best_it}")
                for k, v in parts.items():
                    psim.TextUnformatted(f"  {k:10s} {v:.4e}")
                hist = np.log10(np.maximum([h[1] for h in fitter.history[-300:]], 1e-30)).astype(np.float32)
                try:
                    psim.PlotLines("log10 loss", hist.tolist(), graph_size=(240, 80))
                except Exception:  # noqa: BLE001 -- older polyscope builds
                    pass
            vals = ", ".join(f"{x:+.3f}" for x in fitter.values)
            psim.TextWrapped(f"values [{vals}]")
            if t.values is not None:
                psim.TextUnformatted(f"angle to truth {values_close_up_to_scale(fitter.values, t.values):.3f} deg")
        psim.PopItemWidth()
        if st["running"]:
            for _ in range(st["steps_per_frame"]):
                fitter.step()
            draw()

    draw()
    ps.reset_camera_to_home_view()
    ps.set_user_callback(ui)
    ps.show()


# ============================================================================
def final_error(fitter: Fitter, metrics=None) -> dict:
    """Error of the values the fitter returns (the best iterate), in one place:
    the optimization objective (deterministic) and true distances to the target."""
    total, parts = fitter.objective()
    m = metrics if metrics is not None else evaluate(fitter.values, fitter.target)
    width = fitter.cube.size
    m0 = evaluate(fitter.init_values, fitter.target)          # same measurement as `m`
    err = {"objective": total, "terms": parts, "weights": dict(fitter.cfg.losses),
           "best_iteration": fitter.best_it, "iterations": fitter.it,
           "cube_width": width, "initial_rms": m0.get("rms", float("inf"))}
    if not m.get("empty"):
        err.update({k: m[k] for k in ("rms", "fit_to_target_max", "target_to_fit_max")})
        err["rms_percent_of_cube"] = 100 * m["rms"] / width
        err["max_percent_of_cube"] = 100 * max(m["fit_to_target_max"], m["target_to_fit_max"]) / width
    if fitter.target.values is not None:
        err["angle_to_ground_truth_deg"] = values_close_up_to_scale(fitter.values, fitter.target.values)
    return err


def print_final_error(err: dict):
    w = err["cube_width"]
    terms = " + ".join(f"{err['weights'][k]:g}*{k}" for k in err["terms"])
    print("\n" + "=" * 26 + " final optimization error " + "=" * 26)
    print(f"  objective  {terms:28s} {err['objective']:.4e}   "
          f"(best iterate: {err['best_iteration']} of {err['iterations']})")
    if len(err["terms"]) > 1:
        for k, v in err["terms"].items():
            print(f"    {k:38s} {v:.4e}")
    if "rms" in err:
        print(f"  distance to target (world units, cube width {w:g}):")
        print(f"    RMS                                    {err['rms']:.4e}   ({err['rms_percent_of_cube']:.3f} % of the cube width)")
        print(f"    max  fit -> target                     {err['fit_to_target_max']:.4e}   ({100 * err['fit_to_target_max'] / w:.3f} %)")
        print(f"    max  target -> fit                     {err['target_to_fit_max']:.4e}   ({100 * err['target_to_fit_max'] / w:.3f} %)")
        r0 = err["initial_rms"]
        if np.isfinite(r0) and r0 > 0:
            print(f"  RMS: initial fit {r0:.4e} -> optimized {err['rms']:.4e}  ({100 * (err['rms'] - r0) / r0:+.1f} %)")
    else:
        print("  the fitted surface does not cross the cube")
    if "angle_to_ground_truth_deg" in err:
        print(f"  angle to the ground-truth values               {err['angle_to_ground_truth_deg']:.4f} deg")
    print("=" * 78)


def save_fit(path, fitter: Fitter, metrics=None):
    err = final_error(fitter, metrics)
    data = {"values": fitter.values.tolist(), "depth": fitter.cfg.depth, "target": fitter.target.name,
            "iterations": fitter.it, "losses": fitter.cfg.losses,
            "final_loss": err["objective"], "final_error": err}
    if metrics:
        data["metrics"] = metrics
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def _parse_losses(items):
    if not items:
        return {"chamfer": 1.0}
    out = {}
    for it in items:
        name, _, w = it.partition("=")
        if name not in LOSSES:
            raise SystemExit(f"unknown loss '{name}'; available: {', '.join(LOSSES)}")
        out[name] = float(w) if w else 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--target", default="sphere flank", choices=["trilinear", *FUNCTIONS])
    src.add_argument("--mesh", help="target mesh (.obj, or any format libigl reads if installed)")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--refine-iters", "--iters", dest="refine_iters", type=int, default=300,
                    help="Adam steps of the chamfer refinement (default 300; --iters is an alias). "
                         "The start is a closed-form fit and has no iterations.")
    ap.add_argument("--no-refine", action="store_true", help="closed-form start only (same as --refine-iters 0)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--samples", type=int, default=3000, help="samples on target and on the fitted mesh")
    ap.add_argument("--loss", action="append", metavar="NAME=W", help=f"loss term, repeatable ({', '.join(LOSSES)})")
    ap.add_argument("--init", default="algebraic", choices=["algebraic", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fit.json")
    ap.add_argument("--view", action="store_true", help="open the live Polyscope view instead of running headless")
    ap.add_argument("--list", action="store_true", help="list targets and losses")
    args = ap.parse_args()

    if args.list:
        print("targets:", "trilinear,", ", ".join(FUNCTIONS))
        print("losses: ", ", ".join(LOSSES))
        return

    if args.mesh:
        if args.mesh.lower().endswith(".obj"):
            V, F = load_obj(args.mesh)
        else:
            import igl
            V, F = igl.read_triangle_mesh(args.mesh)
        target = target_from_mesh(V, F, name=args.mesh, n=args.samples, seed=args.seed)
    else:
        target = builtin_target(args.target, n=args.samples, seed=args.seed)

    cfg = FitConfig(depth=args.depth, iters=0 if args.no_refine else args.refine_iters, lr=args.lr,
                    mesh_samples=args.samples,
                    losses=_parse_losses(args.loss), init=args.init, seed=args.seed)
    fitter = Fitter(target, cfg, verbose=True)
    print(f"target {target.name}: {len(target.points)} samples; losses {cfg.losses}; depth {cfg.depth}")
    print("init values", np.round(fitter.values, 4).tolist())
    print("init metrics", evaluate(fitter.values, target))

    if args.view:
        view(fitter)
        return
    fitter.run()
    metrics = evaluate(fitter.values, target)           # same settings as final_error's initial RMS
    print("final values", np.round(fitter.values, 4).tolist())
    print_final_error(final_error(fitter, metrics))
    save_fit(args.out, fitter, metrics)
    print(f"saved {args.out}  (view: python viewer.py --load {args.out})")


if __name__ == "__main__":
    main()
