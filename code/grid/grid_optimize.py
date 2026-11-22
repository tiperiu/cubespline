#!/usr/bin/env python3
"""
grid_optimize.py -- fit a grid of cubes (scalar or warp model, see grid.py) to a target:
an implicit function from a Python file, or a point cloud file (see targets.py).

Models (--model):
    scalar   one value G per grid vertex; f is trilinear in every cell
    warp     a value G and a 3D displacement D per vertex: f(x) = G(x + h D(x))

Methods (--method):
    chamfer  (default) surface distance, three stages:
             1. closed-form start for G (scalar): least squares with
                    f(q) = 0 and h grad f(q) = n at the target points (q, n),
                    a small second-difference smoothness term, and a weak pull of every
                    vertex towards its estimated signed distance (distance to the nearest
                    target point, signed by the target's inside/outside), which keeps the
                    vertices far from the target well defined (no spurious surfaces there);
                the result is about the signed distance, in cell units. It is solved for
                each pull weight in --anchor-weights (0.03 suits shapes several cells
                thick, 0.3 thin ones) and the result with the fewest surface pieces,
                then the lowest RMS, is kept.
             2. Sampson fit (L-BFGS, no mesh): f(q)^2 / |grad f(q)|^2 + a small normal
                term over G; for warp then over G and D together, from that result,
                plus --disp-penalty * mean(D^2).
             3. chamfer refinement through marching cubes (Adam): mesh the grid, sample
                the mesh, and minimise a two-sided chamfer distance measured to the
                surfaces (mesh samples to the target points' tangent planes, target
                points to the mesh triangles), so it has no sampling floor; vertices are
                differentiable (exact crossings for scalar, implicit-function derivative
                of the bisected crossings for warp). The best iterate is kept.
             After stage 1, a result (a stage, or an iterate of stage 3) replaces the
             current one only if it lowers the RMS without adding surface pieces: small
             detached pieces barely change the RMS but are wrong.
    logistic, es, grpo
             inside/outside classification, as in classify_optimize.py: points labelled
             by the target (sign of g, or the side of the nearest point's tangent plane
             for a point cloud), and the field rewarded for the correct sign. The start
             is each vertex's own label (-1 inside, +1 outside), so vertices far from the
             target begin on the right side. For the warp model the scalar grid is
             trained first, then G and D together.
             --chamfer-iters N refines the result with stage 3 afterwards.
             ES and GRPO estimate gradients by random trials; with hundreds of numbers
             (9^3 = 729 vertices for 8^3 cells, 4x that for warp) they are slow to converge.
             Their learning rate on grids is 0.01 (0.05 for one cube): Adam moves every
             number by about the learning rate, and for the many vertices whose estimated
             gradient is pure noise a larger rate becomes a random walk that flips signs
             and creates spurious surface pieces.

Grid (--dims nx [ny nz], cubic cells):
    function targets   over [lo, lo + size]^3 (--lo, --size; default [-1, 1]^3)
    point clouds       over [-1, 1]^3 if the cloud lies inside it, else around the
                       cloud's bounding box (one cell of margin); --lo/--size override,
                       --fit-domain always fits the bounding box

Usage (from the project folder):
    python code/grid/grid_optimize.py --list                                   # example targets
    python code/grid/grid_optimize.py --function sphere_medium                 # scalar grid, 8^3 cells, chamfer
    python code/grid/grid_optimize.py --function torus_medium --model warp --dims 6
    python code/grid/grid_optimize.py --points ellipsoid_long --dims 10
    python code/grid/grid_optimize.py --points torus_medium_no_normals          # normals estimated
    python code/grid/grid_optimize.py --function my_shape.py --lo -2 -2 -2 --size 4
    python code/grid/grid_optimize.py --function sphere_small --method logistic --chamfer-iters 100
    python code/grid/grid_optimize.py --function sphere_small --method grpo --dims 4
    python code/visualization/grid_viewer.py                                   # look at the results
Fits are written to data/fits/grid/<target>[_cloud]__<model>__<method>__<cells per axis>.json.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE = str(_Path(__file__).resolve().parents[1])          # .../code: makes `import core...` work
if _CODE not in _sys.path:                                  # when this file is run as a script
    _sys.path.insert(0, _CODE)

import argparse
import json
import os
import time
from dataclasses import dataclass, fields

import numpy as np
import torch

import optimization.classify_optimize as C
import optimization.optimize as O
from core.geometry import closest_point_triangle, count_pieces, point_mesh_dist2
from core.paths import GRID_FITS, REPO, output_path
from grid import targets as T
from grid.grid import DT, Grid, GridModel, basis, basis_grad
from grid.grid_marching import extract_topology, mesh_check


@dataclass
class GridFitConfig:
    model: str = "scalar"          # scalar | warp
    method: str = "chamfer"        # chamfer | logistic | es | grpo
    depth: int = 2                 # marching-cubes subdivision of each cell (2^depth per axis)
    normal_weight: float = 1.0     # stage 1: weight of h grad f = n
    smooth_weight: float = 0.01    # stage 1: weight of the second differences of G
    anchor_weights: tuple = (0.03, 0.3)   # stage 1: pulls towards the estimated signed distance (one fit each)
    sampson_iters: int = 100       # stage 2: L-BFGS iterations (0 = skip)
    disp_penalty: float = 1e-3     # warp: lambda * mean(D^2), D in cell units
    iters: int = 300               # stage 3: Adam steps of the chamfer refinement
    lr: float = 1e-3
    lr_final: float = 1e-5
    samples: int = 3000            # mesh / target samples per chamfer step
    check_every: int = 10
    seed: int = 0


# ============================================================================
# Stage 1: closed-form scalar start
# ============================================================================
def second_differences(grid: Grid) -> np.ndarray:
    """(m, 3) vertex id triplets (v - e, v, v + e) along every axis."""
    ijk = grid.vertex_ijk()
    out = []
    for ax in range(3):
        e = np.eye(3, dtype=np.int64)[ax]
        ok = (ijk[:, ax] >= 1) & (ijk[:, ax] <= grid.dims[ax] - 1)
        mid = ijk[ok]
        out.append(np.stack([grid.vertex_id(mid - e), grid.vertex_id(mid), grid.vertex_id(mid + e)], 1))
    return np.concatenate(out, 0) if out else np.zeros((0, 3), np.int64)


def signed_distance_estimate(grid: Grid, target: T.GridTarget) -> torch.Tensor:
    """At every vertex: distance to the nearest target point, in cells, with the sign of
    the target's inside/outside label."""
    X = torch.as_tensor(grid.vertex_positions(), dtype=DT)
    cloud = torch.as_tensor(target.cloud, dtype=DT)
    d = (X - cloud[O.nearest(X, cloud)]).norm(dim=-1)
    return target.label(X) * d / grid.h


def fit_closed_form(grid: Grid, target: T.GridTarget, normal_weight=1.0, smooth_weight=0.01,
                    anchor_weight=0.03, iters=300) -> torch.Tensor:
    """G (V,) minimising  mean f(q)^2 + mu mean |h grad f(q) - n|^2 + lam mean(second diff^2)
    + a mean (G - SDF0)^2, with SDF0 the signed-distance estimate (also the start).
    A convex quadratic, solved with L-BFGS."""
    P, N = target.points, target.normals
    ids, t = grid.locate(P)
    w, dw = basis(t), basis_grad(t)                                   # (n, 8), (n, 3, 8)
    trip = torch.as_tensor(second_differences(grid))
    sdf0 = signed_distance_estimate(grid, target)
    G = sdf0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([G], lr=1, max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-15,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        Gc = G[ids]
        f = (w * Gc).sum(-1)
        gr = (dw * Gc[:, None, :]).sum(-1)                            # h * grad f
        loss = f.square().mean() + normal_weight * (gr - N).square().sum(-1).mean()
        if len(trip) and smooth_weight:
            loss = loss + smooth_weight * (G[trip[:, 0]] - 2 * G[trip[:, 1]] + G[trip[:, 2]]).square().mean()
        if anchor_weight:
            loss = loss + anchor_weight * (G - sdf0).square().mean()
        loss.backward()
        return loss
    opt.step(closure)
    return G.detach()


# ============================================================================
# Stage 2: Sampson fit
# ============================================================================
def sampson_loss(model: GridModel, p, P, N, normal_weight=1e-3, disp_penalty=0.0):
    f = model.field(p, P)
    g = model.gradient(p, P)
    g2 = g.square().sum(-1).clamp_min(1e-12)
    loss = (f.square() / g2).mean() + normal_weight * (1 - (g * N).sum(-1) / g2.sqrt()).mean()
    _, D = model.split(p)
    if D is not None and disp_penalty:
        loss = loss + disp_penalty * D.square().mean()
    return loss


def fit_sampson(model: GridModel, target: T.GridTarget, p0, iters=100, disp_penalty=1e-3, n_points=3000):
    p = p0.clone().requires_grad_(True)
    P, N = target.points[:n_points], target.normals[:n_points]
    opt = torch.optim.LBFGS([p], lr=1, max_iter=iters, history_size=30, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = sampson_loss(model, p, P, N, disp_penalty=disp_penalty)
        loss.backward()
        return loss
    opt.step(closure)
    return p.detach()


# ============================================================================
# Stage 3: chamfer refinement, and the error measures
# ============================================================================
def quick_check(model: GridModel, p, target: T.GridTarget, depth, n=4000) -> tuple:
    """(RMS distance, surface pieces), for keep-best: fit -> target by target.distance at
    mesh samples, target -> fit exact point-to-mesh distance of 2000 target points."""
    with torch.no_grad():
        topo = extract_topology(model, p, depth)
        if not len(topo.F):
            return float("inf"), 0
        V = topo.vertices(p)
        S = O.surface_samples(V, topo.F, n, torch.Generator().manual_seed(11))
        d1 = target.distance(S)
        d2 = point_mesh_dist2(target.points[:2000], V, topo.F).sqrt().numpy()
    return float(np.sqrt((np.mean(d1 ** 2) + np.mean(d2 ** 2)) / 2)), count_pieces(topo.F)


def quick_score(model: GridModel, p, target: T.GridTarget, depth, n=4000) -> float:
    """RMS distance only (see quick_check)."""
    return quick_check(model, p, target, depth, n)[0]


def improves(new: tuple, best: tuple) -> bool:
    """A (RMS, pieces) result replaces the current best only if it lowers the RMS
    without adding surface pieces."""
    return new[0] < best[0] and new[1] <= best[1]


def surface_chamfer(S: torch.Tensor, V: torch.Tensor, F: np.ndarray, target: T.GridTarget, Q: torch.Tensor,
                    cloud: torch.Tensor, cloud_n: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Two-sided chamfer between the fitted mesh (V, F; samples S on it) and the target,
    measured to the surfaces rather than to samples, so it has no sampling floor:
      fit -> target   squared distance from each mesh sample to the tangent plane of
                      the nearest target point (point-to-plane)
      target -> fit   squared distance from each target point Q to the mesh (closest
                      point on the nearest of k candidate triangles), differentiable in V
    """
    with torch.no_grad():
        nn = O.nearest(S, cloud)
    d_fit = ((S - cloud[nn]) * cloud_n[nn]).sum(-1)
    Fi = torch.as_tensor(np.asarray(F, np.int64))
    with torch.no_grad():
        cen = V.detach()[Fi].mean(1)
        cand = torch.cdist(Q.float(), cen.float()).topk(min(k, len(Fi)), largest=False).indices   # (m, k)
    tri = V[Fi[cand]]                                                                     # (m, k, 3, 3)
    cp = closest_point_triangle(Q[:, None, :].expand(-1, cand.shape[1], -1), tri[..., 0, :], tri[..., 1, :],
                                tri[..., 2, :])
    d_tgt = (cp - Q[:, None, :]).square().sum(-1).min(1).values
    return d_fit.square().mean() + d_tgt.mean()


def chamfer_refine(model: GridModel, target: T.GridTarget, p0, cfg: GridFitConfig, log_every=50):
    """Adam on the chamfer distance through marching cubes. Returns (best p, history)."""
    gen = torch.Generator().manual_seed(cfg.seed)
    p = p0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=cfg.lr)
    g_norm = float(p.detach()[model.field_slice].norm())
    h2 = model.grid.h ** 2
    best_p, best = p.detach().clone(), quick_check(model, p.detach(), target, cfg.depth)
    cloud, cloud_n = target.points, target.normals            # tangent planes for the fit -> target term
    history = [(0, None, best[0])]
    t0 = time.time()
    for it in range(cfg.iters):
        for g in opt.param_groups:
            g["lr"] = C._cosine(it, cfg.iters, cfg.lr, cfg.lr_final)
        topo = extract_topology(model, p.detach(), cfg.depth)
        if not len(topo.F):
            print("  the surface left the grid; keeping the best result so far")
            break
        opt.zero_grad()
        V = topo.vertices(p)
        S = O.surface_samples(V, topo.F, cfg.samples, gen)
        Q = target.points[torch.randperm(len(target.points), generator=gen)[:cfg.samples]]
        loss = surface_chamfer(S, V, topo.F, target, Q, cloud, cloud_n) / h2
        _, D = model.split(p)
        if D is not None and cfg.disp_penalty:
            loss = loss + cfg.disp_penalty * D.square().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():                                     # the scale of G is free
            p[model.field_slice] *= g_norm / p[model.field_slice].norm()
        if (it + 1) % cfg.check_every == 0 or it + 1 == cfg.iters:
            s = quick_check(model, p.detach(), target, cfg.depth)
            history.append((it + 1, loss.item(), s[0]))
            if improves(s, best):
                best, best_p = s, p.detach().clone()
            if log_every and ((it + 1) % log_every == 0 or it + 1 == cfg.iters):
                print(f"  iter {it + 1:4d}  chamfer loss {loss.item():.4e}   RMS {s[0]:.4e}, {s[1]} pieces   "
                      f"(best {best[0]:.4e}, {time.time() - t0:.1f}s)")
    return best_p, history


def evaluate(model: GridModel, p, target: T.GridTarget, depth=3, n=20000) -> dict:
    """Final errors: two-sided chamfer distance with exact point-to-mesh distances for
    target -> fit (fit -> target: |g|/|grad g| for a function, nearest point for a
    cloud), RMS, maxima, surface pieces and a watertightness check."""
    p = torch.as_tensor(p, dtype=DT)
    with torch.no_grad():
        topo = extract_topology(model, p, depth)
        if not len(topo.F):
            return {"empty": True}
        V = topo.vertices(p)
        S = O.surface_samples(V, topo.F, n, torch.Generator().manual_seed(7))
        d1 = target.distance(S)
        Q = torch.as_tensor(target.dense(n), dtype=DT)
        d2 = point_mesh_dist2(Q, V, topo.F).sqrt().numpy()
    cd = float(np.mean(d1 ** 2) + np.mean(d2 ** 2))
    chk = mesh_check(V, topo.F, model.grid)
    out = {"cd": cd, "rms": float(np.sqrt(cd / 2)), "fit_to_target_max": float(d1.max()),
           "target_to_fit_max": float(d2.max()), "rms_in_cells": float(np.sqrt(cd / 2)) / model.grid.h,
           "cell_size": model.grid.h, "surface_pieces": count_pieces(topo.F),
           "target_pieces": count_pieces(target.F) if target.F is not None and len(target.F) else None,
           "watertight": chk["ok"], "boundary_edges": chk["boundary_edges"], "faces": len(topo.F)}
    _, D = model.split(p)
    if D is not None:
        out["max_displacement_cells"] = float(D.norm(dim=-1).max())
    return out


def print_report(err: dict, title="final error"):
    print("\n" + "=" * 24 + f" {title} " + "=" * max(2, 52 - len(title)))
    if err.get("empty"):
        print("  the fitted surface does not cross the grid")
        print("=" * 78)
        return
    for k in ("accuracy", "accuracy_train_lattice"):
        if k in err:
            print(f"  {'accuracy (random validation points)' if k == 'accuracy' else 'accuracy on the training lattice':37s}"
                  f"{100 * err[k]:.3f} %")
    print(f"  chamfer distance (CD)                {err['cd']:.4e}   (mean d^2 both ways, world units^2)")
    print(f"  RMS distance                         {err['rms']:.4e}   ({err['rms_in_cells']:.3f} cells of {err['cell_size']:g})")
    print(f"  max  fit -> target                   {err['fit_to_target_max']:.4e}")
    print(f"  max  target -> fit                   {err['target_to_fit_max']:.4e}")
    tp = err.get("target_pieces")
    flag = "   <-- extra pieces" if tp is not None and err["surface_pieces"] > tp else ""
    print(f"  surface pieces                       {err['surface_pieces']}" + (f"   (target: {tp})" if tp else "") + flag)
    print(f"  closed and manifold inside the grid  {'yes' if err['watertight'] else 'NO'}   ({err['faces']} faces)")
    if "max_displacement_cells" in err:
        print(f"  largest displacement                 {err['max_displacement_cells']:.3f} cells")
    print("=" * 78)


# ============================================================================
# Classification on the grid
# ============================================================================
class GridSampler:
    """Labelled points in the grid domain for the classification methods (same
    interface as classify_optimize.Sampler: draw, draw_random, lattice, describe)."""

    def __init__(self, target: T.GridTarget, grid: Grid, band=0.1, uniform=0.25, sampling="random", k=10,
                 faces=False):
        self.target, self.grid = target, grid
        self.band, self.uniform = grid.scale(band), uniform
        self.sampling, self.k, self.faces = sampling, k, faces
        self.lattice = None
        if sampling == "lattice":
            X = torch.as_tensor(grid.lattice(k, faces), dtype=DT)
            self.lattice = (X, target.label(X))

    def draw(self, n, gen):
        return self.lattice if self.lattice is not None else self.draw_random(n, gen)

    def draw_random(self, n, gen):
        P, N = self.target.points, self.target.normals
        nu = int(round(self.uniform * n))
        idx = torch.randint(len(P), (n - nu,), generator=gen)
        Xs = P[idx] + N[idx] * (self.band * torch.randn(n - nu, 1, dtype=DT, generator=gen))
        X = self.grid.clamp(torch.cat([Xs, self.grid.uniform(nu, gen)]))
        return X, self.target.label(X)

    def validation(self, n=20000, seed=999) -> C.Validation:
        X, y = self.draw_random(n, torch.Generator().manual_seed(seed))
        return self.as_validation(X, y)

    def as_validation(self, X, y) -> C.Validation:
        return C.Validation(X, y, torch.as_tensor(self.target.distance(X) < self.grid.scale(C.Validation.NEAR)))

    def describe(self) -> str:
        if self.lattice is None:
            return f"random: fresh batches, near-surface band {self.band:g} + {100 * self.uniform:g} % uniform"
        return f"lattice: {self.k}^3 = {self.k ** 3} fixed points over the grid"


def fit_classify(model: GridModel, target: T.GridTarget, ccfg: C.ClassifyConfig, scalar_iters=None, verbose=True):
    """Train with logistic / ES / GRPO. Warp: the scalar grid first, then G and D.
    Returns (p, info)."""
    gen = torch.Generator().manual_seed(ccfg.seed)
    sampler = GridSampler(target, model.grid, ccfg.band, ccfg.uniform, ccfg.sampling, ccfg.k, ccfg.lattice_faces)
    val = sampler.as_validation(*sampler.lattice) if sampler.lattice is not None else sampler.validation()
    log = 50 if verbose else 0
    if verbose:
        print(f"  sampling {sampler.describe()}")
    scalar = GridModel(model.grid, "scalar")
    iters = ccfg.iters if model.kind == "scalar" or scalar_iters is None else scalar_iters
    if verbose and model.kind == "warp":
        print(f"  scalar grid first ({iters} steps)")
    # start: each vertex's own inside/outside label (-1 / +1), the same information the
    # reward uses; vertices far from the target then start (and stay) on the right side
    G0 = target.label(torch.as_tensor(model.grid.vertex_positions(), dtype=DT))
    G, r, hist = C.train(scalar, sampler, val, G0, iters, ccfg, gen, log)
    if model.kind == "scalar":
        return G, {"val_reward": r, "history": hist, "sampler": sampler, "val": val}
    if verbose:
        print(f"  then values and displacements together ({ccfg.iters} steps)")
    p, r2, hist2 = C.train(model, sampler, val, model.join(G), ccfg.iters, ccfg, gen, log)
    return p, {"val_reward": r2, "history": hist2, "sampler": sampler, "val": val}


# ============================================================================
# The whole fit
# ============================================================================
def fit(model: GridModel, target: T.GridTarget, cfg: GridFitConfig, ccfg: C.ClassifyConfig | None = None,
        chamfer_iters_after=0, verbose=True):
    """Returns (p, info)."""
    info = {"stages": []}
    t0 = time.time()
    if cfg.method == "chamfer":
        best = None
        for a in np.atleast_1d(cfg.anchor_weights):
            Ga = fit_closed_form(model.grid, target, cfg.normal_weight, cfg.smooth_weight, float(a))
            sa = quick_check(model, model.join(Ga), target, cfg.depth)
            if verbose:
                print(f"  1. closed form, pull {float(a):<6g}   RMS {sa[0]:.4e}, {sa[1]} pieces")
            if best is None or (np.isinf(sa[0]), sa[1], sa[0]) < (np.isinf(best[0]), best[1], best[0]):
                G, best = Ga, sa
        p, s = model.join(G), best
        info["stages"].append(("closed form", s[0], time.time() - t0))
        if verbose:
            print(f"  1. closed-form start          RMS {s[0]:.4e}, {s[1]} pieces   ({time.time() - t0:.1f}s)")
        if cfg.sampson_iters:
            # scalar Sampson first; for the warp model then values and displacements
            # together, from there. Each step is kept only if it improves (see improves).
            scalar = GridModel(model.grid, "scalar")
            G2 = fit_sampson(scalar, target, G, cfg.sampson_iters)
            p2 = model.join(G2)
            s2 = quick_check(model, p2, target, cfg.depth)
            if improves(s2, s):
                p, s = p2, s2
            if model.kind == "warp":
                p3 = fit_sampson(model, target, p, cfg.sampson_iters, cfg.disp_penalty)
                s3 = quick_check(model, p3, target, cfg.depth)
                if verbose:
                    print(f"  2. Sampson fit, scalar        RMS {s[0]:.4e}, {s[1]} pieces")
                if improves(s3, s):
                    p, s = p3, s3
            info["stages"].append(("sampson", s[0], time.time() - t0))
            if verbose:
                print(f"  2. Sampson fit{', with D' if model.kind == 'warp' else ''}          RMS {s[0]:.4e}, {s[1]} pieces   "
                      f"({time.time() - t0:.1f}s)")
        if cfg.iters:
            if verbose:
                print(f"  3. chamfer refinement, {cfg.iters} Adam steps")
            p, hist = chamfer_refine(model, target, p, cfg, log_every=100 if verbose else 0)
            info["history"] = hist
            info["stages"].append(("chamfer", hist and min(h[2] for h in hist), time.time() - t0))
        return model.normalised(p), info
    ccfg = ccfg or C.ClassifyConfig(method=cfg.method, seed=cfg.seed)
    p, cinfo = fit_classify(model, target, ccfg, verbose=verbose)
    info.update({k: v for k, v in cinfo.items() if k in ("val_reward", "history")})
    info["sampler"], info["val"] = cinfo["sampler"], cinfo["val"]
    info["stages"].append((cfg.method, quick_score(model, p, target, cfg.depth), time.time() - t0))
    if chamfer_iters_after:
        if verbose:
            print(f"  chamfer refinement, {chamfer_iters_after} Adam steps")
        c2 = GridFitConfig(**{**{f.name: getattr(cfg, f.name) for f in fields(cfg)}, "iters": chamfer_iters_after})
        p, _ = chamfer_refine(model, target, p, c2, log_every=100 if verbose else 0)
        info["stages"].append(("chamfer", quick_score(model, p, target, cfg.depth), time.time() - t0))
    return model.normalised(p), info


# ============================================================================
# Files
# ============================================================================
def _portable(path):
    """A path relative to the project folder if it is inside it."""
    if path is None:
        return None
    try:
        return str(_Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def save_fit(path, model: GridModel, p, target: T.GridTarget, cfg: GridFitConfig, err: dict, info: dict,
             ccfg: C.ClassifyConfig | None = None):
    data = {"type": "grid", "model": model.kind, "grid": model.grid.to_dict(),
            "params": np.asarray(p, float).tolist(), "depth": cfg.depth,
            "target": {"kind": target.kind, "name": target.name, "source": _portable(target.source)},
            "method": cfg.method, "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
            "stages": [[s, v, t] for s, v, t in info.get("stages", [])], "final_error": err}
    if ccfg is not None and cfg.method != "chamfer":
        data["classify_config"] = {f.name: getattr(ccfg, f.name) for f in fields(ccfg)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)


def load_fit(path) -> dict:
    """{'model', 'params', 'grid', 'target' (GridTarget or None), 'data'} from a grid fit file."""
    with open(path) as fh:
        data = json.load(fh)
    if data.get("type") != "grid":
        raise ValueError(f"{path} is not a grid fit")
    grid = Grid.from_dict(data["grid"])
    model = GridModel(grid, data["model"])
    target = None
    tinfo = data.get("target") or {}
    src = tinfo.get("source")
    if src:
        cand = _Path(src) if _Path(src).is_absolute() else REPO / src
        try:
            if tinfo.get("kind") == "function":
                target = T.load_target(grid, function=str(cand if cand.exists() else tinfo.get("name")))
            else:
                target = T.load_target(grid, points=str(cand if cand.exists() else tinfo.get("name")))
        except Exception as e:  # noqa: BLE001
            print(f"  (target not loaded: {e})")
    return {"model": model, "params": torch.as_tensor(data["params"], dtype=DT), "grid": grid,
            "target": target, "data": data}


def _slug(s):
    return str(s).replace(" ", "_")


def fit_name(target: T.GridTarget, model: GridModel, method: str) -> str:
    """Default fit file name: <target>__<model>__<method>__<cells>.json, with "_cloud"
    after the target name for a point cloud, and cells = 8 for 8x8x8 (nx x ny x nz
    otherwise)."""
    nx, ny, nz = model.grid.dims
    cells = f"{nx}" if nx == ny == nz else f"{nx}x{ny}x{nz}"
    name = _slug(target.name) + ("_cloud" if target.kind == "points" else "")
    return f"{name}__{model.kind}__{method}__{cells}.json"


# ============================================================================
# Command line
# ============================================================================
def add_classify_arguments(ap):
    g = ap.add_argument_group("classification (--method logistic, es, grpo)")
    g.add_argument("--class-iters", type=int, default=400, help="steps (generations for ES/GRPO)")
    g.add_argument("--sampling", default="random", choices=["random", "lattice"])
    g.add_argument("--k", type=int, default=10, help="lattice: k^3 points over the grid")
    g.add_argument("--lattice-faces", action="store_true")
    g.add_argument("--batch", type=int, default=4000)
    g.add_argument("--band", type=float, default=0.1, help="near-surface jitter (for a domain of width 2)")
    g.add_argument("--uniform", type=float, default=0.25)
    g.add_argument("--tau", type=float, default=0.02)
    g.add_argument("--weight-decay", type=float, default=1e-3)
    g.add_argument("--pop", type=int, default=32)
    g.add_argument("--sigma", type=float, default=0.1)
    g.add_argument("--group", type=int, default=32)
    g.add_argument("--grpo-epochs", type=int, default=4)
    g.add_argument("--clip", type=float, default=0.2)
    g.add_argument("--kl-coef", type=float, default=0.0)
    g.add_argument("--ref-every", type=int, default=10)
    g.add_argument("--grpo-sigma", type=float, default=0.2)
    g.add_argument("--grpo-lr", type=float, default=0.01)
    g.add_argument("--class-lr", type=float, help="logistic / ES learning rate (default 0.05 logistic, 0.01 ES)")
    g.add_argument("--chamfer-iters", type=int, default=0, help="chamfer refinement afterwards")


def classify_config(args, method) -> C.ClassifyConfig:
    lr = args.class_lr if args.class_lr is not None else (0.01 if method == "es" else 0.05)
    return C.ClassifyConfig(method=method, iters=args.class_iters, lr=lr, batch=args.batch,
                            band=args.band, uniform=args.uniform, weight_decay=args.weight_decay, tau=args.tau,
                            offset_penalty=args.disp_penalty, pop=args.pop + args.pop % 2, sigma=args.sigma,
                            group=args.group, grpo_epochs=args.grpo_epochs, clip=args.clip, kl_coef=args.kl_coef,
                            ref_every=args.ref_every, grpo_sigma=args.grpo_sigma, grpo_lr=args.grpo_lr,
                            sampling=args.sampling, k=args.k, lattice_faces=args.lattice_faces, seed=args.seed)


def make_grid_and_target(args):
    """The grid for --dims and the target (see 'Grid' in the module docstring)."""
    dims = args.dims * 3 if len(args.dims) == 1 else args.dims
    if len(dims) != 3:
        raise SystemExit("--dims takes 1 or 3 numbers")
    explicit = args.lo is not None or args.size is not None
    if args.function:
        grid = T.default_grid(dims, lo=args.lo if explicit else -1.0, size=args.size if explicit else 2.0)
        return grid, T.load_target(grid, function=args.function, n=args.samples, seed=args.seed)
    path = T.resolve_points(args.points)
    P, N = T.load_point_cloud(path)
    if explicit:
        grid = T.default_grid(dims, lo=args.lo, size=args.size)
    elif not args.fit_domain and np.all(P >= -1) and np.all(P <= 1):
        grid = Grid.cube(dims)
    else:
        grid = T.default_grid(dims, target_bounds=(P.min(0), P.max(0)))
    return grid, T.target_from_points(P, N, path.stem, grid, n=args.samples, seed=args.seed, source=str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--function", help="Python file defining g(X), or the name of an example function")
    src.add_argument("--points", help="point cloud file, or the name of an example point cloud")
    ap.add_argument("--list", action="store_true", help="list the example targets")
    ap.add_argument("--model", default="scalar", choices=["scalar", "warp"])
    ap.add_argument("--method", default="chamfer", choices=["chamfer", "logistic", "es", "grpo"])
    ap.add_argument("--dims", type=int, nargs="+", default=[8], help="cells per axis: n, or nx ny nz")
    ap.add_argument("--lo", type=float, nargs=3, metavar=("X", "Y", "Z"), help="minimum corner of the grid domain")
    ap.add_argument("--size", type=float, help="edge length of the (cubic) domain")
    ap.add_argument("--fit-domain", action="store_true", help="point clouds: fit the grid to the bounding box")
    ap.add_argument("--depth", type=int, default=2, help="marching-cubes subdivision of each cell")
    ap.add_argument("--iters", type=int, default=300, help="chamfer refinement steps")
    ap.add_argument("--lr", type=float, default=1e-3, help="chamfer refinement learning rate")
    ap.add_argument("--samples", type=int, default=3000, help="target / mesh samples")
    ap.add_argument("--sampson-iters", type=int, default=100)
    ap.add_argument("--disp-penalty", type=float, default=1e-3, help="warp: lambda * mean(D^2)")
    ap.add_argument("--normal-weight", type=float, default=1.0)
    ap.add_argument("--smooth-weight", type=float, default=0.01)
    ap.add_argument("--anchor-weights", type=float, nargs="+", default=[0.03, 0.3],
                    help="stage 1: pulls towards the signed-distance estimate (one fit each, the best is kept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="fit file (default data/fits/grid/<target>[_cloud]__<model>__<method>__<cells>.json)")
    add_classify_arguments(ap)
    args = ap.parse_args()

    if args.list:
        ex = T.list_examples()
        print("example functions    (--function NAME):", ", ".join(ex["functions"]))
        print("example point clouds (--points NAME):  ", ", ".join(ex["point_clouds"]))
        return
    if not (args.function or args.points):
        raise SystemExit("give --function or --points (see --list)")
    cfg = GridFitConfig(model=args.model, method=args.method, depth=args.depth, normal_weight=args.normal_weight,
                        smooth_weight=args.smooth_weight, anchor_weights=tuple(args.anchor_weights),
                        sampson_iters=args.sampson_iters,
                        disp_penalty=args.disp_penalty, iters=args.iters, lr=args.lr, samples=args.samples,
                        seed=args.seed)
    grid, target = make_grid_and_target(args)
    model = GridModel(grid, args.model)
    print(f"target: {target.describe()}")
    print(f"grid:   {model.describe()}, domain {np.round(grid.lo_np, 3)} .. {np.round(grid.hi_np, 3)}")
    print(f"method: {args.method}")
    ccfg = classify_config(args, args.method) if args.method != "chamfer" else None
    t0 = time.time()
    p, info = fit(model, target, cfg, ccfg, chamfer_iters_after=args.chamfer_iters)
    err = evaluate(model, p, target, depth=max(cfg.depth, 3))
    if args.method != "chamfer":
        sampler = info["sampler"]
        err.update(info["val"].accuracy(model, p))
        if sampler.lattice is not None:
            err["accuracy_train_lattice"] = sampler.as_validation(*sampler.lattice).accuracy(model, p)["accuracy"]
    err["time"] = time.time() - t0
    print(f"  done in {err['time']:.1f}s")
    print_report(err)
    out = args.out or output_path(fit_name(target, model, args.method), GRID_FITS)
    save_fit(out, model, p, target, cfg, err, info, ccfg)
    print(f"saved {out}   (view: python code/visualization/grid_viewer.py --load {os.path.basename(out)})")


if __name__ == "__main__":
    main()
