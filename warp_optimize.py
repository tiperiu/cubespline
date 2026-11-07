#!/usr/bin/env python3
"""
warp_optimize.py -- fit the 16 values of a WarpCube (8 offset + 8 field) to a target.

Same idea as optimize.py, with the 16-value cube and its mesher:

    start:  multi-start, mesh-free fit: from the 8-value solution (offset 0, field =
            closed-form fit of the 8-value cube) and from several random 16-value
            starts, minimise the first-order point-to-surface (Sampson) distance with
            L-BFGS; keep the start with the smallest true distance. Good fits need
            large offsets, which plain gradient descent from offset 0 does not reach.
    loop:   topology = warp_marching.extract_topology(cube, depth)   no gradients
            V        = topology.vertices(values)                      differentiable in all 16
            loss     = sum_k w_k * LOSSES[k](ctx)                     default: chamfer
            Adam step, then |field| renormalised (the field part is scale invariant;
            the offset part is not, it moves the geometry)

Targets, sampling, chamfer and the final-error report are shared with optimize.py.

Two stages, each with its own iteration count:
    stage 1  --start-iters N    L-BFGS iterations per start of the multi-start fit (default 100)
             --starts K         number of random starts (default 16, plus the 8-value fit)
    stage 2  --refine-iters M   Adam steps of the chamfer refinement (default 300)
             --no-refine        stage 1 only (same as --refine-iters 0)

Usage:
    python warp_optimize.py --target "sphere cap"
    python warp_optimize.py --target "sphere cap" --start-iters 200 --refine-iters 500
    python warp_optimize.py --target "sphere cap" --no-refine
    python warp_optimize.py --target "sphere cap" --init regular   # start at the 8-value fit only
    python warp_optimize.py --target "torus inner flank" --depth 5 --refine-iters 400 --out warp_torus.json
    python warp_optimize.py --target "sphere cap" --view          # live window
    python warp_optimize.py --list
View a result with   python warp_viewer.py   (fits appear in the combo box).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import optimize as O
from cube import Cube, CUBE_EDGES
from marching import to_numpy
from warpcube import WarpCube
import warp_marching as WM

DT = torch.float64


# ============================================================================
# Distances to the target (same definitions as optimize.evaluate)
# ============================================================================
def evaluate(values, target: O.Target, depth=5, n=20000, seed=1):
    """First-order distances between the fitted surface and the target, both ways."""
    cube = WarpCube(values, requires_grad=False)
    topo = WM.extract_topology(cube, depth)
    if not len(topo.F):
        return {"empty": True}
    with torch.no_grad():
        V = topo.vertices(cube.values)
        P = O.surface_samples(V, topo.F, n, torch.Generator().manual_seed(seed)).numpy()
        Q = target.points
        f = cube.evaluate(Q)
        g = cube.gradient(Q).norm(dim=-1)
        d_qp = (f.abs() / g.clamp_min(1e-12)).numpy()
    if target.fn is not None:
        gv, G = target.fn(P), O._fn_gradient(target.fn, P)
        d_pq = np.abs(gv) / np.maximum(np.linalg.norm(G, axis=-1), 1e-12)
    else:
        dense = getattr(target, "_dense", None)
        if dense is None:
            dense = torch.as_tensor(O.sample_surface(target.V, target.F, 50000, np.random.default_rng(seed))[0])
            target._dense = dense
        d_pq = torch.cat([torch.cdist(torch.as_tensor(P[i:i + 4096]), dense).min(1).values
                          for i in range(0, len(P), 4096)]).numpy()
    return {"rms": float(np.sqrt((np.mean(d_pq ** 2) + np.mean(d_qp ** 2)) / 2)),
            "fit_to_target_max": float(d_pq.max()), "target_to_fit_max": float(d_qp.max()),
            "faces": int(len(topo.F))}


# ============================================================================
# Losses (registry, same pattern as optimize.py)
# ============================================================================
@dataclass
class LossContext:
    cube: WarpCube
    values: torch.Tensor          # the 16 values (requires grad)
    V: torch.Tensor
    F: np.ndarray
    mesh_points: torch.Tensor     # area-uniform samples on the fitted mesh, differentiable
    target: O.Target


def loss_chamfer(ctx):
    return O.chamfer(ctx.mesh_points, ctx.target.points)


def loss_algebraic(ctx):
    """First-order (Sampson) distance of the target points to the level set."""
    f = ctx.cube.evaluate(ctx.target.points, ctx.values)
    g = ctx.cube.gradient(ctx.target.points, ctx.values)
    return (f.square() / g.square().sum(-1).clamp_min(1e-12)).mean()


def loss_normal(ctx):
    g = ctx.cube.gradient(ctx.target.points, ctx.values)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (1 - (g * ctx.target.normals).sum(-1)).mean()


def loss_offset(ctx):
    """Keeps the offsets small (mean squared offset value)."""
    return ctx.values[:8].square().mean()


LOSSES = {"chamfer": loss_chamfer, "algebraic": loss_algebraic, "normal": loss_normal,
          "offset": loss_offset}


# ============================================================================
# Starting values: multi-start, mesh-free fit
# ============================================================================
def fit_sampson(target: O.Target, v0, iters=100, normal_weight=1e-3, n_points=1500):
    """L-BFGS on the Sampson distance f^2/|grad f|^2 of the target points to the
    level set (plus a small normal-agreement term that picks the inside side).
    No mesh is involved, so it is fast and smooth."""
    cube = WarpCube(v0, requires_grad=True)
    P, N = target.points[:n_points], target.normals[:n_points]
    opt = torch.optim.LBFGS([cube.values], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        f = cube.evaluate(P)
        g = cube.gradient(P)
        gn = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = (f.square() / g.square().sum(-1).clamp_min(1e-12)).mean() \
            + normal_weight * (1 - (gn * N).sum(-1)).mean()
        loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:  # noqa: BLE001 -- a failed line search just keeps the last values
        pass
    v = cube.values.detach().clone()
    if not torch.isfinite(v).all():
        return None
    return _normalised(v)


def multistart(target: O.Target, starts=16, seed=0, depth=5, verbose=False, start_iters=100):
    """Best of: the 8-value solution and `starts` random starts, each improved by
    fit_sampson. Scored by the true RMS distance. Returns (values, description)."""
    field8, choice = O.fit_algebraic(Cube(), target)
    base = torch.cat([torch.zeros(8, dtype=DT), field8])
    rng = torch.Generator().manual_seed(seed)
    cands = [("8-value fit", base)]
    for i in range(starts):
        scale = (0.1, 0.3, 0.6)[i % 3]
        cands.append((f"random start {i} (offset scale {scale})",
                      torch.cat([scale * torch.randn(8, dtype=DT, generator=rng),
                                 torch.randn(8, dtype=DT, generator=rng)])))
    best = (np.inf, base, "8-value fit, zero offset")
    for name, v0 in cands:
        v = fit_sampson(target, v0, iters=start_iters)
        if v is None:
            continue
        r = evaluate(v.numpy(), target, depth=depth, n=3000).get("rms", np.inf)
        if verbose:
            print(f"    start {name:32s} -> RMS {r:.4e}")
        if r < best[0]:
            best = (r, v, f"multi-start: {name} + Sampson fit")
    return best[1], best[2]


# ============================================================================
# Fitter
# ============================================================================
@dataclass
class FitConfig:
    depth: int = 5
    iters: int = 300              # stage 2: Adam refinement steps (--refine-iters)
    lr: float = 0.02
    lr_final: float = 2e-4
    mesh_samples: int = 3000
    losses: dict = field(default_factory=lambda: {"chamfer": 1.0})
    init: str = "multistart"      # "multistart" | "regular" (8-value fit, zero offset) | "random" | "given"
    starts: int = 16              # random starts for "multistart"
    start_iters: int = 100        # stage 1: L-BFGS iterations per start
    seed: int = 0
    keep_best: bool = True
    check_every: int = 10


def _normalised(v):
    v = torch.as_tensor(v, dtype=DT).clone()
    v[8:] = v[8:] / v[8:].norm() * np.sqrt(8)
    return v


class WarpFitter:
    def __init__(self, target: O.Target, config: FitConfig, values=None, verbose=False):
        self.target, self.cfg, self.verbose = target, config, verbose
        self.cube = WarpCube(requires_grad=True)
        self.gen = torch.Generator().manual_seed(config.seed)
        self.reset(values)

    def initial_values(self):
        field8, choice = O.fit_algebraic(Cube(), self.target)
        return torch.cat([torch.zeros(8, dtype=DT), field8]), f"8-value fit ({choice}), zero offset"

    def reset(self, values=None):
        cfg = self.cfg
        if values is not None:
            v, self.init_choice = torch.as_tensor(values, dtype=DT), "given"
        elif cfg.init == "multistart":
            t0 = time.time()
            v, self.init_choice = multistart(self.target, cfg.starts, cfg.seed, cfg.depth, self.verbose,
                                             start_iters=cfg.start_iters)
            if self.verbose:
                print(f"  multi-start took {time.time() - t0:.1f}s")
        elif cfg.init == "random":
            v = torch.cat([0.2 * torch.randn(8, dtype=DT, generator=self.gen),
                           torch.randn(8, dtype=DT, generator=self.gen)])
            self.init_choice = "random"
        else:
            v, self.init_choice = self.initial_values()
        self.cube.set_values(_normalised(v))
        self.opt = torch.optim.Adam([self.cube.values], lr=cfg.lr)
        self.it = 0
        self.history = []
        self.topo = None
        self.init_values = self.values
        self.best_values, self.best_score, self.best_it = self.values, self.score(), 0
        self.last_score = self.best_score

    @property
    def values(self) -> np.ndarray:
        return self.cube.values.detach().numpy().copy()

    def score(self, values=None) -> float:
        """True RMS distance (deterministic) -- used to keep the best iterate."""
        v = self.values if values is None else values
        return float(evaluate(v, self.target, depth=self.cfg.depth, n=self.cfg.mesh_samples).get("rms", np.inf))

    def objective(self, values=None):
        """Configured losses at `values`, fixed samples -> (total, {term: value})."""
        cube = self.cube if values is None else WarpCube(values, requires_grad=False)
        with torch.no_grad():
            topo = WM.extract_topology(cube, self.cfg.depth)
            if not len(topo.F):
                return float("inf"), {}
            V = topo.vertices(cube.values)
            pts = O.surface_samples(V, topo.F, self.cfg.mesh_samples, torch.Generator().manual_seed(12345))
            ctx = LossContext(cube, cube.values, V, topo.F, pts, self.target)
            parts = {k: float(LOSSES[k](ctx)) for k, w in self.cfg.losses.items() if w}
        return float(sum(self.cfg.losses[k] * v for k, v in parts.items())), parts

    def step(self):
        cfg, cube = self.cfg, self.cube
        self.topo = WM.extract_topology(cube, cfg.depth)
        if not len(self.topo.F):
            v, _ = self.initial_values()           # surface left the cube: restart
            cube.set_values(_normalised(v + 0.05 * torch.randn(16, dtype=DT, generator=self.gen)))
            self.opt = torch.optim.Adam([cube.values], lr=self.opt.param_groups[0]["lr"])
            self.topo = WM.extract_topology(cube, cfg.depth)
            if self.verbose:
                print(f"  iter {self.it}: surface left the cube, restarted")
            if not len(self.topo.F):
                self.history.append((self.it, float("nan"), {}))
                self.it += 1
                return float("nan")
        V = self.topo.vertices(cube.values)
        pts = O.surface_samples(V, self.topo.F, cfg.mesh_samples, self.gen)
        ctx = LossContext(cube, cube.values, V, self.topo.F, pts, self.target)
        parts = {k: LOSSES[k](ctx) for k, w in cfg.losses.items() if w}
        total = sum(cfg.losses[k] * v for k, v in parts.items())
        self.opt.zero_grad()
        total.backward()
        if self.it < cfg.iters:
            w = 0.5 * (1 + np.cos(np.pi * self.it / max(cfg.iters, 1)))
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr_final + (cfg.lr - cfg.lr_final) * w
        self.opt.step()
        with torch.no_grad():                       # field part is scale invariant
            cube.values[8:] *= np.sqrt(8) / cube.values[8:].norm()
        self.history.append((self.it, float(total.detach()), {k: float(v.detach()) for k, v in parts.items()}))
        self.it += 1
        if cfg.keep_best and self.it % cfg.check_every == 0:
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
                print(f"  iter {self.it:4d}  loss {loss:.4e}   {parts}   rms {self.last_score:.3e}   ({time.time() - t0:.1f}s)")
        if self.cfg.keep_best:
            s = self.score()
            if s < self.best_score:
                self.best_values, self.best_score, self.best_it = self.values, s, self.it
            self.cube.set_values(self.best_values)
        return self.values

    def mesh(self):
        topo = WM.extract_topology(self.cube, self.cfg.depth)
        return to_numpy(topo.vertices(self.cube.values)), topo.F


# ============================================================================
# Report and save (same report format as optimize.py)
# ============================================================================
def final_error(fitter: WarpFitter, metrics=None) -> dict:
    total, parts = fitter.objective()
    m = metrics if metrics is not None else evaluate(fitter.values, fitter.target)
    m0 = evaluate(fitter.init_values, fitter.target)
    width = fitter.cube.size
    err = {"objective": total, "terms": parts, "weights": dict(fitter.cfg.losses),
           "best_iteration": fitter.best_it, "iterations": fitter.it, "cube_width": width,
           "initial_rms": m0.get("rms", float("inf")), "initial": fitter.init_choice}
    if not m.get("empty"):
        err.update({k: m[k] for k in ("rms", "fit_to_target_max", "target_to_fit_max")})
        err["rms_percent_of_cube"] = 100 * m["rms"] / width
        err["max_percent_of_cube"] = 100 * max(m["fit_to_target_max"], m["target_to_fit_max"]) / width
    err["max_abs_offset"] = float(np.abs(fitter.values[:8]).max())
    return err


def print_final_error(err):
    O.print_final_error(err)
    print(f"  start: {err['initial']};  largest offset value {err['max_abs_offset']:.3f}")


def save_fit(path, fitter: WarpFitter, metrics=None):
    err = final_error(fitter, metrics)
    v = fitter.values
    data = {"type": "warp16", "values": v.tolist(), "offset": v[:8].tolist(), "field": v[8:].tolist(),
            "depth": fitter.cfg.depth, "target": fitter.target.name, "iterations": fitter.it,
            "losses": fitter.cfg.losses, "final_loss": err["objective"], "final_error": err}
    if metrics:
        data["metrics"] = metrics
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


# ============================================================================
# Live view
# ============================================================================
def view(fitter: WarpFitter):
    import polyscope as ps
    import polyscope.imgui as psim
    from viewer import BASE_COLOR, BACK_COLOR, EDGE_COLOR, CUBE_COLOR, TARGET_COLOR

    st = {"running": False, "steps": 1, "wire": True}
    ps.init()
    ps.set_program_name("cubespline -- warp optimize")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    P = to_numpy(Cube().corner_positions())
    ps.register_curve_network("cube", P, CUBE_EDGES, color=CUBE_COLOR).set_radius(0.004, relative=True)
    t = fitter.target
    tm = ps.register_surface_mesh("target", t.V, t.F, color=TARGET_COLOR, transparency=0.45)
    tm.set_back_face_policy("identical")

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
        Pm = to_numpy(fitter.cube.moved_corners())
        nodes = np.vstack([P, Pm])
        ps.register_curve_network("offsets", nodes, np.c_[np.arange(8), np.arange(8) + 8],
                                  color=(0.2, 0.5, 0.9)).set_radius(0.003, relative=True)
        ps.register_point_cloud("moved corners", Pm, radius=0.01, color=(0.2, 0.5, 0.9))

    def ui():
        psim.PushItemWidth(160)
        if psim.CollapsingHeader("Optimize", psim.ImGuiTreeNodeFlags_DefaultOpen):
            if psim.Button("Pause" if st["running"] else "Run"):
                st["running"] = not st["running"]
            psim.SameLine()
            if psim.Button("Step"):
                fitter.step(); draw()
            psim.SameLine()
            if psim.Button("Reset (8-value start)"):
                fitter.cfg.init = "regular"; fitter.reset(); draw()
            psim.SameLine()
            if psim.Button("Reset (multi-start)"):
                fitter.cfg.init = "multistart"; fitter.reset(); draw()
            c, d = psim.SliderInt("depth", fitter.cfg.depth, 2, 7)
            if c:
                fitter.cfg.depth = d; draw()
            _, st["steps"] = psim.SliderInt("steps / frame", st["steps"], 1, 10)
            c, st["wire"] = psim.Checkbox("Wireframe", st["wire"])
            if c:
                draw()
            for name in LOSSES:
                c, w = psim.InputFloat(f"weight {name}", fitter.cfg.losses.get(name, 0.0))
                if c:
                    fitter.cfg.losses[name] = max(0.0, w)
            if psim.Button("Save warp_fit.json"):
                save_fit("warp_fit.json", fitter)
                print_final_error(final_error(fitter))
                print("saved warp_fit.json")
        if psim.CollapsingHeader("Progress", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.TextUnformatted(f"iteration  {fitter.it}")
            if fitter.history:
                psim.TextUnformatted(f"loss       {fitter.history[-1][1]:.4e}")
            psim.TextUnformatted(f"RMS dist.  {fitter.last_score:.4e}")
            psim.TextUnformatted(f"best RMS   {fitter.best_score:.4e}  at iteration {fitter.best_it}")
            v = fitter.values
            psim.TextWrapped("offset [" + ", ".join(f"{x:+.3f}" for x in v[:8]) + "]")
            psim.TextWrapped("field  [" + ", ".join(f"{x:+.3f}" for x in v[8:]) + "]")
        psim.PopItemWidth()
        if st["running"]:
            for _ in range(st["steps"]):
                fitter.step()
            draw()

    draw()
    ps.reset_camera_to_home_view()
    ps.set_user_callback(ui)
    ps.show()


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--target", default="sphere cap", choices=["trilinear", *O.FUNCTIONS])
    src.add_argument("--mesh", help="target mesh (.obj)")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--refine-iters", "--iters", dest="refine_iters", type=int, default=300,
                    help="stage 2: Adam steps of the chamfer refinement (default 300; --iters is an alias)")
    ap.add_argument("--no-refine", action="store_true", help="stage 1 only (same as --refine-iters 0)")
    ap.add_argument("--start-iters", type=int, default=100,
                    help="stage 1: L-BFGS iterations per start of the multi-start fit (default 100)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--loss", action="append", metavar="NAME=W", help=f"repeatable ({', '.join(LOSSES)})")
    ap.add_argument("--init", default="multistart", choices=["multistart", "regular", "random"],
                    help="multistart (default): best of the 8-value fit and random starts; "
                         "regular: 8-value fit with zero offset")
    ap.add_argument("--starts", type=int, default=16, help="random starts for --init multistart")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="warp_fit.json")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("targets:", "trilinear,", ", ".join(O.FUNCTIONS))
        print("losses: ", ", ".join(LOSSES))
        return
    if args.mesh:
        V, F = O.load_obj(args.mesh)
        target = O.target_from_mesh(V, F, name=args.mesh, n=args.samples, seed=args.seed)
    else:
        target = O.builtin_target(args.target, n=args.samples, seed=args.seed)
    losses = {"chamfer": 1.0}
    if args.loss:
        losses = {}
        for it in args.loss:
            k, _, w = it.partition("=")
            if k not in LOSSES:
                raise SystemExit(f"unknown loss '{k}'; available: {', '.join(LOSSES)}")
            losses[k] = float(w) if w else 1.0
    refine_iters = 0 if args.no_refine else args.refine_iters
    cfg = FitConfig(depth=args.depth, iters=refine_iters, lr=args.lr, lr_final=args.lr / 100,
                    mesh_samples=args.samples, losses=losses, init=args.init, starts=args.starts,
                    start_iters=args.start_iters, seed=args.seed)
    fitter = WarpFitter(target, cfg, verbose=True)
    print(f"target {target.name}: {len(target.points)} samples; losses {cfg.losses}; depth {cfg.depth}")
    print(f"stage 1 ({cfg.init}, {cfg.starts} random starts x {cfg.start_iters} L-BFGS iterations): "
          f"{fitter.init_choice}, RMS {fitter.best_score:.4e}")
    print(f"stage 2: {cfg.iters} refinement steps" + ("  (skipped)" if cfg.iters == 0 else ""))
    if args.view:
        view(fitter)
        return
    fitter.run()
    metrics = evaluate(fitter.values, target)
    v = fitter.values
    print("offset", np.round(v[:8], 4).tolist())
    print("field ", np.round(v[8:], 4).tolist())
    print_final_error(final_error(fitter, metrics))
    save_fit(args.out, fitter, metrics)
    print(f"saved {args.out}  (view: python warp_viewer.py --load {args.out})")


if __name__ == "__main__":
    main()
