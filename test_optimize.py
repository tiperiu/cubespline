"""
Tests for optimize.py.   Run:  python test_optimize.py   (or pytest)
"""

import numpy as np
import torch

import optimize as O
from cube import Cube


def test_algebraic_init_recovers_trilinear():
    """Target is itself trilinear: the closed-form fit must return its values
    (up to positive scale) and the right inside/outside sign."""
    t = O.builtin_target("trilinear", n=3000)
    c = O.fit_taubin(Cube(), t.points, t.normals).numpy()
    ang = O.values_close_up_to_scale(c, t.values)
    assert ang < 0.01, ang                      # degrees; 180 would mean the sign flipped
    assert O.evaluate(c, t)["rms"] < 1e-4


def test_algebraic_init_sign_follows_normals():
    """Open, graph-like targets: the fitted inside/outside must match the target's.
    Closed targets (sphere, cylinder): a trilinear field has no interior extremum,
    so it cannot be negative inside them; only the aggregate sign is guaranteed."""
    for name, need in [("plane", 0.95), ("paraboloid", 0.95), ("sphere", None), ("cylinder", None)]:
        t = O.builtin_target(name, n=2000)
        cube = Cube()
        c, _ = O.fit_algebraic(cube, t)
        dots = ((cube.basis_gradient(t.points) @ c) * t.normals).sum(-1)
        assert float(dots.sum()) > 0, name
        if need:
            assert (dots > 0).float().mean() > need, name


def test_refinement_recovers_from_perturbed_start():
    """Start away from a representable target; the mesh-based chamfer refinement
    alone (through marching cubes) must bring the surface back."""
    t = O.builtin_target("trilinear", n=3000)
    start = t.values + np.random.default_rng(0).normal(scale=0.3, size=8)
    f = O.Fitter(t, O.FitConfig(depth=5, iters=250, lr=0.1, lr_final=1e-3), values=start)
    e0 = O.evaluate(f.values, t)["rms"]
    f.run(verbose=False)
    e1 = O.evaluate(f.values, t)["rms"]
    assert e1 < 0.15 * e0 and e1 < 5e-3, (e0, e1)


def test_gradient_reaches_values():
    t = O.builtin_target("sphere", n=1000)
    f = O.Fitter(t, O.FitConfig(depth=4))
    v = f.cube.values.detach().clone()          # the values the gradient is taken at
    f.step()
    g = f.cube.values.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # scale invariance: vertex positions are homogeneous of degree 0 in the
    # values, so the chamfer gradient is orthogonal to them (Euler)
    assert abs(float(g @ v)) < 1e-2 * float(g.norm() * v.norm()) + 1e-12


def test_empty_mesh_falls_back():
    """All corners positive -> no surface in the cube; the fitter must still move."""
    t = O.builtin_target("plane", n=1000)
    f = O.Fitter(t, O.FitConfig(depth=4, lr=0.05), values=np.ones(8))
    for _ in range(80):
        f.step()
    assert len(f.mesh()[1]) > 0


def test_mesh_target_clipped_to_cube():
    """A sphere mesh sticking out of the cube: only samples inside are kept."""
    t = np.linspace(-1.6, 1.6, 49)
    X = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1)
    V, F = O.mesh_from_grid(np.linalg.norm(X - [0.6, 0, 0], axis=-1) - 0.9, lo=-1.6, size=3.2)
    tgt = O.target_from_mesh(V, F, n=2000)
    P = tgt.points.numpy()
    assert (P >= -1 - 1e-9).all() and (P <= 1 + 1e-9).all()
    assert len(P) == 2000
    f = O.Fitter(tgt, O.FitConfig(depth=4, iters=30))
    f.run(verbose=False)
    assert O.evaluate(f.values, tgt)["rms"] < 0.5


def test_custom_loss_registry():
    calls = []

    def my_loss(ctx):
        calls.append(1)
        return ctx.mesh_points.square().mean() * 0.0 + O.loss_chamfer(ctx)
    O.LOSSES["mine"] = my_loss
    try:
        t = O.builtin_target("plane", n=1000)
        f = O.Fitter(t, O.FitConfig(depth=3, losses={"mine": 1.0, "normal": 0.1}))
        f.step()
        assert calls and set(f.history[-1][2]) == {"mine", "normal"}
    finally:
        del O.LOSSES["mine"]


if __name__ == "__main__":
    import time
    for fn in [test_algebraic_init_recovers_trilinear, test_algebraic_init_sign_follows_normals,
               test_refinement_recovers_from_perturbed_start, test_gradient_reaches_values,
               test_empty_mesh_falls_back, test_mesh_target_clipped_to_cube, test_custom_loss_registry]:
        t = time.time()
        fn()
        print(f"ok  {fn.__name__}  ({time.time() - t:.2f}s)")
