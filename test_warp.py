"""
Tests for warpcube.py, warp_marching.py and warp_optimize.py.
Run:  python test_warp.py   (or pytest)
"""

import numpy as np
import torch

import optimize as O
import warp_marching as WM
import warp_optimize as W
from cube import Cube
from marching import marching_cubes as mc8, mesh_check, to_numpy
from warpcube import WarpCube, WARP_PRESETS, field_local_np

RNG = np.random.default_rng(0)


def _random_warp(scale=0.35):
    return np.r_[RNG.normal(size=8) * scale, RNG.normal(size=8)]


def test_zero_offset_is_the_8_value_cube():
    for _ in range(30):
        c = RNG.normal(size=8)
        x = torch.rand(200, 3, dtype=torch.float64) * 2 - 1
        assert torch.allclose(WarpCube(c).evaluate(x), Cube(c).evaluate(x))
        V1, F1, _ = mc8(Cube(c), 4)
        V2, F2, _ = WM.marching_cubes(WarpCube(c), 4)
        assert len(F1) == len(F2)
        assert np.allclose(np.sort(to_numpy(V1), 0), np.sort(to_numpy(V2), 0), atol=1e-12)


def test_gradient_and_numpy_agree():
    for _ in range(20):
        v = _random_warp(0.5)
        w = WarpCube(v)
        x = (torch.rand(200, 3, dtype=torch.float64) * 2 - 1).requires_grad_(True)
        (g,) = torch.autograd.grad(w.evaluate(x).sum(), x)
        assert torch.allclose(w.gradient(x.detach()), g, atol=1e-10)
        u = ((x.detach() + 1) / 2).numpy()
        assert np.allclose(field_local_np(v, u, 2.0), w.evaluate(x.detach()).detach().numpy())
        assert torch.allclose(w.warp(w.corner_positions()), w.moved_corners())


def test_meshes_watertight_and_exact():
    worst = 0.0
    cases = list(WARP_PRESETS.values()) + [_random_warp(s) for s in (0.2, 0.5, 1.0) for _ in range(40)]
    for v in cases:
        w = WarpCube(v)
        for depth in (1, 3, 5):
            V, F, _ = WM.marching_cubes(w, depth)
            if not len(F):
                continue
            chk = mesh_check(V, F, w.lo, w.size)
            assert chk["ok"], (v, depth, chk)
            with torch.no_grad():
                worst = max(worst, float(w.evaluate(V).abs().max()))
    assert worst < 1e-10, worst
    print(f"  max |f(V)| = {worst:.1e}")


def test_pruning_never_loses_a_sign_change():
    """Every leaf cell whose corners change sign must survive the bound-based pruning."""
    for _ in range(15):
        w = WarpCube(_random_warp(0.6))
        depth = 4
        R = 2 ** depth
        ijk = np.stack(np.meshgrid(*[np.arange(R + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        tol = WM._snap_tol(w.values.detach().numpy(), 0.0)
        g = WM.lattice_values(w.values.detach().numpy(), ijk, R, 2.0, 0.0, tol) < 0
        g = g.reshape(R + 1, R + 1, R + 1)
        mixed = {(i, j, k) for i in range(R) for j in range(R) for k in range(R)
                 if g[i:i + 2, j:j + 2, k:k + 2].any() and not g[i:i + 2, j:j + 2, k:k + 2].all()}
        topo = WM.extract_topology(w, depth)
        assert mixed == set(map(tuple, topo.cells.tolist()))


def test_vertices_differentiable_in_16_values():
    done = 0
    while done < 3:
        w = WarpCube(_random_warp(0.3))
        topo = WM.extract_topology(w, 3)
        if not len(topo.F):
            continue
        v0 = w.values.detach().clone().requires_grad_(True)
        assert torch.autograd.gradcheck(lambda x: topo.vertices(x), (v0,), eps=1e-7, atol=1e-5)
        done += 1


def test_beats_8_value_cube_on_sphere_cap():
    """The 8-value cube cannot bend a level set into a cap inside the cube
    (RMS ~ 0.09); the 16-value cube must do much better."""
    t = O.builtin_target("sphere cap", n=1500)
    f = W.WarpFitter(t, W.FitConfig(depth=4, iters=20, starts=6, mesh_samples=1500))
    f.run(verbose=False)
    r = W.evaluate(f.values, t, depth=5)["rms"]
    assert r < 0.02, r


def test_never_worse_than_8_value_start():
    t = O.builtin_target("sphere flank", n=1500)
    f = W.WarpFitter(t, W.FitConfig(depth=4, iters=20, init="regular", mesh_samples=1500))
    r0 = W.evaluate(f.values, t, depth=4, n=1500)["rms"]
    f.run(verbose=False)
    assert W.evaluate(f.values, t, depth=4, n=1500)["rms"] <= r0 + 1e-12


if __name__ == "__main__":
    import time
    for fn in [test_zero_offset_is_the_8_value_cube, test_gradient_and_numpy_agree,
               test_meshes_watertight_and_exact, test_pruning_never_loses_a_sign_change,
               test_vertices_differentiable_in_16_values, test_beats_8_value_cube_on_sphere_cap,
               test_never_worse_than_8_value_start]:
        t = time.time()
        fn()
        print(f"ok  {fn.__name__}  ({time.time() - t:.2f}s)")
