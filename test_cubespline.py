"""
Sanity tests for cube.py and marching.py.   Run:  python test_cubespline.py
(also works with pytest).
"""

import numpy as np
import torch

from cube import Cube, PRESETS, CORNER_IJK
import marching as M
from marching import extract_topology, marching_cubes, mesh_check, to_numpy, _case_triangles

torch.manual_seed(0)
RNG = np.random.default_rng(0)


def test_cube_basics():
    c = Cube(RNG.normal(size=8))
    P = c.corner_positions()
    assert torch.allclose(c.evaluate(P), c.values)                      # interpolates corners
    x = c.lo + torch.rand(500, 3, dtype=c.dtype) * c.size
    assert torch.allclose(c.basis(x).sum(-1), torch.ones(500, dtype=c.dtype))
    xg = x.clone().requires_grad_(True)
    (g_auto,) = torch.autograd.grad(c.evaluate(xg).sum(), xg)
    assert torch.allclose(c.gradient(x), g_auto)                        # analytic gradient
    # batch of value vectors
    B = torch.randn(4, 8, dtype=c.dtype)
    assert c.evaluate(x, B).shape == (500, 4)


def test_case_table_consistent():
    """All 256 configs x all face decisions build closed, consistently
    oriented polygons (asserts inside _case_triangles)."""
    n = 0
    for cfg in range(256):
        for dec in range(64):
            n += len(_case_triangles(cfg, dec))
    assert n > 0


def _random_cubes(k):
    for name, v in PRESETS.items():
        yield name, v
    for i in range(k):
        yield f"random{i}", RNG.normal(size=8)


def test_meshes_watertight_exact_oriented():
    worst = 0.0
    for name, vals in _random_cubes(150):
        cube = Cube(vals, lo=-1.0, size=2.0)
        for level in (0.0, 0.3):
            for depth in range(0, 6):
                V, F, topo = marching_cubes(cube, depth, level)
                if not len(F):
                    continue
                chk = mesh_check(V, F, cube.lo, cube.size)
                assert chk["ok"], (name, depth, level, chk)
                with torch.no_grad():
                    err = (cube.evaluate(V) - level).abs().max().item()
                worst = max(worst, err)
                assert err < 1e-10, (name, depth, err)                 # vertices on the surface
                # orientation: face normals agree with grad f
                Vn = to_numpy(V)
                n = np.cross(Vn[F[:, 1]] - Vn[F[:, 0]], Vn[F[:, 2]] - Vn[F[:, 0]])
                area = np.linalg.norm(n, axis=1)
                cen = torch.as_tensor(Vn[F].mean(1))
                gr = cube.gradient(cen).detach().numpy()
                good = area > 1e-10 * cube.size ** 2
                dots = (n[good] * gr[good]).sum(1)
                agree = area[good][dots > 0].sum() / area[good].sum()
                # depth 0-1: one coarse polygon can span a strongly curved patch and
                # a fan triangle may fold; from depth 2 on every triangle must agree
                assert agree > (0.999 if depth >= 2 else 0.6), (name, depth, agree)
    print(f"  max |f(V) - level| over all meshes: {worst:.2e}")


def test_pruning_matches_brute_force():
    for name, vals in _random_cubes(30):
        cube = Cube(vals)
        for depth in (2, 4):
            topo = extract_topology(cube, depth)
            R = 2 ** depth
            ijk = np.stack(np.meshgrid(*[np.arange(R + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
            g = (M._lattice_values(cube, ijk, R, 0.0) < 0).reshape(R + 1, R + 1, R + 1)  # [i,j,k]
            mixed = set()
            for i in range(R):
                for j in range(R):
                    for k in range(R):
                        s = g[i:i + 2, j:j + 2, k:k + 2]
                        if s.any() and not s.all():
                            mixed.add((i, j, k))
            assert mixed == set(map(tuple, topo.cells.tolist())), (name, depth)


def test_robust_to_roundoff():
    """Several presets put the surface exactly through lattice points. Different
    platforms / BLAS libraries round those values to +0 or -1e-17 differently;
    inject that noise and require the meshes to stay watertight."""
    orig = M._trilinear_np
    rng = np.random.default_rng(1)
    M._trilinear_np = lambda c, u: orig(c, u) + rng.normal(size=len(u)) * 1e-15
    try:
        for name, vals in PRESETS.items():
            cube = Cube(vals)
            for depth in range(0, 7):
                V, F, _ = marching_cubes(cube, depth)
                if len(F):
                    chk = mesh_check(V, F, cube.lo, cube.size)
                    assert chk["ok"], (name, depth, chk)
    finally:
        M._trilinear_np = orig


def test_vertices_differentiable():
    cube = Cube(RNG.normal(size=8))
    topo = extract_topology(cube, 3)
    v0 = cube.values.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda v: topo.vertices(v), (v0,))
    # frozen topology reproduces the extracted positions exactly
    V1, _, _ = marching_cubes(cube, 3)
    assert torch.allclose(topo.vertices(cube.values), V1)
    # gradient flows into the Cube parameter
    V1.square().sum().backward()
    assert cube.values.grad is not None and cube.values.grad.abs().sum() > 0


def test_area_converges():
    cube = Cube(PRESETS["bowl"])
    areas = []
    for d in range(1, 8):
        V, F, _ = marching_cubes(cube, d)
        V = to_numpy(V)
        areas.append(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]],
                                                   V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
    diffs = np.abs(np.diff(areas))
    assert diffs[-1] < diffs[0] * 0.05, areas
    print("  bowl area by depth:", " ".join(f"{a:.5f}" for a in areas))


if __name__ == "__main__":
    import time
    for fn in [test_cube_basics, test_case_table_consistent, test_meshes_watertight_exact_oriented,
               test_pruning_matches_brute_force, test_robust_to_roundoff,
               test_vertices_differentiable, test_area_converges]:
        t = time.time()
        fn()
        print(f"ok  {fn.__name__}  ({time.time() - t:.2f}s)")
    cube = Cube(PRESETS["checkerboard"])
    for d in (4, 6, 7):
        t = time.time()
        topo = extract_topology(cube, d)
        print(f"  depth {d}: {topo.stats['active_cells']}/{topo.stats['total_cells']} cells, "
              f"{topo.stats['vertices']} V, {topo.stats['faces']} F, {time.time() - t:.3f}s")
