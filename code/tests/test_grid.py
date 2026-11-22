"""
Tests for the grid of cubes (code/grid): the scalar and warp models, marching cubes
over the grid, the targets (function files and point clouds) and the optimizers.
Run:  python -m pytest code/tests/test_grid.py -q      (or  python code/tests/test_grid.py)
"""

import sys as _sys
from pathlib import Path as _Path
_CODE = str(_Path(__file__).resolve().parents[1])          # .../code: makes `import core...` work
if _CODE not in _sys.path:                                  # when this file is run as a script
    _sys.path.insert(0, _CODE)

import os
import tempfile

import numpy as np
import torch

import optimization.classify_optimize as C
import grid.grid_optimize as GO
from core.cube import Cube
from core.geometry import count_pieces
from grid import targets as T
from grid.grid import DT, Grid, GridModel, basis
from grid.grid_marching import extract_topology, marching_cubes, mesh_check

RNG = np.random.default_rng(0)


def _sphere(c=(0.05, -0.1, 0.02), r=0.55):
    c = np.asarray(c)
    return lambda X: np.linalg.norm(X - c, axis=-1) - r


def _random_params(model, scale_d=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(model.nv, dtype=DT, generator=g)
    D = scale_d * torch.randn(model.nv, 3, dtype=DT, generator=g) if model.kind == "warp" else None
    return model.join(G, D)


# ----------------------------------------------------------------------------- the grid
def test_vertex_and_corner_numbering():
    grid = Grid((0.5, -1.0, 2.0), 0.25, (3, 2, 4))
    ijk = grid.vertex_ijk()
    assert len(ijk) == grid.num_vertices == 4 * 3 * 5
    assert np.array_equal(grid.vertex_id(ijk), np.arange(grid.num_vertices))
    assert np.allclose(grid.vertex_positions()[-1], grid.hi_np)
    # the 8 corners of a cell are in the corner order n = i + 2j + 4k of the basis
    X = torch.as_tensor(grid.lo_np + grid.h * np.array([[1.3, 0.6, 2.2]]), dtype=DT)
    ids, t = grid.locate(X)
    corners = grid.vertex_positions()[ids[0].numpy()]
    off = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)])
    assert np.allclose(corners, grid.lo_np + grid.h * (np.array([1, 0, 2]) + off))
    assert np.allclose(t.numpy(), [[0.3, 0.6, 0.2]])
    assert np.allclose(basis(t).sum(-1).numpy(), 1)


def test_one_cell_grid_is_the_single_cube():
    v = RNG.normal(size=8)
    cube = Cube(v, lo=-1.0, size=2.0, requires_grad=False)
    model = GridModel(Grid.cube(1), "scalar")
    X = torch.as_tensor(RNG.uniform(-1, 1, (500, 3)), dtype=DT)
    p = torch.as_tensor(v, dtype=DT)
    assert torch.allclose(model.field(p, X), cube.evaluate(X), atol=1e-14)
    assert torch.allclose(model.gradient(p, X), cube.gradient(X), atol=1e-13)


def test_continuous_across_cell_faces():
    grid = Grid.cube(4)
    for kind in ("scalar", "warp"):
        model = GridModel(grid, kind)
        p = _random_params(model, 0.2)
        X = RNG.uniform(-1, 1, (300, 3))
        for ax in range(3):
            Y = X.copy()
            Y[:, ax] = grid.lo[ax] + grid.h * RNG.integers(1, 4, len(X))      # interior cell faces
            a, b = Y.copy(), Y.copy()
            a[:, ax] -= 1e-10
            b[:, ax] += 1e-10
            fa = model.field(p, torch.as_tensor(a, dtype=DT))
            fb = model.field(p, torch.as_tensor(b, dtype=DT))
            assert (fa - fb).abs().max() < 1e-7, (kind, ax, float((fa - fb).abs().max()))


def test_zero_displacement_is_the_scalar_grid():
    grid = Grid.cube(3)
    s, w = GridModel(grid, "scalar"), GridModel(grid, "warp")
    G = torch.randn(s.nv, dtype=DT, generator=torch.Generator().manual_seed(1))
    X = torch.as_tensor(RNG.uniform(-1, 1, (400, 3)), dtype=DT)
    assert torch.allclose(s.field(G, X), w.field(w.join(G), X), atol=1e-15)
    assert torch.allclose(s.gradient(G, X), w.gradient(w.join(G), X), atol=1e-14)


def test_analytic_gradient_matches_autograd():
    grid = Grid.cube(3)
    for kind, scale in (("scalar", 0.0), ("warp", 0.25), ("warp", 1.0)):     # 1.0: many x' clamped
        model = GridModel(grid, kind)
        p = _random_params(model, scale, seed=2)
        X = torch.as_tensor(RNG.uniform(-0.98, 0.98, (400, 3)), dtype=DT).requires_grad_(True)
        g_auto, = torch.autograd.grad(model.field(p, X).sum(), X)
        assert torch.allclose(model.gradient(p, X), g_auto, atol=1e-12), (kind, scale)


def test_warped_points_stay_in_the_grid():
    grid = Grid.cube(3)
    model = GridModel(grid, "warp")
    p = _random_params(model, 2.0, seed=8)
    X = torch.as_tensor(RNG.uniform(-1, 1, (1000, 3)), dtype=DT)
    Xw = model.warp(p, X)
    assert grid.contains(Xw.numpy(), eps=0).all()
    _, D = model.split(p)
    free = X + grid.h * grid.interp(D, X)
    assert (~grid.contains(free.numpy())).sum() > 100                  # the clamp did something


def test_population_field_matches_single_field():
    grid = Grid.cube(3)
    X = torch.as_tensor(RNG.uniform(-1, 1, (150, 3)), dtype=DT)
    for kind in ("scalar", "warp"):
        model = GridModel(grid, kind)
        P = torch.stack([_random_params(model, 0.2, seed=s) for s in range(5)])
        F = model.field_batch(P, X)
        assert F.shape == (150, 5)
        for j in range(5):
            assert torch.allclose(F[:, j], model.field(P[j], X), atol=1e-13), kind


# ----------------------------------------------------------------------------- marching cubes
def test_sphere_mesh_is_closed_and_on_the_surface():
    grid = Grid.cube(6)
    G = torch.as_tensor(_sphere()(grid.vertex_positions()), dtype=DT)
    for kind, scale in (("scalar", 0.0), ("warp", 0.15)):
        model = GridModel(grid, kind)
        D = scale * torch.randn(model.nv, 3, dtype=DT, generator=torch.Generator().manual_seed(3))
        p = model.join(G, D if kind == "warp" else None)
        V, F, _ = marching_cubes(model, p, depth=2)
        chk = mesh_check(V, F, grid)
        assert chk["ok"] and chk["boundary_edges"] == 0, (kind, chk)
        assert count_pieces(F) == 1
        assert model.field(p, V).abs().max() < 1e-10, kind
    # interpolating the exact distance at the vertices: the sphere comes out a little
    # small (trilinear interpolation of |x - c| overestimates it between vertices)
    V, F, _ = marching_cubes(GridModel(grid, "scalar"), G, depth=2)
    r = np.linalg.norm(V.detach().numpy() - np.array([0.05, -0.1, 0.02]), axis=1)
    assert 0.45 < r.min() and r.max() < 0.551


def test_surface_cut_by_the_domain_has_boundary_only_on_the_faces():
    grid = Grid.cube(4)
    G = torch.as_tensor(_sphere(c=(0.9, 0.0, 0.0), r=0.5)(grid.vertex_positions()), dtype=DT)
    V, F, _ = marching_cubes(GridModel(grid, "scalar"), G, depth=1)
    chk = mesh_check(V, F, grid)
    assert chk["ok"] and chk["boundary_edges"] > 0 and chk["boundary_edges_off_domain_faces"] == 0


def test_vertex_derivatives_match_finite_differences():
    grid = Grid.cube(4)
    G = torch.as_tensor(_sphere()(grid.vertex_positions()), dtype=DT)
    for kind in ("scalar", "warp"):
        model = GridModel(grid, kind)
        D = 0.1 * torch.randn(model.nv, 3, dtype=DT, generator=torch.Generator().manual_seed(4))
        p = model.join(G, D if kind == "warp" else None)
        topo = extract_topology(model, p, depth=1)
        d = torch.randn(model.n, dtype=DT, generator=torch.Generator().manual_seed(5))
        # d/de V(p + e d) at e = 0, by autograd ...
        e = torch.zeros((), dtype=DT, requires_grad=True)
        V = topo.vertices(p + e * d)
        w = torch.randn(V.shape, dtype=DT, generator=torch.Generator().manual_seed(6))
        jvp, = torch.autograd.grad((V * w).sum(), e)
        # ... and by central differences, re-extracting the mesh each time
        eps = 1e-6
        Vp, Fp, _ = marching_cubes(model, p + eps * d, depth=1)
        Vm, Fm, _ = marching_cubes(model, p - eps * d, depth=1)
        assert np.array_equal(Fp, topo.F) and np.array_equal(Fm, topo.F)
        fd = float(((Vp - Vm) * w).sum()) / (2 * eps)
        assert abs(float(jvp) - fd) < 1e-6 * max(1.0, abs(fd)), (kind, float(jvp), fd)


# ----------------------------------------------------------------------------- targets
def test_examples_load():
    ex = T.list_examples()
    names = set(ex["functions"])
    for shape in ("sphere", "ellipsoid", "torus"):
        assert sum(n.startswith(shape) for n in names) >= 3, shape
    grid = Grid.cube(4)
    X = grid.vertex_positions()
    for name in ex["functions"]:
        fn, title = T.load_function(T.resolve_function(name))
        g = fn(X)
        assert g.shape == (len(X),) and (g < 0).any() and (g > 0).any(), name
        assert isinstance(title, str) and title
    assert len(ex["point_clouds"]) >= len(ex["functions"])
    t = T.load_target(grid, points="torus_medium_no_normals", n=500)
    assert t.normals_estimated and len(t.points) == 500


def test_point_cloud_files_and_normal_estimation():
    fn = _sphere()
    P, N = T.function_samples(fn, Grid.cube(1), 3000, 0)[:2]
    with tempfile.TemporaryDirectory() as d:
        with_n = os.path.join(d, "a.xyz")
        np.savetxt(with_n, np.concatenate([P, N], 1), header="x y z nx ny nz")
        without = os.path.join(d, "b.csv")
        np.savetxt(without, P, delimiter=",")
        npy = os.path.join(d, "c.npy")
        np.save(npy, P)
        P1, N1 = T.load_point_cloud(with_n)
        assert np.allclose(P1, P) and np.allclose(N1, N)
        for path in (without, npy):
            P2, N2 = T.load_point_cloud(path)
            assert np.allclose(P2, P) and N2 is None
        props = "".join(f"property float {c}\n" for c in ("x", "y", "z", "nx", "ny", "nz"))
        head = f"ply\nformat {{}} 1.0\nelement vertex {len(P)}\n{props}element face 0\n" \
               "property list uchar int vertex_indices\nend_header\n"
        A = np.concatenate([P, N], 1)
        ply_bin, ply_txt, obj = (os.path.join(d, n) for n in ("d.ply", "e.ply", "f.obj"))
        with open(ply_bin, "wb") as fh:
            fh.write(head.format("binary_little_endian").encode())
            fh.write(A.astype("<f4").tobytes())
        with open(ply_txt, "w") as fh:
            fh.write(head.format("ascii"))
            np.savetxt(fh, A)
        with open(obj, "w") as fh:
            fh.writelines(f"v {x} {y} {z}\n" for x, y, z in P)
            fh.writelines(f"vn {x} {y} {z}\n" for x, y, z in N)
        for path in (ply_bin, ply_txt, obj):
            P3, N3 = T.load_point_cloud(path)
            assert np.allclose(P3, P, atol=1e-6) and np.allclose(N3, N, atol=1e-6), path
    Ne = T.estimate_normals(P)
    cos = (Ne * N).sum(1)
    assert (cos > 0).all() and np.median(cos) > 0.999          # oriented outward, no flips


def test_labels_and_distances():
    grid = Grid.cube(6)
    fn = _sphere()
    X = torch.as_tensor(RNG.uniform(-1, 1, (3000, 3)), dtype=DT)
    truth = np.where(fn(X.numpy()) > 0, 1.0, -1.0)
    tf = T.target_from_function(fn, "sphere", grid, n=1000)
    assert np.array_equal(tf.label(X).numpy(), truth)
    assert np.allclose(tf.distance(X), np.abs(fn(X.numpy())), atol=1e-6)
    P, N = T.function_samples(fn, grid, 5000, 1)[:2]
    tp = T.target_from_points(P, None, "sphere cloud", grid, n=1000)
    assert (tp.label(X).numpy() == truth).mean() > 0.99
    near = np.abs(fn(X.numpy())) < 0.1
    e = np.abs(tp.distance(X)[near] - np.abs(fn(X.numpy()))[near])
    assert np.median(e) < 0.003 and e.max() < 3 * tp.spacing()   # gaps between random samples


# ----------------------------------------------------------------------------- optimizers
def _quick_cfg(kind):
    return GO.GridFitConfig(model=kind, sampson_iters=30, iters=60, samples=1500)


def test_chamfer_fit_of_a_sphere():
    grid = Grid.cube(6)
    target = T.target_from_function(_sphere(), "sphere", grid, n=2000)
    for kind, tol in (("scalar", 0.05), ("warp", 0.03)):
        model = GridModel(grid, kind)
        p, info = GO.fit(model, target, _quick_cfg(kind), verbose=False)
        err = GO.evaluate(model, p, target, depth=2, n=5000)
        assert err["surface_pieces"] == 1 and err["watertight"], (kind, err)
        assert err["rms_in_cells"] < tol, (kind, err["rms_in_cells"])


def test_chamfer_fit_of_a_point_cloud():
    grid = Grid.cube(6)
    P, _ = T.function_samples(_sphere(), grid, 4000, 2)[:2]
    target = T.target_from_points(P, None, "sphere cloud", grid, n=2000)
    model = GridModel(grid, "scalar")
    p, _ = GO.fit(model, target, _quick_cfg("scalar"), verbose=False)
    err = GO.evaluate(model, p, target, depth=2, n=4000)
    assert err["surface_pieces"] == 1 and err["rms_in_cells"] < 0.08, err


def test_logistic_classification_on_the_grid():
    grid = Grid.cube(5)
    target = T.target_from_function(_sphere(), "sphere", grid, n=2000)
    for kind in ("scalar", "warp"):
        model = GridModel(grid, kind)
        ccfg = C.ClassifyConfig(method="logistic", iters=150, seed=0)
        p, info = GO.fit_classify(model, target, ccfg, scalar_iters=100, verbose=False)
        assert p.shape == (model.n,)
        err = GO.evaluate(model, p, target, depth=2, n=4000)
        assert err["surface_pieces"] == 1 and err["rms_in_cells"] < 0.3, (kind, err)
        assert info["val_reward"] > 0.9, (kind, info["val_reward"])


def test_shell_points_are_off_the_target_and_labelled():
    grid = Grid.cube(8)
    fn = _sphere()
    target = T.target_from_function(fn, "sphere", grid, n=2000)
    X, y, d = GO.shell_points(target, grid, n=2000)
    g = fn(X.numpy())
    assert np.array_equal(y.numpy(), np.where(g > 0, 1.0, -1.0))
    assert np.allclose(d.numpy(), np.abs(g) / grid.h, atol=1e-6)
    assert d.min() > 0.2 and d.max() < 1.6 and 0.4 < (y > 0).double().mean() < 0.6


def test_warp_sampson_has_no_bubble():
    # ellipsoid_flat cloud, 8^3, seed 5: without the sign term the warp Sampson fit had a
    # 4-face bubble 1.2 cells off the surface and was rejected (RMS stayed 2.7x higher)
    grid = Grid.cube(8)
    P, N = T.load_point_cloud(T.resolve_points("ellipsoid_flat"))
    target = T.target_from_points(P, N, "ellipsoid_flat", grid, n=3000, seed=5)
    model = GridModel(grid, "warp")
    p, info = GO.fit(model, target, GO.GridFitConfig(model="warp", seed=5, iters=0), verbose=False)
    stages = dict((s[0], s[1]) for s in info["stages"])
    assert stages["sampson"] < 0.5 * stages["closed form"], stages     # the warp stage was kept
    for depth in (2, 3):
        assert count_pieces(extract_topology(model, p, depth).F) == 1, depth


def test_grpo_stays_finite_with_many_numbers():
    # GRPO on a warp 16^3 grid (19652 numbers): the log-ratio of a sample is a sum over
    # all the numbers, and after one Adam epoch it can pass 709, where exp() overflows
    # and the step turned into NaN. The ratio is capped (dual clip), so it stays finite.
    grid = Grid.cube(16)
    target = T.target_from_function(_sphere(), "sphere", grid, n=2000)
    model = GridModel(grid, "warp")
    cfg = C.ClassifyConfig(method="grpo", grpo_lr=0.02, offset_penalty=1e-3)
    gen = torch.Generator().manual_seed(0)
    G0 = target.label(torch.as_tensor(grid.vertex_positions(), dtype=DT))
    mu = model.normalised(model.join(G0)).requires_grad_(True)
    policy = C.GRPOPolicy(mu, cfg)
    sampler = GO.GridSampler(target, grid)
    for it in range(3):
        X, y = sampler.draw(2000, gen)
        policy.step(model, X, y, gen, cfg.grpo_lr)
        assert torch.isfinite(mu).all() and torch.isfinite(policy.log_std).all(), it


def test_keep_rule_rejects_extra_pieces():
    assert GO.improves((0.01, 1), (0.02, 1))
    assert not GO.improves((0.01, 2), (0.02, 1))          # lower RMS, but a detached piece
    assert GO.improves((0.01, 1), (0.02, 3))
    assert not GO.improves((0.03, 1), (0.02, 1))


def test_save_and_load_a_fit():
    grid = Grid.cube(4)
    target = T.load_target(grid, function="sphere_medium", n=1000)
    model = GridModel(grid, "warp")
    cfg = GO.GridFitConfig(model="warp", sampson_iters=10, iters=0)
    p, info = GO.fit(model, target, cfg, verbose=False)
    err = GO.evaluate(model, p, target, depth=2, n=2000)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fit.json")
        GO.save_fit(path, model, p.numpy(), target, cfg, err, info)
        back = GO.load_fit(path)
    assert back["model"].kind == "warp" and back["grid"] == grid
    assert torch.allclose(back["params"], p)
    assert back["target"] is not None and back["target"].name == target.name
    assert back["data"]["final_error"]["rms"] == err["rms"]


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t0:.1f}s)")
