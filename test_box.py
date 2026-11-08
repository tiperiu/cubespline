"""
Tests for cubes at arbitrary positions and sizes (box.py) and the grid preparation.
Run:  python -m pytest test_box.py -q      (or  python test_box.py)
"""

import json
import os
import tempfile

import numpy as np
import torch

import optimize as O
import warp_marching as WM
import warp_optimize as WO
from box import Box, corner_vertex_ids
from cube import Cube
from marching import extract_topology
from warpcube import WarpCube

DT = torch.float64
RNG = np.random.default_rng(0)
MOVED = Box((2.5, -4.0, 0.75), 0.4)


def test_lattice():
    b = MOVED
    P = b.lattice(10)
    assert P.shape == (1000, 3)
    U = b.to_local(P)
    assert np.all((U > 0) & (U < 1))                                  # cell centres: strictly inside
    assert np.allclose(np.unique(U[:, 0]), (np.arange(10) + 0.5) / 10)
    Q = b.lattice(4, faces=True)
    assert Q.shape == (64, 3) and b.contains(Q).all()
    for c in b.corners():                                             # corners are on the lattice
        assert np.isclose(np.linalg.norm(Q - c, axis=1).min(), 0)


def test_files_without_a_box_are_the_default_cube():
    assert Box.from_dict({"values": [0] * 8}) == Box()
    assert Box.from_dict(MOVED.to_dict()) == MOVED


def test_same_values_same_shape_in_any_box():
    """f in a moved, scaled cube at the image of a local point equals f in the default
    cube at that local point -- for 8 values and for 16 (offsets are in half-widths)."""
    D = Box()
    U = torch.rand(300, 3, dtype=DT)
    v8 = RNG.normal(size=8)
    v16 = np.r_[0.4 * RNG.normal(size=8), RNG.normal(size=8)]
    with torch.no_grad():
        assert torch.allclose(Cube(v8, **MOVED.kw).evaluate(MOVED.to_world(U)),
                              Cube(v8, **D.kw).evaluate(D.to_world(U)), atol=1e-12)
        w_m, w_d = WarpCube(v16, **MOVED.kw), WarpCube(v16, **D.kw)
        assert torch.allclose(w_m.evaluate(MOVED.to_world(U)), w_d.evaluate(D.to_world(U)), atol=1e-12)
        assert torch.allclose(MOVED.to_local(w_m.moved_corners()), D.to_local(w_d.moved_corners()), atol=1e-12)
        # the meshes are the same, moved and scaled
        for cube_m, cube_d, mc in ((Cube(v8, **MOVED.kw), Cube(v8), extract_topology),
                                   (w_m, w_d, WM.extract_topology)):
            tm, td = mc(cube_m, 4), mc(cube_d, 4)
            assert np.array_equal(tm.F, td.F)
            assert torch.allclose(MOVED.to_local(tm.vertices(cube_m.values)),
                                  D.to_local(td.vertices(cube_d.values)), atol=1e-9)


def test_target_and_errors_follow_the_box():
    """A target moved and scaled with the cube gives the same samples (mapped) and
    the same errors, scaled by the size ratio -- for both evaluators."""
    D = Box()
    fn = O.FUNCTIONS["sphere cap"]

    def fn_moved(X):                       # the same surface, in the moved cube's frame
        return fn(D.to_world(MOVED.to_local(np.asarray(X))))
    t_d = O.target_from_function(fn, "cap", box=D, n=1500)
    t_m = O.target_from_function(fn_moved, "cap moved", box=MOVED, n=1500)
    assert t_m.box == MOVED and MOVED.contains(t_m.points.numpy()).all()
    assert np.allclose(MOVED.to_local(t_m.points.numpy()), D.to_local(t_d.points.numpy()), atol=1e-6)
    ratio = MOVED.size / D.size
    v8, _ = O.fit_algebraic(None, t_d)
    e_d, e_m = O.evaluate(v8.numpy(), t_d, depth=4), O.evaluate(v8.numpy(), t_m, depth=4)
    assert np.isclose(e_m["rms"], ratio * e_d["rms"], rtol=1e-4), (e_m["rms"], ratio * e_d["rms"])
    v16 = np.r_[0.3 * RNG.normal(size=8), v8.numpy()]
    w_d, w_m = WO.evaluate(v16, t_d, depth=4), WO.evaluate(v16, t_m, depth=4)
    assert np.isclose(w_m["rms"], ratio * w_d["rms"], rtol=1e-4)


def test_fit_in_a_moved_box_and_save():
    t = O.builtin_target("sphere cap", n=1500, box=Box((0, -1, 0), 1.0))
    f = O.Fitter(t, O.FitConfig(depth=4, iters=30, mesh_samples=1500))
    f.run(verbose=False)
    V, F = f.mesh()
    assert len(F) and t.box.contains(V).all()
    assert f.score() < 0.01 * t.box.size
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fit.json")
        O.save_fit(path, f)
        with open(path) as fh:
            assert Box.from_dict(json.load(fh)) == t.box


def test_grid_corner_ids_are_shared_and_8_value_cells_join_continuously():
    """Preparation for grids: neighbouring cells get the same vertex ids on their
    common corners, those ids are the same world points, and 8-value cells that
    take their values from one array per grid vertex agree on the common face."""
    dims = (2, 2, 1)
    origin, h = (0.0, 0.0, 0.0), 0.5
    nv = 3 * 3 * 2
    g = RNG.normal(size=nv)
    world = {}
    for ijk in np.ndindex(*dims):
        ids = corner_vertex_ids(ijk, dims)
        for vid, p in zip(ids, Box.grid_cell(origin, h, ijk).corners()):
            world.setdefault(int(vid), p)
            assert np.allclose(world[int(vid)], p)                    # one position per id
    assert len(world) == nv
    face = torch.as_tensor(np.c_[np.full(200, 0.5), RNG.random((200, 2)) * 0.5])   # x = 0.5, cells (0,0,0)|(1,0,0)
    a, b = (0, 0, 0), (1, 0, 0)
    fa = Cube(g[corner_vertex_ids(a, dims)], requires_grad=False, **Box.grid_cell(origin, h, a).kw).evaluate(face)
    fb = Cube(g[corner_vertex_ids(b, dims)], requires_grad=False, **Box.grid_cell(origin, h, b).kw).evaluate(face)
    assert torch.allclose(fa, fb, atol=1e-12)


def test_topology_keeps_lattice_edges():
    """The 8-value topology stores each vertex's lattice edge (for welding grid cells)."""
    cube = Cube(RNG.normal(size=8), requires_grad=False, **MOVED.kw)
    topo = extract_topology(cube, 3)
    R = topo.resolution
    assert topo.a_ijk.shape == (topo.num_vertices, 3)
    assert torch.allclose(topo.pa, MOVED.to_world(torch.as_tensor(topo.a_ijk, dtype=DT) / R))
    assert np.all(np.abs(topo.b_ijk - topo.a_ijk).sum(1) == 1)        # unit lattice edges


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t0:.1f}s)")
