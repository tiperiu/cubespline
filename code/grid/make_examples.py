#!/usr/bin/env python3
"""
make_examples.py -- write the example point clouds (examples/point_clouds/*.xyz) from
the example functions (examples/functions/*.py).

Each cloud has --points area-uniform points on the surface with unit outward normals,
one per line:  x y z nx ny nz. The fits read only the positions; the normals are kept in
the files for other tools. Two extra variants:
    torus_medium_no_normals.xyz   x y z only (the same points as torus_medium.xyz, so it
                                  now fits exactly like it)
    sphere_medium_noisy.xyz       points moved by Gaussian noise (std 0.005), no normals

Usage (from the project folder):
    python code/grid/make_examples.py
    python code/grid/make_examples.py --points 10000
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE = str(_Path(__file__).resolve().parents[1])          # .../code: makes `import core...` work
if _CODE not in _sys.path:                                  # when this file is run as a script
    _sys.path.insert(0, _CODE)

import argparse

import numpy as np

from grid import targets as T
from grid.grid import Grid


def write_xyz(path, P, N=None):
    A = P if N is None else np.concatenate([P, N], 1)
    header = "x y z" + ("" if N is None else " nx ny nz")
    np.savetxt(path, A, fmt="%.6f", header=f"{header}  ({len(P)} points)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=5000, help="points per cloud")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    T.POINTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = Grid.cube(1)                                    # the example domain [-1, 1]^3
    for name in T.list_examples()["functions"]:
        fn, _ = T.load_function(T.resolve_function(name))
        P, N = T.function_samples(fn, grid, args.points, args.seed)[:2]
        path = T.POINTS_DIR / f"{name}.xyz"
        write_xyz(path, P, N)
        print(f"wrote {path.relative_to(T.EXAMPLES.parent)}  ({len(P)} points)")
        if name == "torus_medium":
            path = T.POINTS_DIR / f"{name}_no_normals.xyz"
            write_xyz(path, P)
            print(f"wrote {path.relative_to(T.EXAMPLES.parent)}  (no normals)")
        if name == "sphere_medium":
            rng = np.random.default_rng(args.seed + 1)
            path = T.POINTS_DIR / f"{name}_noisy.xyz"
            write_xyz(path, P + rng.normal(scale=0.005, size=P.shape))
            print(f"wrote {path.relative_to(T.EXAMPLES.parent)}  (noise 0.005, no normals)")


if __name__ == "__main__":
    main()
