#!/usr/bin/env python3
"""
fit_examples.py -- fit every example target with the grid optimizer and print a table.

Each fit is saved as data/fits/grid/<example>[_cloud]__<model>__<method>__<cells>.json, so the grid
viewer lists them all. With the defaults (9 example functions, scalar and warp,
8 cells per axis, chamfer) it takes about 10 minutes.

A point cloud fit is measured against the cloud (to the nearest point, which has a
floor), and also against the example function the cloud was sampled from ("exact":
sphere_medium_noisy and torus_medium_no_normals against sphere_medium and torus_medium).

Usage (from the project folder):
    python code/grid/fit_examples.py                          # example functions, scalar + warp
    python code/grid/fit_examples.py --points                 # example point clouds instead
    python code/grid/fit_examples.py --dims 12 --models warp
    python code/grid/fit_examples.py --examples sphere_small torus_small --dims 16
    python code/grid/fit_examples.py --method logistic
    python code/visualization/grid_viewer.py                  # then look at them
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE = str(_Path(__file__).resolve().parents[1])          # .../code: makes `import core...` work
if _CODE not in _sys.path:                                  # when this file is run as a script
    _sys.path.insert(0, _CODE)

import argparse
import time

import optimization.classify_optimize as C
import grid.grid_optimize as GO
from core.paths import GRID_FITS, output_path
from grid import targets as T
from grid.grid import Grid, GridModel


def source_function(cloud_name: str):
    """The example function an example cloud was sampled from (see make_examples.py), or None."""
    base = cloud_name.replace("_noisy", "").replace("_no_normals", "")
    return base if base in T.list_examples()["functions"] else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", action="store_true", help="the example point clouds instead of the functions")
    ap.add_argument("--examples", nargs="+", help="only these examples (names)")
    ap.add_argument("--models", nargs="+", default=["scalar", "warp"], choices=["scalar", "warp"])
    ap.add_argument("--method", default="chamfer", choices=["chamfer", "logistic", "es", "grpo"])
    ap.add_argument("--dims", type=int, default=8, help="cells per axis")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inside", default="rays", choices=["rays", "flood"], help="point clouds: inside/outside test")
    args = ap.parse_args()

    ex = T.list_examples()
    names = ex["point_clouds"] if args.points else ex["functions"]
    if args.examples:
        names = [n for n in names if n in args.examples or _Path(n).stem in args.examples]
    rows = []
    t_all = time.time()
    for name in names:
        for kind in args.models:
            if args.points:
                path = T.resolve_points(name)
                P = T.load_point_cloud(path)
                grid = Grid.cube(args.dims) if (P.min() >= -1 and P.max() <= 1) else \
                    T.default_grid((args.dims,) * 3, target_bounds=(P.min(0), P.max(0)))
                target = T.target_from_points(P, path.stem, grid, seed=args.seed, source=str(path), inside=args.inside)
            else:
                grid = Grid.cube(args.dims)
                target = T.load_target(grid, function=name, seed=args.seed)
            model = GridModel(grid, kind)
            cfg = GO.GridFitConfig(model=kind, method=args.method, depth=args.depth, inside=args.inside, seed=args.seed)
            ccfg = None
            if args.method != "chamfer":
                ccfg = C.ClassifyConfig(method=args.method, seed=args.seed, offset_penalty=cfg.disp_penalty,
                                        lr=0.01 if args.method == "es" else 0.05, grpo_lr=0.01)
            print(f"\n=== {target.name}, {model.describe()}, {args.method}")
            t0 = time.time()
            p, info = GO.fit(model, target, cfg, ccfg, verbose=False)
            err = GO.evaluate(model, p, target, depth=max(args.depth, 3))
            err["time"] = time.time() - t0
            source = source_function(target.name) if args.points else None
            if source and not err.get("empty"):
                ex = GO.evaluate(model, p, T.load_target(grid, function=source), depth=max(args.depth, 3))
                err["exact_function"], err["rms_exact"] = source, ex["rms"]
                err["rms_exact_in_cells"] = ex["rms_in_cells"]
            out = output_path(GO.fit_name(target, model, args.method), GRID_FITS)
            GO.save_fit(out, model, p, target, cfg, err, info, ccfg)
            rows.append((target.name, kind, err))
            if err.get("empty"):
                print("    empty surface")
            else:
                exact = f", exact {err['rms_exact']:.3e} ({err['rms_exact_in_cells']:.3f} cells)" if "rms_exact" in err else ""
                print(f"    RMS {err['rms']:.3e} ({err['rms_in_cells']:.3f} cells){exact}, pieces {err['surface_pieces']}, "
                      f"{err['time']:.0f}s -> {_Path(out).name}")

    print(f"\n{len(rows)} fits in {time.time() - t_all:.0f}s, {args.dims} cells per axis, method {args.method}\n")
    print(f"| {'example':24s} | model  | RMS       | RMS / cell | exact RMS | pieces | time |")
    print(f"|{'-' * 26}|--------|-----------|------------|-----------|--------|------|")
    for name, kind, e in rows:
        if e.get("empty"):
            print(f"| {name:24s} | {kind:6s} | empty     |            |           |        |      |")
            continue
        tp = e.get("target_pieces")
        pcs = f"{e['surface_pieces']}" + (f"/{tp}" if tp else "")
        exact = f"{e['rms_exact']:.3e}" if "rms_exact" in e else ""
        print(f"| {name:24s} | {kind:6s} | {e['rms']:.3e} | {e['rms_in_cells']:10.3f} | {exact:9s} | {pcs:6s} | {e['time']:3.0f}s |")
    print("\nview them: python code/visualization/grid_viewer.py")


if __name__ == "__main__":
    main()
