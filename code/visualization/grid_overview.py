#!/usr/bin/env python3
"""
grid_overview.py -- images of the grid fits: a row per target, the target and then
the fitted surface of each fit of it (matplotlib, no window needed).

Writes data/fits/grid/overview_functions.png and overview_point_clouds.png (one
image per kind of target).

Usage (from the project folder):
    python code/visualization/grid_overview.py                       # every fit in data/fits/grid
    python code/visualization/grid_overview.py --method chamfer --kind function
    python code/visualization/grid_overview.py --fits a.json b.json --out-dir .
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_CODE = str(_Path(__file__).resolve().parents[1])          # .../code: makes `import core...` work
if _CODE not in _sys.path:                                  # when this file is run as a script
    _sys.path.insert(0, _CODE)

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import grid.grid_optimize as GO
from core.paths import GRID_FITS
from grid.grid_marching import marching_cubes

LIGHT = np.array([0.4, 0.5, 0.75]) / np.linalg.norm([0.4, 0.5, 0.75])


def draw_mesh(ax, V, F, color, alpha=1.0):
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    shade = 0.35 + 0.65 * np.abs(n @ LIGHT)
    fc = np.clip(np.asarray(color)[None] * shade[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(T, facecolors=np.c_[fc, np.full(len(fc), alpha)], linewidths=0))


def draw_points(ax, P, color, n=3000):
    if len(P) > n:
        P = P[np.random.default_rng(0).choice(len(P), n, replace=False)]
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.5, c=[color], depthshade=False)


def frame(ax, box, title):
    """Same view for every panel of a row: the box (lo, hi) the target fills."""
    lo, hi = box
    c, r = (lo + hi) / 2, 0.6 * (hi - lo).max()
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def render(rows: dict, models: list, depth: int, out: str):
    """One figure: a row per target (the target, then one panel per model column)."""
    fig = plt.figure(figsize=(2.6 * (1 + len(models)), 2.5 * len(rows)))
    for r, (key, cols) in enumerate(sorted(rows.items())):
        first = GO.load_fit(next(iter(cols.values())))
        target = first["target"]
        box = (target.cloud.min(0), target.cloud.max(0)) if target is not None else \
            (first["grid"].lo_np, first["grid"].hi_np)
        for c, col in enumerate(models):
            ax = fig.add_subplot(len(rows), 1 + len(models), r * (1 + len(models)) + c + 2, projection="3d")
            if col not in cols:
                ax.set_axis_off()
                continue
            fit = GO.load_fit(cols[col])
            V, F, _ = marching_cubes(fit["model"], fit["params"], depth)
            if len(F):
                draw_mesh(ax, V.detach().numpy(), F, (0.35, 0.55, 0.9))
            err = fit["data"].get("final_error", {})
            sub = "empty" if err.get("empty") or not len(F) else \
                f"RMS {err['rms']:.1e}, {err['surface_pieces']} piece{'s' * (err['surface_pieces'] != 1)}"
            frame(ax, box, f"{col}\n{sub}")
        ax = fig.add_subplot(len(rows), 1 + len(models), r * (1 + len(models)) + 1, projection="3d")
        ax.set_axis_off()
        if target is not None:
            if target.V is not None and len(target.F):
                draw_mesh(ax, target.V, target.F, (0.95, 0.6, 0.25))
            else:
                draw_points(ax, target.cloud, (0.9, 0.5, 0.15))
            frame(ax, box, f"target: {key[0]}\n({'function' if key[1] == 'function' else 'point cloud'})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} targets x {len(models)} fits each)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", nargs="+", help="fit files (default: every grid fit in data/fits/grid)")
    ap.add_argument("--method", help="only fits made with this method")
    ap.add_argument("--kind", choices=["function", "points"], help="only function or only point-cloud targets")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out-dir", default=str(GRID_FITS),
                    help="writes overview_functions.png and overview_point_clouds.png there")
    args = ap.parse_args()

    paths = args.fits or sorted(glob.glob(str(GRID_FITS / "*.json")))
    by_kind = {}
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        if data.get("type") != "grid" or (args.method and data.get("method") != args.method):
            continue
        kind = data["target"]["kind"]
        if args.kind and kind != args.kind:
            continue
        rows, models = by_kind.setdefault(kind, ({}, []))
        nx, ny, nz = data["grid"]["dims"]
        col = f"{data['model']} {data['method']} {nx}x{ny}x{nz}"
        rows.setdefault((data["target"]["name"], kind), {})[col] = path
        if col not in models:
            models.append(col)
    if not by_kind:
        raise SystemExit("no grid fits found")
    for kind, (rows, models) in sorted(by_kind.items()):
        models.sort(key=lambda c: (c.split()[1] != "chamfer", int(c.split()[2].split("x")[0]),
                                   c.split()[0] != "scalar", c))
        name = "overview_functions.png" if kind == "function" else "overview_point_clouds.png"
        render(rows, models, args.depth, os.path.join(args.out_dir, name))


if __name__ == "__main__":
    main()
