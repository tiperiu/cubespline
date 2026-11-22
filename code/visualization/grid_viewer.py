#!/usr/bin/env python3
"""
grid_viewer.py -- Polyscope viewer for grids of cubes (code/grid).

Shows, for the entry selected in the combo box:
    the target        an example function (orange surface) or point cloud (orange points)
    the grid          the domain box and, optionally, every cell edge
    the fitted mesh   (for fits) marching cubes of the fitted field, at the chosen depth
    displacements     (warp fits) arrows h * D at the grid vertices

The combo box lists the example targets (examples/functions, examples/point_clouds)
and the grid fits in data/fits/grid. "Fit" fits the selected example with the
chamfer method (model and cells per axis from the controls; it takes 20-60 s, the
window waits) and adds the result to the list.

Usage (from the project folder):
    python code/visualization/grid_viewer.py
    python code/visualization/grid_viewer.py --load sphere_medium__warp__chamfer__8.json
    python code/visualization/grid_viewer.py --function torus_medium --dims 6
    python code/visualization/grid_viewer.py --points torus_medium_no_normals
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
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

import grid.grid_optimize as GO
from core.paths import GRID_FITS, find_input
from grid import targets as T
from grid.grid import DT, Grid, GridModel
from grid.grid_marching import marching_cubes
from visualization.viewer import (BACK_COLOR, BASE_COLOR, CUBE_COLOR, TARGET_COLOR, choose_file, clean_path,
                                  clipboard_text)

LATTICE_COLOR = (0.55, 0.55, 0.55)
ARROW_COLOR = (0.20, 0.45, 0.85)


class S:
    entries = []          # {"name", "kind": "function" | "points" | "fit", "path"}
    idx = 0
    dims = 8
    model_kind = 0        # 0 scalar, 1 warp (for Fit)
    depth = 2
    level = 0.0
    show_target = True
    show_lattice = False
    show_disp = True
    show_mesh = True
    show_backface = True
    grid: Grid | None = None
    model: GridModel | None = None
    p: torch.Tensor | None = None
    target: T.GridTarget | None = None
    err: dict | None = None
    fit_data: dict | None = None
    fit_path = ""
    message = ""


# ----------------------------------------------------------------------------
# Entries
# ----------------------------------------------------------------------------
def scan():
    """Examples first, then every grid fit in data/fits/grid (and the current folder)."""
    known = {e["path"] for e in S.entries}
    ex = T.list_examples()
    for n in ex["functions"]:
        p = str(T.FUNCTION_DIR / f"{n}.py")
        if p not in known:
            S.entries.append({"name": f"function: {n}", "kind": "function", "path": p})
    for n in ex["point_clouds"]:
        p = str(T.POINTS_DIR / n)
        if p not in known:
            S.entries.append({"name": f"points: {n}", "kind": "points", "path": p})
    for folder in dict.fromkeys([str(GRID_FITS), os.getcwd()]):
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            path = os.path.abspath(path)
            if path in known:
                continue
            try:
                with open(path) as fh:
                    if json.load(fh).get("type") != "grid":
                        continue
            except Exception:  # noqa: BLE001
                continue
            S.entries.append({"name": f"fit: {os.path.basename(path)}", "kind": "fit", "path": path})
            known.add(path)


def example_grid(kind, path) -> Grid:
    dims = (S.dims,) * 3
    if kind == "points":
        P, _ = T.load_point_cloud(path)
        if np.all(P >= -1) and np.all(P <= 1):
            return Grid.cube(dims)
        return T.default_grid(dims, target_bounds=(P.min(0), P.max(0)))
    return Grid.cube(dims)


def select(i):
    S.idx = i
    e = S.entries[i]
    S.model = S.p = S.err = S.fit_data = None
    t0 = time.time()
    try:
        if e["kind"] == "fit":
            fit = GO.load_fit(e["path"])
            S.model, S.p, S.grid, S.target = fit["model"], fit["params"], fit["grid"], fit["target"]
            S.fit_data = fit["data"]
            S.err = S.fit_data.get("final_error")
            S.depth = int(S.fit_data.get("depth", 2))
            S.dims = int(max(S.grid.dims))
            S.model_kind = 0 if S.model.kind == "scalar" else 1
            S.fit_path = e["path"]
        else:
            S.grid = example_grid(e["kind"], e["path"])
            S.target = T.load_target(S.grid, **{("function" if e["kind"] == "function" else "points"): e["path"]})
        S.message = f"{e['name']}  ({time.time() - t0:.1f}s)"
    except Exception as ex:  # noqa: BLE001
        S.message = f"Could not load {e['name']}: {ex}"
    draw_all()


def load_path(path) -> bool:
    path = find_input(clean_path(path))
    if not os.path.isfile(path):
        cand = os.path.join(str(GRID_FITS), os.path.basename(path))
        path = cand if os.path.isfile(cand) else path
    path = os.path.abspath(path)
    for i, e in enumerate(S.entries):
        if e["path"] == path:
            select(i)
            return True
    try:
        with open(path) as fh:
            ok = json.load(fh).get("type") == "grid"
    except Exception as ex:  # noqa: BLE001
        S.message = f"Could not read {path}: {ex}"
        return False
    if not ok:
        S.message = f"{path} is not a grid fit"
        return False
    S.entries.append({"name": f"fit: {os.path.basename(path)}", "kind": "fit", "path": path})
    select(len(S.entries) - 1)
    return True


def fit_selected():
    """Chamfer fit of the selected example (or of a fit's target) with the current controls."""
    if S.target is None:
        S.message = "Nothing to fit: no target"
        return
    kind = ("scalar", "warp")[S.model_kind]
    grid = Grid.cube((S.dims,) * 3) if S.grid is None else Grid.covering(S.grid.lo_np, S.grid.hi_np, (S.dims,) * 3)
    target = S.target
    e = S.entries[S.idx]
    if e["kind"] != "fit":                                     # the example on the new grid
        target = T.load_target(grid, **{("function" if e["kind"] == "function" else "points"): e["path"]})
    model = GridModel(grid, kind)
    cfg = GO.GridFitConfig(model=kind, depth=S.depth)
    print(f"\nfitting {target.describe()} with a {model.describe()}")
    t0 = time.time()
    p, info = GO.fit(model, target, cfg)
    err = GO.evaluate(model, p, target, depth=max(cfg.depth, 3))
    err["time"] = time.time() - t0
    GO.print_report(err)
    out = os.path.join(str(GRID_FITS), GO.fit_name(target, model, "chamfer"))
    GO.save_fit(out, model, p, target, cfg, err, info)
    print(f"saved {out}")
    load_path(out)
    S.message = f"Fitted in {err['time']:.0f}s: RMS {err['rms']:.3e}; saved {os.path.basename(out)}"


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_all():
    ps.remove_all_structures()
    if S.grid is not None:
        C, E = S.grid.box_edges()
        ps.register_curve_network("domain", C, E, color=CUBE_COLOR).set_radius(0.003, relative=True)
        if S.show_lattice:
            P, E = S.grid.edges()
            ps.register_curve_network("cells", P, E, color=LATTICE_COLOR).set_radius(0.0008, relative=True)
    if S.target is not None and S.show_target:
        t = S.target
        if t.kind == "function" and t.F is not None and len(t.F):
            m = ps.register_surface_mesh("target", t.V, t.F, color=TARGET_COLOR, transparency=0.35)
            m.set_back_face_policy("identical")
        else:
            pc = ps.register_point_cloud("target points", t.cloud, color=TARGET_COLOR)
            pc.set_radius(0.0025, relative=True)
    if S.model is not None and S.p is not None:
        if S.show_mesh:
            with torch.no_grad():
                V, F, topo = marching_cubes(S.model, S.p, depth=S.depth, level=S.level)
            if len(F):
                m = ps.register_surface_mesh("fit", V.numpy(), F, color=BASE_COLOR)
                if S.show_backface:
                    m.set_back_face_policy("custom")
                    m.set_back_face_color(BACK_COLOR)
        disp = S.model.displacement_at_vertices(S.p)
        if disp is not None and S.show_disp:
            P = S.grid.vertex_positions()
            mag = np.linalg.norm(disp, axis=1)
            keep = mag > 1e-3 * S.grid.h
            if keep.any():
                pc = ps.register_point_cloud("displaced vertices", P[keep], color=ARROW_COLOR)
                pc.set_radius(0.002, relative=True)
                pc.add_vector_quantity("displacement h*D", disp[keep], vectortype="ambient", enabled=True,
                                       color=ARROW_COLOR, radius=0.002)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def ui():
    redraw = False
    names = [e["name"] for e in S.entries]
    if names:
        changed, i = psim.Combo("example / fit", S.idx, names)
        if changed:
            select(i)
    if S.model is not None:
        psim.TextUnformatted(S.model.describe())
    elif S.grid is not None:
        psim.TextUnformatted(f"grid {S.grid.dims[0]}x{S.grid.dims[1]}x{S.grid.dims[2]} cells of {S.grid.h:g}")
    if S.target is not None:
        psim.TextUnformatted("target: " + S.target.describe())

    psim.Separator()
    _, S.fit_path = psim.InputText("fit file", S.fit_path)
    if psim.Button("Browse..."):
        p = choose_file(str(GRID_FITS) if GRID_FITS.is_dir() else ".")
        if p:
            S.fit_path = p
            load_path(p)
    psim.SameLine()
    if psim.Button("Paste"):
        p = clean_path(clipboard_text())
        if p:
            S.fit_path = p
    psim.SameLine()
    if psim.Button("Load fit"):
        load_path(S.fit_path)
    psim.SameLine()
    if psim.Button("Rescan"):
        n = len(S.entries)
        scan()
        S.message = f"Found {len(S.entries) - n} new entries"

    psim.Separator()
    c, S.dims = psim.SliderInt("cells per axis", S.dims, 1, 24)
    if c and S.entries and S.entries[S.idx]["kind"] != "fit":
        select(S.idx)
    c, S.model_kind = psim.Combo("model", S.model_kind, ["scalar (value per vertex)", "warp (value + 3D displacement)"])
    if psim.Button("Fit (chamfer)"):
        S.message = "Fitting... (see the terminal)"
        fit_selected()

    psim.Separator()
    c, S.depth = psim.SliderInt("mesh depth per cell", S.depth, 0, 4); redraw |= c
    c, S.level = psim.SliderFloat("iso level", S.level, -1.0, 1.0); redraw |= c
    if psim.Button("Level = 0"):
        S.level, redraw = 0.0, True
    c, S.show_mesh = psim.Checkbox("Fitted surface", S.show_mesh); redraw |= c
    c, S.show_backface = psim.Checkbox("Inside side in orange", S.show_backface); redraw |= c
    c, S.show_target = psim.Checkbox("Target (orange)", S.show_target); redraw |= c
    c, S.show_lattice = psim.Checkbox("Grid cells", S.show_lattice); redraw |= c
    c, S.show_disp = psim.Checkbox("Displacements (warp)", S.show_disp); redraw |= c
    if redraw:
        draw_all()

    if S.err:
        psim.Separator()
        e = S.err
        if e.get("empty"):
            psim.TextUnformatted("Fit error: the surface does not cross the grid")
        else:
            psim.TextUnformatted("Fit error (world units):")
            psim.TextUnformatted(f"  RMS          {e['rms']:.3e}   ({e['rms_in_cells']:.3f} cells)")
            psim.TextUnformatted(f"  chamfer (CD) {e['cd']:.3e}")
            psim.TextUnformatted(f"  max fit->tgt {e['fit_to_target_max']:.3e}")
            psim.TextUnformatted(f"  max tgt->fit {e['target_to_fit_max']:.3e}")
            tp = e.get("target_pieces")
            psim.TextUnformatted(f"  pieces       {e['surface_pieces']}" + (f" (target {tp})" if tp else ""))
            if "max_displacement_cells" in e:
                psim.TextUnformatted(f"  largest displacement {e['max_displacement_cells']:.2f} cells")
            if S.fit_data:
                psim.TextUnformatted(f"  method {S.fit_data.get('method')}, time {e.get('time', float('nan')):.0f} s")
    if S.message:
        psim.Separator()
        psim.TextUnformatted(S.message)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", help="grid fit file (a name is also looked up in data/fits/grid)")
    ap.add_argument("--function", help="show this example function (name or file)")
    ap.add_argument("--points", help="show this example point cloud (name or file)")
    ap.add_argument("--dims", type=int, default=8, help="cells per axis for examples")
    args = ap.parse_args()
    S.dims = args.dims

    ps.init()
    ps.set_program_name("cubespline grid")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    scan()
    if args.load:
        load_path(args.load)
    elif args.function or args.points:
        path = str(T.resolve_function(args.function) if args.function else T.resolve_points(args.points))
        for i, e in enumerate(S.entries):
            if e["path"] == path:
                select(i)
                break
        else:
            S.entries.append({"name": os.path.basename(path), "kind": "function" if args.function else "points",
                              "path": path})
            select(len(S.entries) - 1)
    elif S.entries:
        select(0)
    ps.set_user_callback(ui)
    ps.show()


if __name__ == "__main__":
    main()
