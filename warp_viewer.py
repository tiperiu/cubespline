#!/usr/bin/env python3
"""
warp_viewer.py -- interactive viewer for the 16-value WarpCube.

Same look and layout as viewer.py (Polyscope, y-up, shadow-only ground), with
16 sliders: the 8 offsets (each moves its corner along the cube diagonal) and
the 8 field values. The cube is drawn as it is and as it is warped (the image
of its edges under x -> x + D(x)), with the moved corners in blue.

Fits written by warp_optimize.py (16 values) and by optimize.py (8 values, shown
with zero offset) appear in the "preset / fit" combo box.

Usage:
    python warp_viewer.py
    python warp_viewer.py --preset "twisted plane" --depth 6
    python warp_viewer.py --load warp_fit.json
    python warp_viewer.py --preset "twisted plane" --lo 2 0 0 --size 0.5   # elsewhere, smaller
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

from box import Box, add_box_arguments, box_from_args
from cube import CUBE_EDGES, CORNER_IJK
from marching import mesh_check, save_obj, to_numpy
from viewer import (BASE_COLOR, BACK_COLOR, EDGE_COLOR, CUBE_COLOR, CELL_COLOR, TARGET_COLOR,
                    clean_path, clipboard_text, choose_file)
from warpcube import WarpCube, WARP_PRESETS
import warp_marching as WM

WARP_COLOR = (0.2, 0.5, 0.9)
MAX_CELLS_DRAWN = 40000


class State:
    entries = [{"name": k, "values": list(v), "depth": None, "target": None, "path": None, "error": None,
                "box": None} for k, v in WARP_PRESETS.items()]
    box = Box()             # where the cube is now (fits carry their own; presets use preset_box)
    preset_box = Box()      # --lo / --size
    entry_idx = 0
    values = list(entries[0]["values"])
    depth = 5
    level = 0.0
    offset_range = 1.0
    field_range = 2.0
    show_surface = True
    smooth = False
    wireframe = True
    show_backface = True
    show_cube = True
    show_warped = True
    show_corners = True
    show_cells = False
    show_target = True
    fit_path = "warp_fit.json"
    obj_path = "warp_mesh.obj"
    target_name = None
    cube = None
    topo = None
    V = None
    F = None
    stats = {}
    check = {}
    max_err = 0.0
    t_extract = 0.0
    message = ""


S = State()


# --------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------
def _entry_from_fit(path, fit):
    values = [float(v) for v in fit["values"]]
    if len(values) == 8:                       # 8-value fit from optimize.py: zero offset
        values = [0.0] * 8 + values
    target = fit.get("target")
    tag = "fit16" if fit.get("type") == "warp16" else "fit8"
    return {"name": f"{tag}: {os.path.basename(path)}" + (f" ({os.path.basename(str(target))})" if target else ""),
            "values": values, "depth": fit.get("depth"), "target": target,
            "path": os.path.abspath(path), "error": fit.get("final_error") or fit.get("metrics"),
            "box": Box.from_dict(fit)}


def _read_fit(path):
    with open(path) as fh:
        fit = json.load(fh)
    if not (isinstance(fit, dict) and len(fit.get("values", [])) in (8, 16)):
        raise ValueError("not a fit file (needs 'values' with 8 or 16 numbers)")
    return fit


def load_fit(path):
    path = clean_path(path)
    try:
        entry = _entry_from_fit(path, _read_fit(path))
    except Exception as e:  # noqa: BLE001
        S.message = f"Could not load {path}: {e}"
        print(S.message)
        return False
    for i, e in enumerate(S.entries):
        if e["path"] == entry["path"]:
            S.entries[i] = entry
            break
    else:
        S.entries.append(entry)
        i = len(S.entries) - 1
    select_entry(i)
    rms = (entry["error"] or {}).get("rms")
    S.message = f"Loaded {os.path.basename(path)}" + (f", rms {rms:.3g}" if rms else "")
    print(S.message)
    return True


def scan_fits(folder="."):
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        if any(e["path"] == os.path.abspath(path) for e in S.entries):
            continue
        try:
            S.entries.append(_entry_from_fit(path, _read_fit(path)))
        except Exception:  # noqa: BLE001
            continue


def select_entry(idx):
    S.entry_idx = idx
    e = S.entries[idx]
    S.values = list(e["values"])
    if e["depth"] is not None:
        S.depth = int(e["depth"])
    S.offset_range = max(S.offset_range, float(np.abs(S.values[:8]).max()) * 1.2)
    S.field_range = max(S.field_range, float(np.abs(S.values[8:]).max()) * 1.2)
    _apply_box(e.get("box") or S.preset_box)
    draw_target(e["target"])


def _apply_box(box):
    """Place the cube (and the target drawn with it) in `box`; reset the camera if it moved."""
    if box == S.box:
        return
    S.box = box
    S.target_name = None                        # the target is clipped to the box: rebuild it
    if ps.is_initialized():
        draw_cube_frame()
        ps.reset_camera_to_home_view()


def draw_target(target):
    if target == S.target_name:
        if ps.has_surface_mesh("target"):
            ps.get_surface_mesh("target").set_enabled(S.show_target)
        return
    if ps.has_surface_mesh("target"):
        ps.remove_surface_mesh("target")
    S.target_name = target
    if not target:
        return
    try:
        import optimize as O
        if target in O.FUNCTIONS or target == "trilinear":
            t = O.builtin_target(target, n=500, box=S.box)
        elif os.path.isfile(target):
            V, F = O.load_obj(target)
            t = O.target_from_mesh(V, F, name=target, n=500, box=S.box)
        else:
            return
    except Exception as e:  # noqa: BLE001
        print(f"  could not rebuild target '{target}': {e}")
        return
    m = ps.register_surface_mesh("target", t.V, t.F, color=TARGET_COLOR, transparency=0.45)
    m.set_back_face_policy("identical")
    m.set_enabled(S.show_target)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def rebuild():
    S.cube = WarpCube(S.values, requires_grad=False, **S.box.kw)
    t = time.time()
    S.topo = WM.extract_topology(S.cube, S.depth, S.level)
    S.V = to_numpy(S.topo.vertices(S.cube.values))
    S.F = S.topo.F
    S.t_extract = time.time() - t
    S.stats = S.topo.stats
    S.check = mesh_check(S.V, S.F, S.cube.lo, S.cube.size) if len(S.F) else {"ok": True, "boundary_edges": 0}
    with torch.no_grad():
        S.max_err = float((S.cube.evaluate(torch.as_tensor(S.V)) - S.level).abs().max()) if len(S.V) else 0.0
    draw_mesh()
    draw_corners()
    draw_warped_cube()
    draw_cells()
    apply_display_options()


def draw_cube_frame():
    P = S.box.corners()
    ps.register_curve_network("cube", P, CUBE_EDGES, color=CUBE_COLOR).set_radius(0.004, relative=True)


def draw_mesh():
    if ps.has_surface_mesh("mesh"):
        ps.remove_surface_mesh("mesh")
    if not len(S.F):
        return
    m = ps.register_surface_mesh("mesh", S.V, S.F, color=BASE_COLOR)
    m.set_edge_color(EDGE_COLOR)


def draw_corners():
    P = to_numpy(S.cube.corner_positions())
    Pm = to_numpy(S.cube.moved_corners())
    pc = ps.register_point_cloud("corners", P, radius=0.012)
    pc.set_point_render_mode("sphere")
    v = np.asarray(S.values[8:], float) - S.level
    m = max(np.abs(v).max(), 1e-9)
    pc.add_scalar_quantity("field - level", v, cmap="coolwarm", vminmax=(-m, m), enabled=True)
    mc = ps.register_point_cloud("moved corners", Pm, radius=0.009, color=WARP_COLOR)
    mc.set_point_render_mode("sphere")
    ps.register_curve_network("offsets", np.vstack([P, Pm]), np.c_[np.arange(8), np.arange(8) + 8],
                              color=WARP_COLOR).set_radius(0.0025, relative=True)


def draw_warped_cube(samples=24):
    """Image of the 12 cube edges under x -> x + D(x) (curves, since D is trilinear)."""
    P = to_numpy(S.cube.corner_positions())
    t = np.linspace(0, 1, samples + 1)
    pts = np.concatenate([P[a] + t[:, None] * (P[b] - P[a]) for a, b in CUBE_EDGES])
    with torch.no_grad():
        W = to_numpy(S.cube.warp(torch.as_tensor(pts)))
    k = samples + 1
    edges = np.concatenate([np.c_[np.arange(samples), np.arange(1, samples + 1)] + e * k for e in range(12)])
    ps.register_curve_network("warped cube", W, edges, color=WARP_COLOR).set_radius(0.002, relative=True)


def draw_cells():
    if ps.has_curve_network("active cells"):
        ps.remove_curve_network("active cells")
    if not S.show_cells or not len(S.topo.cells):
        return
    mins, h = S.topo.cell_bounds()
    mins = mins[:MAX_CELLS_DRAWN]
    nodes = (mins[:, None, :] + h * CORNER_IJK[None]).reshape(-1, 3)
    edges = (np.arange(len(mins))[:, None, None] * 8 + CUBE_EDGES[None]).reshape(-1, 2)
    ps.register_curve_network("active cells", nodes, edges, color=CELL_COLOR).set_radius(0.0012, relative=True)


def apply_display_options():
    if ps.has_surface_mesh("mesh"):
        m = ps.get_surface_mesh("mesh")
        m.set_enabled(S.show_surface)
        m.set_smooth_shade(S.smooth)
        m.set_edge_width(1.0 if S.wireframe else 0.0)
        if S.show_backface:
            m.set_back_face_policy("custom")
            m.set_back_face_color(BACK_COLOR)
        else:
            m.set_back_face_policy("identical")
    for name, on in (("cube", S.show_cube), ("warped cube", S.show_warped)):
        if ps.has_curve_network(name):
            ps.get_curve_network(name).set_enabled(on)
    if ps.has_curve_network("offsets"):
        ps.get_curve_network("offsets").set_enabled(S.show_corners)
    for name in ("corners", "moved corners"):
        if ps.has_point_cloud(name):
            ps.get_point_cloud(name).set_enabled(S.show_corners)
    if ps.has_surface_mesh("target"):
        ps.get_surface_mesh("target").set_enabled(S.show_target)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def ui():
    psim.PushItemWidth(170)
    dirty = False

    if psim.CollapsingHeader("Values", psim.ImGuiTreeNodeFlags_DefaultOpen):
        changed, idx = psim.Combo("preset / fit", S.entry_idx, [e["name"] for e in S.entries])
        if changed:
            select_entry(idx)
            dirty = True
        psim.TextUnformatted(f"cube {S.box}")
        _, S.fit_path = psim.InputText("fit file", S.fit_path)
        if psim.Button("Browse..."):
            p = choose_file(".")
            if p:
                S.fit_path = p
                dirty |= load_fit(p)
        psim.SameLine()
        if psim.Button("Paste"):
            p = clean_path(clipboard_text())
            if p:
                S.fit_path = p
        psim.SameLine()
        if psim.Button("Load fit"):
            dirty |= load_fit(S.fit_path)
        psim.SameLine()
        if psim.Button("Rescan *.json"):
            scan_fits(".")
        if psim.TreeNodeEx("Offsets (corner moves along its diagonal)", psim.ImGuiTreeNodeFlags_DefaultOpen):
            for n in range(8):
                i, j, k = CORNER_IJK[n]
                c, S.values[n] = psim.SliderFloat(f"d{n}  ({i}{j}{k})", S.values[n], -S.offset_range, S.offset_range)
                dirty |= c
            if psim.Button("Zero offsets"):
                S.values[:8] = [0.0] * 8
                dirty = True
            psim.SameLine()
            if psim.Button("Random offsets"):
                S.values[:8] = (np.random.default_rng().normal(size=8) * 0.3).round(3).tolist()
                dirty = True
            _, S.offset_range = psim.SliderFloat("offset range", S.offset_range, 0.1, 5.0)
            psim.TreePop()
        if psim.TreeNodeEx("Field values", psim.ImGuiTreeNodeFlags_DefaultOpen):
            for n in range(8):
                i, j, k = CORNER_IJK[n]
                c, S.values[8 + n] = psim.SliderFloat(f"v{n}  ({i}{j}{k})", S.values[8 + n], -S.field_range, S.field_range)
                dirty |= c
            if psim.Button("Random field"):
                S.values[8:] = np.random.default_rng().normal(size=8).round(3).tolist()
                dirty = True
            psim.SameLine()
            if psim.Button("Print"):
                print("values =", "[" + ", ".join(f"{v:.4f}" for v in S.values) + "]")
            _, S.field_range = psim.SliderFloat("field range", S.field_range, 0.5, 10.0)
            psim.TreePop()
        if S.message:
            psim.TextUnformatted(S.message)

    if psim.CollapsingHeader("Marching cubes", psim.ImGuiTreeNodeFlags_DefaultOpen):
        c, S.depth = psim.SliderInt("subdivision depth", S.depth, 0, 8); dirty |= c
        psim.TextDisabled(f"{2 ** S.depth}^3 sub-cubes")
        c, S.level = psim.SliderFloat("iso level", S.level, -S.field_range, S.field_range); dirty |= c
        _, S.obj_path = psim.InputText("obj path", S.obj_path)
        psim.SameLine()
        if psim.Button("Save OBJ"):
            save_obj(S.obj_path, S.V, S.F)
            S.message = f"Saved {len(S.V)} V / {len(S.F)} F to {S.obj_path}"

    if dirty:
        rebuild()

    if psim.CollapsingHeader("Display", psim.ImGuiTreeNodeFlags_DefaultOpen):
        ch = False
        c, S.show_surface = psim.Checkbox("Surface", S.show_surface); ch |= c
        psim.Indent()
        if psim.RadioButton("Flat", not S.smooth):
            S.smooth = False; ch = True
        psim.SameLine()
        if psim.RadioButton("Smooth", S.smooth):
            S.smooth = True; ch = True
        psim.Unindent()
        c, S.wireframe = psim.Checkbox("Wireframe", S.wireframe); ch |= c
        c, S.show_backface = psim.Checkbox("Inside side in orange", S.show_backface); ch |= c
        c, S.show_cube = psim.Checkbox("Cube", S.show_cube); ch |= c
        c, S.show_warped = psim.Checkbox("Warped cube (blue)", S.show_warped); ch |= c
        c, S.show_corners = psim.Checkbox("Corners and offsets", S.show_corners); ch |= c
        c, S.show_target = psim.Checkbox("Target of the fit (orange)", S.show_target); ch |= c
        c, S.show_cells = psim.Checkbox("Active sub-cubes", S.show_cells)
        if c:
            draw_cells()
        if ch:
            apply_display_options()

    if psim.CollapsingHeader("Statistics", psim.ImGuiTreeNodeFlags_DefaultOpen):
        st = S.stats
        psim.TextUnformatted(f"Vertices        {st.get('vertices', 0)}")
        psim.TextUnformatted(f"Faces           {st.get('faces', 0)}")
        psim.TextUnformatted(f"Active cubes    {st.get('active_cells', 0)} / {st.get('total_cells', 1)}")
        psim.TextUnformatted(f"max |f(V)-lvl|  {S.max_err:.1e}")
        psim.TextUnformatted(f"Mesh check      {'ok' if S.check.get('ok', True) else 'FAILED'}")
        psim.TextUnformatted(f"Extraction      {1000 * S.t_extract:.1f} ms")
        err = S.entries[S.entry_idx].get("error")
        if err and "rms" in err:
            psim.Separator()
            psim.TextUnformatted("Fit error (saved by the optimizer)")
            psim.TextUnformatted(f"  RMS           {err['rms']:.3e}  ({100 * err['rms'] / err.get('cube_width', 2.0):.2f} % of cube)")
            if "fit_to_target_max" in err:
                psim.TextUnformatted(f"  max fit->tgt  {err['fit_to_target_max']:.3e}")
                psim.TextUnformatted(f"  max tgt->fit  {err['target_to_fit_max']:.3e}")
            if "cd" in err:                                    # written by benchmark_warp.py
                psim.TextUnformatted(f"  chamfer (CD)  {err['cd']:.3e}   pieces {err.get('pieces', '?')}")
            if "method" in err:
                psim.TextUnformatted(f"  method        {err['method']}, seed {err.get('seed', '?')}")
                psim.TextUnformatted(f"  own loss      {err.get('own_loss', float('nan')):.3e}   "
                                     f"time {err.get('time', float('nan')):.1f} s")
        if psim.TreeNode("Pruning per level"):
            for lvl, tested, kept in st.get("levels", []):
                psim.TextUnformatted(f"level {lvl}: {kept:6d} of {tested:6d} kept")
            psim.TreePop()
    psim.PopItemWidth()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", type=float, nargs=16, help="16 values: 8 offsets then 8 field values")
    ap.add_argument("--preset", choices=list(WARP_PRESETS), default=list(WARP_PRESETS)[0])
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--load", help="fit file from warp_optimize.py or optimize.py")
    add_box_arguments(ap)
    args = ap.parse_args()
    S.box = S.preset_box = box_from_args(args)

    S.entry_idx = list(WARP_PRESETS).index(args.preset)
    S.values = list(args.values) if args.values else list(WARP_PRESETS[args.preset])
    S.depth, S.level = args.depth, args.level

    ps.init()
    ps.set_program_name("cubespline -- 16-value cube")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    draw_cube_frame()
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in dict.fromkeys([os.getcwd(), here]):
        scan_fits(folder)
    if args.load:
        S.fit_path = args.load
        load_fit(args.load)
    rebuild()
    ps.reset_camera_to_home_view()
    ps.set_user_callback(ui)
    ps.show()


if __name__ == "__main__":
    main()
