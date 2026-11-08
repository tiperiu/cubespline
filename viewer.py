#!/usr/bin/env python3
"""
cubespline -- interactive viewer for a trilinear cube and its iso-surface.

Set the 8 corner values with sliders (or on the command line); the surface is
re-extracted with subdivided marching cubes (marching.py) on every change.
Rendering follows the libigl playground viewer: Polyscope, y-up, shadow-only
ground plane, same mesh colours and panel layout.

Usage:
    python viewer.py                                  # preset "bowl", depth 4
    python viewer.py --preset saddle --depth 6
    python viewer.py --values -1 1 1 1 1 1 1 -1 --depth 5 --level 0.2
    python viewer.py --load fit.json                  # result of optimize.py
    python viewer.py --preset saddle --lo 2 0 0 --size 0.5   # the cube elsewhere, smaller
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from box import Box, add_box_arguments, box_from_args
from cube import Cube, CUBE_EDGES, CORNER_IJK, PRESETS
from marching import extract_topology, mesh_check, save_obj, to_numpy

BASE_COLOR = (0.75, 0.80, 0.95)        # same as libigl playground
BACK_COLOR = (0.95, 0.70, 0.45)        # inside (f < level) side of the surface
EDGE_COLOR = (0.10, 0.10, 0.10)
CUBE_COLOR = (0.25, 0.25, 0.25)
CELL_COLOR = (0.45, 0.60, 0.45)
TARGET_COLOR = (0.95, 0.55, 0.30)      # same as the optimize.py live view
MAX_CELLS_DRAWN = 40000


# --------------------------------------------------------------------------
# Viewer state
# --------------------------------------------------------------------------
class State:
    values = list(PRESETS["bowl"])
    depth = 4
    level = 0.0
    show_surface = True
    smooth = False          # flat shading shows the marching-cubes facets
    wireframe = True
    show_backface = True
    show_cube = True
    show_corners = True
    show_cells = False
    show_vertices = False
    slider_range = 2.0
    # combo-box entries: the presets, then every fit loaded from a fit.json
    # entry = {"name", "values", "depth" (or None), "target" (or None), "path" (or None)}
    entries = [{"name": k, "values": list(v), "depth": None, "target": None, "path": None, "box": None}
               for k, v in PRESETS.items()]
    box = Box()             # where the cube is now (fits carry their own; presets use preset_box)
    preset_box = Box()      # --lo / --size
    entry_idx = list(PRESETS).index("bowl")
    fit_path = "fit.json"
    show_target = True
    target_name = None      # target of the loaded fit currently drawn
    obj_path = "cubespline_mesh.obj"
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
# Geometry
# --------------------------------------------------------------------------
def rebuild():
    """Re-extract the surface for the current values / depth / level."""
    S.cube = Cube(S.values, requires_grad=False, **S.box.kw)
    t = time.time()
    S.topo = extract_topology(S.cube, S.depth, S.level)
    S.V = to_numpy(S.topo.vertices(S.cube.values))
    S.F = S.topo.F
    S.t_extract = time.time() - t
    S.stats = S.topo.stats
    S.check = mesh_check(S.V, S.F, S.cube.lo, S.cube.size)
    S.max_err = float(np.abs(S.cube.evaluate(S.V).detach().numpy() - S.level).max()) if len(S.V) else 0.0
    draw_mesh()
    draw_corners()
    draw_cells()
    apply_display_options()


# --------------------------------------------------------------------------
# Fits written by optimize.py
# --------------------------------------------------------------------------
def entry_names():
    return [e["name"] for e in S.entries]


def select_entry(idx):
    """Show combo-box entry idx: its values, its depth and (for a fit) its target."""
    S.entry_idx = idx
    e = S.entries[idx]
    S.values = list(e["values"])
    if e["depth"] is not None:
        S.depth = int(e["depth"])
    S.slider_range = max(S.slider_range, float(np.abs(S.values).max()) * 1.2)
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


def clean_path(text):
    """Turn whatever was pasted into a plain path: strip spaces and quotes, a
    file:// prefix, and a trailing newline; expand ~."""
    p = (text or "").strip().strip('"').strip("'").strip()
    if p.startswith("file://"):
        from urllib.parse import unquote, urlparse
        p = unquote(urlparse(p).path)
    return os.path.expanduser(p)


def clipboard_text():
    """System clipboard as text. ImGui text fields in Polyscope often ignore
    Cmd+V on macOS, so read the clipboard directly."""
    try:
        if sys.platform == "darwin":
            return subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2).stdout
        if sys.platform.startswith("win"):
            return subprocess.run(["powershell", "-command", "Get-Clipboard"],
                                  capture_output=True, text=True, timeout=5).stdout
        for cmd in (["xclip", "-o", "-selection", "clipboard"], ["wl-paste", "-n"]):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout
            except FileNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    try:
        return psim.GetClipboardText() or ""
    except Exception:  # noqa: BLE001
        return ""


def choose_file(start_dir="."):
    """Native file picker; returns a path or "" if cancelled."""
    start_dir = os.path.abspath(start_dir)
    try:
        if sys.platform == "darwin":
            script = ('POSIX path of (choose file with prompt "Choose a fit (.json)" '
                      f'default location (POSIX file "{start_dir}"))')
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            return r.stdout.strip()
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(initialdir=start_dir, title="Choose a fit",
                                          filetypes=[("fit files", "*.json"), ("all files", "*")])
        root.destroy()
        return path or ""
    except Exception as e:  # noqa: BLE001
        print(f"  file picker unavailable: {e}")
        return ""


def load_fit(path):
    """Read a fit.json from optimize.py, add it to the combo box and show it."""
    path = clean_path(path)
    try:
        with open(path) as fh:
            fit = json.load(fh)
        values = [float(v) for v in fit["values"]]
        assert len(values) == 8
    except Exception as e:  # noqa: BLE001
        S.message = f"Could not load {path}: {e}"
        print(S.message)
        return False
    target = fit.get("target")
    name = f"fit: {os.path.basename(path)}" + (f" ({os.path.basename(str(target))})" if target else "")
    entry = {"name": name, "values": values, "depth": fit.get("depth"), "target": target,
             "path": os.path.abspath(path), "error": fit.get("final_error") or fit.get("metrics"),
             "box": Box.from_dict(fit)}
    # reloading the same file replaces its entry instead of adding a duplicate
    for i, e in enumerate(S.entries):
        if e["path"] == entry["path"]:
            S.entries[i] = entry
            break
    else:
        S.entries.append(entry)
        i = len(S.entries) - 1
    select_entry(i)
    rms = fit.get("metrics", {}).get("rms")
    S.message = f"Loaded {os.path.basename(path)}: target {target}" + (f", rms {rms:.3g}" if rms else "")
    print(S.message)
    return True


def scan_fits(folder="."):
    """Add every fit-looking .json in `folder` to the combo box (does not select them)."""
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            with open(path) as fh:
                fit = json.load(fh)
            if not (isinstance(fit, dict) and len(fit.get("values", [])) == 8):
                continue
        except Exception:  # noqa: BLE001
            continue
        if any(e["path"] == os.path.abspath(path) for e in S.entries):
            continue
        target = fit.get("target")
        S.entries.append({"name": f"fit: {os.path.basename(path)}" + (f" ({os.path.basename(str(target))})" if target else ""),
                          "values": [float(v) for v in fit["values"]], "depth": fit.get("depth"),
                          "target": target, "path": os.path.abspath(path),
                          "error": fit.get("final_error") or fit.get("metrics"), "box": Box.from_dict(fit)})


def draw_target(target):
    """Draw the surface a fit was optimized against (built-in name or mesh path)."""
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
            V, F = O.load_obj(target) if target.lower().endswith(".obj") else __import__("igl").read_triangle_mesh(target)
            t = O.target_from_mesh(V, F, name=target, n=500, box=S.box)
        else:
            print(f"  target '{target}' not found; showing the fit only")
            return
    except Exception as e:  # noqa: BLE001
        print(f"  could not rebuild target '{target}': {e}")
        return
    m = ps.register_surface_mesh("target", t.V, t.F, color=TARGET_COLOR, transparency=0.45)
    m.set_back_face_policy("identical")
    m.set_enabled(S.show_target)


def draw_cube_frame():
    P = S.box.corners()
    cn = ps.register_curve_network("cube", P, CUBE_EDGES, color=CUBE_COLOR)
    cn.set_radius(0.004, relative=True)


def draw_corners():
    P = to_numpy(S.cube.corner_positions())
    pc = ps.register_point_cloud("corners", P, radius=0.012, color=(0.9, 0.3, 0.2))
    pc.set_point_render_mode("sphere")
    v = np.asarray(S.values, dtype=float) - S.level
    m = max(np.abs(v).max(), 1e-9)
    pc.add_scalar_quantity("value - level", v, cmap="coolwarm", vminmax=(-m, m), enabled=True)


def draw_mesh():
    for name in ("mesh", "vertices"):
        if name == "mesh" and ps.has_surface_mesh(name):
            ps.remove_surface_mesh(name)
        if name == "vertices" and ps.has_point_cloud(name):
            ps.remove_point_cloud(name)
    if not len(S.F):
        return
    m = ps.register_surface_mesh("mesh", S.V, S.F, color=BASE_COLOR)
    m.set_edge_color(EDGE_COLOR)
    pr = 0.004 if len(S.V) < 5000 else 0.002
    pc = ps.register_point_cloud("vertices", S.V, radius=pr, color=(0.9, 0.3, 0.2))
    pc.set_point_render_mode("sphere")


def draw_cells():
    if ps.has_curve_network("active cells"):
        ps.remove_curve_network("active cells")
    if not S.show_cells or not len(S.topo.cells):
        return
    mins, h = S.topo.cell_bounds()
    mins = mins[:MAX_CELLS_DRAWN]
    nodes = (mins[:, None, :] + h * CORNER_IJK[None]).reshape(-1, 3)
    edges = (np.arange(len(mins))[:, None, None] * 8 + CUBE_EDGES[None]).reshape(-1, 2)
    cn = ps.register_curve_network("active cells", nodes, edges, color=CELL_COLOR)
    cn.set_radius(0.0012, relative=True)


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
    if ps.has_point_cloud("vertices"):
        ps.get_point_cloud("vertices").set_enabled(S.show_vertices)
    if ps.has_curve_network("cube"):
        ps.get_curve_network("cube").set_enabled(S.show_cube)
    if ps.has_point_cloud("corners"):
        ps.get_point_cloud("corners").set_enabled(S.show_corners)


# --------------------------------------------------------------------------
# UI (ImGui panel, drawn every frame)
# --------------------------------------------------------------------------
def ui():
    psim.PushItemWidth(180)
    dirty = False

    # ---- Corner values ----
    if psim.CollapsingHeader("Corner values", psim.ImGuiTreeNodeFlags_DefaultOpen):
        changed, idx = psim.Combo("preset / fit", S.entry_idx, entry_names())
        if changed:
            select_entry(idx)
            dirty = True
        psim.TextUnformatted(f"cube {S.box}")
        _, S.fit_path = psim.InputText("fit file", S.fit_path)
        if psim.Button("Browse..."):
            start = os.path.dirname(clean_path(S.fit_path)) or "."
            p = choose_file(start if os.path.isdir(start) else ".")
            if p:
                S.fit_path = p
                dirty |= load_fit(p)
        psim.SameLine()
        if psim.Button("Paste"):
            text = clean_path(clipboard_text())
            if text:
                S.fit_path = text
                S.message = "Pasted path; press Load fit"
            else:
                S.message = "Clipboard is empty"
        psim.SameLine()
        if psim.Button("Load fit"):
            dirty |= load_fit(S.fit_path)
        psim.SameLine()
        if psim.Button("Rescan *.json"):
            n = len(S.entries)
            scan_fits(os.path.dirname(os.path.abspath(S.fit_path)) or ".")
            S.message = f"Found {len(S.entries) - n} new fit file(s)"
        r = S.slider_range
        for n in range(8):
            i, j, k = CORNER_IJK[n]
            c, S.values[n] = psim.SliderFloat(f"v{n}  ({i}{j}{k})", S.values[n], -r, r)
            dirty |= c
        if psim.Button("Random"):
            S.values = np.random.default_rng().normal(size=8).round(3).tolist()
            dirty = True
        psim.SameLine()
        if psim.Button("Negate"):
            S.values = [-v for v in S.values]
            dirty = True
        psim.SameLine()
        if psim.Button("Normalize"):
            nrm = np.linalg.norm(S.values)
            if nrm > 0:
                S.values = (np.asarray(S.values) / nrm * np.sqrt(8)).tolist()
                dirty = True
        psim.SameLine()
        if psim.Button("Print"):
            print("values =", "[" + ", ".join(f"{v:.4f}" for v in S.values) + "]")
        _, S.slider_range = psim.SliderFloat("slider range", S.slider_range, 0.5, 10.0)

    # ---- Extraction ----
    if psim.CollapsingHeader("Marching cubes", psim.ImGuiTreeNodeFlags_DefaultOpen):
        c, S.depth = psim.SliderInt("subdivision depth", S.depth, 0, 8); dirty |= c
        psim.TextDisabled(f"{2 ** S.depth}^3 sub-cubes")
        c, S.level = psim.SliderFloat("iso level", S.level, -S.slider_range, S.slider_range); dirty |= c
        if psim.Button("Level = 0"):
            S.level = 0.0
            dirty = True
        _, S.obj_path = psim.InputText("obj path", S.obj_path)
        psim.SameLine()
        if psim.Button("Save OBJ"):
            save_obj(S.obj_path, S.V, S.F)
            S.message = f"Saved {len(S.V)} V / {len(S.F)} F to {S.obj_path}"
            print(S.message)
        if S.message:
            psim.TextUnformatted(S.message)

    if dirty:
        rebuild()

    # ---- Display ----
    if psim.CollapsingHeader("Display", psim.ImGuiTreeNodeFlags_DefaultOpen):
        changed = False
        c, S.show_surface = psim.Checkbox("Surface (shading on/off)", S.show_surface); changed |= c
        psim.Indent()
        if psim.RadioButton("Flat", not S.smooth):
            S.smooth = False; changed = True
        psim.SameLine()
        if psim.RadioButton("Smooth", S.smooth):
            S.smooth = True; changed = True
        psim.Unindent()
        c, S.wireframe = psim.Checkbox("Wireframe", S.wireframe); changed |= c
        c, S.show_backface = psim.Checkbox("Inside side in orange", S.show_backface); changed |= c
        c, S.show_vertices = psim.Checkbox("Vertices", S.show_vertices); changed |= c
        c, S.show_cube = psim.Checkbox("Cube", S.show_cube); changed |= c
        c, S.show_corners = psim.Checkbox("Corner values (blue < level < red)", S.show_corners); changed |= c
        c, S.show_target = psim.Checkbox("Target of the fit (orange)", S.show_target)
        if c and ps.has_surface_mesh("target"):
            ps.get_surface_mesh("target").set_enabled(S.show_target)
        c, S.show_cells = psim.Checkbox("Active sub-cubes", S.show_cells)
        if c:
            draw_cells()
        if changed:
            apply_display_options()

    # ---- Stats ----
    if psim.CollapsingHeader("Statistics", psim.ImGuiTreeNodeFlags_DefaultOpen):
        st = S.stats
        psim.TextUnformatted(f"Vertices        {st.get('vertices', 0)}")
        psim.TextUnformatted(f"Faces           {st.get('faces', 0)}")
        psim.TextUnformatted(f"Active cubes    {st.get('active_cells', 0)} / {st.get('total_cells', 1)}")
        psim.TextUnformatted(f"Ambiguous faces {st.get('ambiguous_faces', 0)}")
        psim.TextUnformatted(f"Boundary edges  {S.check.get('boundary_edges', 0)}")
        psim.TextUnformatted(f"max |f(V)-lvl|  {S.max_err:.1e}")
        psim.TextUnformatted(f"Mesh check      {'ok' if S.check.get('ok', True) else 'FAILED'}")
        psim.TextUnformatted(f"Extraction      {1000 * S.t_extract:.1f} ms")
        err = S.entries[S.entry_idx].get("error")
        if err and "rms" in err:
            psim.Separator()
            psim.TextUnformatted("Fit error (saved by optimize.py)")
            psim.TextUnformatted(f"  RMS           {err['rms']:.3e}  ({100 * err['rms'] / err.get('cube_width', 2.0):.2f} % of cube)")
            psim.TextUnformatted(f"  max fit->tgt  {err['fit_to_target_max']:.3e}")
            psim.TextUnformatted(f"  max tgt->fit  {err['target_to_fit_max']:.3e}")
            if "objective" in err:
                psim.TextUnformatted(f"  objective     {err['objective']:.3e}")
        if psim.TreeNode("Pruning per level"):
            for lvl, tested, active in st.get("levels", []):
                psim.TextUnformatted(f"level {lvl}: {active:6d} of {tested:6d} kept")
            psim.TreePop()

    psim.PopItemWidth()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", type=float, nargs=8, help="8 corner values, n = i + 2j + 4k")
    ap.add_argument("--preset", choices=list(PRESETS), default="bowl")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--load", help="values from a fit.json written by optimize.py")
    add_box_arguments(ap)
    args = ap.parse_args()
    S.box = S.preset_box = box_from_args(args)

    S.entry_idx = list(PRESETS).index(args.preset)
    S.values = list(args.values) if args.values else list(PRESETS[args.preset])
    S.depth, S.level = args.depth, args.level

    ps.init()
    ps.set_program_name("cubespline")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    draw_cube_frame()
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in dict.fromkeys([os.getcwd(), here]):   # fits where you run it and next to the code
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
