# cubespline

A single cube with 8 corner values. Inside the cube the field is the trilinear
interpolation of the values; its level set `f = level` (default 0) is a surface.
This repo extracts that surface with subdivided marching cubes, views it with
Polyscope, and optimizes the 8 values so the surface matches a target.

| file | what it does |
|---|---|
| `cube.py` | `Cube` (PyTorch `nn.Module`): 8 values as a parameter, `evaluate`, `basis`, `gradient`, `basis_gradient`, presets |
| `marching.py` | octree subdivision with exact pruning + marching cubes → `V` (positions), `F` (connectivity); `mesh_from_grid` for any sampled function |
| `viewer.py` | Polyscope viewer: cube, corner values, mesh, sliders for the values / depth / level; `--load fit.json` |
| `optimize.py` | fit the 8 values to a target surface (function or mesh), live view with `--view` |
| `test_cubespline.py`, `test_optimize.py` | tests |
| `warpcube.py` | `WarpCube`: 16 values, 8 offsets + 8 field values (second version, see below) |
| `warp_marching.py` | subdivided marching cubes for the 16-value cube |
| `warp_optimize.py` | fit the 16 values to a target |
| `warp_viewer.py` | viewer with 16 sliders, the warped cube and fit loading |
| `test_warp.py` | tests for the 16-value cube |

## Setup

```bash
cd cubespline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_cubespline.py && python test_optimize.py   # quick self-check
```

## Use

```bash
python viewer.py                                   # preset "bowl", depth 4
python viewer.py --preset checkerboard --depth 6
python viewer.py --values -1 1 1 1 1 1 1 -1 --depth 5
```

```python
from cube import Cube
from marching import marching_cubes, save_obj

cube = Cube([-1, 1, 1, 1, 1, 1, 1, -1])     # corner n = i + 2j + 4k, cube [-1,1]^3
V, F, topo = marching_cubes(cube, depth=5)  # V: torch (nv,3), F: int64 (nf,3)
save_obj("surface.obj", V, F)
```

### Corner numbering

Corner `n = i + 2j + 4k`, with `(i, j, k) ∈ {0,1}³` along `(x, y, z)`:

```
n   : 0    1    2    3    4    5    6    7
ijk : 000  100  010  110  001  101  011  111
```

"Inside" is `f < level`. Triangle normals point towards increasing `f`
(the viewer paints the inside face orange).

## How the mesher works

1. **Subdivision with exact pruning.** The cube is split level by level into
   `2^depth` sub-cubes per axis. A trilinear function on any box has its min and
   max at the box corners, so a sub-cube whose 8 corners are on one side of the
   level cannot contain surface and is discarded with all its children. This is
   exact, not an approximation. All leaves are at the same depth, so they form a
   conforming grid: no cracks, no T-junctions.
2. **Marching cubes with the asymptotic decider.** On a sub-cube face the field is
   exactly bilinear, so the face saddle value decides correctly whether the two
   inside corners of an ambiguous face are connected. Both neighbours sharing
   that face make the same decision, so the mesh is watertight. The triangle
   table is generated from these face rules in code, not typed in.
3. **Exact vertices.** Along an axis-aligned edge the trilinear field is linear,
   so linear interpolation finds the exact crossing: all vertices lie on the true
   surface (|f(V)| ≈ 1e-16) at every depth. Increasing the depth only refines the
   triangulation between them, and resolves the interior ambiguities (tunnels)
   that a single marching cube cannot see.

A trilinear field has no interior extrema, so the surface is never closed inside
the cube: every component reaches the cube boundary, and the mesh boundary lies
on the cube faces.

## Fitting the 8 values (`optimize.py`)

```bash
python optimize.py --list                          # built-in targets and loss terms
python optimize.py --target 'sphere flank'         # headless: prints progress, writes fit.json
python optimize.py --target 'sphere flank' --no-refine   # closed-form start only
python optimize.py --target torus --view           # live: Run / Step / Reset, loss plot
python optimize.py --mesh my.obj --loss chamfer=1 --loss normal=0.1
python viewer.py --load fit.json                   # inspect the result
```

In the viewer, every `*.json` fit in the folder you run it from appears in the
**preset / fit** combo box as `fit: <file> (<target>)`. Selecting one sets its values
and depth and draws its target surface in transparent orange. **Load fit** opens
any other fit file by path; **Rescan \*.json** picks up fits written while the viewer
is open.


**Target.** A function `g(x)` (negative inside) or a triangle mesh with outward
normals, clipped to the cube. Function targets are meshed with the same marching
cubes and sampled area-uniformly, then the samples are projected onto `g = 0`
with Newton steps. Mesh targets are sampled and only samples inside the cube are
kept. Each target provides points `q_i` and unit normals `n_i`.

**Initialisation (closed form).** `f` and `grad f` are linear in the values, so two
cheap fits are tried and the one with the smallest true surface distance is kept:

* Taubin: minimise `Σ f(q_i)² / Σ |∇f(q_i)|²`, a generalised 8×8 eigenproblem. It is
  exact when the target is itself trilinear. It ignores orientation, so the sign is
  fixed from the normals afterwards.
* With normals: least squares `Σ f(q_i)² + μ Σ |∇f(q_i) − n_i|²` for a few μ. It
  follows the target's inside/outside and does better on e.g. the paraboloid.

**Refinement (through marching cubes).** Each iteration:

```
topology = extract_topology(cube, depth)        # discrete, no gradients
V        = topology.vertices(values)            # differentiable in the 8 values
P        = area-uniform samples on (V, F)       # differentiable in V
loss     = Σ w_k LOSSES[k](ctx)                 # default: two-sided chamfer(P, target points)
Adam step (cosine-decayed lr), then |values| = √8 (the surface is scale invariant)
```

The chamfer nearest neighbours are found without gradients and the loss is formed
from the matched pairs only. That gives the same gradient as the full distance
matrix and is about 7× faster (≈ 33 ms per iteration at depth 5 with 3000 samples).
Every 10 iterations a deterministic version of the loss is evaluated, and the best
iterate is returned, so refinement never ends worse than it started.

**Adding a loss.** Write `def my_loss(ctx) -> scalar tensor` using the fields of
`LossContext` (cube, values, mesh `V`/`F`, mesh samples, target), register it with
`LOSSES["my_loss"] = my_loss`, and select it with `--loss my_loss=0.5`. Built in:
`chamfer`, `algebraic` (Sampson distance of the target points to the level set) and
`normal` (1 − cos between ∇f and the target normals).

**Metrics** (`evaluate`): first-order distances to the true surfaces in both
directions, `|f(q)|/|∇f(q)|` and `|g(p)|/|∇g(p)|`, reported as RMS and max. For mesh
targets the fit-to-target direction uses 50k dense target samples.

### Results on the built-in targets

Defaults (depth 5, 300 iterations, lr 0.05 → 5e-4, 3000 samples), cube [-1, 1]³ (width 2).
RMS and max are true distances to the surfaces, in world units:

| target | init chosen | RMS after init | RMS after refinement | max distance |
|---|---|---|---|---|
| trilinear | Taubin | 7.6e-05 | 7.6e-05 | 0.00 |
| plane | normals μ=0.001 | 2.8e-14 | 2.8e-14 | 0.00 |
| paraboloid | normals μ=0.001 | 0.30 | 0.27 | 0.92 |
| hyperboloid | Taubin | 0.29 | 0.23 | 1.58 |
| cylinder | Taubin | 0.26 | 0.25 | 1.02 |
| sphere | Taubin | 0.25 | 0.24 | 1.29 |
| small sphere | Taubin | 0.33 | 0.33 | 1.15 |
| torus | Taubin | 0.27 | 0.26 | 0.82 |

Each run takes about 10 s on a laptop CPU.

What this shows:

* **Representable targets are recovered.** A trilinear target comes back to 0.0001
  of the cube width; the closed-form start is already there and refinement keeps it.
* **Curved targets hit the limit of one cube, not of the optimizer.** Random starts
  plus refinement end at the same ≈ 0.23–0.27 RMS as the closed-form start. A
  trilinear field has no interior extremum, so it cannot be negative inside a
  closed shape. For the sphere, cylinder and torus the best fit is a saddle-like
  surface whose inside/outside agrees with the target's at only about half the
  points. Open, graph-like targets (plane, paraboloid) keep the target's orientation.
* **Refinement from a poor start works, but it needs a good basin.** From a start
  perturbed by 0.3 per value, chamfer refinement alone recovers a trilinear target
  from 0.44 to 8e-4 RMS. From fully random starts it often stops at 0.04–0.6, so
  the closed-form initialisation matters.

### Slices of a large sphere or torus

One cube cannot close a surface, so the realistic targets are patches of big
shapes clipped by the cube. These slices are built in (`python optimize.py --list`):

| target | apex / crest | RMS | max distance |
|---|---|---|---|
| sphere cap (r 2.5, centred below) | inside the cube | 0.087 | 0.29 |
| large sphere cap (r 5) | inside in z | 0.0094 | 0.039 |
| torus outer (R 3, r 1.5) | inside | 0.098 | 0.35 |
| torus inner, saddle side (R 3.5, r 1.5) | inside | 0.13 | 0.35 |
| **sphere flank** (r 2.5) | outside | **0.0084** | 0.047 |
| **large sphere flank** (r 5) | outside | **0.0035** | 0.021 |
| **torus outer flank** (R 3, r 2.6) | outside | **0.0085** | 0.044 |
| **torus inner flank**, saddle side (R 4.4, r 2.6) | outside | **0.0066** | 0.024 |

Making the shape bigger helps, but where the patch sits matters more. On every
plane `z = const` the field is bilinear in `(x, y)`, so the fitted curve there is
`y(x) = −(A + Bx)/(C + Dx)`, a Möbius function, which is monotone (likewise for the
other two plane families). Checked numerically on 2000 random cubes: no turning
points. So a single cube's surface cannot have a bump, dip or crest inside the
cube; equivalently, no component of its normal can change sign there.

* In a **cap** the apex is inside the cube. However large the sphere, one cube
  must flatten it, and the error stays at a fixed fraction of the cap's height.
* In a **flank** the sphere centre, or the torus's tube centre and axis, lies outside
  the cube's range in x, y and z. The normal keeps its signs and one cube follows
  the patch closely. The `sphere flank` has the same radius as the `sphere cap` and
  fits 10× better.

This also decides how a grid of cubes must be laid out: every cell has to see a
monotone piece, so cells must be small enough, or placed so that no apex or crest
falls strictly inside one.

The natural next step for curved targets is several cubes sharing corner values,
a grid of trilinear cells (the "cube spline"). The pieces here, the mesher, the
differentiable vertices and the loss registry, carry over unchanged to that setting.

## Second version: the 16-value cube (`warpcube.py`)

```
values = [ offset c1 (8) | field c2 (8) ]
x'   = x + D(x),   D(x) = Σ_n w_n(x) c1[n] e_n      e_n = corner n's diagonal direction (±1, ±1, ±1)
f(x) = T2(x')      T2 = trilinear interpolation of c2 (extended outside the cube if x' leaves it)
```

Each offset value moves its corner along the cube diagonal, outward if positive.
Trilinear interpolation of the moved corners gives the new point x'; the 8 field
values are then interpolated at x'. With all offsets 0 this is exactly the 8-value
cube; the tests check that the mesher gives identical meshes then. The offset rule
is the single function `offset_local` in `warpcube.py`.

```bash
python warp_optimize.py --target "sphere cap"              # writes warp_fit.json
python warp_optimize.py --target "sphere cap" --start-iters 200 --refine-iters 500
python warp_optimize.py --target "sphere cap" --no-refine  # stage 1 only
python warp_optimize.py --target "torus inner" --out warp_torus_inner.json
python warp_optimize.py --target "sphere cap" --view       # live window
python warp_viewer.py                                      # 16 sliders; fits in the combo box
python test_warp.py
```

**Mesher (`warp_marching.py`).** f is no longer trilinear, so three parts change:

* *Pruning* uses a rigorous bound. On a sub-cube each component of x' is trilinear,
  so x' stays in the box spanned by its 8 corner values. T2 is trilinear, so its
  range over that box is spanned by the box's 8 corners. If that range excludes the
  level, the sub-cube is empty. The bound never discards surface (tested).
* *Vertices* are found by bisection along each lattice edge, in float64, to round-off.
  |f(V)| ≤ 1e-14 in the tests.
* *Gradients*: after bisection, `t = t* − f/f_t`, with `f_t` detached, gives the
  implicit-function-theorem derivative, so the vertices are differentiable in all
  16 values (gradcheck).

As in any marching cubes, a surface piece entirely inside one leaf cell with all
corners on one side is not captured; the depth controls that.

**Optimizer (`warp_optimize.py`).** Good fits need large offsets (values of ±1–2),
which gradient descent from zero offset does not reach. So the start is a
multi-start, mesh-free fit: from the 8-value solution and 16 random starts, minimise
the point-to-surface (Sampson) distance with L-BFGS, and keep the best by true
distance. Then the same chamfer refinement through marching cubes runs as before,
keeping the best iterate. `--init regular` starts at the 8-value fit only.

Iteration counts, per stage:

| option | stage | default |
|---|---|---|
| `--starts K` | 1: number of random starts (plus the 8-value fit) | 16 |
| `--start-iters N` | 1: L-BFGS iterations per start | 100 |
| `--refine-iters M` | 2: Adam steps of the chamfer refinement (`--iters` is an alias) | 300 |
| `--no-refine` | skip stage 2 (same as `--refine-iters 0`) | | Losses:
`chamfer`, `algebraic`, `normal`, and `offset` (keeps offsets small). The final error
report and the `final_error` block in the fit file have the same format as `optimize.py`.

**Why it helps.** The 8-value cube's level set is monotone in every axis-aligned
slice (a Möbius function), so it cannot have a bump inside the cube. The 16-value
level set is T2(x + D(x)): products such as x'·y' with y' depending on x create x²
terms, and the field can bend.

### Results: 16-value vs 8-value cube

Default settings (depth 5, 300 iterations, 16 random starts), cube [-1, 1]³ (width 2).
RMS = true distance to the target, both directions:

| target | 8-value RMS | 16-value RMS | 16-value max | gain | largest offset |
|---|---|---|---|---|---|
| sphere flank | 8.4e-03 | 2.4e-03 | 0.011 | 3.5× | 2.17 |
| large sphere flank | 3.5e-03 | 7.2e-04 | 0.003 | 4.9× | 1.46 |
| torus outer flank | 8.5e-03 | 3.0e-03 | 0.013 | 2.8× | 0.55 |
| torus inner flank (saddle) | 6.7e-03 | 3.2e-03 | 0.018 | 2.1× | 0.92 |
| **sphere cap** (apex inside) | 8.7e-02 | **5.8e-03** | 0.031 | **15×** | 1.27 |
| large sphere cap | 9.4e-03 | 3.0e-03 | 0.027 | 3.1× | 1.20 |
| **torus outer** (crest inside) | 9.8e-02 | **1.2e-02** | 0.056 | **8.6×** | 1.19 |
| **torus inner** (saddle, crest inside) | 1.3e-01 | **1.2e-02** | 0.069 | **11×** | 2.28 |

Each run takes 40–60 s on a laptop CPU.

* On the flanks, which the 8-value cube already handles, the extra 8 values still
  reduce the error 2–5×.
* On patches with an apex or crest inside the cube, which the 8-value cube cannot
  bend into, the 16-value cube is 9–15× better. The cap and the centred tori come
  down from 9–13 % of the cube width to about 0.6 %.
* The good fits use large offsets (±1–2, on a cube of width 2): the warp moves
  points well outside the cube before the field is evaluated. The multi-start
  matters; from zero offset alone, the refinement stays near the 8-value solution.
* The two "(fitted)" presets in `warp_viewer.py` are the sphere-cap and torus-inner
  results from this table.
