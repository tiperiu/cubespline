# cubespline

A cube with 8 corner values, and grids of such cubes. Inside a cube the field is the
trilinear interpolation of the values; its level set `f = level` (default 0) is a
surface. This repo extracts that surface with subdivided marching cubes, views it
with Polyscope, and optimizes the values so the surface matches a target: one cube
(`code/core`, `code/optimization`), or a whole grid of cubes that share their corner
values (`code/grid`, see [Grids of cubes](#grids-of-cubes-codegrid)).

## Project layout

```
cubespline/
  README.md  CLAUDE.md  requirements.txt
  code/
    core/           the single cube: representation, meshing, placement; mesh geometry
    optimization/   the single-cube optimizers
    grid/           grids of cubes: models, meshing, targets, optimizer, examples script
    visualization/  the viewers
    tests/          tests
  examples/
    functions/      example implicit functions g(x): spheres, ellipsoids, tori
    point_clouds/   the same shapes as point clouds (.xyz), plus two harder variants
  data/
    fits/           fit files of the single-cube optimizers (default --out)
    fits/grid/      fit files of the grid optimizer
```

| file | what it does |
|---|---|
| `code/core/cube.py` | `Cube` (PyTorch `nn.Module`): 8 values as a parameter, `evaluate`, `basis`, `gradient`, `basis_gradient`, presets |
| `code/core/marching.py` | octree subdivision with exact pruning + marching cubes → `V` (positions), `F` (connectivity); `mesh_from_grid` for any sampled function |
| `code/core/box.py` | `Box`: where a cube sits (`lo`, `size`), lattices, `--lo/--size` options, grid helpers (`grid_cell`, `corner_vertex_ids`) |
| `code/core/geometry.py` | mesh helpers: connected pieces, closest point on a triangle, exact point-to-mesh distances |
| `code/core/paths.py` | the project folders (`data/fits`, `data/fits/grid`, `examples`) used for default outputs and by the viewers |
| `code/optimization/optimize.py` | targets, and the fit of the 8 values to a target surface (function or mesh); live view with `--view` |
| `code/optimization/classify_optimize.py` | fit the cube by inside/outside classification: logistic loss + Adam, evolution strategies, or GRPO (reward only); the grid optimizer reuses it |
| `code/grid/grid.py` | `Grid` (cells, vertex ids, interpolation) and `GridModel`: `scalar` (a value per vertex) or `warp` (a value and a 3D displacement per vertex) |
| `code/grid/grid_marching.py` | marching cubes over the whole grid, differentiable vertices, watertightness check |
| `code/grid/targets.py` | grid targets: function files `g(X)` and point clouds (.xyz, .ply, .obj, …), normal estimation, inside/outside labels, distances |
| `code/grid/grid_optimize.py` | fit a grid to a target: chamfer (closed form → Sampson → chamfer through marching cubes), logistic, ES, GRPO |
| `code/grid/fit_examples.py` | fit every example, save the fits and print a results table |
| `code/grid/make_examples.py` | write the example point clouds from the example functions |
| `code/visualization/viewer.py` | Polyscope viewer for one cube: cube, corner values, mesh, sliders for the values / depth / level; `--load fit.json` |
| `code/visualization/grid_viewer.py` | Polyscope viewer for grids: examples, fits, cells, displacements; fit from the window |
| `code/tests/test_*.py` | tests: `test_cubespline`, `test_optimize`, `test_classify`, `test_box` (one cube), `test_grid` (grids) |

**Running.** Run every script from anywhere, e.g. `python code/optimization/optimize.py`
from the project folder: each script puts `code/` on the Python path itself, and
inside the code the modules import each other as packages (`from core.cube import
Cube`). In PyCharm, optionally mark `code/` as a Sources Root so it resolves the
imports in the editor.

**Data.** Unless `--out` says otherwise, the single-cube optimizers write their fit
files to `data/fits/` and the grid optimizer to `data/fits/grid/`. The viewers list
the fit files of the folder they are started from and of those folders, and
`--load NAME` also finds a file by name there.

## Setup

```bash
cd cubespline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest code/tests -q                         # all tests (49), about 1 min
python code/tests/test_cubespline.py                   # or one file at a time
python code/tests/test_grid.py                         # only the grid tests (18), about 20 s
```

## Use

```bash
python code/visualization/viewer.py                                   # preset "bowl", depth 4
python code/visualization/viewer.py --preset checkerboard --depth 6
python code/visualization/viewer.py --values -1 1 1 1 1 1 1 -1 --depth 5
```

```python
# with code/ on the Python path (e.g. run from code/, or code/ as a PyCharm Sources Root)
from core.cube import Cube
from core.marching import marching_cubes, save_obj

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
python code/optimization/optimize.py --list                          # built-in targets and loss terms
python code/optimization/optimize.py --target 'sphere flank'         # prints progress, writes data/fits/fit.json
python code/optimization/optimize.py --target 'sphere flank' --no-refine   # closed-form start only
python code/optimization/optimize.py --target torus --view           # live: Run / Step / Reset, loss plot
python code/optimization/optimize.py --mesh my.obj --loss chamfer=1 --loss normal=0.1
python code/visualization/viewer.py --load fit.json                   # inspect the result
```

In the viewer, every `*.json` fit in the folder you run it from and in `data/fits`
appears in the **preset / fit** combo box as `fit: <file> (<target>)`. Selecting one sets its values
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
shapes clipped by the cube. These slices are built in (`python code/optimization/optimize.py --list`):

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
a grid of trilinear cells (the "cube spline"). That is `code/grid`, described
[below](#grids-of-cubes-codegrid). It reuses the mesher's case table, the
differentiable vertices and the classification optimizers.

## Cube position and size (`box.py`)

Every cube is an axis-aligned cube `[lo, lo + size]³`, by default `[-1, 1]³`. All
the single-cube optimizers and the viewer take `--lo X Y Z` and `--size S`:

```bash
python code/optimization/optimize.py --target "sphere cap" --lo 0 -1 0 --size 1        # a corner octant of the default cube
python code/optimization/classify_optimize.py --target "sphere cap" --lo 0 -1 0 --size 1
python code/visualization/viewer.py --preset saddle --lo 2 0 0 --size 0.5
```

* **Targets live in world space** and are clipped to the cube. The built-in
  functions were designed around `[-1, 1]³`, so a cube placed elsewhere sees a
  different part of them, or nothing. Then you get the error "no surface inside
  the cube".
* **Values are dimensionless.** The same values give the same shape, moved and
  scaled, in any cube.
* **Saved fits** store `"lo"` and `"size"`. Files without them load as the
  default cube. The viewers move the cube, the target and the camera to each
  fit's box. Presets use `--lo/--size`. The current box is shown under the combo
  box.
* **Errors** (RMS, max) are in world units, and as a percentage of the cube width.

Example: the corner octant `--lo 0 -1 0 --size 1` of the sphere cap. The fit
reaches RMS 1.8e-3, well under 1 % of the cube width. That octant contains only a
flank-like part of the cap.

`Box.grid_cell(origin, h, ijk)` is cell `ijk` of a regular grid, and
`corner_vertex_ids(ijk, dims)` gives the ids of its 8 corners (order `n = i + 2j + 4k`)
in the grid's `(nx+1)(ny+1)(nz+1)` vertex array. Neighbouring cells that share their
corner values meet without a seam (`test_box.py` checks that the field agrees
exactly on the common face). `code/grid` uses the same numbering.

## Second optimizer: inside/outside classification (`classify_optimize.py`)

Instead of distances to the target surface, points are sampled in the cube and
labelled by the analytic target g: `y = +1` outside (g > 0), `y = -1` inside. The
cube is right at x when `sign(f(x)) == y`. The reward is

    R = mean_x  y · sign(f(x))   =  2 · accuracy − 1

R is a step function of the values. A small change almost never flips a sample,
so its gradient is zero almost everywhere and Adam cannot use it directly. There
are three ways around that:

| `--method` | what it optimizes | how |
|---|---|---|
| `logistic` (default) | a smooth version of the reward: `softplus(−y f / τ)` + weight decay | Adam on its gradient. This is logistic regression; for the 8-value cube the features are the 8 trilinear weights, so the loss is **convex**: one optimum, reached from zero values. |
| `es` | the true ±1 reward | evolution strategies (REINFORCE / policy gradient on the values): sample `θ = μ + σε` in ± pairs, rank the rewards, step μ along `Σ u_i ε_i / (pop σ)` with Adam. No derivative of f is used. |
| `grpo` | the true ±1 reward | Group Relative Policy Optimization (as in DeepSeek-R1 training): a Gaussian policy N(μ, diag σ²) over the values, with **μ and σ both learned**. Each iteration samples a group of G value vectors on the same batch of points, takes advantages `A_i = (R_i − mean R) / std R` (no critic), and runs `--grpo-epochs` Adam steps on the PPO clipped surrogate `min(r_i A_i, clip(r_i, 1±ε) A_i)`, optionally minus β·KL(π‖π_ref). |

Samples (`--sampling`):

* `random` (default): a fresh batch every step. Most points lie near the target
  surface (surface points moved along the normal by N(0, band²)), and `--uniform`
  (25 %) lie anywhere in the cube. Far from the surface everything is classified
  correctly and carries no information. The iterate with the best reward on a
  fixed set of random validation points is kept.
* `lattice`: the same regular `k × k × k` lattice every step (`--k`, default 10,
  i.e. 1000 points). By default the points are cell centres, `(i + ½)/k`, all
  strictly inside the cube. With `--lattice-faces` they are `i/(k−1)`, which puts
  points on the faces, edges and corners too. The reward is the accuracy on
  exactly these points, and the iterate kept is the best on the lattice. The
  report also gives the accuracy on independent random points, to show how well
  the lattice fit generalises.

```bash
python code/optimization/classify_optimize.py --target "sphere flank"                      # logistic
python code/optimization/classify_optimize.py --target "sphere flank" --method es
python code/optimization/classify_optimize.py --target "torus inner flank" --method grpo
python code/optimization/classify_optimize.py --target "sphere cap" --chamfer-iters 200      # then chamfer refinement
python code/optimization/classify_optimize.py --target "sphere flank" --sampling lattice     # 10x10x10 = 1000 points
python code/optimization/classify_optimize.py --target "sphere flank" --sampling lattice --k 20 --lattice-faces
python code/visualization/viewer.py --load classify_fit.json                                 # view the result
```

| option | default | meaning |
|---|---|---|
| `--iters` | 400 | steps (ES: generations) |
| `--sampling` | random | `random` (fresh batches) or `lattice` (fixed k³ points) |
| `--k`, `--lattice-faces` | 10, off | lattice: points per axis; include the faces, edges and corners |
| `--batch` | 4000 | random: labelled samples per step |
| `--band`, `--uniform` | 0.1, 0.25 | random: near-surface jitter (for a cube of width 2, scales with the cube), fraction of uniform samples |
| `--lo`, `--size` | -1 -1 -1, 2 | the cube |
| `--tau`, `--weight-decay` | 0.02, 1e-3 | logistic: soft-boundary width, scale of the field |
| `--pop`, `--sigma` | 32, 0.1 | ES: perturbations per generation, perturbation size (decays to 0.02) |
| `--group`, `--grpo-epochs`, `--clip` | 32, 4, 0.2 | GRPO: group size G, Adam steps per group, ratio clipping ε |
| `--grpo-sigma`, `--grpo-lr` | 0.2, 0.02 | GRPO: initial policy std (then learned), learning rate for μ and log σ |
| `--kl-coef`, `--ref-every` | 0, 10 | GRPO: β of KL(π‖π_ref), and how often π_ref is refreshed |
| `--chamfer-iters` | 0 | refine afterwards with the chamfer optimizer of `optimize.py` |

The report gives the validation accuracy (overall, and within 0.05 of the
surface), the same RMS distance as the other optimizers, and the number of
pieces of the fitted surface.

**GRPO settings.** The KL term is off by default. In LLM training it keeps the
model close to the pretrained one. Here the policy has to travel far from its
start, and β = 0.04 (the DeepSeek value) made the fits worse. Without it, the
reference policy plays no role. The learned σ shrinks by itself (0.2 → about
0.003 over a run), which replaces the fixed σ schedule of ES.

### Results (default settings, about 1–2 s per run)

RMS distance to the target (cube width 2); the last column is the chamfer
optimizer (`optimize.py`) from the tables above.

| target | logistic | ES | GRPO | chamfer |
|---|---|---|---|---|
| sphere flank | 8.9e-3 | 9.5e-3 | 9.5e-3 | 8.4e-3 |
| large sphere flank | 3.7e-3 | 3.8e-3 | 3.7e-3 | 3.5e-3 |
| torus outer flank | 8.8e-3 | 9.0e-3 | 9.3e-3 | 8.5e-3 |
| torus inner flank | 7.0e-3 | 6.7e-3 | 7.4e-3 | 6.7e-3 |
| sphere cap | 8.8e-2 | 1.0e-1 | 1.1e-1 | 8.7e-2 |
| large sphere cap | 1.1e-2 | 1.0e-2 | 9.8e-3 | 9.4e-3 |
| torus outer | 1.1e-1 | 1.2e-1 | 1.1e-1 | 9.8e-2 |
| torus inner | 1.3e-1 | 1.5e-1 | 1.4e-1 | 1.3e-1 |

* All three methods match the chamfer optimizer, starting from zero values, with
  no closed-form start and no mesh.
* ES and GRPO need no derivatives and do about as well as logistic.

### Lattice sampling (`--sampling lattice`)

RMS distance to the target, with random sampling for comparison (from the table
above). The lattice is 10³ = 1000 or 20³ = 8000 cell-centre points.

| target | logistic k=10 / 20 | ES k=10 / 20 | GRPO k=10 / 20 | random (logistic / ES / GRPO) |
|---|---|---|---|---|
| sphere flank | 1.5e-2 / 1.1e-2 | 1.1e-2 / 9.7e-3 | 1.0e-2 / 1.1e-2 | 8.9e-3 / 9.5e-3 / 9.5e-3 |

* At k = 10 nearly every lattice point ends up on the right side, but the surface
  is still free to move anywhere between neighbouring lattice points, about 0.2
  apart. So the RMS is up to 1.7× that of random near-surface sampling (logistic).
* k = 20 closes part of that gap, at about 3× the time for ES and GRPO.

## Grids of cubes (`code/grid`)

A grid of `nx × ny × nz` cubic cells of edge `h` over a box, with the numbers stored
at the `(nx+1)(ny+1)(nz+1)` grid vertices. Every cell is a cube as above, and
neighbouring cells share the numbers on their common corners, so the field is
continuous over the whole grid and the surface can close: a sphere, an ellipsoid or a
torus comes out as one watertight mesh. The single-cube code above is unchanged and
keeps its own tests; a `1 × 1 × 1` grid is exactly the 8-value cube (tested).

```bash
python code/grid/grid_optimize.py --function sphere_medium                 # fit: scalar grid, 8³ cells
python code/grid/grid_optimize.py --function torus_medium --model warp     # value + displacement per vertex
python code/grid/grid_optimize.py --points my_scan.ply --dims 16           # a point cloud
python code/visualization/grid_viewer.py                                   # look at examples and fits
python code/tests/test_grid.py
```

### Two models (`--model`)

| model | numbers per vertex | field |
|---|---|---|
| `scalar` (default) | 1: a value `G` | `f(x) = Σ_n w_n(t) G[corner n of x's cell]`, trilinear in every cell |
| `warp` | 4: a value `G` and a 3D displacement `D`, in cell units | `x' = x + h Σ_n w_n(t) D[corner n]`, then `f(x) = G(x')`, the scalar grid at `x'` |

`t ∈ [0, 1]³` are the local coordinates of `x` in its cell and `w_n` the trilinear
weights, with the corner order `n = i + 2j + 4k` used everywhere.

* The displacement and `G` are both interpolated over the whole grid, so the warp
  field is continuous too. `D = 0` gives the scalar model. On a single cell the warp
  model has 8 values + 8 displacements = 32 numbers.
* `x'` may land in another cell. It is clamped to the grid domain: `G` is only
  defined on the grid, and extending a boundary cell's polynomial outwards made small
  spurious surface pieces at the domain faces.
* Gradients with respect to `x` are analytic (`∇f = Jᵀ ∇G(x')`, with
  `J = I + h ∂D/∂x`), and everything is differentiable in the numbers.

### Meshing (`grid_marching.py`)

Every cell is split into `2^depth` sub-cells per axis (`--depth`, default 2), and the
field is evaluated once at every lattice point, so neighbouring cells always agree.
The sub-cells whose corners change sign are triangulated with the case table, face
decider and welding of the single-cube mesher, and one vertex is placed on each
crossed lattice edge:

* **scalar**: exact, because the field is linear along a lattice edge;
* **warp**: found by bisection, then written as `t = t* − f/f_t` with `f_t`
  detached, so the vertices are differentiable (implicit-function derivative).

The mesh is watertight wherever the surface is inside the grid. `mesh_check` checks
it: every edge is shared by two consistently oriented triangles, except edges on the
domain faces where a surface leaves the grid.

### Targets (`targets.py`)

`--function FILE` or `--points FILE`. Wherever a target is asked for, the name of an
example (below) works as well as a path.

* **Function file**: a Python file defining `g(X)` for an `(..., 3)` NumPy array,
  negative inside, and optionally a display name:

  ```python
  import numpy as np
  NAME = "my sphere"

  def g(X):
      return np.linalg.norm(X - np.array([0.1, 0.0, 0.0]), axis=-1) - 0.5
  ```

  It is meshed on a fine lattice over the domain, sampled area-uniformly, and the
  samples are projected onto `g = 0`, as for the single cube.
* **Point cloud**: `.xyz`, `.txt`, `.csv`, `.pts` (columns `x y z [nx ny nz]`),
  `.npy`, `.ply` (ASCII or binary) or `.obj` (vertices and vertex normals). Without
  normals they are estimated: PCA over the 16 nearest neighbours, oriented
  consistently by propagating over the neighbour graph, and outward at the point of
  largest x of each piece.
* **Inside/outside** (for the classification methods): the sign of `g`, or for a
  cloud the side of the nearest point's tangent plane.
* **Distances** (for the errors): `|g| / |∇g|`, or for a cloud the distance to the
  nearest point's tangent plane, unless the offset along that plane is more than 4
  point spacings (beyond the edge of an open cloud), then to the point itself. A fit
  from the sphere cloud measures the same against the cloud as against the exact
  sphere (2.9e-3). With a 2-spacing limit it measured 6.4e-3: 6 % of the points of a
  perfect surface fall in a gap between random samples that wide.

**The grid domain.** For a function, `[-1, 1]³` or `--lo X Y Z --size S`. For a
point cloud, `[-1, 1]³` if the cloud lies inside it, otherwise its bounding box plus
one cell of margin (`--fit-domain` always does that). `--dims n` (or `nx ny nz`) sets
the cells per axis; cells are always cubes.

### Examples (`examples/`)

Nine functions in `examples/functions/`, and the same shapes as point clouds in
`examples/point_clouds/` (5000 points each, with normals, written by
`make_examples.py`), all inside `[-1, 1]³`:

| examples | sizes |
|---|---|
| `sphere_small`, `sphere_medium`, `sphere_large` | radius 0.3, 0.55, 0.8 |
| `ellipsoid_small`, `ellipsoid_flat`, `ellipsoid_long` | semi-axes (0.4, 0.25, 0.18), (0.8, 0.5, 0.25), (0.85, 0.3, 0.3); the small and long ones rotated |
| `torus_small`, `torus_medium`, `torus_large_thin` | major / tube radius 0.35 / 0.12, 0.55 / 0.2, 0.75 / 0.12; the small and thin ones rotated |

Two more point clouds test harder input: `torus_medium_no_normals` (positions only,
the normals are estimated) and `sphere_medium_noisy` (Gaussian noise of std 0.005,
no normals).

### Fitting (`grid_optimize.py`)

`--method chamfer` (default) has three stages:

1. **Closed-form start** for `G`: least squares on `f(q) = 0` and `h ∇f(q) = n` at
   the target points `(q, n)`, a small smoothness term (second differences of `G`),
   and a pull of every vertex towards its estimated signed distance (distance to the
   nearest target point, with the target's inside/outside sign). The pull keeps the
   vertices far from the target well defined. It is solved for each weight in
   `--anchor-weights` (0.03 and 0.3), and the result with the fewest surface pieces,
   then the lowest RMS, is kept. 0.03 is best for shapes several cells thick, 0.3 for
   thin ones: on the small ellipsoid (0.36 thick, 1.4 cells) 0.03 gave 3 pieces at
   RMS 6e-2, 0.3 one piece at 2.2e-2.
2. **Sampson fit** (L-BFGS, no mesh) of `f(q)² / |∇f(q)|²`: the scalar grid first,
   then for the warp model `G` and `D` together, plus `--disp-penalty · mean(D²)`.
3. **Chamfer refinement through marching cubes** (Adam, `--iters`): mesh samples to
   the target's tangent planes and target points to the mesh triangles, so the loss
   has no sampling floor. The best iterate is kept.

After stage 1, a result replaces the current one only if it lowers the RMS **without
adding surface pieces**. Small detached pieces barely change the RMS but are wrong;
without this rule the warp Sampson fit of the small ellipsoid "improved" the RMS
while making 9 pieces.

`--method logistic | es | grpo` are the classification optimizers of
`classify_optimize.py`, with the points labelled by the target. They start from each
vertex's own label (±1), and for the warp model the scalar grid is trained first.
On grids ES and GRPO use learning rate 0.01: with hundreds of numbers, most of whose
estimated gradients are noise, a larger rate becomes a random walk that flips signs
and creates extra pieces. `--chamfer-iters N` refines the result with stage 3.

```bash
python code/grid/grid_optimize.py --list                                    # the example targets
python code/grid/grid_optimize.py --function sphere_medium                  # scalar, 8³ cells, chamfer
python code/grid/grid_optimize.py --function torus_small --model warp --dims 12
python code/grid/grid_optimize.py --function my_shape.py --lo -2 -2 -2 --size 4
python code/grid/grid_optimize.py --points ellipsoid_long --dims 10
python code/grid/grid_optimize.py --points my_scan.ply --fit-domain          # grid around the cloud
python code/grid/grid_optimize.py --points torus_medium_no_normals           # normals estimated
python code/grid/grid_optimize.py --function sphere_small --method logistic --chamfer-iters 100
python code/grid/grid_optimize.py --function sphere_small --method grpo --dims 4
```

| option | default | meaning |
|---|---|---|
| `--function`, `--points` | | the target: a file, or an example name |
| `--model` | scalar | `scalar` or `warp` |
| `--dims` | 8 | cells per axis: `n`, or `nx ny nz` |
| `--lo`, `--size`, `--fit-domain` | | the grid domain (see Targets) |
| `--depth` | 2 | marching-cubes subdivision of each cell |
| `--method` | chamfer | `chamfer`, `logistic`, `es`, `grpo` |
| `--anchor-weights`, `--normal-weight`, `--smooth-weight` | 0.03 0.3, 1, 0.01 | chamfer stage 1 |
| `--sampson-iters` | 100 | chamfer stage 2 |
| `--iters`, `--lr` | 300, 1e-3 | chamfer stage 3 |
| `--disp-penalty` | 1e-3 | warp: λ · mean(D²), D in cells (also used by the classifiers) |
| `--samples` | 3000 | target points used by the optimizers |
| `--class-iters`, `--class-lr`, `--sampling`, … | 400, … | classification: the options of `classify_optimize.py` |
| `--chamfer-iters` | 0 | classification: chamfer refinement afterwards |
| `--out` | | the fit file; default `data/fits/grid/<target>__<model>__<method>__<cells>.json`, with `_cloud` after the target name for a point cloud |

The report gives the two-sided RMS distance (exact point-to-mesh distances), the
maxima in both directions, the RMS in cells, the number of surface pieces, the
watertightness check and, for warp, the largest displacement.

### Viewing

```bash
python code/visualization/grid_viewer.py                                   # examples and fits in the combo box
python code/visualization/grid_viewer.py --function torus_medium --dims 6
python code/visualization/grid_viewer.py --points sphere_medium_noisy
python code/visualization/grid_viewer.py --load sphere_medium__warp__chamfer__8.json
python code/visualization/grid_viewer.py --load torus_medium_cloud__scalar__chamfer__8.json
python code/visualization/grid_overview.py                                 # images of all fits
```

* The combo box lists the example functions and point clouds and every grid fit in
  `data/fits/grid` and the current folder.
* The target is drawn in transparent orange (a mesh, or points for a cloud), the
  fitted surface in blue (inside face orange). Also drawn: the domain box, optionally
  every cell edge, and for warp fits the displacement arrows `h · D` at the vertices.
* With an example selected, the cells-per-axis slider and the model choice set up a
  grid, and **Fit (chamfer)** fits it from the window (20–60 s) and saves it to
  `data/fits/grid`. Sliders for the meshing depth and the iso level, and the error
  panel, work for any fit.
* `grid_overview.py` renders the fits in `data/fits/grid` without a window, into
  `overview_functions.png` and `overview_point_clouds.png` there: a row per target,
  the target first, then every fit of it.

### All examples (`fit_examples.py`)

```bash
python code/grid/fit_examples.py                         # 9 functions x (scalar, warp), 8³ cells: about 10 min
python code/grid/fit_examples.py --points                # the 11 point clouds: about 13 min
python code/grid/fit_examples.py --dims 16 --examples torus_small ellipsoid_small   # 1-4 min per fit
python code/grid/fit_examples.py --method logistic
```

**Results** (chamfer, depth 2, default settings). RMS distance in world units
(the domain is 2 wide), and in cells in parentheses (cell 0.25 at 8³, 0.125 at
16³). All 80 fits are one closed, watertight piece, like their targets. A fit takes
20–70 s at 8³ and 1–4 min at 16³ (warp about twice as long as scalar), on 2 CPU cores.

Functions:

| example | scalar 8³ | warp 8³ | scalar 16³ | warp 16³ |
|---|---|---|---|---|
| `ellipsoid_flat` | 3.3e-03 (0.013) | 1.5e-03 (0.006) | 1.0e-03 (0.008) | 8.8e-04 (0.007) |
| `ellipsoid_long` | 8.5e-03 (0.034) | 4.5e-03 (0.018) | 2.0e-03 (0.016) | 1.3e-03 (0.010) |
| `ellipsoid_small` | 1.1e-02 (0.043) | 5.7e-03 (0.023) | 4.1e-03 (0.033) | 8.1e-04 (0.006) |
| `sphere_large` | 1.2e-03 (0.005) | 1.1e-03 (0.005) | 8.1e-04 (0.006) | 7.7e-04 (0.006) |
| `sphere_medium` | 2.8e-03 (0.011) | 1.8e-03 (0.007) | 1.0e-03 (0.008) | 9.2e-04 (0.007) |
| `sphere_small` | 5.3e-03 (0.021) | 3.5e-03 (0.014) | 2.0e-03 (0.016) | 1.3e-03 (0.010) |
| `torus_large_thin` | 1.2e-02 (0.047) | 3.4e-03 (0.013) | 2.3e-03 (0.018) | 1.4e-03 (0.011) |
| `torus_medium` | 3.7e-03 (0.015) | 2.6e-03 (0.010) | 1.7e-03 (0.014) | 1.2e-03 (0.009) |
| `torus_small` | 2.4e-02 (0.098) | 4.0e-03 (0.016) | 3.3e-03 (0.026) | 1.8e-03 (0.014) |

Point clouds (5000 points each):

| point cloud | scalar 8³ | warp 8³ | scalar 16³ | warp 16³ |
|---|---|---|---|---|
| `ellipsoid_flat` | 3.2e-03 (0.013) | 2.4e-03 (0.009) | 1.1e-03 (0.009) | 9.6e-04 (0.008) |
| `ellipsoid_long` | 8.4e-03 (0.034) | 4.5e-03 (0.018) | 2.0e-03 (0.016) | 1.3e-03 (0.010) |
| `ellipsoid_small` | 1.0e-02 (0.042) | 5.6e-03 (0.022) | 4.0e-03 (0.032) | 2.3e-03 (0.019) |
| `sphere_large` | 1.1e-03 (0.004) | 1.0e-03 (0.004) | 7.3e-04 (0.006) | 6.4e-04 (0.005) |
| `sphere_medium` | 2.8e-03 (0.011) | 1.8e-03 (0.007) | 1.1e-03 (0.009) | 9.6e-04 (0.008) |
| `sphere_medium_noisy` | 6.1e-03 (0.024) | 4.8e-03 (0.019) | 4.7e-03 (0.038) | 4.6e-03 (0.037) |
| `sphere_small` | 5.2e-03 (0.021) | 3.5e-03 (0.014) | 2.0e-03 (0.016) | 1.2e-03 (0.009) |
| `torus_large_thin` | 1.2e-02 (0.047) | 3.3e-03 (0.013) | 2.5e-03 (0.020) | 1.1e-03 (0.009) |
| `torus_medium` | 3.6e-03 (0.015) | 2.5e-03 (0.010) | 1.7e-03 (0.013) | 1.1e-03 (0.009) |
| `torus_medium_no_normals` | 3.7e-03 (0.015) | 2.6e-03 (0.010) | 1.7e-03 (0.014) | 1.2e-03 (0.010) |
| `torus_small` | 2.5e-02 (0.098) | 1.4e-02 (0.055) | 3.9e-03 (0.031) | 2.3e-03 (0.018) |

* **16³ against 8³.** Halving the cell size cuts the error 1.5–4× on the shapes that
  8³ already resolves, and 2–7× on the thin ones (small ellipsoid, small torus, thin
  torus), which are crude with the scalar model at 8³: the small torus's tube radius
  (0.12) is under half a cell there.
* **Warp against scalar.** At 8³ warp is better on every example: 1.0–1.6× on the
  spheres and the medium torus, about 2× on the ellipsoids, 3.5–6× on the two thin
  tori. At 16³ the gap shrinks to 1.05–1.8× (5× on the small ellipsoid). A finer grid
  still helps more than the warp: scalar 16³ (4913 numbers) beats warp 8³ (2916
  numbers) on every example.
* **Point clouds fit about as well as the functions** they were sampled from, also
  with estimated normals (`torus_medium_no_normals`). The noisy cloud (std 0.005)
  stays at about 4.6e-3 even at 16³: the error is measured to the noisy points, so
  the noise is a floor.
* **Open issue: warp fits of some point clouds.** Three are 1.6–3× behind the same
  shape's function fit: the small torus at 8³ (1.4e-2 against 4.0e-3), the flat
  ellipsoid at 8³ (2.4e-3 against 1.5e-3) and the small ellipsoid at 16³ (2.3e-3
  against 8.1e-4). In the two checked (small torus 8³, small ellipsoid 16³) the warp
  Sampson stage was rejected, so the refinement started from the scalar start. For the
  small torus the rejected result had a tiny detached piece (16 triangles). A possible
  fix is to also refine from the rejected result, or to remove tiny pieces before
  comparing.

`grid_overview.py` draws all of these; `overview_functions.png` and
`overview_point_clouds.png` in `data/fits/grid` show them next to their targets.
