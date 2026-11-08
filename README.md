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
| `classify_optimize.py` | fit either cube by inside/outside classification: logistic loss + Adam, evolution strategies, or GRPO (reward only) |
| `test_classify.py` | tests for the classification optimizers |
| `box.py` | `Box`: where a cube sits (`lo`, `size`), lattices, `--lo/--size` options, grid helpers (`grid_cell`, `corner_vertex_ids`) |
| `test_box.py` | tests for cubes at any position and size, and the grid preparation |
| `benchmark_warp.py` | compares every 16-value optimizer (time, own loss, common chamfer distance) |

## Setup

```bash
cd cubespline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_cubespline.py && python test_optimize.py   # quick self-check
python test_warp.py && python test_classify.py && python test_box.py
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

## Cube position and size (`box.py`)

Every cube is an axis-aligned cube `[lo, lo + size]³`, by default `[-1, 1]³`. All
the optimizers and both viewers take `--lo X Y Z` and `--size S`:

```bash
python optimize.py --target "sphere cap" --lo 0 -1 0 --size 1        # a corner octant of the default cube
python warp_optimize.py --target "sphere cap" --lo 0 -1 0 --size 1
python classify_optimize.py --target "sphere cap" --lo 0 -1 0 --size 1
python viewer.py --preset saddle --lo 2 0 0 --size 0.5
```

* **Targets live in world space** and are clipped to the cube. The built-in
  functions were designed around `[-1, 1]³`, so a cube placed elsewhere sees a
  different part of them, or nothing. Then you get the error "no surface inside
  the cube".
* **Values are dimensionless.** The same values give the same shape, moved and
  scaled, in any cube. The 16-value offsets are measured in half-widths: corner n
  moves by `c1[n] · size/2` along its diagonal. For the default cube this is the
  world unit used before, so earlier fits and presets are unchanged.
* **Saved fits** store `"lo"` and `"size"`. Files without them load as the
  default cube. The viewers move the cube, the target and the camera to each
  fit's box. Presets use `--lo/--size`. The current box is shown under the combo
  box.
* **Errors** (RMS, max) are in world units, and as a percentage of the cube width.

Example: the corner octant `--lo 0 -1 0 --size 1` of the sphere cap. The 8-value
fit reaches RMS 1.8e-3 and the 16-value fit 1.1e-3, both well under 1 % of the
cube width. That octant contains only a flank-like part of the cap.

### Preparing for grids of cubes

Not implemented yet. The pieces in place:

* `Box.grid_cell(origin, h, ijk)` is cell `ijk` of a regular grid.
  `corner_vertex_ids(ijk, dims)` gives the ids of its 8 corners (same order
  `n = i + 2j + 4k`) in the grid's `(nx+1)(ny+1)(nz+1)` vertex array. A grid
  stores one value per grid vertex, and a cell uses `values[corner_vertex_ids(ijk, dims)]`.
* The 8-value mesh topology stores each mesh vertex's lattice edge `(a_ijk, b_ijk)`
  at resolution `2^depth`. In a grid, `ijk · 2^depth + a_ijk` gives a global key,
  which joins the meshes of neighbouring cells on their common face.
* **8 values:** neighbouring cells sharing their corner values meet without a
  seam (`test_box.py` checks the field agrees exactly on the common face).
* **16 values:** not yet. Corner offsets move along the cell's own diagonal,
  `e_n = (2i−1, 2j−1, 2k−1)`, which points outward from that cell. A grid
  vertex shared by two cells is therefore moved in different directions by
  each, so the field jumps across the face, even with shared offsets and
  values (by about 0.09 in a random test). Making 16-value grids continuous
  needs a choice of offset rule first. For example, a 3-vector displacement
  per grid vertex instead of a scalar along each cell's diagonal, with the
  displacement kept tangential on the faces.

## Second version: the 16-value cube (`warpcube.py`)

```
values = [ offset c1 (8) | field c2 (8) ]
x'   = x + D(x),   D(x) = (size/2) Σ_n w_n(x) c1[n] e_n      e_n = corner n's diagonal direction (±1, ±1, ±1)
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

## Third optimizer: inside/outside classification (`classify_optimize.py`)

Instead of distances to the target surface, points are sampled in the cube and
labelled by the analytic target g: `y = +1` outside (g > 0), `y = -1` inside. The
cube is right at x when `sign(f(x)) == y`. The reward is

    R = mean_x  y · sign(f(x))   =  2 · accuracy − 1

R is a step function of the values. A small change almost never flips a sample,
so its gradient is zero almost everywhere and Adam cannot use it directly. There
are three ways around that, and all work with the 8-value cube (`--cube 8`,
default) and the 16-value cube (`--cube 16`):

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

16-value cube: the problem is not convex, so stage 1 runs `--starts` random
starts plus the 8-value solution (zero offset) for `--start-iters` steps each.
The best one continues for `--iters` steps.

```bash
python classify_optimize.py --target "sphere flank"                       # 8 values, logistic
python classify_optimize.py --target "sphere flank" --method es
python classify_optimize.py --cube 16 --target "sphere cap"               # 16 values
python classify_optimize.py --cube 16 --target "sphere cap" --method es --iters 600
python classify_optimize.py --cube 16 --target "torus inner" --method grpo
python classify_optimize.py --cube 16 --target "sphere cap" --chamfer-iters 200   # then chamfer refinement
python classify_optimize.py --cube 16 --target "sphere cap" --sampling lattice           # 10x10x10 = 1000 points
python classify_optimize.py --cube 16 --target "sphere cap" --sampling lattice --k 20 --lattice-faces
python warp_viewer.py --load classify_warp_fit.json                       # view (8-value: viewer.py --load classify_fit.json)
```

| option | default | meaning |
|---|---|---|
| `--iters` | 400 | steps (ES: generations); stage 2 for `--cube 16` |
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
| `--starts`, `--start-iters` | 8, 100 | `--cube 16` only: stage 1 |
| `--offset-penalty` | 0.01 | `--cube 16` only: λ · mean(offset²), against corner bubbles (see below) |
| `--chamfer-iters` | 0 | refine afterwards with the chamfer optimizer of `optimize.py` / `warp_optimize.py` |

The report gives the validation accuracy (overall, and within 0.05 of the
surface), the same RMS distance as the other optimizers, and the number of
pieces of the fitted surface.

**Corner bubbles.** Accuracy measures volume, not distance. Without the offset
penalty, ES on the 16-value cube reached 99.3 % accuracy on the sphere flank but
RMS 0.14. A corner offset of 3.9 sends the points near that corner far outside the
cube. The second trilinear field is extrapolated there and changes sign again,
which creates two small closed bubbles. They hold 0.17 % of the volume, so they
cost almost nothing in accuracy, and the chamfer refinement cannot remove a
detached piece. The default penalty λ = 0.01 removes them (RMS 3.2e-3), and the
"surface pieces" line in the report flags any that remain.

**GRPO settings.** The KL term is off by default. In LLM training it keeps the
model close to the pretrained one. Here the policy has to travel far from its
start, and β = 0.04 (the DeepSeek value) roughly doubled the error on the caps.
Without it, the reference policy plays no role. The learned σ shrinks by itself
(0.2 → about 0.003 over a run), which replaces the fixed σ schedule of ES.
Starting σ at 0.1 instead of 0.2 let it narrow too early on torus inner (RMS
8e-2 vs 2e-2).

### Results (default settings, about 1–2 s for 8 values, 5 s logistic / 11–14 s ES and GRPO for 16)

RMS distance to the target (cube width 2); the last column is the chamfer
optimizers (`optimize.py` / `warp_optimize.py`) from the tables above.

| target | 8 logistic | 8 ES | 8 GRPO | 16 logistic | 16 ES | 16 GRPO | chamfer 8 / 16 |
|---|---|---|---|---|---|---|---|
| sphere flank | 8.9e-3 | 9.5e-3 | 9.5e-3 | 5.6e-3 | 3.2e-3 | 7.7e-3 | 8.4e-3 / 2.4e-3 |
| large sphere flank | 3.7e-3 | 3.8e-3 | 3.7e-3 | 3.1e-3 | 3.5e-3 | 2.8e-3 | 3.5e-3 / 7.2e-4 |
| torus outer flank | 8.8e-3 | 9.0e-3 | 9.3e-3 | 4.0e-3 | 7.0e-3 | 6.8e-3 | 8.5e-3 / 3.0e-3 |
| torus inner flank | 7.0e-3 | 6.7e-3 | 7.4e-3 | 4.1e-3 | 6.9e-3 | 5.5e-3 | 6.7e-3 / 3.2e-3 |
| sphere cap | 8.8e-2 | 1.0e-1 | 1.1e-1 | 1.6e-2 | 1.8e-2 | 2.4e-2 | 8.7e-2 / 5.8e-3 |
| large sphere cap | 1.1e-2 | 1.0e-2 | 9.8e-3 | 6.7e-3 | 8.6e-3 | 8.4e-3 | 9.4e-3 / 3.0e-3 |
| torus outer | 1.1e-1 | 1.2e-1 | 1.1e-1 | 3.8e-2 | 4.2e-2 | 4.5e-2 | 9.8e-2 / 1.2e-2 |
| torus inner | 1.3e-1 | 1.5e-1 | 1.4e-1 | 6.0e-2 | 4.7e-2 | 2.0e-2 | 1.3e-1 / 1.2e-2 |

* 8 values: both methods match the chamfer optimizer starting from zero values,
  with no closed-form start and no mesh.
* 16 values: classification gets most of the gain over 8 values, but stays 1.5–4×
  behind the chamfer optimizer on caps and tori. It optimizes sign agreement over
  a band of width ~0.1, not the exact position of the surface.
  `--chamfer-iters` helps a little (sphere cap 1.6e-2 → 1.4e-2); the chamfer
  optimizer's own multi-start finds a better basin.
* ES and GRPO need no derivatives and do about as well as logistic, at 2–3× the
  time for 16 values. No single method wins everywhere: GRPO is best of the
  classifiers on torus inner, ES on the sphere flank, logistic on the caps.

### Lattice sampling (`--sampling lattice`)

RMS distance to the target, with random sampling for comparison (from the table
above). The lattice is 10³ = 1000 or 20³ = 8000 cell-centre points.

| target | cube | logistic k=10 / 20 | ES k=10 / 20 | GRPO k=10 / 20 | random (logistic / ES / GRPO) |
|---|---|---|---|---|---|
| sphere flank | 8 | 1.5e-2 / 1.1e-2 | 1.1e-2 / 9.7e-3 | 1.0e-2 / 1.1e-2 | 8.9e-3 / 9.5e-3 / 9.5e-3 |
| sphere flank | 16 | 1.0e-2 / 6.8e-3 | 8.9e-3 / 8.5e-3 | 1.1e-2 / 8.1e-3 | 5.6e-3 / 3.2e-3 / 7.7e-3 |
| sphere cap | 16 | 1.8e-2 / 3.1e-2 | 2.2e-2 / 1.5e-2 | 4.7e-2 / 2.1e-2 | 1.6e-2 / 1.8e-2 / 2.4e-2 |
| torus inner | 16 | 5.3e-2 / 5.5e-2 | 7.5e-2 / 4.7e-2 | 8.6e-2 / 4.1e-2 | 6.0e-2 / 4.7e-2 / 2.0e-2 |

* At k = 10 nearly every lattice point ends up on the right side: 100 % on the
  sphere flank and cap with 16 values, 98–99 % on torus inner. The surface is
  still free to move anywhere between neighbouring lattice points, about 0.2
  apart. So on flanks the RMS is about 1.5× that of random near-surface
  sampling, and the accuracy on random validation points is lower (95–98 %
  instead of 98 %).
* k = 20 usually closes much of that gap (ES and GRPO on the cap and torus
  inner), but costs about 3× more time for ES and GRPO.
* One 16-value GRPO run at k = 20 (sphere flank) had a small extra surface piece.
  The report flags these.

## Benchmark: all optimizers of the 16-value cube on the three sphere targets

`python benchmark_warp.py` (about 12 min) runs every method with its default
settings on `sphere`, `sphere cap` and `sphere flank`, with 3 seeds each.

* **Own loss** is the objective each method minimises, at its result. It can only
  be compared within a row:
  * Sampson: f²/|∇f|² + normal term;
  * chamfer: the chamfer loss;
  * logistic: the softplus loss on the last batch;
  * ES / GRPO: (1 − R)/2, the misclassification rate of the last population.
* **CD** is the same measure for all methods: the two-sided chamfer distance
  `mean d(p, target)² + mean d(q, fit)²`, in world units² (cube width 2), with
  exact point-to-surface distances.
  * p are 20 000 samples of the fitted mesh at depth 6; q are 20 000 target
    samples that no method was trained on.
  * Point-to-point distances between two sample sets would add a floor of about
    0.008, larger than the differences measured here.
* Values are medians over the 3 seeds; the worst seed is in parentheses.
  `*` marks one seed whose surface had extra pieces.

| method | kind | time (s) | sphere: own loss | sphere: CD ×1e-4 | cap: own loss | cap: CD ×1e-4 | flank: own loss | flank: CD ×1e-4 |
|---|---|---|---|---|---|---|---|---|
| Sampson multi-start (stage 1) | differentiable | 11 | 5.2e-3 | 440 (1400) ** | 2.0e-3 | **0.78** (0.94) | 5.5e-6 | **0.11** (0.12) |
| chamfer (stage 1 + 300 Adam) | differentiable | 35 | 4.1e-2 | 440 (1400) *** | 8.8e-4 | **0.74** (0.80) | 9.8e-4 | **0.11** (0.12) |
| logistic, random | differentiable | 4 | 3.6e-1 | 93 (790) ** | 1.1e-1 | 3.7 (5.0) | 5.6e-2 | 0.71 (0.72) |
| logistic, lattice 10³ | differentiable | 2 | 9.6e-2 | 62 (62) | 2.1e-2 | 11 (15) | 1.1e-2 | 2.0 (2.0) |
| ES, random | reward only | 11 | 1.2e-1 | **26** (27) ** | 5.6e-2 | 6.8 (7.8) | 2.5e-2 | 0.8 (120) * |
| ES, lattice 10³ | reward only | 4 | 2.8e-2 | **23** (23) * | 1.1e-2 | 25 (230) | 4.7e-3 | 1.5 (1.6) |
| GRPO, random | reward only | 15 | 8.8e-2 | 28 (700) * | 3.7e-2 | 12 (29) | 1.9e-2 | 1.3 (1.6) |
| GRPO, lattice 10³ | reward only | 7 | 5.4e-2 | 210 (580) | 6.7e-3 | 43 (45) | 8.6e-4 | 2.3 (2.4) |

(Times are medians over all runs; `sphere` runs take about 1.5× longer for the
chamfer method.)

**What it shows**

* **Cap and flank (the surface fits in one cube).** The surface-based,
  differentiable methods win clearly.
  * The multi-start Sampson fit reaches CD 0.78e-4 / 0.11e-4 (RMS 6e-3 / 2e-3) in
    11 s. The chamfer refinement adds 20–25 s for a small gain (cap 0.78 → 0.74,
    flank unchanged).
  * The classifiers are 5–100× worse in CD. Logistic is the best and fastest of
    them (cap 3.7e-4 in 4 s).
  * ES and GRPO, which only see the reward, are 2–3× slower, worse, and vary
    more between seeds.
  * Lattice sampling is the fastest but loses accuracy: 1000 fixed points
    leave the surface free between them.
* **Whole sphere (a closed surface, not representable by one cube).** Every
  method is poor (RMS ≥ 0.03 on a sphere of radius 0.8), but the order is
  reversed.
  * The Sampson and chamfer fits fall into solutions with extra closed pieces
    in 2–3 of 3 seeds (CD 4e-2 to 1.4e-1).
  * ES is the most reliable (CD about 2.5e-3 for every seed, random or
    lattice). A volume reward penalises wrong inside/outside everywhere in the
    cube, while the surface losses only look near the target samples, and
    bubbles away from them cost little.
* **Own loss vs CD.** Within a method, a lower own loss does not mean a better
  surface. Every lattice variant has a lower own loss than its random version,
  because 1000 fixed points are easier to classify, yet it has a higher CD.
  On the sphere, no chamfer step beat the multi-start result, so keep-best
  returned it unchanged and the two rows have the same CD.
