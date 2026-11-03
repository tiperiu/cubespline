# cubespline

A single cube with 8 corner values. Inside the cube the field is the trilinear
interpolation of the values; its level set `f = level` (default 0) is a surface.
This repo extracts that surface with subdivided marching cubes, views it with
Polyscope, and (next step) optimizes the 8 values so the surface matches a target.

| file | what it does |
|---|---|
| `cube.py` | `Cube` (PyTorch `nn.Module`): 8 values as a parameter, `evaluate`, `basis`, `gradient`, presets |
| `marching.py` | octree subdivision with exact pruning + marching cubes → `V` (positions), `F` (connectivity) |
| `viewer.py` | Polyscope viewer: cube, corner values, mesh, sliders for the values / depth / level |
| `test_cubespline.py` | watertightness, orientation, exactness, pruning and gradient tests |
| `optimize.py` | *(part 3, next)* fit the 8 values to a target surface |

## Setup

```bash
cd cubespline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_cubespline.py        # quick self-check
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

## Part 3 plan: fitting the 8 values

The mesher is built for this step. `extract_topology` returns the connectivity
plus, for each vertex, the endpoints of its lattice edge and their trilinear basis
rows `Wa`, `Wb`. Then

```
t = (Wa·c − level) / ((Wa − Wb)·c),     V = pa + t (pb − pa)
```

is a smooth PyTorch function of the values `c`. An optimization loop is:

```python
for it in range(n_iters):
    if it % k == 0:
        topo = extract_topology(cube, depth)          # discrete step, no gradients
    V = topo.vertices(cube.values)                    # differentiable in the values
    loss = chamfer(V, target_points)                  # or point-to-mesh distance
    loss.backward(); opt.step(); opt.zero_grad()
```

Things to handle in `optimize.py`:

* **Scale invariance.** `c` and `s·c` (s > 0) give the same surface, so the
  loss is flat along `c`. Fix it by normalizing (`‖c‖ = const`, or project the
  gradient), otherwise the optimizer drifts in scale.
* **Target as a function.** If the target is an implicit `g(x)`, sample points on
  `g = 0`; then fitting `f_c(x_i) = 0` is linear in `c` (smallest singular
  vector of the basis matrix), which gives a strong initialization before the
  mesh-based refinement.
* **Target as a mesh.** Sample points on it and use a two-sided chamfer
  distance; the target-to-surface direction keeps the surface from shrinking
  away from parts of the target.
* **Expressiveness.** 8 values span a small family (planes, saddles, bowls,
  xyz-type surfaces). The fit is the best trilinear approximation; richer
  targets need more cubes (a grid of cubes sharing corner values, the natural
  "cube spline" extension, which this code is structured to grow into).
