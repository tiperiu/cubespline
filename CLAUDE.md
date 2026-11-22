# cubespline: notes for Claude Code

Research code: trilinear cubes ("cube spline"), iso-surface extraction with subdivided
marching cubes, and optimizing the stored values so the surface matches a target.
`README.md` is the full documentation (models, optimizers, options, results tables);
read the relevant section before changing a part of the code.

## Layout and running

- `code/core` (single cube: `cube.py`, `marching.py`, `box.py`, `geometry.py`, `paths.py`),
  `code/optimization` (single-cube optimizers), `code/grid` (grids of cubes),
  `code/visualization` (Polyscope viewers, `grid_overview.py` renders images without a
  window), `code/tests`.
- `examples/functions/*.py` (each defines `g(X)`, negative inside, optional `NAME`) and
  `examples/point_clouds/*.xyz` (written by `code/grid/make_examples.py`).
- Run scripts from the project folder, e.g. `python code/grid/grid_optimize.py ...`.
  Each script puts `code/` on `sys.path` itself; modules import as packages
  (`from core.cube import Cube`, `from grid.grid import Grid`).
- Tests: `python -m pytest code/tests -q` (52 tests, about 1 min);
  `python code/tests/test_grid.py` for the grid only (about 25 s).
- Python: use `.venv/bin/python` (or activate `.venv`); the `python` on PATH is miniconda
  without torch.
- Outputs: single-cube fits go to `data/fits/`, grid fits to `data/fits/grid/`
  (`<target>[_cloud]__<model>__<method>__<cells>.json`), overview images there too.

## Conventions

- Corner order `n = i + 2j + 4k`, `(i, j, k)` along `(x, y, z)`. Inside is `f < level`
  (default 0). Everything in float64.
- Grid vertex id `a + (nx+1) * (b + (ny+1) * c)`; cells are cubic (edge `h`).
- Grid models: `scalar` (value G per vertex) and `warp` (G plus a 3D displacement D per
  vertex, in cell units): `f(x) = G(x + h D(x))`, the warped point clamped to the grid.
- Errors are reported as RMS distance in world units and in cells, plus the number of
  surface pieces and a watertightness check.

## Decisions (keep them)

- The 16-value single cube (8 offsets along the cube diagonals + 8 values) was removed
  on purpose. Do not reintroduce it. A single cube with displacements is a 1x1x1 warp
  grid (32 numbers). Old 16-value fits in `data/` no longer load.
- Grid chamfer fit: closed-form start for several `--anchor-weights` (0.03, 0.3), keep
  the one with the fewest pieces, then the lowest RMS; after that a result is kept only
  if it lowers the RMS without adding surface pieces.
- ES and GRPO on grids use learning rate 0.01 (larger values flip signs of noisy
  vertices and create extra pieces); classification starts from the vertex labels.
- Point-cloud distance: to the nearest point's tangent plane unless the offset along
  the plane exceeds 4 point spacings (smaller limits overstate the error).
- Warp Sampson stage (chamfer stage 2): sign term `--shell-weight 1e-2`, margin 1, at
  points 0.25-1.5 cells off the target, and tight L-BFGS tolerances. Without it D makes
  wrong-sign bubbles off the surface, the stage is rejected, and the fit depends on the
  seed and machine (README, "The warp Sampson stage"). Refining from the rejected
  result was tried and does not help.
- GRPO: the importance ratio is capped at 3 (`GRPOPolicy.DUAL_CLIP`, dual-clip PPO).
  The log-ratio is a sum over all the numbers; uncapped, `exp` overflowed on warp 16^3
  grids (NaN) and single samples dominated the gradient (README, "The ratio cap").

## Working with Tiberiu

- Commit only when asked, to `main`, and only code, README and examples: not `data/` or
  `docs/`. He pushes to GitHub (tiperiu/cubespline) himself.
- On his Mac he uses MacPorts, not Homebrew.
- `docs/cubespline_methods.docx` (not in git) still describes the removed 16-value cube.

## Open issues

- The scalar Sampson step adds pieces and is rejected for most shapes; the sign term
  (above) is not used there yet. Tight tolerances alone already help one case (torus
  medium cloud 16^3: L-BFGS stopped after one step). Changing it changes the scalar
  columns of the README tables.
- The keep rule counts pieces at depth 2, the final report at depth 3; small pieces
  can show only at depth 3.
- GRPO steps far outside the PPO trust region: approximate KL 0.3-0.5 after one
  epoch on the 8-value cube, log-ratios of tens to hundreds on grids. A KL target or a
  learning rate that shrinks with the number of values is untested; either would also
  change the single-cube results.
- The fits in `data/fits/grid` and its overview images are from before the sign term
  (2026-11-22); the README warp columns are newer.
