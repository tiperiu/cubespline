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
- Tests: `python -m pytest code/tests -q` (55 tests, about 1 min);
  `python code/tests/test_grid.py` for the grid only (about 30 s).
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
- No normals (2026-11-23, Tiberiu's choice): point clouds are positions only (normals in
  files are ignored, none are estimated) and no stage uses normals, for functions
  either. Function targets keep the exact distance `|g| / |grad g|` (and `sign g`) as
  the reference the cloud fits are judged by (Tiberiu, 2026-11-23). Inside/outside of a cloud: `grid/inside_outside.py`, a 26-ray parity vote
  (default, shell 3 point spacings) or a flood fill (shell 5; it leaks at 3). Distances
  to a cloud are to the nearest point, also in the error report, which therefore has a
  floor `sqrt(area / (2 pi N))` (printed as `rms_floor`). Do not bring back tangent
  planes or normal estimation without asking.
- Chamfer stages 1-2 use 6000 labelled points 0.25-1.5 cells off the target (random
  directions, half inside, half outside): stage 1 fits `f = y d` there instead of
  `h grad f = n`; stage 2 (scalar and warp Sampson) has the sign term
  `--shell-weight 1e-2` on the first-order distance, `y f / (h |grad f|) >= d` (margin
  1), and tight L-BFGS tolerances. Without the sign term the field makes wrong-sign
  bubbles off the surface and the stage is rejected (README, "The warp Sampson stage");
  on `f` itself (`y f >= d`) L-BFGS scaled G up to ~1e4 and made extra pieces (test
  `test_scalar_sampson_keeps_the_scale`). Margins 0.5 and 0.25 broke on thin shapes.
  Refining from a rejected result was tried and does not help.
- Keep rule for point clouds: its RMS leaves out the floor (fit -> target counts only
  beyond 3 point spacings, `KEEP_SPACINGS`); with the full point-to-point distance the
  floor's sampling noise decided keep/reject.
- GRPO: the importance ratio is capped at 3 (`GRPOPolicy.DUAL_CLIP`, dual-clip PPO).
  The log-ratio is a sum over all the numbers; uncapped, `exp` overflowed on warp 16^3
  grids (NaN) and single samples dominated the gradient (README, "The ratio cap").

## Working with Tiberiu

- Commit only when asked, to `main`, and only code, README and examples: not `data/` or
  `docs/`. He pushes to GitHub (tiperiu/cubespline) himself.
- On his Mac he uses MacPorts, not Homebrew.
- `docs/cubespline_methods.docx` (not in git) still describes the removed 16-value cube.

## Open issues

- Without normals the fits are 1.0-1.24x the error of the fits with normals (geometric
  means per setup, README "Without normals"); worst are the spheres at scalar 8^3
  (1.25-1.47x) and the small sphere and ellipsoid at warp 16^3 (1.55-1.63x). The sign
  term's margin 1 bends small curved shapes; smaller margins made extra pieces.
- Logistic / ES / GRPO on grids were not rerun since the change to no normals (their
  labels now come from inside_outside.py); the README grid GRPO numbers are older.
- The keep rule counts pieces at depth 2, the final report at depth 3; small pieces
  can show only at depth 3.
- GRPO steps far outside the PPO trust region: approximate KL 0.3-0.5 after one
  epoch on the 8-value cube, log-ratios of tens to hundreds on grids. A KL target or a
  learning rate that shrinks with the number of values is untested; either would also
  change the single-cube results.
- The fits in `data/fits/grid` and its overview images are from before the sign term
  (2026-11-22) and use normals; the README tables are newer (2026-11-23, no normals).
