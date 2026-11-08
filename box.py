"""
box.py -- where a cube sits in space: the axis-aligned cube [lo, lo + size]^3.

Every cube (Cube, WarpCube), target, optimizer run and saved fit carries a Box.
Box() is [-1, 1]^3 (lo = (-1, -1, -1), size 2), the cube used so far. Files saved
before boxes existed have no "lo"/"size" keys and load as Box().

Units
    Corner values are dimensionless: values fitted in one box give the same
    shape, moved and scaled, in any other box. Lengths that belong to the fit
    rather than the world are relative to the cube:
      * warp offsets (warpcube.py) are in half-widths: corner n moves by
        c1[n] * size/2 along its diagonal (for the default cube, world units as before);
      * the --band, --tau and near-surface lengths of classify_optimize.py are
        given for a cube of width 2 and scale with the cube (Box.scale).
    Distances to the target (RMS, max) are reported in world units and as a
    percentage of the cube width. Targets live in world space; a target is
    clipped to the box it is fitted in.

Preparing for grids of cubes (cells sharing corners, not implemented yet)
    * Box.grid_cell(origin, cell_size, ijk) is the box of cell ijk of a regular
      grid that starts at `origin`.
    * corner_vertex_ids(ijk, dims) maps a cell's 8 corners (local order
      n = i + 2j + 4k, as everywhere in this project) to ids in the grid's
      (nx+1)(ny+1)(nz+1) vertex lattice. A grid then stores one value per grid
      vertex, and cell ijk evaluates with values[corner_vertex_ids(ijk, dims)];
      neighbouring cells share the values on their common corners.
    * The marching-cubes topologies keep, per mesh vertex, the integer endpoints
      (a_ijk, b_ijk) of its lattice edge in the cell's 2^depth subdivision. In a
      grid, cell ijk's subdivision starts at ijk * 2^depth, so
      (ijk * 2^depth + a_ijk, ijk * 2^depth + b_ijk) is a global edge key that welds
      the meshes of neighbouring cells along their common face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# corner n = i + 2j + 4k  ->  (i, j, k); same table as cube.CORNER_IJK
_CORNER_IJK = np.array([[n & 1, (n >> 1) & 1, (n >> 2) & 1] for n in range(8)], np.int64)


@dataclass(frozen=True)
class Box:
    lo: tuple = (-1.0, -1.0, -1.0)
    size: float = 2.0

    def __post_init__(self):
        lo = np.broadcast_to(np.asarray(self.lo, np.float64), (3,))
        object.__setattr__(self, "lo", tuple(float(x) for x in lo))
        object.__setattr__(self, "size", float(self.size))
        if not self.size > 0:
            raise ValueError(f"cube size must be positive, got {self.size}")

    # ---------------------------------------------------------------- geometry
    @property
    def lo_np(self) -> np.ndarray:
        return np.array(self.lo)

    @property
    def hi(self) -> tuple:
        return tuple(x + self.size for x in self.lo)

    @property
    def center(self) -> np.ndarray:
        return self.lo_np + self.size / 2

    @property
    def kw(self) -> dict:
        """Keyword arguments for Cube / WarpCube:  Cube(values, **box.kw)."""
        return {"lo": list(self.lo), "size": self.size}

    def scale(self, length: float) -> float:
        """A length given for the default cube (width 2), scaled to this cube."""
        return length * self.size / 2

    def to_local(self, X):
        """World points -> local coordinates in [0, 1]^3 (numpy or torch)."""
        lo = torch.as_tensor(self.lo, dtype=X.dtype, device=X.device) if torch.is_tensor(X) else self.lo_np
        return (X - lo) / self.size

    def to_world(self, U):
        lo = torch.as_tensor(self.lo, dtype=U.dtype, device=U.device) if torch.is_tensor(U) else self.lo_np
        return lo + U * self.size

    def contains(self, X, eps=1e-9) -> np.ndarray:
        X = np.asarray(X)
        return np.all((X >= self.lo_np - eps) & (X <= self.lo_np + self.size + eps), axis=-1)

    def clamp(self, X: torch.Tensor) -> torch.Tensor:
        lo = torch.as_tensor(self.lo, dtype=X.dtype, device=X.device)
        return torch.maximum(torch.minimum(X, lo + self.size), lo)

    def corners(self) -> np.ndarray:
        """(8, 3) corner positions, order n = i + 2j + 4k."""
        return self.lo_np + _CORNER_IJK * self.size

    # ---------------------------------------------------------------- sampling
    def uniform(self, n: int, gen: torch.Generator, dtype=torch.float64) -> torch.Tensor:
        return self.to_world(torch.rand(n, 3, dtype=dtype, generator=gen))

    def lattice(self, k: int, faces: bool = False) -> np.ndarray:
        """Regular k x k x k lattice of points in the cube, (k^3, 3).
        faces=False: cell centres (i + 1/2) / k, all strictly inside.
        faces=True:  i / (k - 1), including points on the faces, edges and corners."""
        if k < 1:
            raise ValueError("lattice k must be >= 1")
        if faces and k > 1:
            t = np.arange(k) / (k - 1)
        else:
            t = (np.arange(k) + 0.5) / k
        U = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
        return self.to_world(U)

    # ---------------------------------------------------------------- grids
    @classmethod
    def grid_cell(cls, origin, cell_size: float, ijk) -> "Box":
        """Box of cell ijk of a regular grid of cubes of edge cell_size starting at origin."""
        return cls(tuple(np.asarray(origin, float) + np.asarray(ijk, float) * cell_size), cell_size)

    # ---------------------------------------------------------------- files / CLI
    def to_dict(self) -> dict:
        return {"lo": list(self.lo), "size": self.size}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Box":
        """Box stored in a fit file; files without one are the default cube."""
        if not d or "lo" not in d:
            return cls()
        return cls(tuple(d["lo"]), d.get("size", 2.0))

    def __str__(self):
        lo = ", ".join(f"{x:g}" for x in self.lo)
        hi = ", ".join(f"{x:g}" for x in self.hi)
        return f"[{lo}] .. [{hi}] (size {self.size:g})"


def corner_vertex_ids(ijk, dims) -> np.ndarray:
    """Ids of cell ijk's 8 corners in the vertex lattice of a grid of dims = (nx, ny, nz)
    cells, (nx+1)(ny+1)(nz+1) vertices numbered a + (nx+1) * (b + (ny+1) * c).
    Corner order n = i + 2j + 4k, as for a single cube."""
    nx, ny, nz = (int(d) for d in dims)
    ijk = np.asarray(ijk, np.int64)
    if np.any(ijk < 0) or np.any(ijk >= (nx, ny, nz)):
        raise IndexError(f"cell {tuple(ijk)} outside a grid of {dims} cells")
    v = ijk + _CORNER_IJK                                  # (8, 3) vertex coordinates
    return v[:, 0] + (nx + 1) * (v[:, 1] + (ny + 1) * v[:, 2])


def add_box_arguments(ap):
    """--lo X Y Z and --size S on an argparse parser."""
    ap.add_argument("--lo", type=float, nargs=3, metavar=("X", "Y", "Z"), default=[-1.0, -1.0, -1.0],
                    help="minimum corner of the cube (default -1 -1 -1)")
    ap.add_argument("--size", type=float, default=2.0, help="edge length of the cube (default 2)")


def box_from_args(args) -> Box:
    return Box(tuple(args.lo), args.size)
