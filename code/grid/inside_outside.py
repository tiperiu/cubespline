"""
inside_outside.py -- inside or outside of a point cloud, from the point positions only
(no normals).

Both methods work on a voxel grid around the cloud (voxel size half a point spacing, at
most --max-voxels per axis):

  1. the shell: every voxel whose centre is within r of a cloud point (r = `radius` point
     spacings, at least 1.5 voxels). Where the gaps between points are narrower than 2r
     the shell is closed.
  2. the voxels off the shell are labelled
       flood  outside if they connect to the border of the voxel grid (6-neighbours), inside
              otherwise. Exact for a closed shell; a hole wider than about 2r lets the
              outside in.
       rays   a vote of 26 rays along the lattice directions (the 6 axes, 12 face and 8
              body diagonals): a ray that enters the shell an odd number of times before it
              leaves the grid votes inside, and the majority wins. A hole misleads only the
              rays that pass through it. A part thinner than 2r is a single shell crossing
              instead of two, so it reads as outside.
  3. the shell voxels take the side that is nearer: the two labelled regions grow into the
     shell one voxel per step, and
         phi = (steps from the inside region) - (steps from the outside region)
     changes sign half-way through the shell, which is on the cloud.

label(X) is the sign of phi at X, interpolated trilinearly: +1 outside, -1 inside. Points
beyond the voxel grid are outside. Near the cloud (within about a point spacing) the side
is only as good as the cloud itself.
"""

from __future__ import annotations

import numpy as np
import torch

METHODS = ("flood", "rays")
DIRS6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
DIRS26 = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def shift(A: np.ndarray, s) -> np.ndarray:
    """B[v] = A[v + s] (zero where v + s is outside the array)."""
    B = np.zeros_like(A)
    src, dst = [], []
    for k, n in zip(s, A.shape):
        k = int(k)
        if abs(k) >= n:
            return B
        src.append(slice(max(k, 0), n + min(k, 0)))
        dst.append(slice(max(-k, 0), n + min(-k, 0)))
    B[tuple(dst)] = A[tuple(src)]
    return B


def dilate6(A: np.ndarray) -> np.ndarray:
    out = A.copy()
    for d in DIRS6:
        out |= shift(A, d)
    return out


class InsideOutside:
    def __init__(self, P, spacing: float, method="flood", radius=3.0, lo=None, hi=None, max_voxels=192):
        if method not in METHODS:
            raise ValueError(f"inside/outside method '{method}': use one of {', '.join(METHODS)}")
        P = np.asarray(P, float)
        lo_c, hi_c = P.min(0), P.max(0)
        if lo is not None:                                   # cover the fitting domain too
            lo_c, hi_c = np.minimum(lo_c, lo), np.maximum(hi_c, hi)
        extent = float((hi_c - lo_c).max())
        v = max(spacing / 2, extent / max_voxels)
        r = max(radius * spacing, 1.5 * v)
        margin = r + 3 * v
        self.method, self.v, self.r, self.spacing = method, v, r, spacing
        self.lo = lo_c - margin
        self.n = tuple(int(k) for k in np.ceil((hi_c - lo_c + 2 * margin) / v).astype(int) + 1)
        shell = self._shell(P)
        free = ~shell
        inside = self._flood(free) if method == "flood" else self._rays(shell, free)
        self.inside_voxels, self.shell_voxels = int(inside.sum()), int(shell.sum())
        self.phi = torch.as_tensor(self._grow(shell, inside), dtype=torch.float32)

    # -- the shell -------------------------------------------------------------
    def _shell(self, P) -> np.ndarray:
        n, v, r = np.array(self.n), self.v, self.r
        k = int(np.ceil(r / v)) + 1
        o = np.stack(np.meshgrid(*[np.arange(-k, k + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        o = o[np.linalg.norm(o, axis=1) * v <= r + 1.8 * v]
        shell = np.zeros(self.n, bool)
        for i in range(0, len(P), 20000):
            Q = P[i:i + 20000]
            base = np.round((Q - self.lo) / v).astype(np.int64)
            idx = base[:, None, :] + o[None, :, :]                       # (m, K, 3)
            centre = self.lo + idx * v
            ok = (np.linalg.norm(centre - Q[:, None, :], axis=-1) <= r) & (idx >= 0).all(-1) & (idx < n).all(-1)
            idx = idx[ok]
            shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return shell

    # -- labelling the voxels off the shell -----------------------------------
    @staticmethod
    def _flood(free) -> np.ndarray:
        out = np.zeros_like(free)
        for ax in range(3):                                   # the border of the voxel grid
            idx = [slice(None)] * 3
            for end in (0, -1):
                idx[ax] = end
                out[tuple(idx)] = free[tuple(idx)]
        count = int(out.sum())
        while True:
            for _ in range(8):
                out = dilate6(out) & free
            c = int(out.sum())
            if c == count:
                return free & ~out
            count = c

    def _rays(self, shell, free) -> np.ndarray:
        votes = np.zeros(self.n, np.uint8)
        longest = max(self.n)
        for d in DIRS26:
            entry = shell & ~shift(shell, tuple(-k for k in d))     # stepping along d into the shell
            odd = shift(entry, d)                                   # crossings at v + d
            m = 1
            while m < longest:                                      # ... and at v + 2d, v + 3d, ...
                odd ^= shift(odd, tuple(m * k for k in d))
                m *= 2
            votes += odd
        # average the votes over the free voxels of each 3x3x3 block: rays that graze the
        # shell flip a few votes next to it
        num = np.zeros(self.n, np.float32)
        den = np.zeros(self.n, np.float32)
        vf, ff = (votes * free).astype(np.float32), free.astype(np.float32)
        for d in DIRS26 + [(0, 0, 0)]:
            num += shift(vf, d)
            den += shift(ff, d)
        return free & (num > (len(DIRS26) / 2) * np.maximum(den, 1))

    # -- the side of the shell voxels -----------------------------------------
    def _grow(self, shell, inside) -> np.ndarray:
        cap = int(np.ceil(2 * self.r / self.v)) + 3
        outside = ~shell & ~inside

        def steps(seed):
            dist = np.full(self.n, cap, np.int16)
            reached = seed.copy()
            dist[reached] = 0
            for s in range(1, cap):
                new = dilate6(reached) & ~reached
                if not new.any():
                    break
                dist[new] = s
                reached |= new
            return dist

        return (steps(inside).astype(np.float32) - steps(outside).astype(np.float32)) * self.v

    # -- queries ------------------------------------------------------------------
    def field(self, X: torch.Tensor) -> torch.Tensor:
        """phi at X (trilinear), positive outside; points beyond the voxel grid get +v."""
        Xf = X.detach().to(torch.float64)
        u = (Xf - torch.as_tensor(self.lo)) / self.v
        n = torch.as_tensor(self.n)
        beyond = ((u < 0) | (u > n - 1)).any(-1)
        i0 = u.floor().long().clamp(torch.zeros(3, dtype=torch.long), n - 2)
        t = (u - i0).clamp(0, 1).to(torch.float32)
        out = torch.zeros(len(Xf), dtype=torch.float32)
        for c in range(8):
            o = torch.tensor([c & 1, (c >> 1) & 1, (c >> 2) & 1])
            w = torch.where(o.bool(), t, 1 - t).prod(-1)
            j = i0 + o
            out += w * self.phi[j[:, 0], j[:, 1], j[:, 2]]
        return torch.where(beyond, torch.full_like(out, self.v), out)

    def label(self, X: torch.Tensor) -> torch.Tensor:
        """+1 outside, -1 inside (0 counts as outside)."""
        return torch.where(self.field(X) >= 0, 1.0, -1.0).to(torch.float64)

    def describe(self) -> str:
        how = {"flood": "flood fill from the border", "rays": "26-ray parity vote"}[self.method]
        return (f"inside/outside: {how}, voxels {self.v:.4g} ({'x'.join(map(str, self.n))}), "
                f"shell radius {self.r:.4g} ({self.r / self.spacing:.1f} point spacings)")
