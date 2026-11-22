"""
geometry.py -- small mesh utilities shared by the optimizers: connected pieces of a
triangle mesh and exact point-to-mesh distances.
"""

from __future__ import annotations

import numpy as np
import torch


def count_pieces(F: np.ndarray) -> int:
    """Connected pieces of a triangle mesh (label propagation, no scipy)."""
    F = np.asarray(F, np.int64)
    if not len(F):
        return 0
    lab = np.arange(F.max() + 1)
    while True:
        new = lab.copy()
        np.minimum.at(new, F.ravel(), np.repeat(lab[F].min(1), 3))
        new = new[new]
        if np.array_equal(new, lab):
            return len(np.unique(lab[F.ravel()]))
        lab = new


def closest_point_triangle(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (..., 3) (Ericson,
    Real-Time Collision Detection 5.1.5), vectorised with masks."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def safe(x, y):
        return x / torch.where(y.abs() < 1e-300, torch.full_like(y, 1e-300), y)
    den = va + vb + vc
    out = a + ab * safe(vb, den)[..., None] + ac * safe(vc, den)[..., None]          # interior
    t_bc = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    out = torch.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * t_bc[..., None], out)
    t_ac = safe(d2, d2 - d6)
    out = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * t_ac[..., None], out)
    t_ab = safe(d1, d1 - d3)
    out = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * t_ab[..., None], out)
    out = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)
    out = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)
    out = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)
    return out


def point_mesh_dist2(Q: torch.Tensor, V: torch.Tensor, F: np.ndarray, k=16, chunk=4096) -> torch.Tensor:
    """Squared distance from points Q to a triangle mesh: exact distance to each of the
    k triangles with the nearest centroids (enough for a fine, regular mesh)."""
    Q, V = torch.as_tensor(Q, dtype=torch.float64), torch.as_tensor(V, dtype=torch.float64)
    T = V[torch.as_tensor(np.asarray(F, np.int64))]                     # (nf, 3, 3)
    cen = T.mean(1)
    k = min(k, len(T))
    out = []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        idx = torch.cdist(q, cen).topk(k, largest=False).indices
        t = T[idx]
        cp = closest_point_triangle(q[:, None, :].expand(-1, k, -1), t[..., 0, :], t[..., 1, :], t[..., 2, :])
        out.append((cp - q[:, None, :]).square().sum(-1).min(1).values)
    return torch.cat(out)
