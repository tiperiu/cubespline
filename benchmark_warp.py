#!/usr/bin/env python3
"""
benchmark_warp.py -- compare every optimizer of the 16-value (warp) cube on the same targets.

Methods (all with their default settings):
  differentiable
    sampson      warp_optimize.py stage 1 only: multi-start (16 random starts + the
                 8-value fit), each start L-BFGS on the Sampson distance
    chamfer      warp_optimize.py stage 1 + stage 2 (300 Adam steps on the chamfer loss)
    logistic     classify_optimize.py --method logistic (smooth inside/outside loss, Adam)
  non-differentiable (reward only)
    es           classify_optimize.py --method es
    grpo         classify_optimize.py --method grpo
  the three classifiers run with --sampling random and with --sampling lattice (k = 10)

Columns
  time       wall-clock seconds for the whole fit
  own loss   the objective the method minimises, at its result (not comparable
             across methods):
               sampson   mean f^2/|grad f|^2 + 1e-3 (1 - cos(grad f, n)) on 1500 target points
               chamfer   mean |p - q|^2 both ways, 3000 + 3000 fixed samples (world units^2)
               logistic  softplus loss + regularisers, on the last training batch
               es, grpo  (1 - R) / 2 for the mean reward R of the last population / group:
                         the misclassification rate (plus half the offset penalty)
  CD         common measure for all methods: two-sided chamfer distance
             mean_p d(p, target)^2 + mean_q d(q, fit)^2, in world units^2 (cube width 2),
             with exact point-to-surface distances: p = 20000 area-uniform samples of
             the fitted mesh (depth 6), q = 20000 independent samples of the target
             (none of them used in training). Same form as the chamfer loss, without
             the floor that sample-to-sample distances have.
  RMS        sqrt(CD / 2): root-mean-square distance, world units
  pieces     connected pieces of the fitted surface (target: 1)

Usage:
    python benchmark_warp.py                                   # sphere, sphere cap, sphere flank; seeds 0 1 2
    python benchmark_warp.py --targets "torus inner" --seeds 0
    python benchmark_warp.py --out benchmark/benchmark_warp.json --save-fits benchmark
                                   # also save the best seed of every method as a fit file + overview_best.png
    python benchmark_warp.py --save-fits benchmark --save-all-seeds   # every run, + median / worst overviews
    python benchmark_warp.py --from-json benchmark/benchmark_warp.json --save-fits benchmark   # from an earlier run
    cd benchmark && python ../warp_viewer.py                          # the saved fits in the combo box
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import classify_optimize as C
import optimize as O
import warp_optimize as WO

DT = torch.float64
METHODS = ["sampson", "chamfer", "logistic", "logistic lattice", "es", "es lattice", "grpo", "grpo lattice"]
KIND = {"sampson": "differentiable", "chamfer": "differentiable", "logistic": "differentiable",
        "logistic lattice": "differentiable", "es": "reward only", "es lattice": "reward only",
        "grpo": "reward only", "grpo lattice": "reward only"}


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
    den = (va + vb + vc)
    den = torch.where(den.abs() < 1e-300, torch.full_like(den, 1e-300), den)
    v_, w_ = vb / den, vc / den
    out = a + ab * v_[..., None] + ac * w_[..., None]                       # interior
    def safe(x, y):
        return x / torch.where(y.abs() < 1e-300, torch.full_like(y, 1e-300), y)
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
    """Squared distance from points Q to a triangle mesh: exact distance to each of
    the k triangles with the nearest centroids (enough for a fine, regular mesh)."""
    T = V[torch.as_tensor(F)]                                               # (nf, 3, 3)
    cen = T.mean(1)
    k = min(k, len(F))
    out = []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        idx = torch.cdist(q, cen).topk(k, largest=False).indices          # (m, k)
        t = T[idx]                                                          # (m, k, 3, 3)
        cp = closest_point_triangle(q[:, None, :].expand(-1, k, -1), t[..., 0, :], t[..., 1, :], t[..., 2, :])
        out.append((cp - q[:, None, :]).square().sum(-1).min(1).values)
    return torch.cat(out)


def result_metrics(v, target, dense_Q, n=20000, depth=6) -> dict:
    """CD, RMS, both max distances and the number of surface pieces (see result_chamfer)."""
    V, F = C.fitted_mesh(C.Model(16, target.box), v, depth)
    if not len(F):
        return {"empty": True, "cd": float("inf"), "rms": float("inf"), "pieces": 0}
    V = torch.as_tensor(V, dtype=DT)
    P = O.surface_samples(V, F, n, torch.Generator().manual_seed(7))
    d_pq2 = torch.as_tensor(target.fn(P.numpy()), dtype=DT).square()
    d_qp2 = point_mesh_dist2(dense_Q, V, F)
    cd = float(d_pq2.mean() + d_qp2.mean())
    return {"cd": cd, "rms": float(np.sqrt(cd / 2)), "fit_to_target_max": float(d_pq2.max().sqrt()),
            "target_to_fit_max": float(d_qp2.max().sqrt()), "pieces": C.count_pieces(F)}


def result_chamfer(v, target, dense_Q, n=20000, depth=6):
    """(CD, RMS, pieces) of the 16 values `v`, from exact point-to-surface distances:
    fit -> target: |g(p)| at n area-uniform samples p of the fitted mesh (the sphere
                   targets' g = |x - c| - r is the exact distance to the sphere);
    target -> fit: distance from each dense target sample to the fitted mesh (depth 6).
    Point-to-point distances between two sample sets would add a floor of about half
    the sample spacing (~0.008 here), larger than the differences being measured."""
    V, F = C.fitted_mesh(C.Model(16, target.box), v, depth)
    if not len(F):
        return float("inf"), float("inf"), 0
    V = torch.as_tensor(V, dtype=DT)
    P = O.surface_samples(V, F, n, torch.Generator().manual_seed(7))
    d_pq2 = torch.as_tensor(target.fn(P.numpy()), dtype=DT).square()
    d_qp2 = point_mesh_dist2(dense_Q, V, F)
    cd = float(d_pq2.mean() + d_qp2.mean())
    return cd, float(np.sqrt(cd / 2)), C.count_pieces(F)


def sampson_loss(target, v, n_points=1500, normal_weight=1e-3):
    """The objective of warp_optimize.fit_sampson at v."""
    cube = WO.warp_cube(target, v)
    P, N = target.points[:n_points], target.normals[:n_points]
    with torch.no_grad():
        f, g = cube.evaluate(P), cube.gradient(P)
        gn = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return float((f.square() / g.square().sum(-1).clamp_min(1e-12)).mean()
                     + normal_weight * (1 - (gn * N).sum(-1)).mean())


def run_chamfer_pair(target, seed):
    """One warp_optimize run -> the 'sampson' (after stage 1) and 'chamfer' (after stage 2) rows."""
    t0 = time.time()
    f = WO.WarpFitter(target, WO.FitConfig(seed=seed))
    t1 = time.time()
    v1 = f.values
    rows = {"sampson": dict(values=v1, time=t1 - t0, own_loss=sampson_loss(target, v1),
                            steps="16+1 starts x 100 L-BFGS")}
    f.run(verbose=False)
    v2 = f.values
    rows["chamfer"] = dict(values=v2, time=time.time() - t0, own_loss=f.objective(v2)[0],
                           steps="stage 1 + 300 Adam")
    return rows


def run_classify(target, seed, method, sampling):
    cfg = C.ClassifyConfig(method=method, seed=seed, sampling=sampling)
    t0 = time.time()
    v, info = C.fit(target, 16, cfg, verbose=False)
    t = time.time() - t0
    stat = info["history"][-1][1]                     # last logged: batch loss, or mean reward
    own = stat if method == "logistic" else (1 - stat) / 2
    per = {"logistic": "batch", "es": f"{cfg.pop} x batch", "grpo": f"{cfg.group} x batch"}[method]
    pts = "4000 random" if sampling == "random" else f"{cfg.k}^3 lattice"
    return dict(values=v.numpy(), time=t, own_loss=own,
                steps=f"9 starts x {cfg.start_iters} + {cfg.iters} steps, {per} ({pts})")


def _slug(text):
    return str(text).replace(" ", "_")


def best_runs(results):
    """For every (target, method): the run (seed) with the lowest CD."""
    best = {}
    for r in results:
        key = (r["target"], r["method"])
        if key not in best or r["cd"] < best[key]["cd"]:
            best[key] = r
    return list(best.values())


def save_fits(results, folder, depth=5, all_seeds=False):
    """Fit files for warp_viewer.py (run it from `folder`, or Browse to a file) and
    overview images.
    default:          the best seed (lowest CD) of every method and target, and
                      overview_best.png
    all_seeds=True:   every run, and overview_best / _median / _worst.png"""
    import os
    os.makedirs(folder, exist_ok=True)
    targets = {}
    for r in (results if all_seeds else best_runs(results)):
        name = r["target"]
        if name not in targets:
            t = O.builtin_target(name)
            targets[name] = (t, O.target_from_function(O.FUNCTIONS[name], name, box=t.box, n=20000, seed=123).points)
        t, dense = targets[name]
        m = result_metrics(r["values"], t, dense)
        v = np.asarray(r["values"], float)
        err = {**m, "cube_width": t.box.size, "method": r["method"], "kind": r["kind"], "seed": r["seed"],
               "time": r["time"], "own_loss": r["own_loss"], "steps": r["steps"]}
        data = {"type": "warp16", "values": v.tolist(), "offset": v[:8].tolist(), "field": v[8:].tolist(),
                "depth": depth, "target": name, "optimizer": "benchmark_warp", "method": r["method"],
                "seed": r["seed"], "final_error": err, **t.box.to_dict()}
        path = os.path.join(folder, f"{_slug(name)}__{_slug(r['method'])}__seed{r['seed']}.json")
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1)
        r.update({k: m[k] for k in m if k != "empty"})
    n = len(results) if all_seeds else len(best_runs(results))
    print(f"saved {n} fits to {folder}/" + ("" if all_seeds else "  (best seed of each method and target)"))
    for which in (("best", "median", "worst") if all_seeds else ("best",)):
        path = os.path.join(folder, f"overview_{which}.png")
        overview(results, {n: v[0] for n, v in targets.items()}, path, which, depth)
        print(f"saved {path}")


def overview(results, targets, path, which="median", depth=5):
    """Grid of renders: one row per target, one column per method; the seed with the
    best, median or worst CD. Orange: target (clipped to the cube), blue: the fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    names = list(targets)
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # rotation: world y (up) -> matplotlib z
    def up(X):
        return np.asarray(X) @ R.T
    fig = plt.figure(figsize=(2.6 * len(METHODS), 2.9 * len(names)))
    light = np.array([0.4, 0.8, 0.45]) / np.linalg.norm([0.4, 0.8, 0.45])
    for i, name in enumerate(names):
        t = targets[name]
        for j, m in enumerate(METHODS):
            rs = sorted([r for r in results if r["target"] == name and r["method"] == m], key=lambda r: r["cd"])
            if not rs:
                continue
            r = {"best": rs[0], "median": rs[len(rs) // 2], "worst": rs[-1]}[which]
            ax = fig.add_subplot(len(names), len(METHODS), i * len(METHODS) + j + 1, projection="3d")
            ax.computed_zorder = False                     # fit always over the target: orange shows where it is missed
            ax.add_collection3d(Poly3DCollection(up(t.V)[t.F], facecolor=(0.95, 0.55, 0.30, 0.35), edgecolor="none",
                                                 zorder=1))
            V, F = C.fitted_mesh(C.Model(16, t.box), r["values"], depth)
            if len(F):
                tri = up(V)[F]
                nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                shade = 0.35 + 0.65 * np.abs(nrm @ light)
                col = np.c_[np.outer(shade, (0.55, 0.65, 0.95)), np.ones(len(F))]
                ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none", zorder=2))
            for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
                P = up(t.box.corners()[[a, b]])
                ax.plot(*P.T, color="0.6", lw=0.6)
            lo, hi = t.box.lo_np, t.box.lo_np + t.box.size
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(-hi[2], -lo[2]); ax.set_zlim(lo[1], hi[1])
            ax.view_init(elev=22, azim=-55)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.set_title(f"{m}\nseed {r['seed']}  CD {r['cd'] * 1e4:.3g}e-4" + (f"  {r['pieces']} pieces" if r["pieces"] > 1 else ""),
                         fontsize=7)
        fig.text(0.005, 1 - (i + 0.5) / len(names), name, rotation=90, va="center", fontsize=11)
    fig.suptitle(f"16-value cube, {which} seed per method  (blue: fit, drawn over the target; orange: target where the fit misses it)",
                 fontsize=12)
    fig.tight_layout(rect=(0.015, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["sphere", "sphere cap", "sphere flank"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="benchmark_warp.json")
    ap.add_argument("--save-fits", metavar="DIR", help="also write the best seed of every method as a fit file for warp_viewer.py, "
                                                       "plus overview images, into DIR")
    ap.add_argument("--save-all-seeds", action="store_true",
                    help="with --save-fits: save every seed, not only the best one of each method")
    ap.add_argument("--from-json", metavar="FILE", help="do not run: load results from FILE (an earlier --out)")
    args = ap.parse_args()
    if args.from_json:
        with open(args.from_json) as fh:
            results = json.load(fh)
        if args.save_fits:
            save_fits(results, args.save_fits, all_seeds=args.save_all_seeds)
        print_table(results, list(dict.fromkeys(r["target"] for r in results)))
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = []
    for name in args.targets:
        target = O.builtin_target(name)
        dense = O.target_from_function(O.FUNCTIONS[name], name, box=target.box, n=20000, seed=123).points
        for seed in args.seeds:
            rows = run_chamfer_pair(target, seed)
            for m in ("logistic", "es", "grpo"):
                rows[m] = run_classify(target, seed, m, "random")
                rows[f"{m} lattice"] = run_classify(target, seed, m, "lattice")
            for m in METHODS:
                r = rows[m]
                cd, rms, pieces = result_chamfer(r["values"], target, dense)
                rec = {"target": name, "seed": seed, "method": m, "kind": KIND[m], "time": r["time"],
                       "own_loss": r["own_loss"], "cd": cd, "rms": rms, "pieces": pieces,
                       "steps": r["steps"], "values": np.asarray(r["values"]).tolist()}
                results.append(rec)
                print(f"{name:12s} seed {seed}  {m:16s} {r['time']:6.1f}s  own loss {r['own_loss']:.3e}  "
                      f"CD {cd:.3e}  RMS {rms:.3e}  pieces {pieces}", flush=True)
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1)
    if args.save_fits:
        save_fits(results, args.save_fits, all_seeds=args.save_all_seeds)
    print_table(results, args.targets)


def print_table(results, targets):
    """Markdown table: mean over seeds (CD also min-max)."""
    print()
    for name in targets:
        print(f"\n**{name}** (mean of {len({r['seed'] for r in results if r['target'] == name})} seeds)\n")
        print("| method | kind | time (s) | own loss | CD (×1e-4) | CD min–max (×1e-4) | RMS | pieces |")
        print("|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            rs = [r for r in results if r["target"] == name and r["method"] == m]
            if not rs:
                continue
            cd = np.array([r["cd"] for r in rs]) * 1e4
            print(f"| {m} | {KIND[m]} | {np.mean([r['time'] for r in rs]):.1f} | "
                  f"{np.mean([r['own_loss'] for r in rs]):.2e} | {cd.mean():.2f} | {cd.min():.2f}–{cd.max():.2f} | "
                  f"{np.mean([r['rms'] for r in rs]):.2e} | "
                  f"{'/'.join(str(r['pieces']) for r in rs)} |")


if __name__ == "__main__":
    main()
