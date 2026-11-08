#!/usr/bin/env python3
"""
classify_optimize.py -- fit a cube (8 values, or 16 with the warp) by inside/outside
classification of sample points, instead of by distances to the target surface.

Each sample x gets a label from the analytic target g (negative inside):
    y = +1 if g(x) > 0 (outside),  y = -1 if g(x) < 0 (inside)
and the cube field f is "correct" at x when sign(f(x)) == y.
The reward of a set of values is   R = mean_x  y * sign(f(x))   = 2 * accuracy - 1.

R is a step function of the values: a small change almost never flips a sample,
so its gradient is zero almost everywhere and Adam cannot use it directly.
Three ways around that:

  --method logistic   smooth the reward and use its gradient (supervised learning).
        loss = mean_x softplus(-y f(x) / tau)  +  weight_decay * mean(field^2)
        softplus(-y f / tau) is ~0 when the sign is right with a margin, grows
        linearly when it is wrong. tau sets the width of the soft boundary; with
        the field at its usual scale (|grad f| ~ 1) it is roughly in world units.
        This is logistic regression; for the 8-value cube the features are the 8
        trilinear weights and f is linear in the values, so the loss is convex: one global optimum, reached from zero values. The weight
        decay fixes the scale (otherwise |f| would grow without bound once the
        data are separated). Optimised with Adam, fresh samples every step.

  --method es         keep the true +/-1 reward; estimate a gradient by sampling
        (REINFORCE / policy gradient, also called evolution strategies, ES).
        The values are treated as random:  theta = mu + sigma * eps,  eps ~ N(0, I).
        For each generation: draw `pop` perturbations (in +/- pairs), compute the
        reward of each on the same batch of samples, rank them, and move mu along
        grad ~ sum_i u_i eps_i / (pop * sigma)    (u_i = centred rank of reward i).
        The estimate is passed to Adam as the gradient of mu. sigma decays over
        the run. No derivative of f is used.

  --method grpo       Group Relative Policy Optimization (the RL method used to
        train reasoning LLMs), with the same +/-1 reward. The policy is a Gaussian
        over the values, N(mu, diag sigma^2), and both mu and sigma are learned.
        Each iteration:
          group      draw G value vectors theta_i from the current policy and
                     score each on the same batch of points (the "prompt"; the
                     theta_i are the G "answers")
          advantage  A_i = (R_i - mean R) / std R   within the group, no critic
          update     --grpo-epochs Adam steps on the same group, maximising the
                     clipped surrogate  mean_i min(r_i A_i, clip(r_i, 1-eps, 1+eps) A_i)
                     with r_i = pi(theta_i) / pi_old(theta_i), minus
                     beta * KL(pi || pi_ref)
          reference  pi_ref is refreshed every --ref-every iterations
        The KL term is off by default (--kl-coef 0). In LLM training it keeps the
        model near the pretrained one; here the policy must move far from its
        start, and beta = 0.04 (the DeepSeek value) roughly doubled the error.
        Compared with ES: sigma is learned (it shrinks as the fit converges, no
        schedule), advantages are z-scores instead of ranks, and the clipped
        ratio lets each group be reused for several steps.

16-value cube: large offsets move a corner far outside the cube, where the second
trilinear field is extrapolated and can change sign again. That makes small
closed bubbles near the corners: a tiny volume, so almost free in accuracy, but
far from the target. --offset-penalty (lambda, default 0.01) subtracts
lambda * mean(offset^2) from the reward (adds it to the logistic loss). The report
counts the pieces of the fitted surface so a bubble is visible.

Samples (--sampling):
  random (default)  a fresh batch every step: most points near the target surface
        (surface points moved along the normal by N(0, band^2)), the rest uniform
        in the cube (--uniform). Far from the surface everything is classified
        correctly and carries no information.
  lattice           the same regular k x k x k lattice every step (--k, default 10,
        i.e. 1000 points): cell centres, or with --lattice-faces points i/(k-1)
        that include the faces, edges and corners. The reward is the accuracy on
        exactly these points; the iterate kept is the best on the lattice. The
        report also gives the accuracy on independent random points, which shows
        how well the lattice fit generalises.

The cube can be anywhere (--lo X Y Z, --size S; default [-1, 1]^3). Targets are
in world space and clipped to the cube; --band, --tau and the "near the surface"
distance are given for a cube of width 2 and scale with the cube.

16-value cube: the problem is not convex (f is nonlinear in the offsets), so
stage 1 tries several starts (--starts, each --start-iters steps): the 8-value
solution of the same method with zero offset, and random starts. The start with
the best validation reward continues for --iters steps.

Classification fixes which side is inside and the topology, not the distance.
The report gives both: accuracy (overall and within 0.05 of the surface) and the
same RMS distance as optimize.py / warp_optimize.py. --chamfer-iters N then
refines the result with the chamfer optimizer of those modules.

Usage:
    python classify_optimize.py --target "sphere flank"                         # 8 values, logistic
    python classify_optimize.py --target "sphere flank" --method es
    python classify_optimize.py --cube 16 --target "sphere cap"                 # 16 values, logistic
    python classify_optimize.py --cube 16 --target "sphere cap" --method es --iters 600
    python classify_optimize.py --cube 16 --target "sphere cap" --method grpo
    python classify_optimize.py --cube 16 --target "torus inner" --chamfer-iters 200
    python classify_optimize.py --cube 16 --target "sphere cap" --method grpo --sampling lattice --k 10
    python classify_optimize.py --target "sphere cap" --lo 0 -1 0 --size 1     # a smaller cube elsewhere
    python classify_optimize.py --list
Fits are saved in the same format as optimize.py (8) / warp_optimize.py (16), so
viewer.py and warp_viewer.py show them in their combo boxes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fn

import optimize as O
import warp_optimize as WO
from box import Box, add_box_arguments, box_from_args
from cube import Cube
from warpcube import CORNER_DIR, field_world

DT = torch.float64


# ============================================================================
# The two cubes behind one interface
# ============================================================================
class Model:
    """f(values, X) for the 8-value cube or the 16-value warp cube, placed in `box`."""

    def __init__(self, n_values: int, box: Box | None = None):
        assert n_values in (8, 16)
        self.n = n_values
        self.box = box or Box()
        self.lo = torch.as_tensor(self.box.lo, dtype=DT)

    @property
    def field_slice(self):
        return slice(0, 8) if self.n == 8 else slice(8, 16)

    def field(self, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if self.n == 8:
            return Cube.basis_local(self.box.to_local(X)) @ v
        return field_world(v, X, self.lo, self.box.size)

    def field_batch(self, V: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """f for a population of value vectors V (pop, n) at X (m, 3) -> (m, pop)."""
        u = self.box.to_local(X)
        B = Cube.basis_local(u)                                    # (m, 8)
        if self.n == 8:                                            # linear in v: one matmul
            return B @ V.T
        E = torch.as_tensor(CORNER_DIR, dtype=DT)
        D = (B[None] * V[:, None, :8]) @ E / 2                     # local offsets, half-widths (pop, m, 3)
        return (Cube.basis_local(u[None] + D) * V[:, None, 8:]).sum(-1).T

    def normalised(self, v: torch.Tensor) -> torch.Tensor:
        """Field part scaled to norm sqrt(8) (the surface does not depend on it)."""
        v = v.detach().clone()
        s = self.field_slice
        nrm = v[s].norm()
        if nrm > 0:
            v[s] *= np.sqrt(8) / nrm
        return v

    def evaluate(self, v, target, depth):
        v = np.asarray(v, float)
        return O.evaluate(v, target, depth=depth) if self.n == 8 else WO.evaluate(v, target, depth=depth)


# ============================================================================
# Samples and labels
# ============================================================================
class Sampler:
    """Labelled points in the target's cube. Label +1 outside (g > 0), -1 inside.

    sampling="random":  a fresh batch every step -- points near the target surface
        (normal jitter ~ N(0, band^2), band scaled with the cube) plus a fraction
        uniform in the cube.
    sampling="lattice": the same k x k x k regular lattice every step (k^3 points;
        cell centres, or with faces=True a lattice that includes the cube's
        faces, edges and corners). The optimizers then maximise the reward on
        exactly these points."""

    def __init__(self, target: O.Target, band=0.1, uniform=0.25, sampling="random", k=10, faces=False):
        if target.fn is None:
            raise SystemExit("classification needs a function target (inside/outside); "
                             "mesh targets have no inside")
        if sampling not in ("random", "lattice"):
            raise ValueError(f"unknown sampling '{sampling}'")
        self.target, self.box = target, target.box
        self.band, self.uniform = self.box.scale(band), uniform
        self.sampling, self.k, self.faces = sampling, k, faces
        self.lattice = None
        if sampling == "lattice":
            X = torch.as_tensor(self.box.lattice(k, faces), dtype=DT)
            self.lattice = (X, self.label(X))

    def label(self, X: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(np.where(self.target.fn(X.numpy()) > 0, 1.0, -1.0), dtype=DT)

    def draw(self, n, gen: torch.Generator):
        """Training points: the lattice in lattice mode, else draw_random(n)."""
        return self.lattice if self.lattice is not None else self.draw_random(n, gen)

    def draw_random(self, n, gen: torch.Generator):
        P, N = self.target.points, self.target.normals
        nu = int(round(self.uniform * n))
        idx = torch.randint(len(P), (n - nu,), generator=gen)
        Xs = P[idx] + N[idx] * (self.band * torch.randn(n - nu, 1, dtype=DT, generator=gen))
        Xu = self.box.uniform(nu, gen)
        X = self.box.clamp(torch.cat([Xs, Xu]))
        return X, self.label(X)

    def describe(self) -> str:
        if self.lattice is None:
            return f"random: fresh batches, near-surface band {self.band:g} + {100 * self.uniform:g} % uniform"
        where = "including faces, edges and corners" if self.faces else "cell centres"
        return f"lattice: {self.k}^3 = {self.k ** 3} fixed points ({where})"

    def distance(self, X: torch.Tensor) -> np.ndarray:
        """First-order distance |g| / |grad g| of points to the target surface."""
        Xn = X.numpy()
        G = O._fn_gradient(self.target.fn, Xn)
        return np.abs(self.target.fn(Xn)) / np.maximum(np.linalg.norm(G, axis=-1), 1e-12)


def reward(model: Model, v, X, y) -> float:
    """mean y * sign(f) in [-1, 1]  (= 2 * accuracy - 1); f == 0 counts as wrong."""
    with torch.no_grad():
        s = torch.sign(model.field(v, X))
        return float((y * torch.where(s == 0, -y, s)).mean())


@dataclass
class Validation:
    """Fixed labelled points for keep-best and for the report."""
    X: torch.Tensor
    y: torch.Tensor
    near: torch.Tensor       # bool, within NEAR (scaled with the cube) of the target surface

    NEAR = 0.05              # for a cube of width 2

    @classmethod
    def build(cls, sampler: Sampler, n=20000, seed=999):
        """Random points (same distribution as random sampling), independent of
        the training points, also in lattice mode."""
        X, y = sampler.draw_random(n, torch.Generator().manual_seed(seed))
        return cls.of(sampler, X, y)

    @classmethod
    def of(cls, sampler: Sampler, X, y):
        return cls(X, y, torch.as_tensor(sampler.distance(X) < sampler.box.scale(cls.NEAR)))

    def accuracy(self, model: Model, v) -> dict:
        with torch.no_grad():
            ok = (torch.sign(model.field(torch.as_tensor(v, dtype=DT), self.X)) == self.y)
        return {"accuracy": float(ok.double().mean()),
                "accuracy_near": float(ok[self.near].double().mean()) if self.near.any() else float("nan"),
                "near_fraction": float(self.near.double().mean())}


# ============================================================================
# Configuration and the three methods
# ============================================================================
@dataclass
class ClassifyConfig:
    method: str = "logistic"     # "logistic" | "es" | "grpo"
    iters: int = 400
    lr: float = 0.05
    lr_final: float = 1e-3
    batch: int = 4000
    band: float = 0.1            # std of the normal jitter of near-surface samples (cube of width 2; scaled)
    uniform: float = 0.25        # fraction of uniform samples
    weight_decay: float = 1e-3   # logistic: fixes the scale of the field
    tau: float = 0.02            # logistic: temperature, softplus(-y f / tau)
    offset_penalty: float = 0.01 # 16 values: lambda * mean(offset^2), subtracted from the reward
    pop: int = 32                # es: perturbations per generation (even)
    sigma: float = 0.1           # es: perturbation size, decays to sigma_final
    sigma_final: float = 0.02
    group: int = 32              # grpo: group size G
    grpo_epochs: int = 4         # grpo: Adam steps per group
    clip: float = 0.2            # grpo: ratio clipping eps
    kl_coef: float = 0.0         # grpo: beta, KL(pi || pi_ref); 0 = off (see the docstring)
    ref_every: int = 10          # grpo: refresh the reference policy every k iterations
    grpo_sigma: float = 0.2      # grpo: initial std of the policy (learned afterwards)
    grpo_lr: float = 0.02        # grpo: Adam learning rate (mu and log sigma)
    sampling: str = "random"     # "random" | "lattice"
    k: int = 10                  # lattice: k x k x k points
    lattice_faces: bool = False  # lattice: include the faces (i / (k-1)) instead of cell centres
    starts: int = 8              # 16-value cube: random starts for stage 1
    start_iters: int = 100       # 16-value cube: steps per start
    check_every: int = 10
    seed: int = 0


def _cosine(it, iters, a, b):
    return b + (a - b) * 0.5 * (1 + np.cos(np.pi * min(it, iters) / max(iters, 1)))


def offset_cost(model: Model, v, lam):
    """lam * mean(offset^2) for the 16-value cube (0 for 8). v may be (..., n)."""
    if model.n == 8 or not lam:
        return 0.0
    return lam * v[..., :8].square().mean(-1)


def logistic_loss(model: Model, v, X, y, cfg: ClassifyConfig):
    f = model.field(v, X)
    return (Fn.softplus(-y * f / cfg.tau).mean() + cfg.weight_decay * v[model.field_slice].square().mean()
            + offset_cost(model, v, cfg.offset_penalty))


def population_rewards(model: Model, Theta, X, y, lam=0.0):
    """Reward of each row of Theta (pop, n) on the same samples -> (pop,)."""
    with torch.no_grad():
        S = torch.sign(model.field_batch(Theta, X))       # (m, pop)
        S = torch.where(S == 0, -y[:, None], S)           # f == 0 counts as wrong
        return (y[:, None] * S).mean(0) - offset_cost(model, Theta, lam)


def es_gradient(model: Model, mu, X, y, pop, sigma, gen, lam=0.0):
    """Antithetic ES estimate of grad_mu E[R(mu + sigma eps)], with centred ranks.
    Returns (gradient, mean reward of the population)."""
    half = pop // 2
    eps = torch.randn(half, model.n, dtype=DT, generator=gen)
    eps = torch.cat([eps, -eps])
    R = population_rewards(model, mu + sigma * eps, X, y, lam)
    ranks = torch.empty_like(R)
    ranks[R.argsort()] = torch.arange(len(R), dtype=DT)
    u = ranks / (len(R) - 1) - 0.5                        # centred ranks in [-0.5, 0.5]
    return (u[:, None] * eps).sum(0) / (len(R) * sigma), float(R.mean())


class GRPOPolicy:
    """Gaussian policy N(mu, diag exp(log_std)^2) over the values, trained by GRPO."""

    LOG_STD_MIN, LOG_STD_MAX = np.log(1e-3), np.log(1.0)

    def __init__(self, mu: torch.Tensor, cfg: ClassifyConfig):
        self.cfg = cfg
        self.mu = mu                                       # shared with train() (requires grad)
        self.log_std = torch.full_like(mu.detach(), np.log(cfg.grpo_sigma)).requires_grad_(True)
        self.opt = torch.optim.Adam([self.mu, self.log_std], lr=cfg.grpo_lr)
        self.refresh_reference()

    def dist(self):
        return torch.distributions.Normal(self.mu, self.log_std.exp())

    def refresh_reference(self):
        self.ref = torch.distributions.Normal(self.mu.detach().clone(), self.log_std.detach().exp().clone())

    def step(self, model: Model, X, y, gen, lr):
        """One GRPO iteration on one batch of points. Returns the group's mean reward."""
        cfg = self.cfg
        with torch.no_grad():                              # 1. group from the old policy
            old = self.dist()
            eps = torch.randn(cfg.group, model.n, dtype=DT, generator=gen)
            theta = old.loc + old.scale * eps
            old_logp = old.log_prob(theta).sum(-1)
            R = population_rewards(model, theta, X, y, cfg.offset_penalty)
        if R.std() < 1e-12:                                # all equal: no signal in this group
            return float(R.mean())
        A = (R - R.mean()) / R.std()                       # 2. group-relative advantage
        for g in self.opt.param_groups:
            g["lr"] = lr
        for _ in range(cfg.grpo_epochs):                   # 3. clipped surrogate + KL
            self.opt.zero_grad()
            d = self.dist()
            ratio = (d.log_prob(theta).sum(-1) - old_logp).exp()
            surrogate = torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
            kl = torch.distributions.kl_divergence(d, self.ref).sum()
            (-surrogate + cfg.kl_coef * kl).backward()
            self.opt.step()
            with torch.no_grad():
                self.log_std.clamp_(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return float(R.mean())

    @property
    def sigma(self) -> float:
        return float(self.log_std.detach().exp().mean())


def train(model: Model, sampler: Sampler, val: Validation, v0, iters, cfg: ClassifyConfig,
          gen: torch.Generator, log_every=0, label=""):
    """Run `iters` steps of cfg.method from v0. Keeps the iterate with the best
    validation reward. Returns (best values, best validation reward, history)."""
    mu = model.normalised(torch.as_tensor(v0, dtype=DT)).requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=cfg.lr)
    policy = GRPOPolicy(mu, cfg) if cfg.method == "grpo" else None
    def score(v):                                         # validation reward - offset cost
        return reward(model, v, val.X, val.y) - float(offset_cost(model, v, cfg.offset_penalty))
    best_v, best_r = mu.detach().clone(), score(mu.detach())
    history = []
    t0 = time.time()
    for it in range(iters):
        for g in opt.param_groups:
            g["lr"] = _cosine(it, iters, cfg.lr, cfg.lr_final)
        X, y = sampler.draw(cfg.batch, gen)
        opt.zero_grad()
        if cfg.method == "logistic":
            loss = logistic_loss(model, mu, X, y, cfg)
            loss.backward()
            stat = float(loss.detach())
            opt.step()
        elif cfg.method == "es":
            sigma = _cosine(it, iters, cfg.sigma, cfg.sigma_final)
            grad, stat = es_gradient(model, mu.detach(), X, y, cfg.pop, sigma, gen, cfg.offset_penalty)
            mu.grad = -grad                               # Adam minimises: ascend the reward
            opt.step()
        else:
            stat = policy.step(model, X, y, gen, _cosine(it, iters, cfg.grpo_lr, cfg.grpo_lr / 20))
            if (it + 1) % cfg.ref_every == 0:
                policy.refresh_reference()
        if cfg.method != "logistic":                      # the reward is scale invariant
            with torch.no_grad():
                mu.copy_(model.normalised(mu))
        if (it + 1) % cfg.check_every == 0 or it + 1 == iters:
            r = score(mu.detach())
            history.append((it + 1, stat, r))
            if r > best_r:
                best_v, best_r = mu.detach().clone(), r
            if log_every and ((it + 1) % log_every == 0 or it + 1 == iters):
                what = "loss" if cfg.method == "logistic" else "batch reward"
                extra = f"   policy sigma {policy.sigma:.4f}" if policy else ""
                print(f"  {label}iter {it + 1:4d}  {what} {stat:+.4e}   val reward {r:+.4f} "
                      f"(acc {(1 + r) / 2:.4f}){extra}   ({time.time() - t0:.1f}s)")
    return model.normalised(best_v), best_r, history


def make_sampler(target: O.Target, cfg: ClassifyConfig) -> Sampler:
    return Sampler(target, cfg.band, cfg.uniform, cfg.sampling, cfg.k, cfg.lattice_faces)


def fit(target: O.Target, n_values: int, cfg: ClassifyConfig, verbose=True):
    """Full run. Returns (values, info).
    The iterate kept (keep-best, choice of start) is the one with the best reward on
    the training points in lattice mode, on fixed random validation points otherwise."""
    gen = torch.Generator().manual_seed(cfg.seed)
    sampler = make_sampler(target, cfg)
    val = Validation.of(sampler, *sampler.lattice) if sampler.lattice is not None else Validation.build(sampler)
    log = 50 if verbose else 0
    box = target.box
    if verbose:
        print(f"  sampling {sampler.describe()}")
    if n_values == 8:
        # logistic: convex, start from zero. es: zero gives reward -1 for every
        # sample (f = 0 counts as wrong) but the perturbations do not, so it starts.
        v0 = torch.zeros(8, dtype=DT)
        if verbose:
            print(f"  {cfg.method}, 8 values, start: zero values")
        v, r, hist = train(Model(8, box), sampler, val, v0, cfg.iters, cfg, gen, log)
        return v, {"start": "zero values", "val_reward": r, "history": hist}

    model = Model(16, box)
    if verbose:
        print(f"  stage 1: {cfg.starts} random starts + the 8-value solution, {cfg.start_iters} steps each")
    v8, _, _ = train(Model(8, box), sampler, val, torch.zeros(8, dtype=DT), max(cfg.start_iters, 1), cfg, gen)
    cands = [("8-value solution, zero offset", torch.cat([torch.zeros(8, dtype=DT), v8]))]
    for i in range(cfg.starts):
        scale = (0.3, 0.6, 1.0)[i % 3]
        cands.append((f"random start {i} (offset scale {scale})",
                      torch.cat([scale * torch.randn(8, dtype=DT, generator=gen),
                                 torch.randn(8, dtype=DT, generator=gen)])))
    best = (-np.inf, None, None)
    for name, v0 in cands:
        v, r, _ = train(model, sampler, val, v0, cfg.start_iters, cfg, gen)
        if verbose:
            print(f"    {name:34s} -> val reward {r:+.4f} (acc {(1 + r) / 2:.4f})")
        if r > best[0]:
            best = (r, v, name)
    if verbose:
        print(f"  stage 2: {cfg.iters} steps from '{best[2]}'")
    v, r, hist = train(model, sampler, val, best[1], cfg.iters, cfg, gen, log)
    if best[0] > r:                                       # stage 2 kept the start
        v, r = best[1], best[0]
    return v, {"start": best[2], "val_reward": r, "history": hist}


# ============================================================================
# Report, chamfer refinement, save
# ============================================================================
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


def fitted_mesh(model: Model, v, depth):
    v = np.asarray(v, float)
    if model.n == 8:
        from marching import extract_topology
        cube = Cube(v, requires_grad=False, **model.box.kw)
        topo = extract_topology(cube, depth)
    else:
        import warp_marching as WM
        from warpcube import WarpCube
        cube = WarpCube(v, requires_grad=False, **model.box.kw)
        topo = WM.extract_topology(cube, depth)
    with torch.no_grad():
        return topo.vertices(cube.values).numpy(), topo.F


# ----------------------------------------------------------------------------
# ============================================================================
def report(model: Model, v, target, val: Validation, depth, train_set: Validation | None = None) -> dict:
    """Accuracy on random validation points (and on the training lattice, if given),
    distances to the target, surface pieces."""
    out = val.accuracy(model, v)
    if train_set is not None:
        out["accuracy_train_lattice"] = train_set.accuracy(model, v)["accuracy"]
        out["train_points"] = len(train_set.X)
    m = model.evaluate(v, target, depth)
    width = target.box.size
    if not m.get("empty"):
        out.update({k: m[k] for k in ("rms", "fit_to_target_max", "target_to_fit_max")})
        out["rms_percent_of_cube"] = 100 * m["rms"] / width
    out["cube_width"] = width
    out["cube"] = target.box.to_dict()
    out["surface_pieces"] = count_pieces(fitted_mesh(model, v, depth)[1])
    out["target_pieces"] = count_pieces(target.F)
    if model.n == 16:
        out["max_abs_offset"] = float(np.abs(np.asarray(v)[:8]).max())
    return out


def print_report(err: dict, title="final classification result"):
    print("\n" + "=" * 24 + f" {title} " + "=" * max(2, 52 - len(title)))
    if "accuracy_train_lattice" in err:
        print(f"  accuracy on the training lattice     {100 * err['accuracy_train_lattice']:.3f} %   "
              f"({err['train_points']} points)")
    print(f"  accuracy (random validation points)  {100 * err['accuracy']:.3f} %")
    near = Validation.NEAR * err["cube_width"] / 2
    print(f"  accuracy within {near:g} of the surface  {100 * err['accuracy_near']:.3f} %   "
          f"({100 * err['near_fraction']:.1f} % of the validation points)")
    if "rms" in err:
        w = err["cube_width"]
        print(f"  distance to target (world units, cube width {w:g}):")
        print(f"    RMS                                {err['rms']:.4e}   ({err['rms_percent_of_cube']:.3f} % of the cube width)")
        print(f"    max  fit -> target                 {err['fit_to_target_max']:.4e}")
        print(f"    max  target -> fit                 {err['target_to_fit_max']:.4e}")
    else:
        print("  the fitted surface does not cross the cube")
    print(f"  surface pieces                       {err['surface_pieces']}   (target: {err['target_pieces']})"
          + ("   <-- extra pieces (bubbles)" if err['surface_pieces'] > err['target_pieces'] else ""))
    if "max_abs_offset" in err:
        print(f"  largest offset value                 {err['max_abs_offset']:.3f}")
    print("=" * 78)


def chamfer_refine(n_values, target, v, iters, depth, seed):
    """Refine with the chamfer optimizer of optimize.py / warp_optimize.py."""
    if n_values == 8:
        f = O.Fitter(target, O.FitConfig(depth=depth, iters=iters, seed=seed), values=v)
    else:
        f = WO.WarpFitter(target, WO.FitConfig(depth=depth, iters=iters, seed=seed), values=v)
    f.run(iters, log_every=50)
    return f.values


def save(path, n_values, v, target, depth, cfg: ClassifyConfig, err, info, chamfer_iters):
    v = np.asarray(v, float)
    data = {"values": v.tolist(), "depth": depth, "target": target.name,
            "optimizer": "classify", "method": cfg.method, "iterations": cfg.iters,
            "chamfer_iters": chamfer_iters, "start": info["start"],
            "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "final_error": err, **target.box.to_dict()}
    if n_values == 16:
        data.update({"type": "warp16", "offset": v[:8].tolist(), "field": v[8:].tolist()})
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", type=int, default=8, choices=[8, 16], help="8 values, or 16 (warp cube)")
    ap.add_argument("--method", default="logistic", choices=["logistic", "es", "grpo"])
    ap.add_argument("--target", default="sphere flank", choices=["trilinear", *O.FUNCTIONS])
    ap.add_argument("--iters", type=int, default=400, help="steps (generations for es); stage 2 for --cube 16")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4000, help="labelled samples per step")
    ap.add_argument("--sampling", default="random", choices=["random", "lattice"],
                    help="random: fresh near-surface + uniform batches each step; lattice: a fixed k^3 lattice")
    ap.add_argument("--k", type=int, default=10, help="lattice: k x k x k points (default 10 = 1000 points)")
    ap.add_argument("--lattice-faces", action="store_true",
                    help="lattice: points i/(k-1), including the faces, edges and corners (default: cell centres)")
    ap.add_argument("--band", type=float, default=0.1,
                    help="random: std of the near-surface jitter (for a cube of width 2; scales with the cube)")
    ap.add_argument("--uniform", type=float, default=0.25, help="random: fraction of uniform samples")
    ap.add_argument("--weight-decay", type=float, default=1e-3, help="logistic only")
    ap.add_argument("--tau", type=float, default=0.02, help="logistic only: temperature (soft boundary width)")
    ap.add_argument("--offset-penalty", type=float, default=0.01,
                    help="--cube 16 only: lambda * mean(offset^2) against corner bubbles (0 = off)")
    ap.add_argument("--pop", type=int, default=32, help="es only: perturbations per generation")
    ap.add_argument("--sigma", type=float, default=0.1, help="es only: initial perturbation size")
    ap.add_argument("--group", type=int, default=32, help="grpo only: group size G")
    ap.add_argument("--grpo-epochs", type=int, default=4, help="grpo only: Adam steps per group")
    ap.add_argument("--clip", type=float, default=0.2, help="grpo only: ratio clipping eps")
    ap.add_argument("--kl-coef", type=float, default=0.0, help="grpo only: beta of KL(pi || pi_ref)")
    ap.add_argument("--ref-every", type=int, default=10, help="grpo only: refresh the reference every k iterations")
    ap.add_argument("--grpo-sigma", type=float, default=0.2, help="grpo only: initial policy std (then learned)")
    ap.add_argument("--grpo-lr", type=float, default=0.02, help="grpo only: Adam learning rate")
    ap.add_argument("--starts", type=int, default=8, help="--cube 16 only: random starts")
    ap.add_argument("--start-iters", type=int, default=100, help="--cube 16 only: steps per start")
    ap.add_argument("--chamfer-iters", type=int, default=0, help="then refine with the chamfer optimizer")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    add_box_arguments(ap)
    ap.add_argument("--out", help="default classify_fit.json (8) / classify_warp_fit.json (16)")
    ap.add_argument("--list", action="store_true", help="list targets")
    args = ap.parse_args()

    if args.list:
        print("targets:", "trilinear,", ", ".join(O.FUNCTIONS))
        return
    cfg = ClassifyConfig(method=args.method, iters=args.iters, lr=args.lr, batch=args.batch,
                         band=args.band, uniform=args.uniform, weight_decay=args.weight_decay, tau=args.tau, offset_penalty=args.offset_penalty,
                         pop=args.pop + args.pop % 2, sigma=args.sigma, starts=args.starts,
                         start_iters=args.start_iters, seed=args.seed,
                         group=args.group, grpo_epochs=args.grpo_epochs, clip=args.clip, kl_coef=args.kl_coef,
                         ref_every=args.ref_every, grpo_sigma=args.grpo_sigma, grpo_lr=args.grpo_lr,
                         sampling=args.sampling, k=args.k, lattice_faces=args.lattice_faces)
    out = args.out or ("classify_fit.json" if args.cube == 8 else "classify_warp_fit.json")
    target = O.builtin_target(args.target, box=box_from_args(args))
    print(f"target '{target.name}', {args.cube}-value cube in {target.box}, method {args.method}")
    t0 = time.time()
    v, info = fit(target, args.cube, cfg)
    model = Model(args.cube, target.box)
    sampler = make_sampler(target, cfg)
    val = Validation.build(sampler)
    lat = Validation.of(sampler, *sampler.lattice) if sampler.lattice is not None else None
    err = report(model, v, target, val, args.depth, lat)
    print(f"  done in {time.time() - t0:.1f}s")
    print_report(err)
    if args.chamfer_iters > 0:
        print(f"\nchamfer refinement, {args.chamfer_iters} Adam steps")
        v = model.normalised(torch.as_tensor(chamfer_refine(args.cube, target, v, args.chamfer_iters,
                                                            args.depth, args.seed)))
        err2 = report(model, v, target, val, args.depth, lat)
        err2["before_chamfer"] = err
        err = err2
        print_report(err, "after chamfer refinement")
    save(out, args.cube, v.numpy(), target, args.depth, cfg, err, info, args.chamfer_iters)
    print(f"saved {out}   (view: python {'viewer.py' if args.cube == 8 else 'warp_viewer.py'} --load {out})")


if __name__ == "__main__":
    main()
