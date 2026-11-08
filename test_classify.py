"""
Tests for classify_optimize.py (inside/outside classification: logistic and ES).
Run:  python -m pytest test_classify.py -q      (or  python test_classify.py)
"""

import json
import os
import tempfile

import numpy as np
import torch

import classify_optimize as C
import optimize as O
from box import Box

DT = torch.float64


def test_population_field_matches_single_field():
    g = torch.Generator().manual_seed(0)
    X = 2 * torch.rand(200, 3, dtype=DT, generator=g) - 1
    for n in (8, 16):
        m = C.Model(n)
        V = torch.randn(6, n, dtype=DT, generator=g)
        batch = m.field_batch(V, X)
        loop = torch.stack([m.field(v, X) for v in V], -1)
        assert torch.allclose(batch, loop, atol=1e-12), n


def test_labels_match_the_target_sign():
    t = O.builtin_target("sphere cap", n=1000)
    X, y = C.Sampler(t).draw(2000, torch.Generator().manual_seed(1))
    g = t.fn(X.numpy())
    assert np.all((g > 0) == (y.numpy() > 0))
    assert ((X >= -1) & (X <= 1)).all()


def test_count_pieces():
    F = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
    assert C.count_pieces(F) == 2
    assert C.count_pieces(np.zeros((0, 3), np.int64)) == 0


def test_8_values_all_methods_fit_a_flank():
    """From zero values, all three methods classify a sphere flank well and the surface
    is close to the chamfer fit's accuracy (RMS ~ 8e-3 there)."""
    t = O.builtin_target("sphere flank", n=2000)
    for method in ("logistic", "es", "grpo"):
        cfg = C.ClassifyConfig(method=method, iters=300)
        v, info = C.fit(t, 8, cfg, verbose=False)
        e = C.report(C.Model(8), v, t, C.Validation.build(C.Sampler(t)), 5)
        assert e["accuracy"] > 0.97, (method, e)
        assert e["rms"] < 0.02, (method, e)
        assert e["surface_pieces"] == 1


def test_es_gradient_points_uphill():
    """A step along the ES estimate increases the reward on the same samples."""
    t = O.builtin_target("sphere flank", n=2000)
    m, gen = C.Model(8), torch.Generator().manual_seed(3)
    X, y = C.Sampler(t).draw(4000, gen)
    mu = m.normalised(torch.randn(8, dtype=DT, generator=gen))
    r0 = C.reward(m, mu, X, y)
    grad, _ = C.es_gradient(m, mu, X, y, 64, 0.1, gen)
    r1 = C.reward(m, m.normalised(mu + 0.05 * grad / grad.norm()), X, y)
    assert r1 > r0, (r0, r1)


def test_grpo_improves_the_policy_and_learns_sigma():
    """A few GRPO iterations raise the reward of the policy mean, and sigma (learned,
    no schedule) shrinks as the group rewards concentrate."""
    t = O.builtin_target("sphere flank", n=2000)
    m, gen = C.Model(8), torch.Generator().manual_seed(5)
    X, y = C.Sampler(t).draw(4000, gen)
    cfg = C.ClassifyConfig(method="grpo")
    mu = m.normalised(torch.randn(8, dtype=DT, generator=gen)).requires_grad_(True)
    pol = C.GRPOPolicy(mu, cfg)
    r0, s0 = C.reward(m, mu.detach(), X, y), pol.sigma
    for _ in range(40):
        pol.step(m, X, y, gen, cfg.grpo_lr)
        with torch.no_grad():
            mu.copy_(m.normalised(mu))
    assert C.reward(m, mu.detach(), X, y) > r0 + 0.2, (r0, C.reward(m, mu.detach(), X, y))
    assert pol.sigma < s0


def test_16_values_beat_8_on_sphere_cap_without_bubbles():
    t = O.builtin_target("sphere cap", n=2000)
    cfg = C.ClassifyConfig(method="logistic", iters=200, starts=3, start_iters=60)
    v8, _ = C.fit(t, 8, cfg, verbose=False)
    v16, _ = C.fit(t, 16, cfg, verbose=False)
    val = C.Validation.build(C.Sampler(t))
    e8, e16 = C.report(C.Model(8), v8, t, val, 5), C.report(C.Model(16), v16, t, val, 5)
    assert e16["accuracy"] > e8["accuracy"]
    assert e16["rms"] < 0.5 * e8["rms"], (e8["rms"], e16["rms"])
    assert e16["surface_pieces"] == e16["target_pieces"]


def test_lattice_sampling_uses_the_same_k3_points():
    t = O.builtin_target("sphere flank", n=1000)
    sm = C.Sampler(t, sampling="lattice", k=7)
    g = torch.Generator().manual_seed(0)
    (X1, y1), (X2, y2) = sm.draw(4000, g), sm.draw(4000, g)
    assert X1.shape == (343, 3) and torch.equal(X1, X2) and torch.equal(y1, y2)
    assert np.all((t.fn(X1.numpy()) > 0) == (y1.numpy() > 0))
    assert C.Sampler(t, sampling="lattice", k=5, faces=True).draw(1, g)[0].shape == (125, 3)


def test_lattice_fit_classifies_the_lattice():
    """All three methods, 8 values, k = 10: nearly every lattice point is on the right side."""
    t = O.builtin_target("sphere flank", n=1500)
    for method in ("logistic", "es", "grpo"):
        cfg = C.ClassifyConfig(method=method, iters=300, sampling="lattice", k=10)
        v, _ = C.fit(t, 8, cfg, verbose=False)
        sm = C.make_sampler(t, cfg)
        e = C.report(C.Model(8, t.box), v, t, C.Validation.build(sm), 4, C.Validation.of(sm, *sm.lattice))
        assert e["accuracy_train_lattice"] > 0.98, (method, e)
        assert e["rms"] < 0.05, (method, e)


def test_classify_in_a_moved_box():
    box = Box((0.0, -1.0, 0.0), 1.0)
    t = O.builtin_target("sphere cap", n=1500, box=box)
    v, _ = C.fit(t, 8, C.ClassifyConfig(iters=200), verbose=False)
    e = C.report(C.Model(8, box), v, t, C.Validation.build(C.Sampler(t)), 4)
    assert e["accuracy"] > 0.94 and e["rms"] < 0.01 * box.size, e
    assert box.contains(C.fitted_mesh(C.Model(8, box), v, 4)[0]).all()


def test_saved_files_have_the_viewer_formats():
    t = O.builtin_target("sphere flank", n=1000)
    cfg = C.ClassifyConfig(iters=20, starts=1, start_iters=10)
    with tempfile.TemporaryDirectory() as d:
        for n in (8, 16):
            v, info = C.fit(t, n, cfg, verbose=False)
            err = C.report(C.Model(n), v, t, C.Validation.build(C.Sampler(t)), 4)
            path = os.path.join(d, f"fit{n}.json")
            C.save(path, n, v.numpy(), t, 4, cfg, err, info, 0)
            with open(path) as fh:
                data = json.load(fh)
            assert len(data["values"]) == n and data["target"] == "sphere flank"
            assert "rms" in data["final_error"]
            assert (data.get("type") == "warp16") == (n == 16)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok  {name}  ({time.time() - t0:.1f}s)")
