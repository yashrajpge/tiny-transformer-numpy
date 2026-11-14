"""Numerical gradient check for the from-scratch transformer.

Validates manual backprop against central finite differences on a tiny config.
Run: python tests/test_gradients.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformer import TinyTransformer  # noqa: E402


def test_gradient_check():
    rng = np.random.default_rng(42)
    V, B, T = 17, 2, 8
    # Finite differences need local smoothness: gradient-check at a generic
    # (deterministically perturbed) parameter point whose ReLU pre-activations
    # stay clear of the kink at 0 (|act| >> eps). Backprop must hold at any
    # parameter point, not only at init.
    model = idx = targets = None
    prng = np.random.default_rng(7)
    for seed in range(200):
        m = TinyTransformer(vocab_size=V, dim=8, n_heads=2, n_layers=2, ctx=T, seed=seed)
        for _, p, _ in m.params():
            p += prng.normal(0.0, 0.1, size=p.shape)
        i = rng.integers(0, V, size=(B, T))
        t = rng.integers(0, V, size=(B, T))
        m.zero_grads()
        m.loss_and_grads(i, t)
        min_act = min(np.abs(blk.act).min() for blk in m.blocks)
        if min_act > 1e-3:
            model, idx, targets = m, i, t
            break
    assert model is not None, "no locally smooth config found in 200 seeds"
    model.zero_grads()
    loss = model.loss_and_grads(idx, targets)
    assert np.isfinite(loss)

    eps = 1e-5
    worst = 0.0
    checked = 0
    for name, p, g in model.params():
        flat_p = p.reshape(-1)
        flat_g = g.reshape(-1)
        # check a deterministic spread of entries (full check for small arrays)
        n = flat_p.size
        positions = np.unique(np.linspace(0, n - 1, min(n, 40)).astype(int))
        for i in positions:
            orig = flat_p[i]
            flat_p[i] = orig + eps
            lp = _loss_only(model, idx, targets)
            flat_p[i] = orig - eps
            lm = _loss_only(model, idx, targets)
            flat_p[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = flat_g[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            checked += 1
    print(f"gradient check: {checked} entries, worst relative error {worst:.3e}")
    assert worst < 1e-4, f"gradient check failed: worst rel err {worst:.3e}"


def _loss_only(model, idx, targets):
    logits = model.forward(idx)
    B, T, V = logits.shape
    z = logits.reshape(-1, V)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    t = targets.reshape(-1)
    return -np.log(p[np.arange(t.size), t] + 1e-12).mean()


def test_overfit_tiny_batch():
    """A tiny model must be able to overfit one fixed batch: mechanics sanity check."""
    rng = np.random.default_rng(1)
    V, B, T = 11, 4, 12
    model = TinyTransformer(vocab_size=V, dim=32, n_heads=4, n_layers=1, ctx=T, seed=1)
    idx = rng.integers(0, V, size=(B, T))
    targets = rng.integers(0, V, size=(B, T))
    losses = []
    for step in range(1, 301):
        model.zero_grads()
        loss = model.loss_and_grads(idx, targets)
        losses.append(loss)
        for _, p, g in model.params():
            p -= 0.02 * g  # plain SGD is enough for the sanity check
    print(f"overfit check: first loss {losses[0]:.4f} -> last loss {losses[-1]:.4f}")
    assert losses[-1] < 0.2 * losses[0], "model failed to fit a fixed batch"


def test_causality():
    """Changing a future token must not change logits at earlier positions."""
    model = TinyTransformer(vocab_size=13, dim=16, n_heads=2, n_layers=2, ctx=8, seed=7)
    idx = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    out1 = model.forward(idx.copy())
    idx2 = idx.copy()
    idx2[0, 6] = 9  # change a future token
    out2 = model.forward(idx2)
    assert np.allclose(out1[0, :6], out2[0, :6]), "causal mask broken: past depends on future"


if __name__ == "__main__":
    test_gradient_check()
    test_overfit_tiny_batch()
    test_causality()
    print("all tests passed")
