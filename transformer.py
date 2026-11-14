"""A decoder-only Transformer (GPT-style) implemented from scratch in pure NumPy.

No deep-learning frameworks. Forward pass, cross-entropy loss, and full manual
backpropagation through multi-head causal self-attention, layernorm, and the MLP,
trained with Adam.

Author: Yashraj Pangave
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    def __init__(self, dim):
        self.g = np.ones(dim)
        self.b = np.zeros(dim)
        self.dg = np.zeros_like(self.g)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.mu = x.mean(axis=-1, keepdims=True)
        xc = x - self.mu
        self.var = (xc ** 2).mean(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + 1e-5)
        self.xhat = xc / self.std
        self.x = x
        return self.g * self.xhat + self.b

    def backward(self, dout):
        n = dout.shape[-1]
        self.dg += np.sum(dout * self.xhat, axis=tuple(range(dout.ndim - 1)))
        self.db += np.sum(dout, axis=tuple(range(dout.ndim - 1)))
        dxhat = dout * self.g
        # layernorm backward (Ba et al., 2016)
        dx = (1.0 / self.std) * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )
        return dx

    def params(self):
        return [("g", self.g, self.dg), ("b", self.b, self.db)]


class Linear:
    def __init__(self, fan_in, fan_out, rng):
        self.W = rng.normal(0.0, 0.02, size=(fan_in, fan_out))
        self.b = np.zeros(fan_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW += self.x.reshape(-1, self.x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        self.db += dout.reshape(-1, dout.shape[-1]).sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class MultiHeadAttention:
    def __init__(self, dim, n_heads, rng):
        assert dim % n_heads == 0
        self.dim = dim
        self.nh = n_heads
        self.hd = dim // n_heads
        self.qkv = Linear(dim, 3 * dim, rng)
        self.proj = Linear(dim, dim, rng)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv.forward(x)                       # (B, T, 3C)
        q, k, v = np.split(qkv, 3, axis=-1)
        # (B, nh, T, hd)
        self.q = q.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)
        self.k = k.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)
        self.v = v.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)

        scores = (self.q @ self.k.transpose(0, 1, 3, 2)) / np.sqrt(self.hd)
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -1e10)
        self.attn = softmax(scores, axis=-1)            # (B, nh, T, T)
        ctx = self.attn @ self.v                        # (B, nh, T, hd)
        self.ctx_shape = ctx.shape
        merged = ctx.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj.forward(merged)

    def backward(self, dout):
        B, T, C = dout.shape
        dmerged = self.proj.backward(dout)
        dctx = dmerged.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)

        dattn = dctx @ self.v.transpose(0, 1, 3, 2)     # (B, nh, T, T)
        dv = self.attn.transpose(0, 1, 3, 2) @ dctx     # (B, nh, T, hd)
        # softmax backward
        dscores = self.attn * (dattn - np.sum(dattn * self.attn, axis=-1, keepdims=True))
        dq = dscores @ self.k / np.sqrt(self.hd)
        dk = dscores.transpose(0, 1, 3, 2) @ self.q / np.sqrt(self.hd)

        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, C)
        dx = self.qkv.backward(np.concatenate([dq, dk, dv], axis=-1))
        return dx

    def params(self):
        return self.qkv.params() + self.proj.params()


class Block:
    def __init__(self, dim, n_heads, rng):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, rng)
        self.ln2 = LayerNorm(dim)
        self.fc1 = Linear(dim, 4 * dim, rng)
        self.fc2 = Linear(4 * dim, dim, rng)

    def forward(self, x):
        self.x1 = x
        h = self.ln1.forward(x)
        x = x + self.attn.forward(h)
        self.x2 = x
        h2 = self.ln2.forward(x)
        self.act = h2 @ self.fc1.W + self.fc1.b
        self.fc1.x = h2                              # keep for Linear.backward
        self.relu = np.maximum(self.act, 0.0)
        out = self.fc2.forward(self.relu)
        return x + out

    def backward(self, dout):
        d = self.fc2.backward(dout)
        d = d * (self.act > 0)
        d = self.fc1.backward(d)
        d = self.ln2.backward(d)
        dout = dout + d                              # residual
        d = self.ln1.backward(self.attn.backward(dout))
        return dout + d                              # residual

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + \
            self.fc1.params() + self.fc2.params()


class TinyTransformer:
    def __init__(self, vocab_size, dim=64, n_heads=4, n_layers=2, ctx=64, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.wte = rng.normal(0.0, 0.02, size=(vocab_size, dim))
        self.wpe = rng.normal(0.0, 0.02, size=(ctx, dim))
        self.dwte = np.zeros_like(self.wte)
        self.dwpe = np.zeros_like(self.wpe)
        self.blocks = [Block(dim, n_heads, rng) for _ in range(n_layers)]
        self.lnf = LayerNorm(dim)
        self.head = Linear(dim, vocab_size, rng)

    def forward(self, idx):
        """idx: (B, T) int. Returns logits (B, T, V)."""
        B, T = idx.shape
        self.T = T
        h = self.wte[idx] + self.wpe[np.arange(T)]
        self.idx = idx
        for blk in self.blocks:
            h = blk.forward(h)
        h = self.lnf.forward(h)
        return self.head.forward(h)

    def loss_and_grads(self, idx, targets):
        """idx, targets: (B, T) int. Returns scalar mean cross-entropy; fills grads."""
        logits = self.forward(idx)
        B, T, V = logits.shape
        probs = softmax(logits)
        flat_p = probs.reshape(-1, V)
        flat_t = targets.reshape(-1)
        loss = -np.log(flat_p[np.arange(flat_t.size), flat_t] + 1e-12).mean()

        dlogits = probs
        dlogits.reshape(-1, V)[np.arange(flat_t.size), flat_t] -= 1.0
        dlogits /= flat_t.size

        dh = self.head.backward(dlogits)
        dh = self.lnf.backward(dh)
        for blk in reversed(self.blocks):
            dh = blk.backward(dh)
        # token + position embedding grads
        np.add.at(self.dwte, self.idx, dh)
        self.dwpe[:] = dh.sum(axis=0)
        return loss

    def params(self):
        ps = [("wte", self.wte, self.dwte), ("wpe", self.wpe, self.dwpe)]
        for blk in self.blocks:
            ps += blk.params()
        ps += self.lnf.params() + self.head.params()
        return ps

    def zero_grads(self):
        self.dwte[:] = 0.0
        self.dwpe[:] = 0.0
        for _, _, g in self.params():
            g[:] = 0.0

    def count_params(self):
        return int(sum(p.size for _, p, _ in self.params()))

    def generate(self, idx, max_new_tokens, rng):
        """Autoregressive sampling. idx: (1, T0) int seed sequence."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.ctx:]
            logits = self.forward(ctx)
            next_id = np.array([[int(rng.choice(self.vocab_size, p=softmax(logits[0, -1])))]])
            idx = np.concatenate([idx, next_id], axis=1)
        return idx
