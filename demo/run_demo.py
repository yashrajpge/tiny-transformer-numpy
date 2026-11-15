"""Staged-training demo: records the loss trajectory and generated samples at
training milestones (0 / 250 / 500 / 1000 / 1500 steps) in one CPU run.

Usage:  python demo/run_demo.py
Output: demo_stages.json (loss per step + verbatim 220-char samples per stage)

Config matches README: dim=64, 4 heads, 2 layers, ctx=48, batch=16, lr=3e-3,
seed=0, 250k-char Tiny Shakespeare corpus (data/input.txt), 220-char samples
drawn with a fixed generator seed so every stage is directly comparable.
"""
import json
import numpy as np
from transformer import TinyTransformer
from train import load_corpus, get_batch

STAGES = [0, 250, 500, 1000, 1500]
DIM, HEADS, LAYERS, CTX, BATCH, LR = 64, 4, 2, 48, 16, 3e-3

text = load_corpus()[:250_000]
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = np.array([stoi[c] for c in text], dtype=np.int64)
train_data = data[: int(0.9 * len(data))]

model = TinyTransformer(vocab_size=len(chars), dim=DIM, n_heads=HEADS,
                        n_layers=LAYERS, ctx=CTX, seed=0)
rng = np.random.default_rng(0)
m = {id(p): np.zeros_like(p) for _, p, _ in model.params()}
v = {id(p): np.zeros_like(p) for _, p, _ in model.params()}
b1, b2, eps = 0.9, 0.999, 1e-8

def sample():
    g = np.random.default_rng(1234)  # fixed seed: comparable across stages
    out = model.generate(np.zeros((1, 1), dtype=np.int64), 220, g)
    return ''.join(itos[int(i)] for i in out[0])

out = {'losses': [], 'stages': {'0': {'loss': None, 'sample': sample()}}}
step = 0
for target in STAGES[1:]:
    while step < target:
        step += 1
        x, y = get_batch(train_data, BATCH, CTX, rng)
        model.zero_grads()
        loss = model.loss_and_grads(x, y)
        for _, p, g in model.params():
            k = id(p)
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g ** 2
            p -= LR * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)
        out['losses'].append([step, round(float(loss), 4)])
    out['stages'][str(target)] = {'loss': round(float(loss), 4), 'sample': sample()}
    print(f'step {target}: loss {loss:.4f}')

json.dump(out, open('demo_stages.json', 'w'), indent=1)
