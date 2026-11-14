"""Train the tiny transformer on a character-level corpus (Tiny Shakespeare).

Reproducible: fixed seed, fixed split, exact measured numbers printed at the end.

Author: Yashraj Pangave
"""
import argparse
import os
import urllib.request

import numpy as np

from transformer import TinyTransformer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "input.txt")


def load_corpus():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(data, batch_size, ctx, rng):
    ix = rng.integers(0, len(data) - ctx - 1, size=batch_size)
    x = np.stack([data[i:i + ctx] for i in ix])
    y = np.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x, y


def evaluate(model, data, batch_size, ctx, n_batches, rng):
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, ctx, rng)
        model.zero_grads()
        losses.append(model.loss_and_grads(x, y))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=48)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=250_000,
                    help="truncate corpus to keep pure-NumPy training time bounded")
    args = ap.parse_args()

    text = load_corpus()[: args.max_chars]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    rng = np.random.default_rng(args.seed)
    model = TinyTransformer(vocab_size=len(chars), dim=args.dim, n_heads=args.heads,
                            n_layers=args.layers, ctx=args.ctx, seed=args.seed)
    print(f"corpus chars: {len(data):,} | vocab: {len(chars)} | "
          f"train/val: {len(train_data):,}/{len(val_data):,} | "
          f"parameters: {model.count_params():,}")

    # Adam state
    m = {id(p): np.zeros_like(p) for _, p, _ in model.params()}
    v = {id(p): np.zeros_like(p) for _, p, _ in model.params()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    eval_rng = np.random.default_rng(args.seed + 1)
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.ctx, rng)
        model.zero_grads()
        loss = model.loss_and_grads(x, y)
        for _, p, g in model.params():
            k = id(p)
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g ** 2
            mh = m[k] / (1 - b1 ** step)
            vh = v[k] / (1 - b2 ** step)
            p -= args.lr * mh / (np.sqrt(vh) + eps)
        if step % 250 == 0 or step == 1:
            print(f"step {step:5d}/{args.steps}  train loss {loss:.4f}")

    train_loss = evaluate(model, train_data, args.batch_size, args.ctx, 20, eval_rng)
    val_loss = evaluate(model, val_data, args.batch_size, args.ctx, 20, eval_rng)
    print(f"\nFINAL measured (seed={args.seed}, split=90/10 first-block, "
          f"steps={args.steps}, batch={args.batch_size}, ctx={args.ctx}):")
    print(f"  train loss: {train_loss:.4f}")
    print(f"  val loss:   {val_loss:.4f}")
    print(f"  val perplexity: {np.exp(val_loss):.2f}")

    gen_rng = np.random.default_rng(args.seed + 2)
    seed_idx = np.zeros((1, 1), dtype=np.int64)  # newline-ish start; vocab[0]
    out = model.generate(seed_idx, 300, gen_rng)
    print("\nsample generation (300 chars):\n")
    print("".join(itos[int(i)] for i in out[0]))


if __name__ == "__main__":
    main()
