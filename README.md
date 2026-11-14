# tiny-transformer-numpy

A decoder-only Transformer (GPT-style) implemented **from scratch in pure NumPy** — no PyTorch, no TensorFlow, no autograd. Forward pass, causal multi-head self-attention, layernorm, cross-entropy loss, full manual backpropagation, and Adam, all written out explicitly and validated by numerical gradient checking.

Character-level language model trained on a 250,000-character slice of Tiny Shakespeare (public domain).

## Why

Frameworks hide the math. This repo makes every gradient explicit: attention scores, softmax backward, layernorm backward, residual streams, embedding gradients. The test suite proves the hand-written derivatives match central finite differences to a worst-case relative error of **1.7e-6** over 681 checked parameter entries.

## Measured results

Exact numbers from the committed training run (protocol below, fully reproducible):

| Metric | Value |
|---|---|
| Parameters | 111,166 |
| Vocabulary | 62 characters |
| Corpus | 250,000 chars (train 225,000 / val 25,000, first-block 90/10 split) |
| Final train loss (eval, 20 batches) | **1.7203** |
| Final val loss (eval, 20 batches) | **1.8188** |
| Val perplexity | **6.16** |
| Training wall time | ~49 s (CPU only, NumPy) |

Training protocol: `dim=64, heads=4, layers=2, ctx=48, batch=16, steps=1500, Adam(lr=3e-3, betas=0.9/0.999), seed=0`. Batch train loss trajectory: 4.1250 (step 1) → 2.2239 (250) → 2.0082 (500) → 1.9272 (750) → 1.8733 (1000) → 1.8504 (1250) → 1.8793 (1500, single-batch).

Sample generation after training (unedited):

```
Folly Marced:
Sun consult thaterme? do, still facpest you deal,
WeUld with you.

Than thaum I co cendeest Touch stoo whast but thy way!
```

The model has learned line structure, character-name formatting, and word-like statistics of Shakespeare — not meaning. That is the expected behavior of a 111k-parameter character model trained for ~1.15M tokens.

## Run it

```bash
pip install numpy
python train.py            # downloads Tiny Shakespeare on first run, trains, prints measured numbers
python tests/test_gradients.py
```

## Tests

`tests/test_gradients.py` contains three checks:

1. **Gradient check** — analytic backprop vs central finite differences at a generic (deterministically perturbed) parameter point; worst relative error 1.672e-06 over 681 entries. The perturbation keeps ReLU pre-activations clear of the kink at 0, which finite differences cannot straddle.
2. **Overfit check** — a tiny model must drive one fixed batch's loss from 2.41 to 0.06 with plain SGD.
3. **Causality check** — mutating a future token must not change logits at earlier positions (verifies the attention mask).

All three pass.

## Honest limitations

- Pure NumPy means no GPU: training is CPU-bound and the default config is deliberately small (~49 s) so anyone can reproduce it on a laptop.
- Character-level modeling with 111k parameters is a teaching-scale model. Perplexity 6.16 on this slice is a measured fact about this protocol, not a claim of competitiveness with framework-scale models.
- The corpus is a 250k-character truncation of Tiny Shakespeare to keep runtime bounded; the exact truncation is part of the protocol (`--max-chars 250000`).

## License

MIT — see LICENSE. Tiny Shakespeare is public domain (Project Gutenberg Shakespeare, as packaged by karpathy/char-rnn).

Author: Yashraj Pangave
