# Stacked gated DeltaNet

**Date:** 2026-06-21
**Status:** Design approved, pending spec review
**Author:** Alexander Towell (lex@metafunctor.com)

## Motivation

The single-layer learned fast-weights model plateaued at 3.53 bits-per-byte on enwik8,
between the order-1 (3.91) and order-2 (3.31) context-model baselines. The bottleneck is not
training budget (the loss was flat over the last 4000 of 10000 steps) but the architecture:
one fast-weights layer with no depth and no MLP. This spec scales it into the standard
small-language-model recipe, a stack of pre-norm blocks where fast-weights token mixing
replaces softmax attention, plus the learned input-dependent write gate that makes DeltaNet
train. Depth and the MLP are the dominant bits-per-byte levers, so this is the version that
could actually cross order-2.

Honest cost: a multi-block d=128 model is roughly n_layers times the single layer's cost
(about 0.83 s/step single-layer on this CPU), so a real enwik8 run is multi-hour to a day.
The build is the deliverable here; the long training run is a separate, deliberate
experiment.

## Win condition

By the end:

- A stacked gated DeltaNet (N pre-norm blocks of fast-weights mixing + MLP) trains via the
  shared experiment runner and is scored through the bench on enwik8 bits-per-byte and the
  capability diagnostics, recording `arch:"deltanet"`.
- The records are directly comparable to `fast-weights-learned` (3.53) and the context-model
  (order-2 3.31), so we can see whether depth + MLP + gating crosses order-2.

## Scope and isolation

In scope: the DeltaNet block model, its online eval adapter, tests, runner, and CMake build,
all under `arch/deltanet/`, reusing the shared runner (`arch/common/runner.hpp`) and the
bench. The bench core stays dependency-free; libtorch stays under `arch/`.

Deferred: multi-head mixing, learned `lambda` / forget gate, chunk-parallel recurrence, GPU,
and the long enwik8 training run itself.

## The model (`arch/deltanet/deltanet_model.hpp`, `.cpp`)

```
emb(256 -> d) -> [ block ] x N -> final LayerNorm -> readout(d -> 256)

block(x):                          # pre-norm residual block, x is [B,T,d]
  x = x + FWMix(LayerNorm(x))      # fast-weights token mixing
  x = x + MLP(LayerNorm(x))        # position-wise MLP, ratio 4, GELU
```

`FWMix(h)` maps `[B,T,d] -> [B,T,d]` (single head for this version):

- `k = normalize(Wk h, last dim)`, `v = Wv h`, `q = Wq h`   (each `[B,T,d]`)
- `beta = sigmoid(Wbeta h)`                                  (`[B,T,1]`, learned write gate)
- recurrence, `W_{-1} = 0` (state `W` is `[B,d,d]`):
  `W_t = lambda * W_{t-1} + beta_t * (v_t - W_{t-1} k_t) k_t^T`
- read `o_t = W_t q_t`; output `out = Wo(o)`                 (`Wo: d -> d`)

`MLP(h) = Linear(d, 4d) -> GELU -> Linear(4d, d)`. The mixing reuses the delta recurrence
from the single-layer model (key L2-normalized, `lambda` a fixed knob), now wrapped in the
residual/norm block and gated by the learned `beta`. The block module owns `Wk, Wv, Wq,
Wbeta, Wo`, two LayerNorms, and the MLP; the model owns the embedding, `N` blocks, a final
LayerNorm, and the readout. Knobs: `d` (model dim), `n_layers` (N), `lambda`. Records
`arch:"deltanet"`, `config.d`, `config.n_layers`, `config.lambda`.

Forward is causal next-byte: `FWMix`'s `o_t` incorporates inputs up to and including `b_t`
(the recurrence update and the query both use `x_t`), and the residual stream and MLP are
position-wise, so the hidden at position `t` depends only on inputs `<= t`. `logits_t`
predicts `b_{t+1}`, same shift as the single-layer model.

## Online eval adapter (`DeltaNetEval`)

The only cross-time state is the `N` fast-weight matrices `W_1..W_N` (the residual stream,
LayerNorms, and MLPs are all position-wise, so applying them to a single position online
exactly matches the batched per-position computation). The adapter holds those `N` matrices
and the post-stack hidden:

- `observe(b)`: `x = emb[b]`; for each block, run `FWMix` on `LayerNorm(x)` (updating that
  block's `W`, reading `o`), add the residual, then add the MLP of `LayerNorm(x)`; store the
  final post-stack `x`.
- `predict(logits)`: `readout(finalLayerNorm(x))` if a byte has been observed, else the same
  applied to zeros.

Runs under libtorch `no_grad`. This reproduces the batched forward exactly, verified by the
train/eval consistency test.

## Build (`arch/deltanet/CMakeLists.txt`)

Mirrors `arch/gru/`: `find_package(Torch)`, compile `bench_core`, build `train_deltanet`
(from `train_deltanet.cpp` + `deltanet_model.cpp`) and `deltanet_test` (from
`deltanet_test.cpp` + `deltanet_model.cpp`), with `${BENCH_ROOT}/include`,
`${BENCH_ROOT}/tests`, `${BENCH_ROOT}/arch` (for `common/runner.hpp`), and the deltanet dir
on the include path. The runner is `train_deltanet.cpp`: parse `--d`/`--n-layers`/`--lambda`
+ the common args, build the model, call `run_experiment<DeltaNet>(...)` with `bpb_loss` and
a `DeltaNetEval` factory. The bench Makefile and `make test` stay untouched and
libtorch-free.

## Tests (`arch/deltanet/deltanet_test.cpp`)

- Forward produces `[*, 256]` finite logits; deterministic given a seed.
- Overfit a tiny repeating sequence: loss collapses below 1.0, proving the stacked block
  model trains end to end (BPTT through N blocks).
- Train/eval consistency: the online multi-block adapter's per-byte bits match the batched
  forward on a short sequence within a small tolerance.
- No test asserts a bits-per-byte win; that is the recorded experiment. Thresholds are
  verified-achievable; a mismatch is reported BLOCKED with the measured numbers, never
  relaxed.

## Non-goals

Multi-head mixing, learned `lambda`/forget gate, chunk-parallel BPTT, GPU, and the long
enwik8 training run that would actually test whether DeltaNet beats order-2.

## Future directions (context)

If a small stack beats order-2, the next levers are multi-head mixing, learned decay/forget
gates, more layers/dim along the bench's loss-vs-compute frontier, and GPU to make the long
runs cheap. If it does not, that is itself informative about the fast-weights mixing as a
language-model primitive at this scale.
