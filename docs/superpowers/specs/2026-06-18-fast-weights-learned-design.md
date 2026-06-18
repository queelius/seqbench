# Learned fast-weights (DeltaNet-style) on libtorch

**Date:** 2026-06-18
**Status:** Design approved, pending spec review
**Author:** Alexander Towell (lex@metafunctor.com)

## Motivation

The gradient-free fast-weights cut (fixed random key/value embeddings, delta-rule memory,
tied readout) was built and measured on the bench. The finding was clean and negative: it
is a lossy soft-bigram, and the exact-count context-model baseline beats it on the small
synthetic recall diagnostics (on the bench's induction diagnostic it scored worse than the
uniform baseline, because a fixed point-estimate recaller cannot represent the calibrated
"I do not know" that the smoothed n-gram gets for free). One real fix surfaced along the
way: the readout needs a retrieval inverse-temperature, without which even a perfect recall
sits near the 8-bits-per-byte floor.

This spec builds the version that should actually compete: a **learned** fast-weights layer
whose projections, embeddings, and readout are trained by backpropagation through the
fast-weight recurrence. Learning the projections lets the model represent calibrated
predictions (sharp where the data is predictable, uncertain where it is not), and a learned
readout absorbs the output-scale (the gain) that had to be hand-tuned before. We use
libtorch for autodiff and optimization so the effort goes into the architecture, not the
gradient plumbing.

## Win condition

By the end of this project:

- A single learned fast-weights (DeltaNet-style) layer trains on enwik8 via libtorch
  backprop-through-time.
- The trained model plugs into the existing bench (scored by `run_adaptive` and
  `score_diagnostic`) and emits a provenance-complete `RunRecord` to `runs/results.jsonl`.
- We can read off, from the records, the research answer: does learning the projections let
  fast-weights beat the context-model baseline on enwik8 bits-per-byte and capture the
  induction diagnostic (which the gradient-free version could not)?
- libtorch is confined to `arch/fast-weights/`; the bench core stays dependency-free and its
  `make test` still runs without libtorch.

This is a research run: whether the model wins is the question we are answering, not an
assertion we bake into a test.

## Scope and integration

In scope: one learned fast-weights layer, its libtorch training loop, the bench-eval
adapter, the runner that trains-evaluates-records, and the CMake build.

Two isolated worlds:

- **Bench core** (existing): the `Model` contract, metric, corpus, diagnostics, sweep, and
  experiment-tracking layer. Dependency-free, plain Makefile, untouched. `make test` does
  not need libtorch.
- **Learned model** (new): everything libtorch lives under `arch/fast-weights/`, built by its
  own CMake build that links libtorch and the bench's compiled objects + headers. It trains
  via libtorch autograd, then satisfies the bench `Model` contract for evaluation, reusing
  `run_adaptive` / `score_diagnostic` / the JSONL run records unchanged.

Deferred (later specs): learned input-dependent `beta`/`lambda` gates; chunk-parallel BPTT;
multi-layer stacks; stateful cross-chunk BPTT; GPU; tokenization; exporting weights to a
dependency-free eval path.

## The model (single DeltaNet-style layer)

Let `d` be the model/state dimension. Learned parameters: a byte embedding table
`Emb` (256 x d), three linear projections `W_k`, `W_v`, `W_q` (d x d), and a separate
learned readout `Readout` (Linear d -> 256, with bias). Fixed hyperparameters for this cut:
the write strength `beta`, the decay `lambda`, and `d`.

For a chunk of bytes `b_0 .. b_{T-1}`:

- `x_t = Emb[b_t]`
- `k_t = normalize(W_k x_t)` (L2-normalized key, for delta-rule stability), `v_t = W_v x_t`,
  `q_t = W_q x_t`
- fast-weight recurrence, `W_{-1} = 0`:
  `W_t = lambda * W_{t-1} + beta * (v_t - W_{t-1} k_t) k_t^T`   (state W is d x d)
- read with the query: `o_t = W_t q_t`
- `logits_t = Readout(o_t)` in R^256, which predicts `b_{t+1}`
- loss = mean over `t = 0 .. T-2` of cross-entropy(`logits_t`, `b_{t+1}`), reported in
  bits-per-byte (natural-log CE divided by ln 2)

`o_t` incorporates inputs up to and including `b_t` (the update at step t and the query
`q_t` both use `x_t`), so `logits_t` predicting `b_{t+1}` is causal next-byte prediction.

The retrieval gain that was a hand-tuned knob in the gradient-free version is no longer a
knob: the learned `Readout` sets the output scale during training.

Default starting hyperparameters (all recorded per run): `d = 128`, `beta = 1.0`,
`lambda = 0.99`, `seq_len = 256`, `batch = 32`.

## Training

- **Data:** sample random `seq_len`-byte chunks from the enwik8 train split (the standard
  90% prefix), batched. If `data/enwik8` is absent, the runner errors with a clear message
  (fetch via the bench's `make data/enwik8`).
- **Truncated BPTT:** the recurrence runs over each chunk with `W` reset to zero at the chunk
  start (independent chunks). `lambda` bounds the effective memory horizon to about
  `1/(1-lambda)`, kept at or below `seq_len`.
- **Optimizer:** Adam, learning rate 1e-3.
- **Budget:** a fixed **step count** (default large and configurable via `--steps`; each step
  is one batched forward/backward/update). Step-count makes runs reproducible. The runner
  periodically evaluates bits-per-byte on a held-out enwik8 slice and keeps the best
  checkpoint by val bpb.
- A fixed seed makes a run reproducible (libtorch manual seed + the data sampler seed).

## Evaluation and records

The best checkpoint is wrapped as a bench `Model`:

- state: `W` (d x d), the last read `o` (length d), and a flag for whether any byte has been
  observed.
- `observe(b)`: `x = Emb[b]`; `k = normalize(W_k x)`; `v = W_v x`; `q = W_q x`;
  `W = lambda*W + beta*(v - W k) k^T`; `o = W q`. Runs under libtorch `no_grad` with frozen
  weights.
- `predict(logits)`: `logits = Readout(o)` if a byte has been observed, else `Readout(0)`
  (a learned prior for the very first byte). Copied into the `float logits[256]` the contract
  expects.

This online recurrence is identical to the training forward, so the streaming eval bpb
matches the batched training bpb on the same data (verified by a test). The wrapped model is
scored by the bench: enwik8 val bits-per-byte (`run_adaptive`), the induction and parity
diagnostics (`score_diagnostic`). The runner emits a `RunRecord`:

```json
{"arch":"fast-weights-learned","version":"v1-deltanet",...,
 "config":{"d":128,"beta":1.0,"lambda":0.99,"seq_len":256,"batch":32,"steps":...,"lr":0.001},
 "corpus":{"name":"enwik8-val","bytes":...},
 "results":{"bpb":...,"induction_fraction":...,"parity_fraction":...,"train_bpb":...}}
```

We compare these numbers against the committed context-model and gradient-free records to
answer the research question.

## Build

`arch/fast-weights/CMakeLists.txt`: `find_package(Torch REQUIRED)` (it sets the include
paths, libraries, and the C++ ABI flags), building a `train_fw` executable that links
libtorch and compiles the bench source files it needs directly (`src/corpus.cpp`,
`src/metric.cpp`, `src/diagnostics.cpp`, `src/experiment.cpp`, `models/context_model.cpp`),
with `include/` on the include path. Compiling the bench sources rather than linking
`make`-built objects keeps the CMake build self-contained. Invoked with the venv torch cmake
prefix:

```
cmake -S arch/fast-weights -B arch/fast-weights/build \
  -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/fast-weights/build
```

`arch/fast-weights/README.md` documents this (and the one-liner that regenerates the prefix:
`python3 -c "import torch;print(torch.utils.cmake_prefix_path)"`, used only at build-config
time, never at runtime). The bench's Makefile and `make test` are untouched and remain
libtorch-free. The CMake build is self-contained (it compiles the bench sources it needs),
so it does not require running `make` first.

## Testing (correctness of the machinery, not the research result)

- **Overfit a tiny sequence:** train the model on a short repeating byte pattern for a small
  number of steps and assert the loss drops to near zero. This proves the whole loop
  (embedding -> recurrence -> readout -> cross-entropy -> backward -> Adam step) actually
  learns, i.e., BPTT through the fast-weight recurrence is wired correctly.
- **Forward sanity:** the forward produces logits of shape `[*, 256]` that are finite; the
  model is deterministic given a fixed seed.
- **Train/eval consistency:** the streaming eval adapter's mean bits-per-byte on a short
  sequence matches the batched training-forward bits-per-byte on the same sequence within a
  small tolerance. This proves the online `predict/observe` recurrence replicates the
  training recurrence.
- Full enwik8 training is a runner invocation, not a unit test (too slow for the suite). The
  tests use tiny data and few steps.
- No test asserts that the learned model beats the context model; that is the recorded
  research outcome.

## Future directions (context, not in scope)

Learned `beta`/`lambda` gates (input-dependent write and decay, the full DeltaNet); a
chunk-parallel BPTT for speed; stacking layers with MLP + residual; carrying state across
chunks; comparison against the SSM / linear-RNN family; and, if the learned layer wins,
scaling `d` and `steps` along the bench's loss-vs-compute frontier.
