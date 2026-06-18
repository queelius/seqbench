# seqbench: a cheap-signal bench for byte-level sequence-prediction architectures

**Date:** 2026-06-18
**Status:** Design approved, pending spec review
**Author:** Alexander Towell (lex@metafunctor.com)

## Motivation

The goal is to design and test *new* next-byte-prediction architectures, not another
GPT, not another diffusion model, under a strong simplicity bias: intelligence is
sequence prediction with the right inductive biases, and the right architecture is a
simple algorithm rather than a pile of hacks.

The central practical obstacle to architecture research is cost. The naive way to know
whether an architecture is good is to gather a huge corpus and train for weeks. This
project removes that obstacle by building the **evaluation substrate first**. The bench
is architecture-agnostic and reusable. Once it exists, every candidate architecture is
cheap to evaluate (hours, one machine, CPU), and architectures are compared on a single
honest number.

This spec covers **only the bench**. The first neural architecture is a separate, later
project that plugs into this bench.

## Win condition

By the end of this project:

- A single command prints honest **bits-per-byte (bpb)** for any model on enwik8.
- A real baseline exists to beat (an order-N context model).
- Two synthetic diagnostics already discriminate architecture *classes* (stateless vs.
  stateful, can-copy vs. cannot).
- A scaling-sweep runner emits loss-vs-compute frontiers, ready for the first neural
  contender.
- All of it is dependency-free C++ built by a plain Makefile.

## Key principle: compression equals prediction

Next-byte prediction *is* lossless compression. The model's cross-entropy in bits,
summed over a stream, *is* the compressed size in bits. So the training objective, the
benchmark, and the philosophical north star (MDL / Solomonoff: the best compressor is
the best predictor) are one number: bits-per-byte. Byte-level keeps it tokenizer-free,
so a 60-line context model and a future neural net are compared on identical footing.

## The Model contract (the keystone)

Every architecture, baseline or neural, satisfies one small interface:

```cpp
struct Model {
  // Fill 256 FINITE logits (any real; unnormalized log-probabilities).
  // The bench applies one canonical log-softmax to score bits.
  // Models never normalize and never materialize probabilities.
  virtual void predict(float logits[256]) = 0;

  // Reveal the actual next byte. The model advances its internal state, and,
  // if it is an adaptive/online model, updates its parameters here too.
  virtual void observe(uint8_t b) = 0;

  // Optional offline fit for two-phase (neural) models. Default: no-op.
  virtual void train(ByteSpan corpus) {}

  virtual ~Model() = default;
};
```

That interface is the entire surface the bench depends on. A count-based context model
and a future SSM look identical from the outside.

### Why logits, not probabilities

- **Efficiency:** training never needs normalized probabilities; the softmax is fused
  into the loss. Real probabilities are materialized only at the (deferred)
  arithmetic-coder boundary, "only at actual inference."
- **Numerical stability, centralized and fair:** the bench scores bits with one
  canonical log-sum-exp. Because every model is scored by the *same* code, no model's
  bespoke softmax can subtly differ from another's. The bpb comparisons are
  apples-to-apples by construction.
- **The zero-frequency problem shrinks to a triviality:** `exp(finite) > 0` always, so
  any finite logit vector softmaxes to a strictly-positive PMF automatically. The
  contract requirement degrades from "emit a strictly-positive normalized distribution"
  to merely "emit finite logits (no inf/NaN)," trivial to satisfy and to check.
- **The representation excludes the pathological distributions:** an exact zero
  probability needs logit minus-infinity and an exact one-hot needs plus-infinity; both
  are infinite-confidence bets that cost infinite bits when wrong. Finite logits cannot
  represent them. The constraint and the representation are the same thing.

### Two measurement protocols, one metric

The same interface scores both kinds of model honestly:

- **Adaptive (one-pass):** `predict -> score -> observe(updates allowed)` over the
  stream. The canonical *compression* number: total bits to code the data from scratch.
  Context models / PPM live here.
- **Train/test (two-phase):** `train(train_split)`, then over the val split
  `predict -> score -> observe(state advances, params frozen)`. The canonical
  *generalization* number. Neural models live here.

Both emit the *same* bits-per-byte, so a context model and a transformer are directly
comparable on one axis. The bench supports both; a run specifies which protocol to use.

## Components (v1)

### Corpus layer
- `mmap` enwik8 (the 100 MB Hutter-Prize Wikipedia byte stream); expose a `ByteSpan`
  (pointer + length), no copies.
- Standard 90/5/5 train/val/test split.
- A tiny in-repo toy corpus for fast unit/integration tests.
- Raw bytes only, no parsing. Going byte-level is the point.
- A small fetch step downloads enwik8 into `data/` (gitignored). The bench never
  requires the network at run time once the file is present.

### Metric
- Drive a model over a stream under a chosen protocol; per position compute
  `bits = (logsumexp(logits) - logits[actual]) / ln(2)`.
- Model output is `float logits[256]` (matches neural float32, half the bandwidth); the
  metric up-casts to compute log-sum-exp and accumulates total bits in `double` with
  **Kahan summation** so the running total stays exact over ~10^8 terms.
- The exp/log-sum-exp is the *only* place transcendental functions run in the eval path.
- Report: total bits, bits-per-byte, throughput (bytes/sec), bytes processed, and
  compute spent (feeds the scaling axis).
- **Contract guard:** check logit finiteness (no inf/NaN); on violation, fail loudly
  with the offending position. No positivity or normalization check is needed.

### Baselines (no gradients, prove the pipeline today)
- **Order-N adaptive context model** with add-α smoothing. One implementation,
  parameterized by order N:
  - The prediction for the current order-N context is `log(count_b + α)` for each byte
    `b` (unnormalized log-probs are valid logits; softmax is invariant to the additive
    `log(total)`; α keeps every logit finite).
  - `observe` increments the count for the seen byte under the current context.
  - **Order-0 is simply N = 0**, the sanity floor (~5 bpb on enwik8), so there is one
    baseline code path, not two.
  - N is the capacity knob the scaling-sweep runner sweeps; the bpb-vs-N curve (gains at
    low orders, sparsity rolloff at high orders) doubles as the sweep-runner demo.
  - No backoff, no context-mixing, no PPM escape machinery (deferred until earned).
- **Documented external reference points** (so we always know where we stand without
  running them): gzip ≈ 3.1, good PPM/CM ≈ 1.8–2.0, strong neural ≈ 1.0–1.3, SOTA
  (cmix) ≈ 0.9 bpb on enwik8.

### Diagnostics (synthetic generators plus a known entropy floor)
Each diagnostic produces a byte stream with known structure and reports the *fraction of
achievable structure captured* (observed bits vs. the entropy floor of the predictable
part).

- **Induction / copy:** sequences where a previously-seen key predicts its associated
  value; a model with in-context copying scores ≈ 0 bits on the predictable positions.
- **Parity / mod-k state-tracking:** the next byte depends on a running count over
  *unbounded* history. A finite-window context model provably cannot do this beyond its
  order; a stateful model can. This is the discriminator that proves the bench can tell
  "stateful" from "stateless."

Expected v1 demonstration: the context model partially succeeds on induction and *fails*
parity, proving the diagnostics discriminate architecture classes rather than report
noise.

### Scaling-sweep runner
- Input: a model factory parameterized by a capacity knob, plus a list of configs.
- For each config: construct, then (train or run adaptively), then measure bpb.
- Output: CSV/JSON of `(config, params, compute, bpb)`, the loss-vs-compute frontier.
- v1 proves it by sweeping the context model's order N; the identical runner later
  sweeps neural width/depth/training-steps.
- Plotting is out of scope for the C++ core (emit data; visualize downstream).

## Build and layout

Plain Makefile (dependency-free, single platform). CMake is deferred until/unless we
pull in BLAS/Eigen/CUDA across machines for the neural architectures.

```
solo/
  Makefile
  include/seqbench/   model.hpp corpus.hpp metric.hpp diagnostics.hpp sweep.hpp
  src/                corpus.cpp metric.cpp diagnostics.cpp sweep.cpp
  models/             context_model.cpp        # order-N, N=0 is the sanity floor
  tools/              bench.cpp                 # CLI: model + corpus -> bpb
                      sweep_cli.cpp             # CLI: sweep a knob -> CSV
  tests/              metric_test.cpp diagnostics_test.cpp baseline_test.cpp
  data/               enwik8 (fetched; gitignored)
```

**Data flow:** corpus bytes, then the bench drives `Model` via predict/observe under a
protocol, then the metric accumulates `-log2 p` (via log-sum-exp on logits), then it
reports bpb plus throughput. Sweeps wrap that over a config list; diagnostics swap the
corpus for a generated stream plus its entropy floor.

## Testing

Per the project rule that features ship with tests:

- **Unit:** metric correctness (a known logit vector to hand-computed bits); the
  finiteness guard fires on inf/NaN; seeded-generator determinism; Kahan accumulation
  matches a high-precision reference on a small stream.
- **Golden:** order-0 (N=0) adaptive bpb on a fixed short string equals the hand-derived
  value.
- **Integration:** order-0 and order-N on the toy corpus and on enwik8, asserting bpb
  falls in expected ranges; the parity diagnostic *fails* for the context model and the
  induction diagnostic partially succeeds, encoding the "diagnostics discriminate"
  property as a test.

## Non-goals (YAGNI, deferred until earned)

- No autodiff engine (deferred until the first gradient-trained architecture; it will be
  a library *behind* the contract, not part of the bench core).
- No neural architectures (the bench is the substrate; the first architecture is the
  next project/spec).
- No BLAS/Eigen/CUDA dependency, no GPU (CPU is fine for enwik8 plus small models).
- No real arithmetic coder / `.bin` output (cross-entropy gives exact bpb without
  coding; the coder is a later add for real compressed files and the inline-PNG
  multimodal vision).
- No context-mixing / PPM-escape baseline (the order-N model is the KISS yardstick).
- No MQAR / Dyck / long-range diagnostics yet (second tier, after the first two prove the
  harness).
- No in-C++ plotting (emit CSV/JSON).

## Future directions (context, not in scope)

Once the bench exists, the architectures to race through it, each tagged by the inductive
bias it bakes in, include: linear/state-space RNNs (the past compresses into a bounded
state), fast-weights / test-time-training models (prediction is online compression of
what you have seen), predictive-coding hierarchies (local learning rules, no global
backprop), modern Hopfield / associative-memory models, and energy-based sequence models.
The bench is deliberately agnostic to which of these comes first and to how each one
learns.
