# Gradient-free fast-weights predictor + experiment tracking

**Date:** 2026-06-18
**Status:** Design approved, pending spec review
**Author:** Alexander Towell (lex@metafunctor.com)

## Motivation

The first new architecture to race through the seqbench bench is **fast-weights**: a
sequence predictor whose recurrent state is a weight matrix updated by a local online
learning rule, read out by query. It is the Schmidhuber-1992 fast-weights idea, the same
object as modern linear attention / DeltaNet / test-time-training. It fits the user's
simplicity bias (prediction as online compression of what has been seen) and is neither
a GPT nor a diffusion model.

A faithful fast-weights model trains its slow (projection) parameters by backpropagation
through the fast-weight recurrence, which needs the autodiff engine the bench deferred.
This spec deliberately avoids that: it builds a **gradient-free** fast-weights model with
fixed projections, which still tests the central question (does the fast-weights
inductive bias help?) and should excel at the bench's induction (recall) diagnostic,
while costing nothing in autodiff infrastructure. Autodiff and learned projections are a
later spec.

This spec also establishes the **experiment-tracking discipline** for all future
architectures: provenance-complete, machine-readable run records, plus a per-architecture
home for conceptual documentation.

## Win condition

By the end of this project:

- A gradient-free fast-weights `Model` plugs into the existing bench unchanged (scored by
  `run_adaptive`, `score_diagnostic`, `run_sweep`).
- A single runner command evaluates a configuration across the full battery (bits-per-byte
  on a corpus, both diagnostics) and appends a provenance-complete JSON record to a
  committed `runs/results.jsonl`.
- Fast-weights demonstrably **beats the context-model baseline on the induction diagnostic**
  and **fails the parity diagnostic** (no unbounded state), showing the bench discriminates
  the architecture class.
- `arch/fast-weights/` holds conceptual markdown docs explaining the mechanism and the
  implementation choices.

## Scope and non-goals

In scope: (a) the gradient-free fast-weights model, (b) a reusable experiment-tracking
layer, (c) the `arch/fast-weights/` subdir with docs and a runner CLI.

Deferred (later specs): the autodiff engine; learned (slow) projections via backprop
through time; multi-head / attention-style structure; order-sensitive keys; a
query/leaderboard tool (jq over the JSONL suffices for v1); GPU; moving the context-model
baseline under `arch/`.

The bench's library and `Model` contract are unchanged. The only edit to existing files is
a one-line Makefile addition to build `arch/*/*.cpp` as tools; everything else is new
files.

## The model: fast-weights as online associative memory

State carried by the model:

- `W`: the fast-weight matrix, shape d x d, initialized to zero. This is the recurrent
  memory.
- `history`: the last `key_width` observed bytes (empty before the first observe). For the
  default `key_width = 1` this is just the most recently observed byte. It is used to build
  the query.

Fixed, seeded-random slow weights (NOT learned in this version):

- `E`: a key embedding table, shape 256 x d, each row scaled to unit L2 norm.
- `V`: a value embedding table, shape 256 x d, each row scaled to unit L2 norm.

Per the bench's predict/observe contract:

- **`observe(b)`**:
  - `k = key(b)` (see key_width below; for key_width = 1, `k = E[b]`)
  - `v = V[b]`
  - error against the current memory: `e = v - W k`
  - delta-rule update with decay: `W <- lambda * W + beta * e * k^T`
  - push `b` into `history`, keeping only the last `key_width` bytes
- **`predict(logits)`**:
  - if `history` is empty: fill all 256 logits with 0 (uniform, the bench's 8-bpb floor)
  - else: `q = key(history)`, `y = W q`, and `logits[c] = <y, V[c]>` for every byte c
    (equivalently `logits = V y`)

For `key_width = 1`, `k = E[b]` in observe and `q = E[last byte]` in predict. For
`key_width > 1`, `key(...)` normalizes the sum of the embeddings of the relevant recent
bytes (in observe, the window ending at the just-arrived `b`; in predict, the window of
bytes observed so far).

### Why a fixed, tied readout works without training

Reading out as `logits[c] = <y, V[c]>` (the readout is the value table itself, "tied")
means the prediction is "which byte's value vector does the recalled vector most
resemble." The fast weights store `key -> value-embedding` associations; prediction is
nearest-value-by-dot-product. No training is required for this to produce sensible logits.
A random untied readout would have no reason to favor the correct byte and the model would
sit near 8 bpb. The tie is what makes the gradient-free model work, and it is exactly why
fast-weights should excel at induction: induction is `key -> perm(key)` recall, the one
operation this object performs natively.

### The update rule (the key architectural choice)

Default: the **delta rule** `e = v - W k; W <- lambda*W + beta*e*k^T`. It is self-correcting
(online ridge regression), does not saturate, and is the modern DeltaNet formulation. With
unit-norm keys and `beta = 1`, one observation stores the association near-exactly
(`W_new k ~= v`); `lambda < 1` lets stale associations fade for non-stationary streams.

Alternative (a comparison knob only): **Hebbian + decay** `W <- lambda*W + beta*v*k^T`.
Simpler but saturates and double-counts repeated associations.

### Knobs (all recorded in every run record)

- `dim` (d): feature dimension. Default 64. Cost is O(d^2) per byte (d = 64 processes
  enwik8 in minutes on CPU). d is the natural sweep axis.
- `beta`: delta-rule step size. Default 1.0.
- `lambda`: decay. Default 0.99.
- `key_width` (w): number of most-recent bytes forming the key. Default 1 (key = E[last
  byte], a bigram-in-embedding-space memory). For w > 1, `key = normalize(sum of E[b] over
  the last w bytes)`, an order-insensitive bag of recent bytes (order-sensitive keys are
  future work).
- `update_rule`: `delta` (default) or `hebbian`.
- `seed`: seeds the fixed E and V tables.

Logits are always finite for finite W; if divergent hyperparameters blow W up, the bench's
existing finiteness guard throws, which correctly surfaces a bad configuration rather than
silently producing garbage.

### Files

- `include/seqbench/fast_weights.hpp`: `class FastWeights : public Model` plus a config
  struct, and a factory.
- `models/fast_weights.cpp`: the implementation (peer of `models/context_model.cpp`, so the
  bench builds it via the existing Makefile glob).

## Experiment-tracking layer (reusable, dependency-free)

Shared infrastructure, usable by every future architecture:

- `include/seqbench/experiment.hpp` + `src/experiment.cpp`:
  - a `RunRecord` struct holding architecture name, version, timestamp, git SHA, seed, a
    config map, a corpus descriptor, and a results map;
  - a minimal hand-written JSON serializer (no third-party JSON library; we own the
    schema) that emits one record as a single line;
  - an appender that writes one record line to a target path (default
    `runs/results.jsonl`), creating the file/dir if needed.
- Provenance captured automatically: `git_sha` via `popen("git rev-parse --short HEAD")`
  (POSIX, no new dependency; falls back to `"unknown"` on failure); timestamp via
  `std::time` formatted as ISO-8601 UTC.

Record shape (one JSON object per line in `runs/results.jsonl`):

```json
{"arch":"fast-weights","version":"v1-gradient-free","timestamp":"2026-06-18T12:00:00Z",
 "git_sha":"abc1234","seed":42,
 "config":{"dim":64,"beta":1.0,"lambda":0.99,"key_width":1,"update_rule":"delta"},
 "corpus":{"name":"toy","bytes":4920},
 "results":{"bpb":3.21,"throughput_mbps":1.4,"induction_fraction":0.96,"parity_fraction":0.0}}
```

The record is the single source of truth: any leaderboard or comparison is derived from
`results.jsonl` (jq today, importable to SQLite or the user's memex tooling later). The
log is global (not per-architecture) so future architectures append to the same queryable
file with their own `arch` field. `runs/results.jsonl` is committed to git as part of the
scientific record.

`results` is a flat map of metric name to number for v1. Sweeps continue to emit their own
CSV via the existing sweep machinery; the run record captures the single-config battery.

## Subdir and documentation

```
arch/fast-weights/
  README.md            # what fast-weights is, the delta rule, the tied-readout trick,
                       # why it should win on recall and fail on state-tracking
  NOTES.md             # implementation choices and a running interpretation of results
  run_fw.cpp           # the runner CLI
runs/results.jsonl     # global, committed experiment log
include/seqbench/fast_weights.hpp
models/fast_weights.cpp
include/seqbench/experiment.hpp
src/experiment.cpp
```

The model source lives with the other models so the bench builds it uniformly; the subdir
is the architecture's home for documentation and its runner. Future architectures get
sibling subdirs (`arch/ssm/`, etc.).

## The runner and integration

`arch/fast-weights/run_fw.cpp`: one CLI that builds a `FastWeights` model from flags (dim,
beta, lambda, key_width, update_rule, seed), runs the full battery through the existing
bench (bits-per-byte on a chosen corpus: `toy` or a file path such as `data/enwik8`, plus
both diagnostics), prints a human-readable summary, and appends a `RunRecord` to
`runs/results.jsonl`. Because `FastWeights` is a `Model`, it reuses `run_adaptive` and
`score_diagnostic` unchanged. The Makefile glob is extended to build `arch/*/*.cpp` as
tools.

## Testing

Per the project rule that features ship with tests, and the discipline that test
thresholds are never weakened to pass (a mismatch is reported, not fudged):

- **Unit:**
  - the delta update reduces reconstruction error: after `observe`-ing a single
    association, `||v - W k||` is smaller than before (and near zero for beta = 1, unit
    keys).
  - the tied readout recovers the right byte: after storing one clean association, the
    argmax logit is the stored value byte.
  - the JSON serializer emits valid, parseable JSON with all required fields and correctly
    escapes strings; numbers round-trip to adequate precision.
  - `FastWeights` produces finite logits and is deterministic given a seed.
- **Integration (the headline):**
  - on the induction diagnostic, fast-weights `fraction_captured` is high and strictly
    greater than the order-1 context model's on the same stream.
  - on the parity diagnostic, fast-weights `fraction_captured` is low (it has no unbounded
    state), matching the context model's failure there.
  - on the toy corpus, fast-weights bpb is below 8 (better than the uniform floor).
  - a run via the runner appends exactly one well-formed line to a temp `results.jsonl`
    with the expected `arch`, `config`, and `results` keys.
  - Thresholds are set to verified-achievable values; any unmet threshold is reported as
    BLOCKED with the measured numbers, never silently relaxed.

## Future directions (context, not in scope)

Once this lands and the mechanism is shown to help: add the minimal reverse-mode autodiff
engine and learn E, V, and the projections via backprop through the fast-weight recurrence;
add order-sensitive and recurrent keys; compare against the SSM / linear-RNN family; build
a small leaderboard view over `results.jsonl`.
