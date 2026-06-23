# seqbench

A dependency-free C++ bench for evaluating *new* byte-level next-byte-prediction
architectures cheaply: hours on one CPU, not weeks. Candidates are compared by
**bits per byte** on real data, so you get signal about an architecture's
inductive biases without a giant training run.

The working premise: intelligence is sequence prediction plus the right
inductive biases, and next-byte prediction on raw bytes is the most universal
form of it (any file is a byte stream). Predicting the next byte well *is*
lossless compression, and bits-per-byte is the minimum-description-length score
of how well a model captures the structure in the data. So the bench measures
one number that means something concrete, and races architectures through it.

This repo is **harness-first**: the evaluation substrate was built before any
neural architecture, so every new idea is held to the same honest yardstick.

## The Model contract

An architecture implements one tiny streaming interface (`include/seqbench/model.hpp`):

```cpp
struct Model {
  virtual void predict(float logits[256]) = 0;  // distribution over the next byte
  virtual void observe(uint8_t byte) = 0;        // then reveal the actual byte
};
```

Logits, not probabilities: a single canonical log-softmax lives in the metric, so
models never normalize, and each logit can span the full float range. The metric
(`src/metric.cpp`) turns a `predict`/`observe` stream into bits-per-byte with a
numerically stable log-sum-exp and Kahan summation.

## Results so far

enwik8 validation bits-per-byte, every number measured on this bench:

| Model | enwik8 val bpb | note |
|-------|----------------|------|
| context order-0 (n-gram) | 5.11 | |
| context order-1 | 3.91 | |
| single-layer fast-weights | 3.53 | best in-context recall |
| context order-3 | 3.37 | |
| context order-2 (best n-gram) | 3.31 | |
| **stacked DeltaNet (4 layers)** | **1.99** | **bpb winner** |

Capability probes (synthetic train-on-task diagnostics) separate *what* each
architecture can do, which raw bpb hides:

- **In-context recall (induction):** single-layer fast-weights captures 0.32 of
  the signal vs 0.25 for an exact-count n-gram. An explicit d-by-d associative
  matrix recalls better than a conflating count table or a GRU's compressed
  hidden state (0.20).
- **State tracking (parity):** the linear delta rule *cannot represent* parity
  (flat 0 at every window). A GRU *can represent* it (2-bit parity 0.99) but SGD
  *cannot learn* it past short windows (0.00 by length 8). The bench cleanly
  distinguishes representability from learnability.

The arc that produced these (gradient-free fast-weights, learned fast-weights,
diagnostics rework, GRU, stacked DeltaNet) is recorded as design specs and plans
under `docs/superpowers/`, with one machine-readable JSON line per run in
`runs/results.jsonl`.

## Build and run

The bench core is plain C++17 with no dependencies (just a Makefile):

```bash
make                       # build tools/bench and tools/sweep_cli
make test                  # build and run the dependency-free test suite
make data/enwik8           # fetch the 100MB enwik8 corpus (gitignored)
./tools/bench ctx:2 data/enwik8     # order-2 context-model baseline -> bpb
```

Each neural architecture lives under `arch/<name>/` and builds separately with
**libtorch** (CMake), so the bench core never links a heavyweight dependency and
`make test` stays fast and pure:

```bash
cmake -S arch/deltanet -B arch/deltanet/build \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build arch/deltanet/build -j
./arch/deltanet/build/deltanet_test                                  # correctness
./arch/deltanet/build/train_deltanet --corpus data/enwik8 \
  --d 128 --n-layers 4 --steps 10000                                 # train + record
```

See each architecture's own README (`arch/*/README.md`) for its knobs and results.

## Repository layout

```
include/seqbench/   bench core headers (model, metric, corpus, diagnostics, ...)
src/  models/       bench core implementation + context-model baseline
tools/              bench, sweep_cli (dependency-free CLIs)
tests/              dependency-free test suite (make test)
arch/common/        shared libtorch training/eval runner (run_experiment)
arch/<name>/        one neural architecture each (fast-weights, gru, deltanet)
runs/results.jsonl  one JSON line per experiment (git SHA + config + results)
docs/superpowers/   design specs and implementation plans
```

## Running long, and on a GPU

Training is resumable and device-portable. `--ckpt-dir` writes a rolling
`latest` plus `best` checkpoint set, `--resume` continues to the `--steps`
target, and `--device cuda` runs on a GPU. A checkpoint written on one machine
loads on another, so the intended flow is: develop and test on CPU, run the
heavy training on a GPU box that `git pull`s the code and points its
`CMAKE_PREFIX_PATH` at a CUDA libtorch:

```bash
./arch/deltanet/build/train_deltanet --corpus data/enwik8 --d 128 --n-layers 4 \
  --steps 44000 --device cuda --ckpt-dir runs/ckpt/deltanet-asymptote --ckpt-every 2000
```

Checkpoints are gitignored and moved between machines with rsync, not committed.

## License

See repository for license terms.
