# GRU (non-linear gated RNN)

A baseline non-linear recurrent architecture (`embedding -> torch::nn::GRU -> readout`),
built to test whether non-linearity in the recurrence captures state-tracking (parity)
that the linear fast-weights delta rule provably cannot. libtorch is confined to this
directory; the bench core stays dependency-free. Shares the experiment runner with
fast-weights (`arch/common/runner.hpp`).

## Build

```
cmake -S arch/gru -B arch/gru/build \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build arch/gru/build -j
```

## Test / run

```
./arch/gru/build/gru_test
./arch/gru/build/train_gru --task parity     --d 64 --steps 5000 --eval-every 500
./arch/gru/build/train_gru --task induction  --d 64 --steps 5000 --eval-every 500
./arch/gru/build/train_gru --corpus data/enwik8 --d 128 --steps 10000
```

Knobs: `--d` (hidden), `--layers`, plus the common `--seq-len/--batch/--steps/--lr/--seed`.
Records `arch:"gru-rnn"` to `runs/results.jsonl`.

## Capability results (2026-06-19, d=64, 5000 steps each)

Fraction-captured on the train-on-task probes (0 = marginal, 1 = entropy floor):

| Task | context (n-gram) | fast-weights (linear) | GRU (non-linear) |
|------|------------------|-----------------------|------------------|
| **parity** (state-tracking) | 0.00 | 0.00 | 0.00 |
| **induction** (recall) | 0.25 | 0.32 | 0.20 |

- **Parity: the GRU did not flip it.** Its training loss stayed pinned at the naive 1.0
  (the model never moved off predicting 50/50). This is the canonical optimization
  difficulty of parity for gradient descent (near-flat gradient until the task is almost
  solved), not an expressivity limit: a GRU CAN represent parity, but SGD did not find it
  at this budget. So both architectures fail parity for DIFFERENT reasons, fast-weights
  cannot represent it (linear recurrence), the GRU could not learn it (stuck at the trivial
  minimum). Representability is not learnability.
- **Induction: the GRU under-performs both fast-weights and the n-gram.** A compressed
  fixed-size hidden state is a weaker content-addressed memory than fast-weights' explicit
  d x d matrix memory, so fast-weights remains the best recall architecture here.

Open: getting an architecture to actually LEARN parity (curriculum, much longer training,
higher learning rate, or a recurrence with an inductive bias toward bounded counters) is
the next state-tracking question; fast-weights remains the recall winner.
