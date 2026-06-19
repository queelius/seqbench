# Learned fast-weights (DeltaNet on libtorch)

A single learned fast-weights layer trained by backprop-through-time, evaluated through
the seqbench bench. libtorch is confined to this directory; the bench core stays
dependency-free.

## Build

```
cmake -S arch/fast-weights -B arch/fast-weights/build \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build arch/fast-weights/build -j
```

(The python call is only used at build-configure time to locate libtorch's CMake config;
nothing here runs Python at runtime. The hard-coded prefix is
`/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake`.)

## Test

```
./arch/fast-weights/build/fw_test
```

## Run

```
# fetch enwik8 once (uses the bench Makefile target)
make data/enwik8

# train and record (big step budget; CPU)
./arch/fast-weights/build/train_fw --d 128 --seq-len 256 --batch 32 --steps 50000

# quick offline wiring check on the toy corpus
./arch/fast-weights/build/train_fw --corpus toy --d 32 --seq-len 64 --batch 8 --steps 300 --eval-every 100
```

Each run appends a `fast-weights-learned` record to `runs/results.jsonl`. Whether the
learned layer beats the context-model and gradient-free baselines is read off from those
records; it is the research question, not an assertion.

## Results (first enwik8 run, 2026-06-18)

Single layer, `d=128`, `seq_len=256`, `batch=32`, 10,000 Adam steps (~2.3h CPU).

| Model | enwik8 val bpb |
|-------|----------------|
| context order-0 | 5.11 |
| context order-1 | 3.91 |
| **learned fast-weights (1 layer, d=128, 10k)** | **3.53** |
| context order-2 | 3.31 (best n-gram) |
| context order-3 | 3.37 |
| context order-4 | 3.92 |

Findings:

- **bpb is modest.** The learned layer lands between order-1 and order-2: it beats the
  low-order n-grams but does not beat the best simple context model (order-2, 3.31), and
  the val curve plateaued (3.72 -> 3.53 over 10k steps). A single ungated fast-weights
  layer is underpowered; the bet "learned beats fixed random features" held (the
  gradient-free version was only an order-1-ish soft-bigram), but "single learned layer
  beats the n-gram baseline" did not at this scale. Likely needs more capacity (larger
  `d`, stacked layers, learned beta/lambda gates) and/or more training.
- **The diagnostics are out-of-distribution for a text-trained model.** `induction` uses
  bytes 0-15 and `parity` uses bytes 0/1; those are only ~1.1% of enwik8. An
  enwik8-trained model has essentially never seen them, so its `induction_fraction=0` /
  `parity_fraction=0` are OOV artifacts, not evidence about whether learned fast-weights
  can recall or track state. To test a trained model fairly, the diagnostics must use
  in-distribution bytes (common ASCII), or the model must be trained on the diagnostic
  distribution. This is a bench-methodology fix for the next iteration.

## Capability results (train-on-task probes, 2026-06-19)

After the diagnostics were reframed as train-on-task probes, fast-weights was trained on
each task (d=64, 5000 steps) and scored on a held-out test stream. Fraction-captured is the
share of the recall/state structure the model captured (0 = no better than the marginal,
1 = the entropy floor).

| Task | context model (best n-gram) | learned fast-weights |
|------|-----------------------------|----------------------|
| **induction** (in-context recall) | 0.25 | **0.32** (still improving) |
| **parity** (state-tracking) | 0.00 | 0.00 |

- **Induction: fast-weights wins.** It captures more in-context recall than any
  finite-order n-gram (0.32 vs 0.25), its training loss was still dropping at 5000 steps,
  and the advantage comes from exactly what the n-gram lacks: a content-addressed, decaying
  memory that re-infers each sequence's fresh mapping instead of conflating mappings in a
  persistent count table. The "right inductive bias" thesis holds for recall.
- **Parity: fast-weights fails, like the n-gram.** Its *training* loss never moved off the
  naive 1.0 (it could not even fit the task), which points to an expressivity/optimization
  wall, not undertraining. This is consistent with the known limit that a linear recurrence
  (the delta rule is linear in W) cannot track parity, which needs a non-linear state. A
  gated or non-linear recurrence would be the architecture to test for state-tracking.

Net: the bench cleanly separated the two capabilities and showed this architecture has the
recall inductive bias but not the state-tracking one, for a principled reason.


