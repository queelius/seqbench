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
records; it is the research question, not an assertion. Note: the diagnostics
(`induction_fraction`, `parity_fraction`) reflect the trained model's frozen projections
applied online to those streams, so they are only meaningful for a model trained on a
representative corpus (enwik8), not the tiny toy smoke run.

