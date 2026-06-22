# Stacked gated DeltaNet

The single fast-weights layer scaled into the standard small-LM recipe: N pre-norm blocks
of `[fast-weights token mixing -> MLP]` with residual connections and LayerNorm, where the
mixing is the delta-rule recurrence with a learned input-dependent write gate. libtorch is
confined to this directory; the bench core stays dependency-free. Shares the experiment
runner (`arch/common/runner.hpp`).

## Build

```
cmake -S arch/deltanet -B arch/deltanet/build \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
cmake --build arch/deltanet/build -j
```

## Test / run

```
./arch/deltanet/build/deltanet_test
./arch/deltanet/build/train_deltanet --corpus data/enwik8 --d 128 --n-layers 4 --steps 10000
```

Knobs: `--d` (model dim), `--n-layers`, `--lambda`, plus the common
`--seq-len/--batch/--steps/--lr/--seed`. Records `arch:"deltanet"`. At d=128, n_layers=4 it
costs about 2.93 s/step on a 12-core CPU.

## Result (first enwik8 run, 2026-06-22, d=128, 4 layers, 10000 steps)

enwik8 validation bits-per-byte, against every baseline measured on the same bench:

| Model | enwik8 val bpb |
|-------|----------------|
| context order-0 (n-gram) | 5.11 |
| context order-1 | 3.91 |
| single-layer fast-weights | 3.53 |
| context order-2 (best n-gram) | 3.31 |
| context order-3 | 3.37 |
| **stacked DeltaNet (4 layers)** | **1.99** |

The depth + MLP + gating scaling decisively beats every n-gram baseline and the single
fast-weights layer: stacking the fast-weights mixing into proper residual blocks turns a
~3.5-bpb soft-bigram into a real ~2.0-bpb language model. `train_bpb` was 1.72 and `val_bpb`
was still dropping at 10000 steps, so more steps, more layers/dim, multi-head mixing, or GPU
would push it further. For reference, strong neural byte models reach ~1.0 to 1.3 and the
SOTA context-mixing compressors ~0.9, so 1.99 from a small CPU run is a solid first result,
not the ceiling.
