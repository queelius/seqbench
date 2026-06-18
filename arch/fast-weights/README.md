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
