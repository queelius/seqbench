# Learned Fast-Weights (DeltaNet on libtorch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a single learned fast-weights (DeltaNet-style) layer by backprop-through-time on libtorch, evaluate it through the existing seqbench bench, and record the result.

**Architecture:** A libtorch `nn::Module` (learned embedding, k/v/q projections, delta-rule fast-weight recurrence, separate learned readout) trained on enwik8 chunks. It is wrapped as a bench `Model` for online evaluation and reuses the bench's diagnostics, metric, and JSONL run records. All libtorch lives under `arch/fast-weights/` with its own CMake build; the bench core stays dependency-free.

**Tech Stack:** C++17, libtorch (PyTorch C++ API, the venv's CPU torch 2.8), CMake for the libtorch part, the existing dependency-free bench (plain Makefile, untouched).

## Global Constraints

- C++17. The bench core stays dependency-free and its plain Makefile / `make test` are untouched and never link libtorch.
- All libtorch code lives under `arch/fast-weights/`, built by `arch/fast-weights/CMakeLists.txt` via `find_package(Torch REQUIRED)`.
- The CMake configure prefix is the venv torch cmake path: `/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake`. No Python at runtime (the binaries are pure C++).
- Apply `${TORCH_CXX_FLAGS}` globally in CMake so the bench sources and the model share libtorch's C++ ABI.
- `runs/results.jsonl` is committed; the runner appends records with `arch:"fast-weights-learned"`.
- NO em-dash characters in any committed file (a repo hook rejects them).
- NEVER weaken a test threshold to pass. If a test does not pass with the planned value, STOP and report BLOCKED with the measured numbers.
- The exact libtorch C++ API can vary slightly by version. The plan's calls target torch 2.8; if a specific call does not compile, adapt it to the installed API while preserving the same math/semantics, and note the change.
- Every commit message ends with these two trailer lines:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015zkyVS7cs6YxjoVn9HG1yH
  ```

### File map

| File | Responsibility |
|------|----------------|
| `arch/fast-weights/CMakeLists.txt` | libtorch build: `bench_core` static lib + `fw_test` + `train_fw` |
| `arch/fast-weights/fast_weights_model.hpp` + `.cpp` | `fw::Config`, `fw::FastWeights` module, `fw::bpb_loss`, `fw::FastWeightsEval` (bench Model adapter) |
| `arch/fast-weights/fw_test.cpp` | libtorch tests (built by CMake, run separately from `make test`) |
| `arch/fast-weights/train_fw.cpp` | runner: train, checkpoint, eval via bench, emit RunRecord |
| `arch/fast-weights/README.md` | build + run instructions |
| `.gitignore` | ignore `arch/fast-weights/build/` |

### How to build and test this component

```
cmake -S arch/fast-weights -B arch/fast-weights/build \
  -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/fast-weights/build -j
./arch/fast-weights/build/fw_test
```

`make test` (the bench suite) must still pass independently and without libtorch.

---

## Task 1: CMake build skeleton + libtorch smoke test

**Files:**
- Create: `arch/fast-weights/CMakeLists.txt`, `arch/fast-weights/fw_test.cpp`, `arch/fast-weights/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a working CMake build that links libtorch + the bench sources and runs a `fw_test` executable.

- [ ] **Step 1: Write the CMakeLists**

`arch/fast-weights/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(fast_weights LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(BENCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# The dependency-free bench sources, compiled here under the Torch ABI flags.
add_library(bench_core STATIC
  ${BENCH_ROOT}/src/corpus.cpp
  ${BENCH_ROOT}/src/metric.cpp
  ${BENCH_ROOT}/src/diagnostics.cpp
  ${BENCH_ROOT}/src/experiment.cpp
  ${BENCH_ROOT}/models/context_model.cpp)
target_include_directories(bench_core PUBLIC ${BENCH_ROOT}/include)

add_executable(fw_test fw_test.cpp)
target_include_directories(fw_test PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/tests ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fw_test bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 2: Write the smoke test**

`arch/fast-weights/fw_test.cpp`:

```cpp
#include "test_util.hpp"
#include <torch/torch.h>
#include "seqbench/corpus.hpp"

static void test_torch_works() {
  torch::manual_seed(0);
  auto a = torch::tensor({1.0, 2.0, 3.0});
  CHECK(a.sum().item<double>() == 6.0);
  auto m = torch::ones({2, 3});
  CHECK(m.size(0) == 2);
  CHECK(m.size(1) == 3);
}

static void test_bench_links() {
  auto toy = seqbench::toy_corpus();
  CHECK(toy.len > 0);
}

int main() {
  RUN(test_torch_works);
  RUN(test_bench_links);
  return test_summary();
}
```

- [ ] **Step 3: Configure, build, and run**

Run:

```bash
cmake -S arch/fast-weights -B arch/fast-weights/build \
  -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/fast-weights/build -j
./arch/fast-weights/build/fw_test
```

Expected: configures (finds Torch), builds `fw_test`, prints `OK`. If `find_package(Torch)` fails, confirm the prefix path with `python3 -c "import torch;print(torch.utils.cmake_prefix_path)"` and use that. If linking fails with ABI errors, confirm `${TORCH_CXX_FLAGS}` is applied (it is in the CMakeLists above). This step exists to surface any toolchain problem now.

- [ ] **Step 4: Confirm the bench suite is still libtorch-free**

Run: `make test`
Expected: `ALL TESTS PASSED` (the bench builds and tests with the plain Makefile, no libtorch).

- [ ] **Step 5: Ignore the build dir and write the README**

In `.gitignore`, under the build-artifacts section, add:

```
/arch/fast-weights/build/
```

`arch/fast-weights/README.md`:

````markdown
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
````

- [ ] **Step 6: Commit**

```bash
git add arch/fast-weights/CMakeLists.txt arch/fast-weights/fw_test.cpp arch/fast-weights/README.md .gitignore
git commit -m "Add libtorch CMake build skeleton and smoke test for fast-weights"
```

---

## Task 2: The FastWeights module (forward + loss)

**Files:**
- Create: `arch/fast-weights/fast_weights_model.hpp`, `arch/fast-weights/fast_weights_model.cpp`
- Modify: `arch/fast-weights/CMakeLists.txt` (add the model source to `fw_test`), `arch/fast-weights/fw_test.cpp` (add tests)

**Interfaces:**
- Produces: `struct fw::Config { int d=128; double beta=1.0; double lambda=0.99; }`
- Produces: `fw::FastWeights` (a `TORCH_MODULE` holder for `FastWeightsImpl`) with `torch::Tensor forward(torch::Tensor x_bt)` taking `[B,T]` int64 byte ids and returning `[B,T,256]` logits, and public submodules `emb, wk, wv, wq, readout`
- Produces: `torch::Tensor fw::bpb_loss(fw::FastWeights model, torch::Tensor x_bt)` (mean next-byte cross-entropy in bits, predicting positions 1..T-1)

- [ ] **Step 1: Write the header (module + loss only; the eval adapter comes in Task 3)**

`arch/fast-weights/fast_weights_model.hpp`:

```cpp
#pragma once
#include <torch/torch.h>
#include "seqbench/model.hpp"
#include <cstdint>

namespace fw {

struct Config {
  int d = 128;
  double beta = 1.0;
  double lambda = 0.99;
};

struct FastWeightsImpl : torch::nn::Module {
  Config cfg;
  torch::nn::Embedding emb{nullptr};
  torch::nn::Linear wk{nullptr}, wv{nullptr}, wq{nullptr}, readout{nullptr};
  explicit FastWeightsImpl(const Config& c);
  // x_bt: [B, T] int64 byte ids -> logits [B, T, 256]
  torch::Tensor forward(torch::Tensor x_bt);
};
TORCH_MODULE(FastWeights);

// Mean next-byte cross-entropy in bits-per-byte for a [B,T] chunk (predicts positions 1..T-1).
torch::Tensor bpb_loss(FastWeights model, torch::Tensor x_bt);

}  // namespace fw
```

- [ ] **Step 2: Add the model source to the build**

In `arch/fast-weights/CMakeLists.txt`, change the `fw_test` target to also compile the model:

```cmake
add_executable(fw_test fw_test.cpp fast_weights_model.cpp)
```

- [ ] **Step 3: Write the failing tests**

Append to `arch/fast-weights/fw_test.cpp` (add the include at the top and the new RUN lines in `main`):

```cpp
#include "fast_weights_model.hpp"
#include <cmath>
```

```cpp
static void test_forward_shape_finite() {
  torch::manual_seed(2);
  fw::Config c; c.d = 16;
  fw::FastWeights model(c);
  auto x = torch::randint(0, 256, {4, 10}, torch::kLong);
  auto logits = model->forward(x);
  CHECK(logits.dim() == 3);
  CHECK(logits.size(0) == 4);
  CHECK(logits.size(1) == 10);
  CHECK(logits.size(2) == 256);
  CHECK(torch::isfinite(logits).all().item<bool>());
}

static void test_deterministic() {
  auto build = []() { torch::manual_seed(7); fw::Config c; c.d = 16; return fw::FastWeights(c); };
  auto m1 = build();
  auto m2 = build();
  auto x = torch::randint(0, 256, {2, 8}, torch::kLong);
  CHECK(torch::allclose(m1->forward(x), m2->forward(x)));
}

// Proves the whole loop (embedding -> recurrence -> readout -> CE -> backward -> Adam)
// actually learns: overfit a tiny repeating pattern and the loss must collapse.
static void test_overfit_tiny() {
  torch::manual_seed(1);
  fw::Config c; c.d = 32; c.beta = 1.0; c.lambda = 0.99;
  fw::FastWeights model(c);
  const int T = 48;
  std::vector<int64_t> buf(T);
  for (int i = 0; i < T; ++i) buf[i] = 97 + (i % 3);  // "abcabc..."
  auto x = torch::tensor(buf, torch::kLong).reshape({1, T});
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  double last = 1e9;
  for (int step = 0; step < 400; ++step) {
    opt.zero_grad();
    auto loss = fw::bpb_loss(model, x);
    loss.backward();
    opt.step();
    last = loss.item<double>();
  }
  CHECK(last < 1.0);  // memorized the period-3 pattern
}
```

And in `main`, add:

```cpp
  RUN(test_forward_shape_finite);
  RUN(test_deterministic);
  RUN(test_overfit_tiny);
```

- [ ] **Step 4: Run to verify failure**

Run: `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test`
Expected: compile/link error (no `fast_weights_model.cpp` implementation yet) or, once the file exists empty, undefined references to `FastWeightsImpl::FastWeightsImpl` / `forward` / `bpb_loss`.

- [ ] **Step 5: Write the implementation**

`arch/fast-weights/fast_weights_model.cpp`:

```cpp
#include "fast_weights_model.hpp"
#include <cmath>
#include <vector>

namespace fw {

namespace F = torch::nn::functional;

FastWeightsImpl::FastWeightsImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(cfg.d, cfg.d).bias(false)));
  wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(cfg.d, cfg.d).bias(false)));
  wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(cfg.d, cfg.d).bias(false)));
  readout = register_module("readout", torch::nn::Linear(cfg.d, 256));
}

torch::Tensor FastWeightsImpl::forward(torch::Tensor x_bt) {
  auto B = x_bt.size(0);
  auto T = x_bt.size(1);
  int d = cfg.d;
  auto x = emb->forward(x_bt);                                  // [B,T,d]
  auto k = F::normalize(wk->forward(x), F::NormalizeFuncOptions().dim(2));  // [B,T,d]
  auto v = wv->forward(x);                                      // [B,T,d]
  auto q = wq->forward(x);                                      // [B,T,d]
  auto W = torch::zeros({B, d, d}, x.options());               // [B,d,d]
  std::vector<torch::Tensor> outs;
  outs.reserve(T);
  for (int64_t t = 0; t < T; ++t) {
    auto kt = k.select(1, t);                                  // [B,d]
    auto vt = v.select(1, t);                                  // [B,d]
    auto qt = q.select(1, t);                                  // [B,d]
    auto Wk = torch::bmm(W, kt.unsqueeze(2)).squeeze(2);       // [B,d]
    auto e = vt - Wk;                                          // [B,d]
    W = cfg.lambda * W + cfg.beta * torch::bmm(e.unsqueeze(2), kt.unsqueeze(1));  // [B,d,d]
    auto ot = torch::bmm(W, qt.unsqueeze(2)).squeeze(2);       // [B,d]
    outs.push_back(ot);
  }
  auto o = torch::stack(outs, 1);                              // [B,T,d]
  return readout->forward(o);                                  // [B,T,256]
}

torch::Tensor bpb_loss(FastWeights model, torch::Tensor x_bt) {
  auto T = x_bt.size(1);
  auto logits = model->forward(x_bt);                          // [B,T,256]
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});    // [B*(T-1),256]
  auto tgt = x_bt.slice(1, 1, T).reshape({-1});                // [B*(T-1)]
  auto ce = F::cross_entropy(pred, tgt);                       // natural-log mean CE
  return ce / std::log(2.0);                                   // bits per byte
}

}  // namespace fw
```

- [ ] **Step 6: Run to verify pass**

Run: `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test`
Expected: `OK`. If `test_overfit_tiny` does not reach `< 1.0`, STOP and report BLOCKED with the final `last` value (do not relax the threshold).

- [ ] **Step 7: Commit**

```bash
git add arch/fast-weights/fast_weights_model.hpp arch/fast-weights/fast_weights_model.cpp arch/fast-weights/CMakeLists.txt arch/fast-weights/fw_test.cpp
git commit -m "Add learned fast-weights module, forward recurrence, and bpb loss"
```

---

## Task 3: The eval adapter + train/eval consistency

**Files:**
- Modify: `arch/fast-weights/fast_weights_model.hpp` + `.cpp` (add `FastWeightsEval`), `arch/fast-weights/fw_test.cpp` (add the consistency test)

**Interfaces:**
- Consumes: `fw::FastWeights`, `fw::Config`, `fw::bpb_loss`, `seqbench::Model`, `seqbench::logit_bits`
- Produces: `class fw::FastWeightsEval : public seqbench::Model` with `FastWeightsEval(fw::FastWeights model, const fw::Config& c)`, `predict(float[256])`, `observe(uint8_t)`

- [ ] **Step 1: Declare the adapter in the header**

In `arch/fast-weights/fast_weights_model.hpp`, before the closing `}  // namespace fw`, add:

```cpp
// Bench Model adapter: runs the trained model online (no_grad), one byte at a time,
// replicating the training forward exactly so streaming and batched bits agree.
class FastWeightsEval : public seqbench::Model {
 public:
  FastWeightsEval(FastWeights model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  FastWeights model_;
  Config cfg_;
  torch::Tensor W_;  // [d, d]
  torch::Tensor o_;  // [d]
  bool seen_ = false;
};
```

- [ ] **Step 2: Write the failing consistency test**

Append to `arch/fast-weights/fw_test.cpp` (add the include and the RUN line):

```cpp
#include "seqbench/metric.hpp"
```

```cpp
// The online adapter must reproduce the batched-forward bits for positions 1..T-1.
static void test_train_eval_consistency() {
  torch::manual_seed(3);
  fw::Config c; c.d = 24; c.beta = 1.0; c.lambda = 0.99;
  fw::FastWeights model(c);
  model->eval();
  const int T = 40;
  std::vector<int64_t> buf(T);
  for (int i = 0; i < T; ++i) buf[i] = (i * 37 + 11) % 256;

  double train_bits = 0.0;
  {
    torch::NoGradGuard ng;
    auto x = torch::tensor(buf, torch::kLong).reshape({1, T});
    auto logits = model->forward(x);
    auto logp = torch::log_softmax(logits.slice(1, 0, T - 1).reshape({-1, 256}), 1).contiguous();
    auto a = logp.accessor<float, 2>();
    for (int t = 0; t < T - 1; ++t)
      train_bits += -double(a[t][buf[t + 1]]) / std::log(2.0);
  }

  double eval_bits = 0.0;
  fw::FastWeightsEval ev(model, c);
  float logits[256];
  for (int i = 0; i < T; ++i) {
    ev.predict(logits);
    if (i >= 1) eval_bits += seqbench::logit_bits(logits, static_cast<uint8_t>(buf[i]));
    ev.observe(static_cast<uint8_t>(buf[i]));
  }

  CHECK_NEAR(train_bits, eval_bits, 0.05 * (T - 1));  // < 0.05 bits/byte average drift
}
```

And in `main`, add:

```cpp
  RUN(test_train_eval_consistency);
```

- [ ] **Step 3: Run to verify failure**

Run: `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test`
Expected: link error (undefined reference to `FastWeightsEval` methods).

- [ ] **Step 4: Implement the adapter**

In `arch/fast-weights/fast_weights_model.cpp`, before the closing `}  // namespace fw`, add:

```cpp
FastWeightsEval::FastWeightsEval(FastWeights model, const Config& c)
    : model_(model), cfg_(c) {
  model_->eval();
  W_ = torch::zeros({cfg_.d, cfg_.d});
  o_ = torch::zeros({cfg_.d});
}

void FastWeightsEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto in = seen_ ? o_ : torch::zeros({cfg_.d});
  auto out = model_->readout->forward(in).contiguous();  // [256]
  float* p = out.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void FastWeightsEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  auto idx = torch::tensor({static_cast<int64_t>(b)}, torch::kLong);
  auto x = model_->emb->forward(idx).squeeze(0);  // [d]
  auto k = F::normalize(model_->wk->forward(x), F::NormalizeFuncOptions().dim(0));  // [d]
  auto v = model_->wv->forward(x);                // [d]
  auto q = model_->wq->forward(x);                // [d]
  auto Wk = torch::mv(W_, k);                      // [d]
  auto e = v - Wk;                                 // [d]
  W_ = cfg_.lambda * W_ + cfg_.beta * torch::outer(e, k);  // [d,d]
  o_ = torch::mv(W_, q);                            // [d]
  seen_ = true;
}
```

- [ ] **Step 5: Run to verify pass**

Run: `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test`
Expected: `OK`. If `test_train_eval_consistency` fails, STOP and report BLOCKED with `train_bits` and `eval_bits` (a large gap means a causality/indexing mismatch between the batched and online recurrences; do not loosen the tolerance to hide it).

- [ ] **Step 6: Commit**

```bash
git add arch/fast-weights/fast_weights_model.hpp arch/fast-weights/fast_weights_model.cpp arch/fast-weights/fw_test.cpp
git commit -m "Add online eval adapter and train/eval consistency test"
```

---

## Task 4: The training runner + first recorded run

**Files:**
- Create: `arch/fast-weights/train_fw.cpp`
- Modify: `arch/fast-weights/CMakeLists.txt` (add the `train_fw` target), `arch/fast-weights/README.md` (run section)

**Interfaces:**
- Consumes: `fw::Config`, `fw::FastWeights`, `fw::bpb_loss`, `fw::FastWeightsEval`, `seqbench::Corpus`, `seqbench::toy_corpus`, `seqbench::ByteSpan`, `seqbench::score_diagnostic`, `seqbench::make_induction`, `seqbench::make_parity`, `seqbench::RunRecord`, `seqbench::JsonValue`, `seqbench::append_record`
- Produces: a `train_fw` executable that trains, evaluates via the bench, and appends a `RunRecord`.

- [ ] **Step 1: Add the runner target to CMake**

In `arch/fast-weights/CMakeLists.txt`, append:

```cmake
add_executable(train_fw train_fw.cpp fast_weights_model.cpp)
target_include_directories(train_fw PRIVATE ${BENCH_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(train_fw bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 2: Write the runner**

`arch/fast-weights/train_fw.cpp`:

```cpp
#include <torch/torch.h>
#include "fast_weights_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/experiment.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace seqbench;

static ByteSpan slice(ByteSpan s, double lo, double hi) {
  std::size_t a = static_cast<std::size_t>(s.len * lo);
  std::size_t b = static_cast<std::size_t>(s.len * hi);
  return ByteSpan{s.data + a, b - a};
}

static torch::Tensor sample_batch(ByteSpan span, int B, int T, std::mt19937_64& rng) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::uniform_int_distribution<std::size_t> start(0, span.len - T - 1);
  for (int b = 0; b < B; ++b) {
    std::size_t s = start(rng);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = span[s + t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

int main(int argc, char** argv) {
  int d = 128, seq_len = 256, batch = 32, steps = 50000, eval_every = 1000;
  double beta = 1.0, lambda = 0.99, lr = 1e-3;
  uint64_t seed = 1;
  std::string corpus = "data/enwik8", out = "runs/results.jsonl";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--beta") beta = std::atof(need("--beta"));
    else if (a == "--lambda") lambda = std::atof(need("--lambda"));
    else if (a == "--seq-len") seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") batch = std::atoi(need("--batch"));
    else if (a == "--steps") steps = std::atoi(need("--steps"));
    else if (a == "--lr") lr = std::atof(need("--lr"));
    else if (a == "--eval-every") eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") corpus = need("--corpus");
    else if (a == "--out") out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  torch::manual_seed(static_cast<int64_t>(seed));
  std::mt19937_64 rng(seed);

  std::unique_ptr<Corpus> cptr;
  ByteSpan full;
  if (corpus == "toy") full = toy_corpus();
  else { cptr = std::make_unique<Corpus>(corpus); full = cptr->bytes(); }
  ByteSpan train = slice(full, 0.0, 0.90);
  ByteSpan val = slice(full, 0.90, 0.95);
  if (train.len <= static_cast<std::size_t>(seq_len + 1) ||
      val.len <= static_cast<std::size_t>(seq_len + 1)) {
    std::fprintf(stderr, "corpus too small for seq_len=%d\n", seq_len);
    return 2;
  }
  std::fprintf(stderr, "corpus=%s train=%zu val=%zu\n", corpus.c_str(), train.len, val.len);

  fw::Config cfg; cfg.d = d; cfg.beta = beta; cfg.lambda = lambda;
  fw::FastWeights model(cfg);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(lr));

  std::vector<torch::Tensor> val_set;
  { std::mt19937_64 vr(seed ^ 0xdeadbeefULL);
    for (int i = 0; i < 8; ++i) val_set.push_back(sample_batch(val, batch, seq_len, vr)); }
  auto eval_val_bpb = [&]() -> double {
    torch::NoGradGuard ng; model->eval();
    double tot = 0.0;
    for (auto& vb : val_set) tot += fw::bpb_loss(model, vb).item<double>();
    model->train();
    return tot / val_set.size();
  };

  const std::string best_path = "/tmp/fw_best.pt";
  double best = std::numeric_limits<double>::infinity();
  for (int step = 1; step <= steps; ++step) {
    model->train();
    auto xb = sample_batch(train, batch, seq_len, rng);
    opt.zero_grad();
    auto loss = fw::bpb_loss(model, xb);
    loss.backward();
    opt.step();
    if (step % eval_every == 0 || step == steps) {
      double vbpb = eval_val_bpb();
      bool improved = vbpb < best;
      std::fprintf(stderr, "step %d train_bpb=%.4f val_bpb=%.4f%s\n",
                   step, loss.item<double>(), vbpb, improved ? " *" : "");
      if (improved) { best = vbpb; torch::save(model, best_path); }
    }
  }
  if (best < std::numeric_limits<double>::infinity()) torch::load(model, best_path);

  double val_bpb = eval_val_bpb();
  double train_bpb;
  { torch::NoGradGuard ng; model->eval();
    train_bpb = fw::bpb_loss(model, sample_batch(train, batch, seq_len, rng)).item<double>(); }

  Diagnostic ind = make_induction(7, 50000, 16);
  Diagnostic par = make_parity(7, 4000, 16);
  fw::FastWeightsEval ev_ind(model, cfg);
  double find = score_diagnostic(ev_ind, ind).fraction_captured;
  fw::FastWeightsEval ev_par(model, cfg);
  double fpar = score_diagnostic(ev_par, par).fraction_captured;

  std::printf("learned fast-weights d=%d beta=%.3g lambda=%.3g steps=%d\n", d, beta, lambda, steps);
  std::printf("  val_bpb=%.4f train_bpb=%.4f induction=%.4f parity=%.4f\n",
              val_bpb, train_bpb, find, fpar);

  RunRecord rec;
  rec.arch = "fast-weights-learned";
  rec.version = "v1-deltanet";
  rec.seed = static_cast<long>(seed);
  rec.config["d"] = JsonValue::n(d);
  rec.config["beta"] = JsonValue::n(beta);
  rec.config["lambda"] = JsonValue::n(lambda);
  rec.config["seq_len"] = JsonValue::n(seq_len);
  rec.config["batch"] = JsonValue::n(batch);
  rec.config["steps"] = JsonValue::n(steps);
  rec.config["lr"] = JsonValue::n(lr);
  rec.corpus_name = (corpus == "toy") ? "toy-val" : "enwik8-val";
  rec.corpus_bytes = val.len;
  rec.results["val_bpb"] = val_bpb;
  rec.results["train_bpb"] = train_bpb;
  rec.results["induction_fraction"] = find;
  rec.results["parity_fraction"] = fpar;
  append_record(rec, out);
  std::printf("  appended run record to %s\n", out.c_str());
  return 0;
}
```

- [ ] **Step 3: Build**

Run: `cmake --build arch/fast-weights/build -j`
Expected: builds `train_fw` and `fw_test`. Then `./arch/fast-weights/build/fw_test` still prints `OK`.

- [ ] **Step 4: Smoke-run on the toy corpus (offline, fast)**

Run: `./arch/fast-weights/build/train_fw --corpus toy --d 32 --seq-len 64 --batch 8 --steps 300 --eval-every 100`
Expected: prints decreasing `train_bpb`, a final summary line with `val_bpb`, `induction`, `parity`, and `appended run record to runs/results.jsonl`. This is a wiring smoke test, not a research result (the toy corpus is tiny).

- [ ] **Step 5: Verify and commit the record**

Run: `tail -1 runs/results.jsonl`
Expected: one JSON line with `"arch":"fast-weights-learned"`, the `config`, `"corpus":{"name":"toy-val",...}`, and a `results` object with `val_bpb`, `train_bpb`, `induction_fraction`, `parity_fraction`.

Add a run section to `arch/fast-weights/README.md`:

````markdown
## Run

```
# fetch enwik8 once (uses the bench Makefile target)
make data/enwik8

# train and record (big step budget; CPU)
./arch/fast-weights/build/train_fw --d 128 --seq-len 256 --batch 32 --steps 50000
```

Each run appends a `fast-weights-learned` record to `runs/results.jsonl`. Whether the
learned layer beats the context-model and gradient-free baselines is read off from those
records; it is the research question, not an assertion.
````

Commit:

```bash
git add arch/fast-weights/train_fw.cpp arch/fast-weights/CMakeLists.txt arch/fast-weights/README.md runs/results.jsonl
git commit -m "Add learned fast-weights training runner and first recorded run"
```

---

## Final verification (after all tasks)

- [ ] `make test` (bench suite) -> `ALL TESTS PASSED`, with no libtorch involvement.
- [ ] `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test` -> `OK`.
- [ ] `./arch/fast-weights/build/train_fw --corpus toy --steps 300 --d 32 --seq-len 64 --batch 8` produces a record.
- [ ] (The real experiment, time-permitting) `make data/enwik8` then `train_fw` with the big step budget on enwik8; read `runs/results.jsonl` and compare `val_bpb` / `induction_fraction` against the context-model and gradient-free records to answer whether learned features help.

## Plan self-review notes

- **Spec coverage:** the single DeltaNet-style layer with learned embedding/projections/readout and the delta-rule recurrence (Task 2); the online eval adapter replicating the training forward (Task 3); training with chunk sampling, Adam, truncated BPTT, step-count budget, best-by-val checkpoint (Task 4); val bpb via batched chunked forward + diagnostics via the adapter + RunRecord (Task 4); libtorch confined to `arch/fast-weights/` with its own CMake build and the bench untouched (Tasks 1-4); the libtorch toolchain de-risked first (Task 1). All spec sections covered.
- **Deferred per spec (not in this plan):** learned beta/lambda gates, chunk-parallel BPTT, multi-layer stacks, stateful cross-chunk BPTT, GPU, tokenization, weight export to a dependency-free eval path.
- **No asserted win:** consistent with the spec, no test asserts the learned model beats the baselines; that is recorded and interpreted. The pass/fail tests cover machinery correctness (overfit-learns, forward shape/finite, determinism, train/eval consistency).
- **Type consistency:** `fw::Config`, `fw::FastWeights`, `fw::bpb_loss(FastWeights, Tensor)`, `fw::FastWeightsEval(FastWeights, const Config&)`, and the bench symbols (`score_diagnostic`, `make_induction`, `make_parity`, `RunRecord`, `JsonValue`, `append_record`, `logit_bits`, `Corpus`, `toy_corpus`, `ByteSpan`) are used with identical signatures across tasks.
- **Anti-fudging:** the two reality-could-differ thresholds (overfit `< 1.0`, consistency tolerance) carry explicit BLOCKED-with-numbers instructions.
