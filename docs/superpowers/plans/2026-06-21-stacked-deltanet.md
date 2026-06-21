# Stacked Gated DeltaNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stacked gated DeltaNet (N pre-norm blocks of fast-weights token mixing + MLP) that plugs into the bench and the shared runner, to scale past the single-layer fast-weights plateau (3.53 bpb).

**Architecture:** `emb -> [x += FWMix(LN(x)); x += MLP(LN(x))] x N -> LN -> readout`. FWMix is the delta-rule recurrence with a learned input-dependent write gate. The online eval adapter carries the N fast-weight matrices; everything else is position-wise.

**Tech Stack:** C++17, libtorch (venv CPU torch 2.8), CMake, the existing bench + shared runner.

## Global Constraints

- C++17. The bench core stays dependency-free; its Makefile and `make test` never link libtorch.
- libtorch stays under `arch/` (CMake, venv prefix `/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake`).
- `runs/results.jsonl` is committed; DeltaNet records carry `arch:"deltanet"`.
- NO em-dash characters in any committed file (a repo hook rejects them).
- NEVER weaken a test threshold to pass. If a test does not pass with the planned value, STOP and report BLOCKED with the measured numbers.
- The libtorch C++ API targets torch 2.8; adapt a call to the installed API if it does not compile, preserving semantics.
- Every commit message ends with these two trailer lines:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015zkyVS7cs6YxjoVn9HG1yH
  ```

### File map

| File | Change |
|------|--------|
| `arch/deltanet/deltanet_model.hpp` + `.cpp` | new: `FWMix`, `Block`, `DeltaNet` modules, `bpb_loss`, `DeltaNetEval` adapter |
| `arch/deltanet/deltanet_test.cpp` | new: forward/determinism/overfit/consistency tests |
| `arch/deltanet/train_deltanet.cpp` | new: runner on the shared runner |
| `arch/deltanet/CMakeLists.txt` | new: builds `bench_core` + `deltanet_test` + `train_deltanet` |

### Build commands

```
cmake -S arch/deltanet -B arch/deltanet/build -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/deltanet/build -j
```

---

## Task 1: The DeltaNet model (FWMix, Block, DeltaNet, bpb_loss)

**Files:**
- Create: `arch/deltanet/deltanet_model.hpp`, `arch/deltanet/deltanet_model.cpp`, `arch/deltanet/deltanet_test.cpp`, `arch/deltanet/CMakeLists.txt`

**Interfaces:**
- Consumes: `seqbench::Model`, `seqbench::logit_bits`
- Produces: `struct dn::Config { int d=128; int n_layers=4; double lambda=0.99; }`; `dn::DeltaNet` (TORCH_MODULE) with `torch::Tensor forward(torch::Tensor x_bt)`; `torch::Tensor dn::bpb_loss(dn::DeltaNet, torch::Tensor)`; `class dn::DeltaNetEval : public seqbench::Model` (declared here, implemented in Task 2)

- [ ] **Step 1: Write the header**

`arch/deltanet/deltanet_model.hpp`:

```cpp
#pragma once
#include <torch/torch.h>
#include "seqbench/model.hpp"
#include <cstdint>
#include <vector>

namespace dn {

struct Config {
  int d = 128;
  int n_layers = 4;
  double lambda = 0.99;
};

// Fast-weights token mixing: delta-rule recurrence with a learned write gate.
struct FWMixImpl : torch::nn::Module {
  int dim;
  double lambda;
  torch::nn::Linear wk{nullptr}, wv{nullptr}, wq{nullptr}, wbeta{nullptr}, wo{nullptr};
  FWMixImpl(int d, double lam);
  torch::Tensor forward(torch::Tensor h);  // [B,T,d] -> [B,T,d]
};
TORCH_MODULE(FWMix);

// Pre-norm residual block: x += FWMix(LN(x)); x += MLP(LN(x)).
struct BlockImpl : torch::nn::Module {
  torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
  FWMix mix{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  BlockImpl(int d, double lam);
  torch::Tensor forward(torch::Tensor x);  // [B,T,d] -> [B,T,d]
};
TORCH_MODULE(Block);

struct DeltaNetImpl : torch::nn::Module {
  Config cfg;
  torch::nn::Embedding emb{nullptr};
  std::vector<Block> blocks;
  torch::nn::LayerNorm ln_f{nullptr};
  torch::nn::Linear readout{nullptr};
  explicit DeltaNetImpl(const Config& c);
  torch::Tensor forward(torch::Tensor x_bt);  // [B,T] int64 -> [B,T,256]
};
TORCH_MODULE(DeltaNet);

torch::Tensor bpb_loss(DeltaNet model, torch::Tensor x_bt);  // next-byte CE in bits, positions 1..T-1

// Online bench adapter: carries the N fast-weight matrices; everything else is position-wise.
class DeltaNetEval : public seqbench::Model {
 public:
  DeltaNetEval(DeltaNet model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  DeltaNet model_;
  Config cfg_;
  std::vector<torch::Tensor> W_;  // n_layers x [d, d]
  torch::Tensor out_;             // [d] post-stack hidden
  bool seen_ = false;
};

}  // namespace dn
```

- [ ] **Step 2: Write the CMakeLists (deltanet_test only)**

`arch/deltanet/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(deltanet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(BENCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_library(bench_core STATIC
  ${BENCH_ROOT}/src/corpus.cpp
  ${BENCH_ROOT}/src/metric.cpp
  ${BENCH_ROOT}/src/diagnostics.cpp
  ${BENCH_ROOT}/src/experiment.cpp
  ${BENCH_ROOT}/models/context_model.cpp)
target_include_directories(bench_core PUBLIC ${BENCH_ROOT}/include)

add_executable(deltanet_test deltanet_test.cpp deltanet_model.cpp)
target_include_directories(deltanet_test PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/tests ${BENCH_ROOT}/arch ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(deltanet_test bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 3: Write the failing tests**

`arch/deltanet/deltanet_test.cpp`:

```cpp
#include "test_util.hpp"
#include <torch/torch.h>
#include "seqbench/metric.hpp"
#include "deltanet_model.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

static void test_forward_shape_finite() {
  torch::manual_seed(2);
  dn::Config c; c.d = 16; c.n_layers = 2;
  dn::DeltaNet model(c);
  auto x = torch::randint(0, 256, {4, 10}, torch::kLong);
  auto logits = model->forward(x);
  CHECK(logits.dim() == 3);
  CHECK(logits.size(0) == 4);
  CHECK(logits.size(1) == 10);
  CHECK(logits.size(2) == 256);
  CHECK(torch::isfinite(logits).all().item<bool>());
}

static void test_deterministic() {
  auto build = []() { torch::manual_seed(7); dn::Config c; c.d = 16; c.n_layers = 2; return dn::DeltaNet(c); };
  auto m1 = build();
  auto m2 = build();
  auto x = torch::randint(0, 256, {2, 8}, torch::kLong);
  CHECK(torch::allclose(m1->forward(x), m2->forward(x)));
}

static void test_overfit_tiny() {
  torch::manual_seed(1);
  dn::Config c; c.d = 32; c.n_layers = 2;
  dn::DeltaNet model(c);
  const int T = 48;
  std::vector<int64_t> buf(T);
  for (int i = 0; i < T; ++i) buf[i] = 97 + (i % 3);  // "abcabc..."
  auto x = torch::tensor(buf, torch::kLong).reshape({1, T});
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  double last = 1e9;
  for (int step = 0; step < 400; ++step) {
    opt.zero_grad();
    auto loss = dn::bpb_loss(model, x);
    loss.backward();
    opt.step();
    last = loss.item<double>();
  }
  std::printf("    [deltanet overfit_tiny final bpb=%.4f]\n", last);
  CHECK(last < 1.0);
}

int main() {
  RUN(test_forward_shape_finite);
  RUN(test_deterministic);
  RUN(test_overfit_tiny);
  return test_summary();
}
```

- [ ] **Step 4: Run to verify failure**

Run:

```bash
cmake -S arch/deltanet -B arch/deltanet/build -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/deltanet/build -j 2>&1 | tail -5
```

Expected: compile/link error (no `deltanet_model.cpp` implementation): undefined references to `dn::FWMixImpl::...`, `dn::BlockImpl::...`, `dn::DeltaNetImpl::...`, `dn::bpb_loss`.

- [ ] **Step 5: Write the implementation**

`arch/deltanet/deltanet_model.cpp`:

```cpp
#include "deltanet_model.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace dn {

namespace F = torch::nn::functional;

FWMixImpl::FWMixImpl(int d, double lam) : dim(d), lambda(lam) {
  wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
  wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
  wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
  wbeta = register_module("wbeta", torch::nn::Linear(d, 1));
  wo = register_module("wo", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
}

torch::Tensor FWMixImpl::forward(torch::Tensor h) {
  auto B = h.size(0);
  auto T = h.size(1);
  int d = dim;
  auto k = F::normalize(wk->forward(h), F::NormalizeFuncOptions().dim(2));  // [B,T,d]
  auto v = wv->forward(h);                                                  // [B,T,d]
  auto q = wq->forward(h);                                                  // [B,T,d]
  auto beta = torch::sigmoid(wbeta->forward(h));                            // [B,T,1]
  auto W = torch::zeros({B, d, d}, h.options());                           // [B,d,d]
  std::vector<torch::Tensor> outs;
  outs.reserve(T);
  for (int64_t t = 0; t < T; ++t) {
    auto kt = k.select(1, t);                                  // [B,d]
    auto vt = v.select(1, t);                                  // [B,d]
    auto qt = q.select(1, t);                                  // [B,d]
    auto bt = beta.select(1, t);                               // [B,1]
    auto Wk = torch::bmm(W, kt.unsqueeze(2)).squeeze(2);       // [B,d]
    auto e = vt - Wk;                                          // [B,d]
    W = lambda * W + torch::bmm((bt * e).unsqueeze(2), kt.unsqueeze(1));  // [B,d,d]
    auto ot = torch::bmm(W, qt.unsqueeze(2)).squeeze(2);       // [B,d]
    outs.push_back(ot);
  }
  auto o = torch::stack(outs, 1);                              // [B,T,d]
  return wo->forward(o);                                       // [B,T,d]
}

BlockImpl::BlockImpl(int d, double lam) {
  ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d})));
  mix = register_module("mix", FWMix(d, lam));
  ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d})));
  fc1 = register_module("fc1", torch::nn::Linear(d, 4 * d));
  fc2 = register_module("fc2", torch::nn::Linear(4 * d, d));
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
  x = x + mix->forward(ln1->forward(x));
  x = x + fc2->forward(torch::gelu(fc1->forward(ln2->forward(x))));
  return x;
}

DeltaNetImpl::DeltaNetImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  for (int i = 0; i < cfg.n_layers; ++i)
    blocks.push_back(register_module("block" + std::to_string(i), Block(cfg.d, cfg.lambda)));
  ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.d})));
  readout = register_module("readout", torch::nn::Linear(cfg.d, 256));
}

torch::Tensor DeltaNetImpl::forward(torch::Tensor x_bt) {
  auto x = emb->forward(x_bt);          // [B,T,d]
  for (auto& blk : blocks) x = blk->forward(x);
  x = ln_f->forward(x);                 // [B,T,d]
  return readout->forward(x);           // [B,T,256]
}

torch::Tensor bpb_loss(DeltaNet model, torch::Tensor x_bt) {
  auto T = x_bt.size(1);
  auto logits = model->forward(x_bt);                       // [B,T,256]
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});
  auto tgt = x_bt.slice(1, 1, T).reshape({-1});
  auto ce = F::cross_entropy(pred, tgt);
  return ce / std::log(2.0);
}

}  // namespace dn
```

- [ ] **Step 6: Run to verify pass**

Run: `cmake --build arch/deltanet/build -j && ./arch/deltanet/build/deltanet_test`
Expected: `OK`. If `test_overfit_tiny` does not reach `< 1.0`, STOP and report BLOCKED with the printed final bpb (do not relax the threshold).

- [ ] **Step 7: Commit**

```bash
git add arch/deltanet/deltanet_model.hpp arch/deltanet/deltanet_model.cpp arch/deltanet/deltanet_test.cpp arch/deltanet/CMakeLists.txt
git commit -m "Add stacked gated DeltaNet model (fast-weights mixing + MLP blocks)"
```

---

## Task 2: The online eval adapter + consistency

**Files:**
- Modify: `arch/deltanet/deltanet_model.cpp` (add `DeltaNetEval`), `arch/deltanet/deltanet_test.cpp` (add the consistency test)

**Interfaces:**
- Consumes: `dn::DeltaNet`, `dn::Config`, the block submodules (`mix->wk/wv/wq/wbeta/wo`, `ln1`, `ln2`, `fc1`, `fc2`, `ln_f`, `readout`), `seqbench::Model`, `seqbench::logit_bits`
- Produces: `dn::DeltaNetEval` method definitions

- [ ] **Step 1: Write the failing consistency test**

Append to `arch/deltanet/deltanet_test.cpp` (and add the `RUN` line in `main`):

```cpp
static void test_train_eval_consistency() {
  torch::manual_seed(3);
  dn::Config c; c.d = 24; c.n_layers = 2;
  dn::DeltaNet model(c);
  model->eval();
  const int T = 40;
  std::vector<int64_t> buf(T);
  for (int i = 0; i < T; ++i) buf[i] = (i * 37 + 11) % 256;

  double train_bits = 0.0;
  {
    torch::NoGradGuard ng;
    auto x = torch::tensor(buf, torch::kLong).reshape({1, T});
    auto logp = torch::log_softmax(model->forward(x).slice(1, 0, T - 1).reshape({-1, 256}), 1).contiguous();
    auto a = logp.accessor<float, 2>();
    for (int t = 0; t < T - 1; ++t) train_bits += -double(a[t][buf[t + 1]]) / std::log(2.0);
  }

  double eval_bits = 0.0;
  dn::DeltaNetEval ev(model, c);
  float logits[256];
  for (int i = 0; i < T; ++i) {
    ev.predict(logits);
    if (i >= 1) eval_bits += seqbench::logit_bits(logits, static_cast<uint8_t>(buf[i]));
    ev.observe(static_cast<uint8_t>(buf[i]));
  }
  std::printf("    [deltanet consistency train_bits=%.4f eval_bits=%.4f]\n", train_bits, eval_bits);
  CHECK_NEAR(train_bits, eval_bits, 0.05 * (T - 1));
}
```

In `main`, add `RUN(test_train_eval_consistency);` before `return test_summary();`.

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build arch/deltanet/build -j 2>&1 | tail -5`
Expected: link error (undefined reference to `dn::DeltaNetEval::...`).

- [ ] **Step 3: Implement the adapter**

In `arch/deltanet/deltanet_model.cpp`, before the closing `}  // namespace dn`, add:

```cpp
DeltaNetEval::DeltaNetEval(DeltaNet model, const Config& c) : model_(model), cfg_(c) {
  model_->eval();
  for (int i = 0; i < cfg_.n_layers; ++i) W_.push_back(torch::zeros({cfg_.d, cfg_.d}));
  out_ = torch::zeros({cfg_.d});
}

void DeltaNetEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto x = seen_ ? out_ : torch::zeros({cfg_.d});
  auto o = model_->readout->forward(model_->ln_f->forward(x)).contiguous();  // [256]
  float* p = o.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void DeltaNetEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  int d = cfg_.d;
  auto x = model_->emb->forward(torch::tensor({static_cast<int64_t>(b)}, torch::kLong)).view({d});  // [d]
  for (int i = 0; i < cfg_.n_layers; ++i) {
    auto& blk = model_->blocks[i];
    // FWMix on ln1(x), updating W_[i].
    auto h = blk->ln1->forward(x);                                                   // [d]
    auto k = F::normalize(blk->mix->wk->forward(h), F::NormalizeFuncOptions().dim(0));  // [d]
    auto vv = blk->mix->wv->forward(h);                                              // [d]
    auto qq = blk->mix->wq->forward(h);                                              // [d]
    auto bt = torch::sigmoid(blk->mix->wbeta->forward(h));                           // [1]
    auto Wk = torch::mv(W_[i], k);                                                   // [d]
    auto e = vv - Wk;                                                                // [d]
    W_[i] = cfg_.lambda * W_[i] + torch::outer(bt * e, k);                           // [d,d]
    auto o = torch::mv(W_[i], qq);                                                   // [d]
    x = x + blk->mix->wo->forward(o);                                                // residual
    // MLP on ln2(x).
    auto h2 = blk->ln2->forward(x);
    x = x + blk->fc2->forward(torch::gelu(blk->fc1->forward(h2)));                   // residual
  }
  out_ = x;
  seen_ = true;
}
```

Note: `bt` is shape `[1]` and `e` is `[d]`; `bt * e` broadcasts to `[d]`, matching the batched `[B,1] * [B,d]`. `F::` is the `torch::nn::functional` alias already defined at the top of the file.

- [ ] **Step 4: Run to verify pass**

Run: `cmake --build arch/deltanet/build -j && ./arch/deltanet/build/deltanet_test`
Expected: `OK`. If `test_train_eval_consistency` fails, STOP and report BLOCKED with the printed `train_bits` and `eval_bits` (a large gap means the online stack does not match the batched forward; the most likely cause is a LayerNorm or residual applied in the wrong order).

- [ ] **Step 5: Commit**

```bash
git add arch/deltanet/deltanet_model.cpp arch/deltanet/deltanet_test.cpp
git commit -m "Add DeltaNet online eval adapter and train/eval consistency test"
```

---

## Task 3: The DeltaNet runner

**Files:**
- Create: `arch/deltanet/train_deltanet.cpp`
- Modify: `arch/deltanet/CMakeLists.txt` (add the `train_deltanet` target)

**Interfaces:**
- Consumes: `dn::{Config, DeltaNet, bpb_loss, DeltaNetEval}`, `archcommon::{RunConfig, run_experiment}`, `seqbench::{JsonValue, Model}`
- Produces: a `train_deltanet` executable accepting `--d`, `--n-layers`, `--lambda`, and the common run args; records `arch:"deltanet"`.

- [ ] **Step 1: Add the runner target to CMake**

In `arch/deltanet/CMakeLists.txt`, append:

```cmake
add_executable(train_deltanet train_deltanet.cpp deltanet_model.cpp)
target_include_directories(train_deltanet PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/arch ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(train_deltanet bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 2: Write the runner**

`arch/deltanet/train_deltanet.cpp`:

```cpp
#include <torch/torch.h>
#include "deltanet_model.hpp"
#include "common/runner.hpp"
#include "seqbench/experiment.hpp"
#include "seqbench/model.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>

using namespace seqbench;

int main(int argc, char** argv) {
  archcommon::RunConfig rc;
  int d = 128, n_layers = 4;
  double lambda = 0.99;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--n-layers") n_layers = std::atoi(need("--n-layers"));
    else if (a == "--lambda") lambda = std::atof(need("--lambda"));
    else if (a == "--seq-len") rc.seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") rc.batch = std::atoi(need("--batch"));
    else if (a == "--steps") rc.steps = std::atoi(need("--steps"));
    else if (a == "--lr") rc.lr = std::atof(need("--lr"));
    else if (a == "--eval-every") rc.eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") rc.seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") rc.corpus = need("--corpus");
    else if (a == "--task") rc.task = need("--task");
    else if (a == "--block-len") rc.block_len = std::atoi(need("--block-len"));
    else if (a == "--out") rc.out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  dn::Config cfg; cfg.d = d; cfg.n_layers = n_layers; cfg.lambda = lambda;
  std::map<std::string, JsonValue> config;
  config["d"] = JsonValue::n(d);
  config["n_layers"] = JsonValue::n(n_layers);
  config["lambda"] = JsonValue::n(lambda);

  dn::DeltaNet model(cfg);
  return archcommon::run_experiment<dn::DeltaNet>(
      rc, "deltanet", "v1-stacked", config, model,
      [](dn::DeltaNet& m, torch::Tensor x) { return dn::bpb_loss(m, x); },
      [cfg](dn::DeltaNet& m) {
        return std::unique_ptr<seqbench::Model>(new dn::DeltaNetEval(m, cfg));
      });
}
```

- [ ] **Step 3: Build**

Run: `cmake --build arch/deltanet/build -j 2>&1 | tail -5`
Expected: `train_deltanet` and `deltanet_test` build; `./arch/deltanet/build/deltanet_test` still prints `OK`.

- [ ] **Step 4: Smoke-run (offline, fast)**

Run: `./arch/deltanet/build/train_deltanet --corpus toy --d 32 --n-layers 2 --seq-len 64 --batch 8 --steps 200 --eval-every 100 --out /tmp/dn_smoke.jsonl`
Expected: decreasing train_bpb, a `deltanet task=enwik8 ...` summary with `val_bpb`, and `appended run record`.

Run: `tail -1 /tmp/dn_smoke.jsonl | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d['arch'],d['config']['n_layers'],sorted(d['results']))" && rm -f /tmp/dn_smoke.jsonl`
Expected: `deltanet 2 ['train_bpb', 'val_bpb']`.

- [ ] **Step 5: Commit**

```bash
git add arch/deltanet/train_deltanet.cpp arch/deltanet/CMakeLists.txt
git commit -m "Add stacked DeltaNet training runner on the shared experiment runner"
```

---

## Final verification (after all tasks)

- [ ] `make test` (bench suite) -> `ALL TESTS PASSED`, libtorch-free.
- [ ] `cmake --build arch/deltanet/build -j && ./arch/deltanet/build/deltanet_test` -> `OK`.
- [ ] `train_deltanet --corpus toy --d 32 --n-layers 2 --steps 200 --seq-len 64 --batch 8` produces a `deltanet` record.

## Plan self-review notes

- **Spec coverage:** the stacked pre-norm block model with FWMix (delta recurrence + learned write gate), MLP, residual + LayerNorm (Task 1); the online adapter carrying the N fast-weight matrices with everything else position-wise (Task 2); the train/eval consistency test (Task 2); the runner recording `arch:"deltanet"` with `d/n_layers/lambda` via the shared runner (Task 3); per-arch CMake with the `arch/` include path (Tasks 1-3); bench untouched. All spec sections covered.
- **Deferred per spec:** multi-head mixing, learned lambda/forget gate, chunk-parallel BPTT, GPU, the long enwik8 training run.
- **Type consistency:** `dn::Config`/`DeltaNet`/`bpb_loss`/`DeltaNetEval`, the block submodule names (`mix`, `ln1`, `ln2`, `fc1`, `fc2`, `ln_f`, `readout`, and `mix->wk/wv/wq/wbeta/wo`), and `archcommon::run_experiment`'s callable types are used identically across tasks; the adapter's online math mirrors `FWMixImpl::forward` and `BlockImpl::forward`.
- **Anti-fudging:** the overfit `< 1.0` and consistency-tolerance thresholds carry BLOCKED-with-numbers instructions and print measured values.
