# GRU Architecture + Shared Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Factor `train_fw`'s model-independent runner into a shared template, migrate `train_fw` onto it, then add a GRU (non-linear gated RNN) architecture that plugs into the bench and train-on-task diagnostics.

**Architecture:** A header-only `template<class ModelT> run_experiment(...)` owns data sampling, the Adam train loop, best-by-val checkpoint, eval (corpus val-bpb or task fraction via `score_diagnostic`), and `RunRecord` assembly; model-specific behavior arrives as `loss_fn`/`make_adapter` callables. The GRU model is `embedding -> torch::nn::GRU -> readout` with an online eval adapter that carries the hidden state.

**Tech Stack:** C++17, libtorch (the venv CPU torch 2.8), CMake, the existing dependency-free bench.

## Global Constraints

- C++17. The bench core stays dependency-free; its Makefile and `make test` never link libtorch.
- libtorch stays under `arch/` (CMake, venv prefix `/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake`).
- `runs/results.jsonl` is committed; GRU records carry `arch:"gru-rnn"`.
- NO em-dash characters in any committed file (a repo hook rejects them).
- NEVER weaken a test threshold to pass. If a test does not pass with the planned value, STOP and report BLOCKED with the measured numbers.
- The libtorch C++ API targets torch 2.8; adapt a call to the installed API if it does not compile, preserving semantics. (In particular, `torch::nn::GRU::forward` returns a `std::tuple<Tensor, Tensor>`; with `batch_first(true)` the output is `[B,T,d]` and the hidden is `[layers,B,d]`.)
- Every commit message ends with these two trailer lines:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015zkyVS7cs6YxjoVn9HG1yH
  ```

### File map

| File | Change |
|------|--------|
| `arch/common/runner.hpp` | new: data helpers + `run_experiment` template |
| `arch/fast-weights/train_fw.cpp` | migrate to the shared runner (shrinks to arg-parse + model + 2 lambdas) |
| `arch/fast-weights/CMakeLists.txt` | add `${BENCH_ROOT}/arch` to `train_fw` include path |
| `arch/gru/gru_model.hpp` + `.cpp` | new: `Gru` module, `bpb_loss`, `GruEval` adapter |
| `arch/gru/gru_test.cpp` | new: forward/determinism/overfit/consistency tests |
| `arch/gru/train_gru.cpp` | new: GRU runner on the shared runner |
| `arch/gru/CMakeLists.txt` | new: builds `bench_core` + `train_gru` + `gru_test` |

### Build commands

```
# fast-weights (existing build dir; reconfigure picks up the migrated train_fw)
cmake --build arch/fast-weights/build -j
# gru (new build dir)
cmake -S arch/gru -B arch/gru/build -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/gru/build -j
```

---

## Task 1: Shared runner + migrate train_fw

**Files:**
- Create: `arch/common/runner.hpp`
- Modify: `arch/fast-weights/train_fw.cpp`, `arch/fast-weights/CMakeLists.txt`

**Interfaces:**
- Consumes: `seqbench::{ByteSpan, Corpus, toy_corpus, Diagnostic, DiagResult, make_parity, make_induction, score_diagnostic, fill_parity, fill_induction, RunRecord, JsonValue, append_record, Model}`, `fw::{Config, FastWeights, bpb_loss, FastWeightsEval}`
- Produces: `struct archcommon::RunConfig`; `archcommon::slice/sample_batch/sample_task_batch`; `template<class ModelT> int archcommon::run_experiment(const RunConfig&, const std::string& arch, const std::string& version, std::map<std::string,seqbench::JsonValue> config, ModelT model, std::function<torch::Tensor(ModelT&, torch::Tensor)> loss_fn, std::function<std::unique_ptr<seqbench::Model>(ModelT&)> make_adapter)`

- [ ] **Step 1: Write the shared runner header**

`arch/common/runner.hpp`:

```cpp
#pragma once
#include <torch/torch.h>
#include "seqbench/corpus.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/experiment.hpp"
#include "seqbench/model.hpp"
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace archcommon {

struct RunConfig {
  int seq_len = 256, batch = 32, steps = 50000, eval_every = 1000;
  double lr = 1e-3;
  uint64_t seed = 1;
  std::string corpus = "data/enwik8", task = "enwik8", out = "runs/results.jsonl";
  int block_len = 16, dict_size = 16, key_len = 4;  // task generator params
};

inline seqbench::ByteSpan slice(seqbench::ByteSpan s, double lo, double hi) {
  std::size_t a = static_cast<std::size_t>(s.len * lo);
  std::size_t b = static_cast<std::size_t>(s.len * hi);
  return seqbench::ByteSpan{s.data + a, b - a};
}

inline torch::Tensor sample_batch(seqbench::ByteSpan span, int B, int T, std::mt19937_64& rng) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::uniform_int_distribution<std::size_t> start(0, span.len - T - 1);
  for (int b = 0; b < B; ++b) {
    std::size_t s = start(rng);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = span[s + t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

inline torch::Tensor sample_task_batch(const std::string& task, int B, int T,
                                       std::mt19937_64& rng, int block_len,
                                       int dict_size, int key_len) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::vector<uint8_t> seq(static_cast<std::size_t>(T));
  for (int b = 0; b < B; ++b) {
    uint64_t s = rng();
    if (task == "parity") seqbench::fill_parity(s, seq.data(), T, block_len);
    else seqbench::fill_induction(s, seq.data(), T, dict_size, key_len);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = seq[t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

template <class ModelT>
int run_experiment(const RunConfig& rc, const std::string& arch, const std::string& version,
                   std::map<std::string, seqbench::JsonValue> config, ModelT model,
                   std::function<torch::Tensor(ModelT&, torch::Tensor)> loss_fn,
                   std::function<std::unique_ptr<seqbench::Model>(ModelT&)> make_adapter) {
  using namespace seqbench;
  torch::manual_seed(static_cast<int64_t>(rc.seed));
  std::mt19937_64 rng(rc.seed);

  const bool is_task = (rc.task == "parity" || rc.task == "induction");

  std::unique_ptr<Corpus> cptr;
  ByteSpan train, val;
  if (!is_task) {
    ByteSpan full;
    if (rc.corpus == "toy") full = toy_corpus();
    else { cptr = std::make_unique<Corpus>(rc.corpus); full = cptr->bytes(); }
    train = slice(full, 0.0, 0.90);
    val = slice(full, 0.90, 0.95);
    if (train.len <= static_cast<std::size_t>(rc.seq_len + 1) ||
        val.len <= static_cast<std::size_t>(rc.seq_len + 1)) {
      std::fprintf(stderr, "corpus too small for seq_len=%d\n", rc.seq_len);
      return 2;
    }
    std::fprintf(stderr, "corpus=%s train=%zu val=%zu\n", rc.corpus.c_str(), train.len, val.len);
  } else {
    std::fprintf(stderr, "task=%s seq_len=%d\n", rc.task.c_str(), rc.seq_len);
  }

  auto train_batch = [&](std::mt19937_64& r) {
    return is_task ? sample_task_batch(rc.task, rc.batch, rc.seq_len, r, rc.block_len,
                                       rc.dict_size, rc.key_len)
                   : sample_batch(train, rc.batch, rc.seq_len, r);
  };

  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(rc.lr));

  std::vector<torch::Tensor> val_set;
  { std::mt19937_64 vr(rc.seed ^ 0xdeadbeefULL);
    for (int i = 0; i < 8; ++i) val_set.push_back(train_batch(vr)); }
  auto eval_val_bpb = [&]() -> double {
    torch::NoGradGuard ng; model->eval();
    double tot = 0.0;
    for (auto& vb : val_set) tot += loss_fn(model, vb).item<double>();
    model->train();
    return tot / val_set.size();
  };

  const std::string best_path = "/tmp/seqbench_best.pt";
  double best = std::numeric_limits<double>::infinity();
  for (int step = 1; step <= rc.steps; ++step) {
    model->train();
    auto xb = train_batch(rng);
    opt.zero_grad();
    auto loss = loss_fn(model, xb);
    loss.backward();
    opt.step();
    if (step % rc.eval_every == 0 || step == rc.steps) {
      double vbpb = eval_val_bpb();
      bool improved = vbpb < best;
      std::fprintf(stderr, "step %d train_bpb=%.4f val_bpb=%.4f%s\n",
                   step, loss.item<double>(), vbpb, improved ? " *" : "");
      if (improved) { best = vbpb; torch::save(model, best_path); }
    }
  }
  if (best < std::numeric_limits<double>::infinity()) torch::load(model, best_path);

  double train_bpb;
  { torch::NoGradGuard ng; model->eval();
    train_bpb = loss_fn(model, train_batch(rng)).item<double>(); }

  RunRecord rec;
  rec.arch = arch;
  rec.version = version;
  rec.seed = static_cast<long>(rc.seed);
  for (const auto& kv : config) rec.config[kv.first] = kv.second;
  rec.config["seq_len"] = JsonValue::n(rc.seq_len);
  rec.config["batch"] = JsonValue::n(rc.batch);
  rec.config["steps"] = JsonValue::n(rc.steps);
  rec.config["lr"] = JsonValue::n(rc.lr);
  rec.config["task"] = JsonValue::s(rc.task);
  rec.results["train_bpb"] = train_bpb;

  if (is_task) {
    Diagnostic test = (rc.task == "parity")
        ? make_parity(rc.seed ^ 0x5eedULL, 4000, rc.block_len)
        : make_induction(rc.seed ^ 0x5eedULL, 400, 51, rc.dict_size, rc.key_len);
    auto adapter = make_adapter(model);
    DiagResult r = score_diagnostic(*adapter, test);
    rec.corpus_name = rc.task + "-test";
    rec.corpus_bytes = test.stream.size();
    rec.results["test_bpb"] = r.observed_bpb;
    rec.results["fraction_captured"] = r.fraction_captured;
    rec.results["floor_bpb"] = test.floor_bpb;
    rec.results["naive_bpb"] = test.naive_bpb;
    std::printf("%s task=%s steps=%d\n", arch.c_str(), rc.task.c_str(), rc.steps);
    std::printf("  test_bpb=%.4f fraction_captured=%.4f (floor=%.4f naive=%.4f) train_bpb=%.4f\n",
                r.observed_bpb, r.fraction_captured, test.floor_bpb, test.naive_bpb, train_bpb);
  } else {
    double val_bpb = eval_val_bpb();
    rec.corpus_name = (rc.corpus == "toy") ? "toy-val" : "enwik8-val";
    rec.corpus_bytes = val.len;
    rec.results["val_bpb"] = val_bpb;
    std::printf("%s task=%s steps=%d\n", arch.c_str(), rc.task.c_str(), rc.steps);
    std::printf("  val_bpb=%.4f train_bpb=%.4f\n", val_bpb, train_bpb);
  }
  append_record(rec, rc.out);
  std::printf("  appended run record to %s\n", rc.out.c_str());
  return 0;
}

}  // namespace archcommon
```

- [ ] **Step 2: Migrate `train_fw.cpp` onto the runner**

Replace the entire contents of `arch/fast-weights/train_fw.cpp` with:

```cpp
#include <torch/torch.h>
#include "fast_weights_model.hpp"
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
  int d = 128;
  double beta = 1.0, lambda = 0.99;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--beta") beta = std::atof(need("--beta"));
    else if (a == "--lambda") lambda = std::atof(need("--lambda"));
    else if (a == "--seq-len") rc.seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") rc.batch = std::atoi(need("--batch"));
    else if (a == "--steps") rc.steps = std::atoi(need("--steps"));
    else if (a == "--lr") rc.lr = std::atof(need("--lr"));
    else if (a == "--eval-every") rc.eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") rc.seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") rc.corpus = need("--corpus");
    else if (a == "--task") rc.task = need("--task");
    else if (a == "--out") rc.out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  fw::Config cfg; cfg.d = d; cfg.beta = beta; cfg.lambda = lambda;
  std::map<std::string, JsonValue> config;
  config["d"] = JsonValue::n(d);
  config["beta"] = JsonValue::n(beta);
  config["lambda"] = JsonValue::n(lambda);

  fw::FastWeights model(cfg);
  return archcommon::run_experiment<fw::FastWeights>(
      rc, "fast-weights-learned", "v1-deltanet", config, model,
      [](fw::FastWeights& m, torch::Tensor x) { return fw::bpb_loss(m, x); },
      [cfg](fw::FastWeights& m) {
        return std::unique_ptr<seqbench::Model>(new fw::FastWeightsEval(m, cfg));
      });
}
```

- [ ] **Step 3: Add the shared-header include path for `train_fw`**

In `arch/fast-weights/CMakeLists.txt`, the `train_fw` target's `target_include_directories` line currently lists `${BENCH_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}`. Add `${BENCH_ROOT}/arch` so `#include "common/runner.hpp"` resolves:

```cmake
target_include_directories(train_fw PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/arch ${CMAKE_CURRENT_SOURCE_DIR})
```

- [ ] **Step 4: Build and verify the migration preserved behavior**

Run: `cmake --build arch/fast-weights/build -j 2>&1 | tail -5`
Expected: `train_fw` and `fw_test` build.

Run: `./arch/fast-weights/build/fw_test`
Expected: `OK` (the model and its tests are unchanged).

Run: `./arch/fast-weights/build/train_fw --corpus toy --d 32 --seq-len 64 --batch 8 --steps 200 --eval-every 100 --out /tmp/mig_toy.jsonl`
Expected: decreasing train_bpb, a `fast-weights-learned task=enwik8 ...` summary with `val_bpb`, and `appended run record`.

Run: `./arch/fast-weights/build/train_fw --task parity --d 32 --seq-len 64 --batch 8 --steps 200 --eval-every 100 --out /tmp/mig_par.jsonl`
Expected: a summary with `test_bpb`/`fraction_captured`/`floor`/`naive`.

Run: `tail -1 /tmp/mig_par.jsonl | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d['arch'],d['config']['task'],sorted(d['results']))" && rm -f /tmp/mig_toy.jsonl /tmp/mig_par.jsonl`
Expected: `fast-weights-learned parity ['floor_bpb', 'fraction_captured', 'naive_bpb', 'test_bpb', 'train_bpb']` (same record shape as before the migration).

- [ ] **Step 5: Commit**

```bash
git add arch/common/runner.hpp arch/fast-weights/train_fw.cpp arch/fast-weights/CMakeLists.txt
git commit -m "Factor shared experiment runner; migrate train_fw onto it"
```

---

## Task 2: GRU model + tests

**Files:**
- Create: `arch/gru/gru_model.hpp`, `arch/gru/gru_model.cpp`, `arch/gru/gru_test.cpp`, `arch/gru/CMakeLists.txt`

**Interfaces:**
- Consumes: `seqbench::Model`, `seqbench::logit_bits`
- Produces: `struct gru::Config { int d=128; int layers=1; }`; `gru::Gru` (TORCH_MODULE) with `torch::Tensor forward(torch::Tensor x_bt)`; `torch::Tensor gru::bpb_loss(gru::Gru, torch::Tensor)`; `class gru::GruEval : public seqbench::Model`

- [ ] **Step 1: Write the model header**

`arch/gru/gru_model.hpp`:

```cpp
#pragma once
#include <torch/torch.h>
#include "seqbench/model.hpp"
#include <cstdint>

namespace gru {

struct Config {
  int d = 128;
  int layers = 1;
};

struct GruImpl : torch::nn::Module {
  Config cfg;
  torch::nn::Embedding emb{nullptr};
  torch::nn::GRU rnn{nullptr};
  torch::nn::Linear readout{nullptr};
  explicit GruImpl(const Config& c);
  torch::Tensor forward(torch::Tensor x_bt);  // [B,T] int64 -> [B,T,256]
};
TORCH_MODULE(Gru);

torch::Tensor bpb_loss(Gru model, torch::Tensor x_bt);  // next-byte CE in bits, positions 1..T-1

// Online bench adapter: carries the GRU hidden state, one byte per observe.
class GruEval : public seqbench::Model {
 public:
  GruEval(Gru model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  Gru model_;
  Config cfg_;
  torch::Tensor h_;    // [layers, 1, d]
  torch::Tensor out_;  // [d] latest GRU output (input to readout)
  bool seen_ = false;
};

}  // namespace gru
```

- [ ] **Step 2: Write the CMakeLists (gru_test only for now)**

`arch/gru/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(gru LANGUAGES CXX)

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

add_executable(gru_test gru_test.cpp gru_model.cpp)
target_include_directories(gru_test PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/tests ${BENCH_ROOT}/arch ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gru_test bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 3: Write the failing tests**

`arch/gru/gru_test.cpp`:

```cpp
#include "test_util.hpp"
#include <torch/torch.h>
#include "seqbench/metric.hpp"
#include "gru_model.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

static void test_forward_shape_finite() {
  torch::manual_seed(2);
  gru::Config c; c.d = 16;
  gru::Gru model(c);
  auto x = torch::randint(0, 256, {4, 10}, torch::kLong);
  auto logits = model->forward(x);
  CHECK(logits.dim() == 3);
  CHECK(logits.size(0) == 4);
  CHECK(logits.size(1) == 10);
  CHECK(logits.size(2) == 256);
  CHECK(torch::isfinite(logits).all().item<bool>());
}

static void test_deterministic() {
  auto build = []() { torch::manual_seed(7); gru::Config c; c.d = 16; return gru::Gru(c); };
  auto m1 = build();
  auto m2 = build();
  auto x = torch::randint(0, 256, {2, 8}, torch::kLong);
  CHECK(torch::allclose(m1->forward(x), m2->forward(x)));
}

static void test_overfit_tiny() {
  torch::manual_seed(1);
  gru::Config c; c.d = 32;
  gru::Gru model(c);
  const int T = 48;
  std::vector<int64_t> buf(T);
  for (int i = 0; i < T; ++i) buf[i] = 97 + (i % 3);  // "abcabc..."
  auto x = torch::tensor(buf, torch::kLong).reshape({1, T});
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  double last = 1e9;
  for (int step = 0; step < 400; ++step) {
    opt.zero_grad();
    auto loss = gru::bpb_loss(model, x);
    loss.backward();
    opt.step();
    last = loss.item<double>();
  }
  std::printf("    [gru overfit_tiny final bpb=%.4f]\n", last);
  CHECK(last < 1.0);
}

static void test_train_eval_consistency() {
  torch::manual_seed(3);
  gru::Config c; c.d = 24;
  gru::Gru model(c);
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
  gru::GruEval ev(model, c);
  float logits[256];
  for (int i = 0; i < T; ++i) {
    ev.predict(logits);
    if (i >= 1) eval_bits += seqbench::logit_bits(logits, static_cast<uint8_t>(buf[i]));
    ev.observe(static_cast<uint8_t>(buf[i]));
  }
  std::printf("    [gru consistency train_bits=%.4f eval_bits=%.4f]\n", train_bits, eval_bits);
  CHECK_NEAR(train_bits, eval_bits, 0.05 * (T - 1));
}

int main() {
  RUN(test_forward_shape_finite);
  RUN(test_deterministic);
  RUN(test_overfit_tiny);
  RUN(test_train_eval_consistency);
  return test_summary();
}
```

- [ ] **Step 4: Run to verify failure**

Run:

```bash
cmake -S arch/gru -B arch/gru/build -DCMAKE_PREFIX_PATH=/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake
cmake --build arch/gru/build -j 2>&1 | tail -5
```

Expected: compile/link error (no `gru_model.cpp` implementation yet): undefined references to `gru::GruImpl::...`, `gru::bpb_loss`, `gru::GruEval::...`. (If `argmax` triggers an unused-function error, remove it.)

- [ ] **Step 5: Write the implementation**

`arch/gru/gru_model.cpp`:

```cpp
#include "gru_model.hpp"
#include <cmath>

namespace gru {

namespace F = torch::nn::functional;

GruImpl::GruImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  rnn = register_module(
      "rnn", torch::nn::GRU(torch::nn::GRUOptions(cfg.d, cfg.d).num_layers(cfg.layers).batch_first(true)));
  readout = register_module("readout", torch::nn::Linear(cfg.d, 256));
}

torch::Tensor GruImpl::forward(torch::Tensor x_bt) {
  auto x = emb->forward(x_bt);                  // [B,T,d]
  auto out = std::get<0>(rnn->forward(x));      // [B,T,d] (batch_first)
  return readout->forward(out);                 // [B,T,256]
}

torch::Tensor bpb_loss(Gru model, torch::Tensor x_bt) {
  auto T = x_bt.size(1);
  auto logits = model->forward(x_bt);                       // [B,T,256]
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});
  auto tgt = x_bt.slice(1, 1, T).reshape({-1});
  auto ce = F::cross_entropy(pred, tgt);
  return ce / std::log(2.0);
}

GruEval::GruEval(Gru model, const Config& c) : model_(model), cfg_(c) {
  model_->eval();
  h_ = torch::zeros({cfg_.layers, 1, cfg_.d});
  out_ = torch::zeros({cfg_.d});
}

void GruEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto in = seen_ ? out_ : torch::zeros({cfg_.d});
  auto o = model_->readout->forward(in).contiguous();  // [256]
  float* p = o.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void GruEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  auto idx = torch::tensor({static_cast<int64_t>(b)}, torch::kLong);  // [1]
  auto x = model_->emb->forward(idx).view({1, 1, cfg_.d});           // [B=1,T=1,d]
  auto res = model_->rnn->forward(x, h_);
  out_ = std::get<0>(res).view({cfg_.d});  // latest output [d]
  h_ = std::get<1>(res);                   // [layers,1,d]
  seen_ = true;
}

}  // namespace gru
```

- [ ] **Step 6: Run to verify pass**

Run: `cmake --build arch/gru/build -j && ./arch/gru/build/gru_test`
Expected: `OK`. If `test_overfit_tiny` does not reach `< 1.0` or `test_train_eval_consistency` fails, STOP and report BLOCKED with the printed numbers (do not relax the thresholds; a consistency gap means the online step does not match the batched GRU forward).

- [ ] **Step 7: Commit**

```bash
git add arch/gru/gru_model.hpp arch/gru/gru_model.cpp arch/gru/gru_test.cpp arch/gru/CMakeLists.txt
git commit -m "Add GRU model (embedding + GRU + readout) with online eval adapter"
```

---

## Task 3: GRU runner

**Files:**
- Create: `arch/gru/train_gru.cpp`
- Modify: `arch/gru/CMakeLists.txt` (add the `train_gru` target)

**Interfaces:**
- Consumes: `gru::{Config, Gru, bpb_loss, GruEval}`, `archcommon::{RunConfig, run_experiment}`, `seqbench::{JsonValue, Model}`
- Produces: a `train_gru` executable accepting `--d`, `--layers`, and the common run args; records `arch:"gru-rnn"`.

- [ ] **Step 1: Add the runner target to CMake**

In `arch/gru/CMakeLists.txt`, append:

```cmake
add_executable(train_gru train_gru.cpp gru_model.cpp)
target_include_directories(train_gru PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/arch ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(train_gru bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 2: Write the runner**

`arch/gru/train_gru.cpp`:

```cpp
#include <torch/torch.h>
#include "gru_model.hpp"
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
  int d = 128, layers = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--layers") layers = std::atoi(need("--layers"));
    else if (a == "--seq-len") rc.seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") rc.batch = std::atoi(need("--batch"));
    else if (a == "--steps") rc.steps = std::atoi(need("--steps"));
    else if (a == "--lr") rc.lr = std::atof(need("--lr"));
    else if (a == "--eval-every") rc.eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") rc.seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") rc.corpus = need("--corpus");
    else if (a == "--task") rc.task = need("--task");
    else if (a == "--out") rc.out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  gru::Config cfg; cfg.d = d; cfg.layers = layers;
  std::map<std::string, JsonValue> config;
  config["d"] = JsonValue::n(d);
  config["layers"] = JsonValue::n(layers);

  gru::Gru model(cfg);
  return archcommon::run_experiment<gru::Gru>(
      rc, "gru-rnn", "v1-gru", config, model,
      [](gru::Gru& m, torch::Tensor x) { return gru::bpb_loss(m, x); },
      [cfg](gru::Gru& m) {
        return std::unique_ptr<seqbench::Model>(new gru::GruEval(m, cfg));
      });
}
```

- [ ] **Step 3: Build**

Run: `cmake --build arch/gru/build -j 2>&1 | tail -5`
Expected: `train_gru` and `gru_test` build; `./arch/gru/build/gru_test` still prints `OK`.

- [ ] **Step 4: Smoke-run the GRU runner (offline, fast)**

Run: `./arch/gru/build/train_gru --task parity --d 32 --seq-len 64 --batch 8 --steps 300 --eval-every 100 --out /tmp/gru_smoke.jsonl`
Expected: decreasing train_bpb, a `gru-rnn task=parity ...` summary line with `test_bpb`/`fraction_captured`, and `appended run record`.

Run: `tail -1 /tmp/gru_smoke.jsonl | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d['arch'],d['config']['task'],d['config']['layers'],sorted(d['results']))" && rm -f /tmp/gru_smoke.jsonl`
Expected: `gru-rnn parity 1 ['floor_bpb', 'fraction_captured', 'naive_bpb', 'test_bpb', 'train_bpb']`.

(Whether the GRU captures parity is the real experiment, not asserted here; this verifies wiring.)

- [ ] **Step 5: Commit**

```bash
git add arch/gru/train_gru.cpp arch/gru/CMakeLists.txt
git commit -m "Add GRU training runner on the shared experiment runner"
```

---

## Final verification (after all tasks)

- [ ] `make test` (bench suite) -> `ALL TESTS PASSED`, libtorch-free.
- [ ] `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test` -> `OK`; `train_fw` smoke still works (migration preserved behavior).
- [ ] `cmake --build arch/gru/build -j && ./arch/gru/build/gru_test` -> `OK`.
- [ ] `train_gru --task parity --steps 300 --d 32 --seq-len 64 --batch 8` produces a `gru-rnn` parity record.

## Plan self-review notes

- **Spec coverage:** shared `run_experiment` template with the `loss_fn`/`make_adapter` hooks and moved data helpers (Task 1); `train_fw` migrated and verified behavior-preserving (Task 1); GRU model `embedding -> GRU -> readout` with online hidden-state adapter (Task 2); GRU tests incl. overfit-learns and train/eval consistency (Task 2); GRU runner recording `arch:"gru-rnn"` via the shared runner (Task 3); per-arch CMake with `arch/` include path (Tasks 1-3); bench untouched. All spec sections covered.
- **Deferred per spec:** the hybrid model, size/layer sweeps, and the capability runs (train GRU on parity/induction/enwik8 and compare).
- **Type consistency:** `archcommon::RunConfig`/`run_experiment`, `gru::Config`/`Gru`/`bpb_loss`/`GruEval`, and the bench symbols are used with identical signatures across tasks; the `run_experiment` callable types match the lambdas passed by both `train_fw` and `train_gru`.
- **Anti-fudging:** the overfit `< 1.0` and consistency-tolerance thresholds carry BLOCKED-with-numbers instructions and print measured values.
