# Resumable, device-aware checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make seqbench training runs crash-safe, resumable, and device-portable (CPU now, GPU later), with the capability living once in the shared `run_experiment`.

**Architecture:** A new header-only serialization layer (`arch/common/checkpoint.hpp`) saves and restores the four pieces of mutable training state (model params, Adam optimizer state, step, and both RNGs) as a small set of files under a checkpoint dir, written atomically with the scalar sidecar as the validity sentinel. The shared runner (`arch/common/runner.hpp`) gains a device (`--device`) and gains checkpoint save/resume (`--ckpt-dir`, `--ckpt-every`, `--resume`), so deltanet, fast-weights, and gru all inherit it. Default behavior is unchanged: no `--ckpt-dir` means no checkpoints, `--device cpu` means CPU.

**Tech Stack:** C++17, libtorch (CPU here; CUDA on the GPU box), CMake per architecture, the repo's `test_util.hpp` macro harness.

## Global Constraints

- libtorch is confined to `arch/`. The bench core (`include/seqbench`, `src`, `models`, `tools`, `tests`) stays dependency-free and `make test` never links libtorch. All new tests here are arch tests built by CMake, not by the Makefile.
- No em-dash characters anywhere in committed files (a commit hook rejects them). Use commas, colons, parentheses, or periods.
- Every commit message ends with the trailer line `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (referred to below as "the standard trailer").
- Never weaken a test threshold to make a test pass. If a value does not meet the asserted bound, report BLOCKED with the numbers and fix the cause.
- `runs/results.jsonl` is the committed experiment log; do not write test records into it (tests write to `/tmp`).
- Checkpoints are gitignored (`/runs/ckpt/` already added); never commit `.pt` or checkpoint files.
- Default behavior must not change for existing commands: empty `ckpt_dir` writes no checkpoints, `device == "cpu"` keeps tensors on CPU.
- `--steps` is the TARGET TOTAL on resume: resuming an N-step checkpoint with `--steps M` trains `M - N` more steps.

---

### Task 1: Checkpoint serialization layer + mechanics tests

Create the save/restore layer and prove the core guarantee (continue-from-checkpoint equals an uninterrupted run) before touching the runner.

**Files:**
- Create: `arch/common/checkpoint.hpp`
- Create: `arch/common/checkpoint_test.cpp`
- Modify: `arch/deltanet/CMakeLists.txt` (add the `checkpoint_test` executable)

**Interfaces:**
- Produces:
  - `struct archcommon::CkptMeta { int step; uint64_t seed; double best; std::string arch; std::string fingerprint; };`
  - `bool archcommon::checkpoint_exists(const std::string& dir, const std::string& prefix);`
  - `template <class ModelT> void archcommon::save_checkpoint(const std::string& dir, const std::string& prefix, ModelT& model, torch::optim::Adam& opt, const CkptMeta& meta, const std::mt19937_64& rng);`
  - `template <class ModelT> bool archcommon::load_checkpoint(const std::string& dir, const std::string& prefix, ModelT& model, torch::optim::Adam& opt, CkptMeta& meta, std::mt19937_64& rng, torch::Device dev);` (returns false if no checkpoint present; loads model/opt/RNGs and fills `meta` if present)
- Consumes: `archcommon::sample_batch` from `arch/common/runner.hpp` (unchanged) for the determinism test.

- [ ] **Step 1: Write `arch/common/checkpoint.hpp`**

```cpp
#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <sys/stat.h>

namespace archcommon {

struct CkptMeta {
  int step = 0;
  uint64_t seed = 0;
  double best = 0.0;
  std::string arch;
  std::string fingerprint;
};

inline bool path_exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

// Create every component of a (possibly nested) directory path. Ignores
// already-exists. Best-effort: a real I/O failure surfaces later on write.
inline void make_dirs(const std::string& dir) {
  std::string acc;
  for (std::size_t i = 0; i < dir.size(); ++i) {
    acc.push_back(dir[i]);
    if (dir[i] == '/' || i + 1 == dir.size()) {
      if (!acc.empty() && acc != "/") ::mkdir(acc.c_str(), 0755);
    }
  }
}

inline bool checkpoint_exists(const std::string& dir, const std::string& prefix) {
  return path_exists(dir + "/" + prefix + ".meta");
}

// Write `src` to `final + ".tmp"` via `fn`, then rename over `final` so a crash
// mid-write cannot corrupt an existing good file.
template <class Fn>
inline void atomic_write(const std::string& final, Fn fn) {
  const std::string tmp = final + ".tmp";
  fn(tmp);
  std::rename(tmp.c_str(), final.c_str());
}

template <class ModelT>
void save_checkpoint(const std::string& dir, const std::string& prefix,
                     ModelT& model, torch::optim::Adam& opt,
                     const CkptMeta& meta, const std::mt19937_64& rng) {
  make_dirs(dir);
  const std::string base = dir + "/" + prefix;
  atomic_write(base + ".model.pt", [&](const std::string& p) { torch::save(model, p); });
  atomic_write(base + ".opt.pt", [&](const std::string& p) { torch::save(opt, p); });
  atomic_write(base + ".rng.pt", [&](const std::string& p) {
    torch::save(torch::get_rng_state(), p);
  });
  atomic_write(base + ".mt", [&](const std::string& p) {
    std::ofstream f(p); f << rng;
  });
  // .meta written LAST: its presence marks the checkpoint set as valid.
  atomic_write(base + ".meta", [&](const std::string& p) {
    std::ofstream f(p);
    f << "step " << meta.step << "\n"
      << "seed " << meta.seed << "\n"
      << "best " << std::setprecision(17) << meta.best << "\n"
      << "arch " << meta.arch << "\n"
      << "fingerprint " << meta.fingerprint << "\n";
  });
}

// Returns true if a checkpoint set was present and loaded; false if absent.
// Fills `meta` (including the stored fingerprint, for the caller to validate).
template <class ModelT>
bool load_checkpoint(const std::string& dir, const std::string& prefix,
                     ModelT& model, torch::optim::Adam& opt,
                     CkptMeta& meta, std::mt19937_64& rng, torch::Device dev) {
  const std::string base = dir + "/" + prefix;
  if (!path_exists(base + ".meta")) return false;
  {
    std::ifstream f(base + ".meta");
    std::string key;
    while (f >> key) {
      if (key == "step") f >> meta.step;
      else if (key == "seed") f >> meta.seed;
      else if (key == "best") f >> meta.best;
      else if (key == "arch") f >> meta.arch;
      else if (key == "fingerprint") f >> meta.fingerprint;
    }
  }
  torch::load(model, base + ".model.pt", dev);
  model->to(dev);
  torch::load(opt, base + ".opt.pt");
  // GPU seam (untested on this box): Adam state tensors load CPU-resident. On a
  // CUDA resume they must be moved onto `dev` here before the first opt.step();
  // the CPU path needs no move. Verify and add the move when the GPU is online.
  {
    torch::Tensor rngst;
    torch::load(rngst, base + ".rng.pt");
    torch::set_rng_state(rngst.to(torch::kCPU));
  }
  { std::ifstream f(base + ".mt"); f >> rng; }
  return true;
}

}  // namespace archcommon
```

- [ ] **Step 2: Write the failing mechanics + determinism tests `arch/common/checkpoint_test.cpp`**

```cpp
#include "test_util.hpp"
#include <torch/torch.h>
#include "common/checkpoint.hpp"
#include "common/runner.hpp"  // archcommon::sample_batch
#include "seqbench/byte_span.hpp"
#include <cstdint>
#include <random>
#include <vector>

// Minimal self-contained next-byte model so these tests do not depend on any
// particular architecture.
struct TinyImpl : torch::nn::Module {
  torch::nn::Embedding emb{nullptr};
  torch::nn::Linear fc{nullptr};
  explicit TinyImpl(int d) {
    emb = register_module("emb", torch::nn::Embedding(256, d));
    fc = register_module("fc", torch::nn::Linear(d, 256));
  }
  torch::Tensor forward(torch::Tensor x) { return fc->forward(emb->forward(x)); }  // [B,T]->[B,T,256]
};
TORCH_MODULE(Tiny);

static torch::Tensor tiny_loss(Tiny m, torch::Tensor x) {
  auto T = x.size(1);
  auto logits = m->forward(x);
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});
  auto tgt = x.slice(1, 1, T).reshape({-1});
  return torch::nn::functional::cross_entropy(pred, tgt);
}

static double max_param_diff(Tiny a, Tiny b) {
  double m = 0.0;
  auto pa = a->parameters(), pb = b->parameters();
  for (size_t i = 0; i < pa.size(); ++i)
    m = std::max(m, (pa[i] - pb[i]).abs().max().item<double>());
  return m;
}

// A 2KB deterministic byte corpus for sampling.
static std::vector<uint8_t> tiny_corpus() {
  std::vector<uint8_t> v(2048);
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>((i * 131 + 17) % 256);
  return v;
}

static void test_roundtrip_params_and_opt() {
  torch::manual_seed(11);
  Tiny model(16);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  auto corp = tiny_corpus();
  seqbench::ByteSpan span{corp.data(), corp.size()};
  std::mt19937_64 rng(5);
  for (int s = 0; s < 5; ++s) {  // build up real Adam moment state
    opt.zero_grad();
    auto loss = tiny_loss(model, archcommon::sample_batch(span, 2, 16, rng));
    loss.backward();
    opt.step();
  }
  archcommon::CkptMeta meta; meta.step = 5; meta.seed = 5; meta.best = 1.25;
  meta.arch = "tiny"; meta.fingerprint = "d=16";
  archcommon::save_checkpoint("/tmp/sb_ckpt_rt", "latest", model, opt, meta, rng);

  // Snapshot current params, then perturb the live model so a successful load
  // must overwrite the perturbation.
  std::vector<torch::Tensor> saved;
  for (auto& p : model->parameters()) saved.push_back(p.detach().clone());
  { torch::NoGradGuard ng; for (auto& p : model->parameters()) p.add_(1.0); }

  archcommon::CkptMeta got;
  std::mt19937_64 rng2(999);
  bool ok = archcommon::load_checkpoint("/tmp/sb_ckpt_rt", "latest", model, opt, got,
                                        rng2, torch::kCPU);
  CHECK(ok);
  CHECK(got.step == 5);
  CHECK_NEAR(got.best, 1.25, 1e-12);
  auto ps = model->parameters();
  double diff = 0.0;
  for (size_t i = 0; i < ps.size(); ++i)
    diff = std::max(diff, (ps[i] - saved[i]).abs().max().item<double>());
  CHECK(diff == 0.0);
}

static void test_rng_restore() {
  torch::manual_seed(3);
  Tiny model(8);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  std::mt19937_64 rng(42);
  for (int i = 0; i < 7; ++i) (void)rng();  // advance
  archcommon::CkptMeta meta; meta.arch = "tiny"; meta.fingerprint = "d=8";
  archcommon::save_checkpoint("/tmp/sb_ckpt_rng", "latest", model, opt, meta, rng);
  uint64_t expect_mt = rng();                  // next draw after save
  auto expect_torch = torch::randint(0, 1000000, {3});

  archcommon::CkptMeta got;
  std::mt19937_64 rng2(0);
  archcommon::load_checkpoint("/tmp/sb_ckpt_rng", "latest", model, opt, got, rng2, torch::kCPU);
  uint64_t got_mt = rng2();
  auto got_torch = torch::randint(0, 1000000, {3});
  CHECK(got_mt == expect_mt);
  CHECK(torch::equal(got_torch, expect_torch));
}

static void test_fingerprint_rejection() {
  torch::manual_seed(1);
  Tiny model(8);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  std::mt19937_64 rng(1);
  archcommon::CkptMeta meta; meta.arch = "tiny"; meta.fingerprint = "tiny|d=128";
  archcommon::save_checkpoint("/tmp/sb_ckpt_fp", "latest", model, opt, meta, rng);

  CHECK(archcommon::checkpoint_exists("/tmp/sb_ckpt_fp", "latest"));
  archcommon::CkptMeta got;
  std::mt19937_64 rng2(2);
  bool ok = archcommon::load_checkpoint("/tmp/sb_ckpt_fp", "latest", model, opt, got, rng2,
                                        torch::kCPU);
  CHECK(ok);
  // The caller decides rejection by comparing the loaded fingerprint:
  CHECK(got.fingerprint == "tiny|d=128");
  CHECK(got.fingerprint != "tiny|d=64");
}

static void test_resume_determinism() {
  torch::set_num_threads(1);  // make CPU math bit-reproducible for an exact check
  auto corp = tiny_corpus();
  seqbench::ByteSpan span{corp.data(), corp.size()};
  const int B = 2, T = 16;
  const uint64_t SEED = 7;

  auto run_continuous = [&]() {
    torch::manual_seed(SEED);
    Tiny m(16);
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(1e-2));
    std::mt19937_64 rng(SEED);
    for (int s = 0; s < 4; ++s) {
      opt.zero_grad();
      tiny_loss(m, archcommon::sample_batch(span, B, T, rng)).backward();
      opt.step();
    }
    return m;
  };

  auto run_resumed = [&]() {
    torch::manual_seed(SEED);
    Tiny m(16);
    torch::optim::Adam opt(m->parameters(), torch::optim::AdamOptions(1e-2));
    std::mt19937_64 rng(SEED);
    for (int s = 0; s < 2; ++s) {  // first half
      opt.zero_grad();
      tiny_loss(m, archcommon::sample_batch(span, B, T, rng)).backward();
      opt.step();
    }
    archcommon::CkptMeta meta; meta.step = 2; meta.arch = "tiny"; meta.fingerprint = "d=16";
    archcommon::save_checkpoint("/tmp/sb_ckpt_det", "latest", m, opt, meta, rng);

    torch::manual_seed(SEED);          // fresh process would re-seed and re-init
    Tiny m2(16);
    torch::optim::Adam opt2(m2->parameters(), torch::optim::AdamOptions(1e-2));
    std::mt19937_64 rng2(0);
    archcommon::CkptMeta got;
    archcommon::load_checkpoint("/tmp/sb_ckpt_det", "latest", m2, opt2, got, rng2, torch::kCPU);
    for (int s = 0; s < 2; ++s) {      // second half, from restored state
      opt2.zero_grad();
      tiny_loss(m2, archcommon::sample_batch(span, B, T, rng2)).backward();
      opt2.step();
    }
    return m2;
  };

  Tiny a = run_continuous();
  Tiny b = run_resumed();
  double d = max_param_diff(a, b);
  std::printf("    [resume determinism max param diff=%.3e]\n", d);
  CHECK(d == 0.0);
}

int main() {
  RUN(test_roundtrip_params_and_opt);
  RUN(test_rng_restore);
  RUN(test_fingerprint_rejection);
  RUN(test_resume_determinism);
  return test_summary();
}
```

- [ ] **Step 3: Add the `checkpoint_test` executable to `arch/deltanet/CMakeLists.txt`**

Append after the `deltanet_test` block (around line 25):

```cmake
add_executable(checkpoint_test ${BENCH_ROOT}/arch/common/checkpoint_test.cpp)
target_include_directories(checkpoint_test PRIVATE ${BENCH_ROOT}/include ${BENCH_ROOT}/tests ${BENCH_ROOT}/arch)
target_link_libraries(checkpoint_test bench_core "${TORCH_LIBRARIES}")
```

- [ ] **Step 4: Build and run the test, expect it to pass**

Run:
```bash
cmake -S arch/deltanet -B arch/deltanet/build \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)") >/dev/null
cmake --build arch/deltanet/build -j --target checkpoint_test
./arch/deltanet/build/checkpoint_test
```
Expected: all four tests print and `OK`. The determinism line shows `max param diff=0.000e+00`.

If `test_resume_determinism` shows a nonzero diff, do NOT relax the bound: investigate which state was not restored (most likely optimizer state or an RNG), fix `checkpoint.hpp`, and re-run.

- [ ] **Step 5: Commit**

```bash
git add arch/common/checkpoint.hpp arch/common/checkpoint_test.cpp arch/deltanet/CMakeLists.txt
git commit  # message below, with the standard trailer
```
Message: `Add resumable checkpoint serialization layer with determinism test`

---

### Task 2: Device support in the shared runner

Add a device the whole run targets, defaulting to CPU so present behavior is identical. This is the piece that lets the GPU box use its GPU.

**Files:**
- Modify: `arch/common/runner.hpp` (RunConfig + `run_experiment` device plumbing)
- Modify: `arch/deltanet/train_deltanet.cpp` (arg parse)
- Modify: `arch/fast-weights/train_fw.cpp` (arg parse)
- Modify: `arch/gru/train_gru.cpp` (arg parse)

**Interfaces:**
- Produces: `RunConfig.device` (string, default `"cpu"`); all three `train_*` accept `--device <cpu|cuda>`.
- Consumes: nothing new.

- [ ] **Step 1: Add `device` to `RunConfig` in `arch/common/runner.hpp`**

In the `RunConfig` struct (currently ending at line 24), add after the `block_len/dict_size/key_len` line:

```cpp
  std::string device = "cpu";  // "cpu" or "cuda"
```

- [ ] **Step 2: Move the model and batches onto the device in `run_experiment`**

After `torch::manual_seed(...)` and `std::mt19937_64 rng(rc.seed);` (around line 63), add:

```cpp
  torch::Device dev(rc.device == "cuda" ? torch::kCUDA : torch::kCPU);
  model->to(dev);
```

Then move every batch to `dev` at its use sites:
- In the training loop, change `auto xb = train_batch(rng);` to `auto xb = train_batch(rng).to(dev);`
- In `eval_val_bpb`, change `tot += loss_fn(model, vb).item().toDouble();` to `tot += loss_fn(model, vb.to(dev)).item().toDouble();`
- In the final train_bpb block, change `loss_fn(model, train_batch(rng))` to `loss_fn(model, train_batch(rng).to(dev))`

(The diagnostic adapter path runs through the model on CPU-resident single bytes; leave it as is. On the CPU default `dev` is CPU and every `.to(dev)` is a no-op.)

- [ ] **Step 3: Add `--device` to each `train_*` arg parser**

In `arch/deltanet/train_deltanet.cpp`, in the arg loop (after the `--corpus` / `--task` lines, near line 35), add:

```cpp
    else if (a == "--device") rc.device = need("--device");
```

Make the identical addition in `arch/fast-weights/train_fw.cpp` and `arch/gru/train_gru.cpp` (each has the same `for` / `need` arg-parse shape; add the line alongside the other `rc.*` assignments).

- [ ] **Step 4: Rebuild and verify CPU behavior is unchanged**

Run:
```bash
cmake --build arch/deltanet/build -j --target deltanet_test train_deltanet
./arch/deltanet/build/deltanet_test
./arch/deltanet/build/train_deltanet --corpus toy --d 16 --n-layers 1 --steps 2 \
  --device cpu --out /tmp/sb_dev_smoke.jsonl
```
Expected: `deltanet_test` prints `OK`; the train smoke prints `val_bpb=... train_bpb=...` and appends a record to `/tmp/sb_dev_smoke.jsonl` (NOT the repo's runs file). The CUDA path is not exercised here (no GPU on this box).

- [ ] **Step 5: Commit**

```bash
git add arch/common/runner.hpp arch/deltanet/train_deltanet.cpp \
        arch/fast-weights/train_fw.cpp arch/gru/train_gru.cpp
git commit  # message below, with the standard trailer
```
Message: `Add --device flag and device plumbing to the shared runner`

---

### Task 3: Checkpoint integration into the runner + flags + docs

Wire save/resume into `run_experiment`, expose the flags, prove the integration with a smoke test, and document the new commands.

**Files:**
- Modify: `arch/common/runner.hpp` (RunConfig fields + save/resume in the loop, replacing the `/tmp/seqbench_best.pt` mechanism)
- Modify: `arch/deltanet/train_deltanet.cpp`, `arch/fast-weights/train_fw.cpp`, `arch/gru/train_gru.cpp` (three new flags each)
- Modify: `arch/deltanet/deltanet_test.cpp` (run_experiment resume smoke test)
- Modify: `README.md` and `arch/deltanet/README.md` (document the flags; drop the "in progress" note)

**Interfaces:**
- Consumes: `archcommon::{CkptMeta, checkpoint_exists, save_checkpoint, load_checkpoint}` from Task 1; `RunConfig.device` and `torch::Device dev` from Task 2.
- Produces: `RunConfig.{ckpt_dir, ckpt_every, resume}`; all three `train_*` accept `--ckpt-dir <dir>`, `--ckpt-every <int>`, `--resume`.

- [ ] **Step 1: Add the checkpoint fields to `RunConfig`**

In `arch/common/runner.hpp`, alongside the `device` field added in Task 2:

```cpp
  std::string ckpt_dir = "";   // "" disables checkpointing
  int ckpt_every = 0;          // 0 means "use eval_every"
  bool resume = false;
```

- [ ] **Step 2: Add `#include "common/checkpoint.hpp"` to `runner.hpp`**

At the top of `arch/common/runner.hpp`, after the existing seqbench includes, add:

```cpp
#include "common/checkpoint.hpp"
```

- [ ] **Step 3: Write the resume smoke test in `arch/deltanet/deltanet_test.cpp`**

Add this include near the top (after `#include "deltanet_model.hpp"`):

```cpp
#include "common/runner.hpp"
```

Add this test function before `main`:

```cpp
static void test_run_experiment_resume() {
  // A tiny deltanet trained on the toy corpus, checkpointing to /tmp, then
  // resumed to a higher target step count. Verifies run_experiment wires
  // save + resume correctly (start_step advances, no error, record written).
  const std::string ckpt = "/tmp/sb_runexp_ckpt";
  std::system(("rm -rf " + ckpt).c_str());
  auto make = [&](int steps, bool resume) {
    archcommon::RunConfig rc;
    rc.corpus = "toy"; rc.seq_len = 16; rc.batch = 2; rc.steps = steps;
    rc.eval_every = 2; rc.ckpt_dir = ckpt; rc.ckpt_every = 2; rc.resume = resume;
    rc.out = "/tmp/sb_runexp.jsonl";
    dn::Config cfg; cfg.d = 16; cfg.n_layers = 1;
    std::map<std::string, seqbench::JsonValue> config;
    config["d"] = seqbench::JsonValue::n(cfg.d);
    config["n_layers"] = seqbench::JsonValue::n(cfg.n_layers);
    config["lambda"] = seqbench::JsonValue::n(cfg.lambda);
    dn::DeltaNet model(cfg);
    return archcommon::run_experiment<dn::DeltaNet>(
        rc, "deltanet", "test", config, model,
        [](dn::DeltaNet& m, torch::Tensor x) { return dn::bpb_loss(m, x); },
        [cfg](dn::DeltaNet& m) {
          return std::unique_ptr<seqbench::Model>(new dn::DeltaNetEval(m, cfg));
        });
  };
  int rc1 = make(4, false);
  CHECK(rc1 == 0);
  CHECK(archcommon::checkpoint_exists(ckpt, "latest"));
  archcommon::CkptMeta m1; std::mt19937_64 r;
  { std::ifstream f(ckpt + "/latest.meta"); std::string k;
    while (f >> k) { if (k == "step") { f >> m1.step; break; } } }
  CHECK(m1.step == 4);
  int rc2 = make(8, true);   // resume to target 8
  CHECK(rc2 == 0);
  archcommon::CkptMeta m2;
  { std::ifstream f(ckpt + "/latest.meta"); std::string k;
    while (f >> k) { if (k == "step") { f >> m2.step; break; } } }
  CHECK(m2.step == 8);
}
```

Add `RUN(test_run_experiment_resume);` to `main` (after the existing `RUN(...)` lines), and add `#include <fstream>`, `#include <random>`, `#include <cstdlib>`, and `#include <map>` near the top if not already present (`std::system` needs `<cstdlib>`; the run_experiment call uses `std::map`).

- [ ] **Step 4: Run the smoke test to verify it FAILS (resume not wired yet)**

Run:
```bash
cmake --build arch/deltanet/build -j --target deltanet_test 2>&1 | tail -5
./arch/deltanet/build/deltanet_test
```
Expected: a FAIL on `test_run_experiment_resume` (no checkpoint is written yet, so `checkpoint_exists` is false and `m1.step` is wrong). This confirms the test exercises the not-yet-built path.

- [ ] **Step 5: Implement the fingerprint, save, and resume in `run_experiment`**

In `arch/common/runner.hpp`, replace the best-checkpoint section (currently lines 104 to 121, from `const std::string best_path ...` through `if (best < ...) torch::load(model, best_path);`) with:

```cpp
  // Checkpoint plumbing. Fingerprint is arch + sorted structural config, so a
  // resume into a structurally different model is rejected.
  std::string fingerprint = arch;
  for (const auto& kv : config) {
    fingerprint += "|" + kv.first + "=";
    fingerprint += (kv.second.type == JsonValue::Number)
                       ? std::to_string(kv.second.num) : kv.second.str;
  }
  const bool ckpt_on = !rc.ckpt_dir.empty();
  const int ckpt_every = rc.ckpt_every > 0 ? rc.ckpt_every : rc.eval_every;
  const std::string fallback_best = "/tmp/seqbench_best.pt";

  double best = std::numeric_limits<double>::infinity();
  int start_step = 1;
  if (rc.resume && ckpt_on && archcommon::checkpoint_exists(rc.ckpt_dir, "latest")) {
    archcommon::CkptMeta meta;
    archcommon::load_checkpoint(rc.ckpt_dir, "latest", model, opt, meta, rng, dev);
    if (meta.fingerprint != fingerprint) {
      std::fprintf(stderr, "resume fingerprint mismatch:\n  ckpt:    %s\n  current: %s\n",
                   meta.fingerprint.c_str(), fingerprint.c_str());
      return 2;
    }
    start_step = meta.step + 1;
    best = meta.best;
    std::fprintf(stderr, "resumed from step %d (best_val_bpb=%.4f), target steps=%d\n",
                 meta.step, best, rc.steps);
  }

  auto save_latest = [&](int step) {
    if (!ckpt_on) return;
    archcommon::CkptMeta m; m.step = step; m.seed = rc.seed; m.best = best;
    m.arch = arch; m.fingerprint = fingerprint;
    archcommon::save_checkpoint(rc.ckpt_dir, "latest", model, opt, m, rng);
  };

  for (int step = start_step; step <= rc.steps; ++step) {
    model->train();
    auto xb = train_batch(rng).to(dev);
    opt.zero_grad();
    auto loss = loss_fn(model, xb);
    loss.backward();
    opt.step();
    if (step % rc.eval_every == 0 || step == rc.steps) {
      double vbpb = eval_val_bpb();
      bool improved = vbpb < best;
      std::fprintf(stderr, "step %d train_bpb=%.4f val_bpb=%.4f%s\n",
                   step, loss.item().toDouble(), vbpb, improved ? " *" : "");
      if (improved) {
        best = vbpb;
        if (ckpt_on) {
          archcommon::CkptMeta m; m.step = step; m.seed = rc.seed; m.best = best;
          m.arch = arch; m.fingerprint = fingerprint;
          archcommon::save_checkpoint(rc.ckpt_dir, "best", model, opt, m, rng);
        } else {
          torch::save(model, fallback_best);
        }
      }
    }
    if (ckpt_on && (step % ckpt_every == 0 || step == rc.steps)) save_latest(step);
  }
  // Restore best params for the final reported metrics.
  if (best < std::numeric_limits<double>::infinity()) {
    if (ckpt_on) {
      archcommon::CkptMeta m; std::mt19937_64 tmp;
      archcommon::load_checkpoint(rc.ckpt_dir, "best", model, opt, m, tmp, dev);
    } else {
      torch::load(model, fallback_best);
    }
  }
```

Note: the `train_batch(rng).to(dev)` line here supersedes the Task 2 edit to the same line (same result). The `opt` variable is already declared above this block (line 91), so it is in scope for both resume-load and save.

- [ ] **Step 6: Add the three checkpoint flags to each `train_*` arg parser**

In `arch/deltanet/train_deltanet.cpp`, alongside the `--device` line from Task 2:

```cpp
    else if (a == "--ckpt-dir") rc.ckpt_dir = need("--ckpt-dir");
    else if (a == "--ckpt-every") rc.ckpt_every = std::atoi(need("--ckpt-every"));
    else if (a == "--resume") rc.resume = true;
```

Make the identical addition in `arch/fast-weights/train_fw.cpp` and `arch/gru/train_gru.cpp`. (`--resume` takes no value, so it does not call `need`.)

- [ ] **Step 7: Build and run all arch tests, expect pass**

Run:
```bash
cmake --build arch/deltanet/build -j --target checkpoint_test deltanet_test train_deltanet
./arch/deltanet/build/checkpoint_test
./arch/deltanet/build/deltanet_test
```
Expected: both print `OK`, including the now-passing `test_run_experiment_resume` (latest.meta step 4 then 8).

- [ ] **Step 8: Manual end-to-end resume smoke**

Run:
```bash
rm -rf /tmp/sb_e2e
./arch/deltanet/build/train_deltanet --corpus toy --d 16 --n-layers 1 --steps 4 \
  --ckpt-dir /tmp/sb_e2e --ckpt-every 2 --out /tmp/sb_e2e.jsonl 2>&1 | tail -3
./arch/deltanet/build/train_deltanet --corpus toy --d 16 --n-layers 1 --steps 8 \
  --ckpt-dir /tmp/sb_e2e --ckpt-every 2 --resume --out /tmp/sb_e2e.jsonl 2>&1 | tail -3
```
Expected: the second run prints `resumed from step 4 ...` and continues to step 8. Confirm a mismatched resume is rejected:
```bash
./arch/deltanet/build/train_deltanet --corpus toy --d 32 --n-layers 1 --steps 8 \
  --ckpt-dir /tmp/sb_e2e --resume --out /tmp/sb_e2e.jsonl 2>&1 | tail -2; echo "exit=$?"
```
Expected: prints `resume fingerprint mismatch` and `exit=2`.

- [ ] **Step 9: Update the docs**

In `arch/deltanet/README.md`, under the Knobs paragraph, add a line documenting the new flags:

```markdown
Long-run flags (shared by all architectures): `--ckpt-dir <dir>` enables
resumable checkpoints (rolling `latest` plus `best`), `--ckpt-every <n>` sets the
save cadence (default: the eval interval), `--resume` continues from `<dir>`'s
latest checkpoint up to `--steps` (the target total), and `--device cpu|cuda`
selects the device. Checkpoints are gitignored; move them between machines with
rsync.
```

In `README.md`, replace the "Running long, and on a GPU" section body so it no longer says "in progress" and shows the real command:

```markdown
## Running long, and on a GPU

Training is resumable and device-portable. `--ckpt-dir` writes a rolling
`latest` plus `best` checkpoint set, `--resume` continues to the `--steps`
target, and `--device cuda` runs on a GPU. A checkpoint written on one machine
loads on another, so the intended flow is: develop and test on CPU, run the
heavy training on a GPU box that `git pull`s the code and points its
`CMAKE_PREFIX_PATH` at a CUDA libtorch:

\```bash
./arch/deltanet/build/train_deltanet --corpus data/enwik8 --d 128 --n-layers 4 \
  --steps 44000 --device cuda --ckpt-dir runs/ckpt/deltanet-asymptote --ckpt-every 2000
\```

Checkpoints are gitignored and moved between machines with rsync, not committed.
```

(Write the bash fence with real triple backticks; the backslashes above are only to show the fence inside this plan.)

- [ ] **Step 10: Commit**

```bash
git add arch/common/runner.hpp arch/deltanet/train_deltanet.cpp \
        arch/fast-weights/train_fw.cpp arch/gru/train_gru.cpp \
        arch/deltanet/deltanet_test.cpp README.md arch/deltanet/README.md
git commit  # message below, with the standard trailer
```
Message: `Wire resumable checkpointing into the shared runner with --ckpt-dir/--resume`

---

## After the plan

Push, then launch the resumable asymptote run (on the GPU box if available, else here):
```bash
./arch/deltanet/build/train_deltanet --corpus data/enwik8 --d 128 --n-layers 4 \
  --steps 44000 --device cuda --ckpt-dir runs/ckpt/deltanet-asymptote --ckpt-every 2000
```
Watch the streaming eval bpb, stop at plateau, commit the run record, then move to sub-project 2 (the ablation and layer-scale battery).
