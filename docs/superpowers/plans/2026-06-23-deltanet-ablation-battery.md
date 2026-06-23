# DeltaNet ablation and layer battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five inductive-bias toggles to the stacked DeltaNet so a cheap screening battery can rank which biases carry the win, built on a shared per-step mix primitive that keeps the training forward and the eval adapter from drifting.

**Architecture:** Task 1 refactors the delta-rule recurrence into one `FWMixImpl::step(W, x)` method that both the batched `forward` and the online `DeltaNetEval` call (no behavior change, guarded by the existing tests). Task 2 adds six `dn::Config` flags (defaulting to current behavior), routes them through `step` and `Block`, exposes CLI flags with run-record provenance, and adds a per-toggle trainability-and-consistency test. The battery itself is a set of runs done after the build.

**Tech Stack:** C++17, libtorch (CPU here), the deltanet CMake build, the repo's `test_util.hpp` harness.

## Global Constraints

- libtorch is confined to `arch/`; the bench core (`include/`, `src/`, `models/`, `tools/`, `tests/`) and the Makefile must not be touched. These are `arch/deltanet` tests.
- No em-dash characters in any committed file (a commit hook rejects them). Use commas, colons, parentheses, periods.
- Every commit message ends with the exact trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Never weaken a test threshold to make a test pass. If a variant fails a bound, report BLOCKED with the number and fix the cause.
- All flags default to the current behavior: `use_gate=true, use_delta=true, learn_lambda=false, no_decay=false, normalize_keys=true, use_mlp=true`. The default config must reproduce today's DeltaNet (guarded by `deltanet_test`).
- Test/smoke records go to `/tmp`, never the repo `runs/results.jsonl`.
- Build: `cmake --build arch/deltanet/build -j --target deltanet_test train_deltanet` (build dir already configured; if not: `cmake -S arch/deltanet -B arch/deltanet/build -DCMAKE_PREFIX_PATH=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")`).

---

### Task 1: Refactor the mix recurrence into a shared step() primitive

Unify the delta-rule recurrence (currently duplicated in `FWMixImpl::forward` and `DeltaNetEval::observe`) into one `step` method, with NO behavior change. This is the de-risking refactor; correctness is proven by the existing tests staying green.

**Files:**
- Modify: `arch/deltanet/deltanet_model.hpp` (declare `FWMixImpl::step`)
- Modify: `arch/deltanet/deltanet_model.cpp` (`FWMixImpl::forward`, `DeltaNetEval` ctor/predict/observe)

**Interfaces:**
- Produces: `torch::Tensor FWMixImpl::step(torch::Tensor& W, torch::Tensor x);` where `W` is `[B,d,d]` (mutated in place), `x` is `[B,d]`, returns `o` `[B,d]` (post-`wo`). Used by `forward` (B=batch) and `DeltaNetEval` (B=1).

- [ ] **Step 1: Declare `step` in the header**

In `arch/deltanet/deltanet_model.hpp`, inside `struct FWMixImpl`, add the declaration just after the `forward` declaration (around line 21):

```cpp
  torch::Tensor step(torch::Tensor& W, torch::Tensor x);  // W:[B,d,d] mutated, x:[B,d] -> o:[B,d]
```

- [ ] **Step 2: Implement `step` and rewrite `forward` to use it**

In `arch/deltanet/deltanet_model.cpp`, replace the entire `FWMixImpl::forward` definition (lines 18-42) with `step` plus a thin `forward`:

```cpp
torch::Tensor FWMixImpl::step(torch::Tensor& W, torch::Tensor x) {
  // x: [B,d], W: [B,d,d] mutated in place, returns o: [B,d].
  auto k = F::normalize(wk->forward(x), F::NormalizeFuncOptions().dim(1));   // [B,d]
  auto v = wv->forward(x);                                                   // [B,d]
  auto q = wq->forward(x);                                                   // [B,d]
  auto beta = torch::sigmoid(wbeta->forward(x));                            // [B,1]
  auto Wk = torch::bmm(W, k.unsqueeze(2)).squeeze(2);                        // [B,d]
  auto e = v - Wk;                                                           // [B,d]
  W = lambda * W + torch::bmm((beta * e).unsqueeze(2), k.unsqueeze(1));      // [B,d,d]
  auto o = torch::bmm(W, q.unsqueeze(2)).squeeze(2);                         // [B,d]
  return wo->forward(o);                                                     // [B,d]
}

torch::Tensor FWMixImpl::forward(torch::Tensor h) {
  auto B = h.size(0);
  auto T = h.size(1);
  int d = dim;
  auto W = torch::zeros({B, d, d}, h.options());
  std::vector<torch::Tensor> outs;
  outs.reserve(T);
  for (int64_t t = 0; t < T; ++t) outs.push_back(step(W, h.select(1, t)));
  return torch::stack(outs, 1);  // [B,T,d]
}
```

Note: `wo` now applies per-step instead of once on the stacked output. Because `wo` is a linear map, `stack(wo(o_t)) == wo(stack(o_t))`, so this is numerically equivalent. The `forward` signature is unchanged.

- [ ] **Step 3: Rewrite the eval adapter to carry [1,d] state and call `step`**

In `arch/deltanet/deltanet_model.cpp`, replace the `DeltaNetEval` constructor, `predict`, and `observe` (lines 82-119) with:

```cpp
DeltaNetEval::DeltaNetEval(DeltaNet model, const Config& c) : model_(model), cfg_(c) {
  model_->eval();
  for (int i = 0; i < cfg_.n_layers; ++i) W_.push_back(torch::zeros({1, cfg_.d, cfg_.d}));
  out_ = torch::zeros({1, cfg_.d});
}

void DeltaNetEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto x = seen_ ? out_ : torch::zeros({1, cfg_.d});                              // [1,d]
  auto o = model_->readout->forward(model_->ln_f->forward(x)).contiguous().view({256});
  float* p = o.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void DeltaNetEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  int d = cfg_.d;
  auto x = model_->emb->forward(torch::tensor({static_cast<int64_t>(b)}, torch::kLong)).view({1, d});  // [1,d]
  for (int i = 0; i < cfg_.n_layers; ++i) {
    auto& blk = model_->blocks[i];
    auto h = blk->ln1->forward(x);              // [1,d]
    x = x + blk->mix->step(W_[i], h);           // step mutates W_[i] [1,d,d], returns [1,d]
    auto h2 = blk->ln2->forward(x);
    x = x + blk->fc2->forward(torch::gelu(blk->fc1->forward(h2)));
  }
  out_ = x;  // [1,d]
  seen_ = true;
}
```

The `W_` vector type in the header (`std::vector<torch::Tensor>`) and `out_` are unchanged; they now hold `[1,d,d]` and `[1,d]` tensors. No header change needed for those members.

- [ ] **Step 4: Build and run `deltanet_test`, expect unchanged green**

Run:
```bash
cmake --build arch/deltanet/build -j --target deltanet_test
./arch/deltanet/build/deltanet_test
```
Expected: `OK`. Specifically `test_overfit_tiny` still reaches about `0.0003`, `test_train_eval_consistency` still shows matching `train_bits` and `eval_bits` (about `318.5757`), and `test_deterministic` passes. If the consistency value or overfit drifts materially, the refactor changed behavior: do NOT adjust the test, find the discrepancy (most likely the `wo` placement or a shape error) and fix `step`.

- [ ] **Step 5: Commit**

```bash
git add arch/deltanet/deltanet_model.hpp arch/deltanet/deltanet_model.cpp
git commit  # message below, with the standard trailer
```
Message: `Refactor DeltaNet mix into a shared step() primitive (no behavior change)`

---

### Task 2: Add inductive-bias toggles, CLI flags, provenance, and per-toggle tests

Add the six `dn::Config` flags, route them through `step` and `Block`, expose CLI flags that record into the run, and add a parametrized per-toggle test.

**Files:**
- Modify: `arch/deltanet/deltanet_model.hpp` (`Config` flags; `FWMixImpl`/`BlockImpl` ctor signatures + members)
- Modify: `arch/deltanet/deltanet_model.cpp` (`FWMixImpl` ctor + `step`; `BlockImpl` ctor + forward; `DeltaNetImpl` ctor; eval adapter MLP guard)
- Modify: `arch/deltanet/train_deltanet.cpp` (CLI flags + config-map provenance)
- Modify: `arch/deltanet/deltanet_test.cpp` (per-toggle test)

**Interfaces:**
- Consumes: `FWMixImpl::step` from Task 1.
- Produces: `dn::Config` with `bool use_gate, use_delta, learn_lambda, no_decay, normalize_keys, use_mlp;`; `FWMixImpl(const Config&)` and `BlockImpl(const Config&)` constructors.

- [ ] **Step 1: Extend `Config` and the module declarations in the header**

In `arch/deltanet/deltanet_model.hpp`, replace the `Config` struct (lines 9-13) with:

```cpp
struct Config {
  int d = 128;
  int n_layers = 4;
  double lambda = 0.99;
  bool use_gate = true;        // false: beta = 1 (no learned write gate)
  bool use_delta = true;       // false: pure Hebbian, e = v (drop the -Wk term)
  bool learn_lambda = false;   // true: lambda = sigmoid(per-layer param), init from `lambda`
  bool no_decay = false;       // true: lambda = 1.0 (no forgetting; overrides learn_lambda)
  bool normalize_keys = true;  // false: raw keys
  bool use_mlp = true;         // false: block is mix-only (MLP sublayer skipped)
};
```

In `struct FWMixImpl`, change the constructor declaration and add flag members + the lambda parameter. Replace lines 16-22 (the `FWMixImpl` body up to and including the `forward`/`step` declarations) so the struct reads:

```cpp
struct FWMixImpl : torch::nn::Module {
  int dim;
  double lambda;
  bool use_gate, use_delta, learn_lambda, no_decay, normalize_keys;
  torch::nn::Linear wk{nullptr}, wv{nullptr}, wq{nullptr}, wbeta{nullptr}, wo{nullptr};
  torch::Tensor lambda_logit;  // registered only when learn_lambda
  explicit FWMixImpl(const Config& c);
  torch::Tensor forward(torch::Tensor h);  // [B,T,d] -> [B,T,d]
  torch::Tensor step(torch::Tensor& W, torch::Tensor x);  // W:[B,d,d] mutated, x:[B,d] -> o:[B,d]
};
TORCH_MODULE(FWMix);
```

In `struct BlockImpl`, change the constructor and add a `use_mlp` member. Replace lines 26-32 so it reads:

```cpp
struct BlockImpl : torch::nn::Module {
  bool use_mlp;
  torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
  FWMix mix{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  explicit BlockImpl(const Config& c);
  torch::Tensor forward(torch::Tensor x);  // [B,T,d] -> [B,T,d]
};
TORCH_MODULE(Block);
```

(`Config` is defined above these structs, so the constructor references resolve.)

- [ ] **Step 2: Update the constructors and `step`/`forward`/Block in the .cpp**

In `arch/deltanet/deltanet_model.cpp`, replace the `FWMixImpl` constructor (lines 10-16) with one taking `const Config&`, storing flags, and conditionally registering `lambda_logit`:

```cpp
FWMixImpl::FWMixImpl(const Config& c)
    : dim(c.d), lambda(c.lambda),
      use_gate(c.use_gate), use_delta(c.use_delta), learn_lambda(c.learn_lambda),
      no_decay(c.no_decay), normalize_keys(c.normalize_keys) {
  wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  wbeta = register_module("wbeta", torch::nn::Linear(dim, 1));
  wo = register_module("wo", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  if (learn_lambda) {
    double logit = std::log(lambda / (1.0 - lambda));  // sigmoid(logit) == lambda
    // float32 to match the model params (a float64 tensor would error when multiplied with W).
    lambda_logit = register_parameter("lambda_logit", torch::full({1}, logit, torch::kFloat32));
  }
}
```

Replace the `step` body (written in Task 1) with the toggle-aware version:

```cpp
torch::Tensor FWMixImpl::step(torch::Tensor& W, torch::Tensor x) {
  auto k = wk->forward(x);
  if (normalize_keys) k = F::normalize(k, F::NormalizeFuncOptions().dim(1));
  auto v = wv->forward(x);
  auto q = wq->forward(x);
  torch::Tensor beta = use_gate ? torch::sigmoid(wbeta->forward(x))
                                : torch::ones({x.size(0), 1}, x.options());
  auto Wk = torch::bmm(W, k.unsqueeze(2)).squeeze(2);
  auto e = use_delta ? (v - Wk) : v;
  auto update = torch::bmm((beta * e).unsqueeze(2), k.unsqueeze(1));  // [B,d,d]
  if (no_decay) {
    W = W + update;                                  // lambda = 1
  } else if (learn_lambda) {
    W = torch::sigmoid(lambda_logit) * W + update;   // lambda_logit broadcasts [1] over [B,d,d]
  } else {
    W = lambda * W + update;                         // fixed lambda
  }
  auto o = torch::bmm(W, q.unsqueeze(2)).squeeze(2);
  return wo->forward(o);
}
```

(`forward` from Task 1 is unchanged; it just calls `step`.)

Replace the `BlockImpl` constructor and forward (lines 44-56) with a `use_mlp`-aware version:

```cpp
BlockImpl::BlockImpl(const Config& c) : use_mlp(c.use_mlp) {
  ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.d})));
  mix = register_module("mix", FWMix(c));
  if (use_mlp) {
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.d})));
    fc1 = register_module("fc1", torch::nn::Linear(c.d, 4 * c.d));
    fc2 = register_module("fc2", torch::nn::Linear(4 * c.d, c.d));
  }
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
  x = x + mix->forward(ln1->forward(x));
  if (use_mlp) x = x + fc2->forward(torch::gelu(fc1->forward(ln2->forward(x))));
  return x;
}
```

Update the `DeltaNetImpl` constructor's block creation (line 60-61) to pass the config:

```cpp
  for (int i = 0; i < cfg.n_layers; ++i)
    blocks.push_back(register_module("block" + std::to_string(i), Block(cfg)));
```

In the eval adapter `observe` (from Task 1), guard the MLP on `cfg_.use_mlp`. Replace the two MLP lines with:

```cpp
    if (cfg_.use_mlp) {
      auto h2 = blk->ln2->forward(x);
      x = x + blk->fc2->forward(torch::gelu(blk->fc1->forward(h2)));
    }
```

Add `#include <cmath>` is already present (line 2); `std::log` is available.

- [ ] **Step 3: Add CLI flags and provenance to `train_deltanet.cpp`**

In `arch/deltanet/train_deltanet.cpp`, add local bool defaults near the existing `int d`, `int n_layers`, `double lambda` declarations:

```cpp
  bool no_gate = false, hebbian = false, learn_lambda = false,
       no_decay = false, no_key_norm = false, no_mlp = false;
```

In the arg-parse loop, add (alongside the existing `--lambda` line):

```cpp
    else if (a == "--no-gate") no_gate = true;
    else if (a == "--hebbian") hebbian = true;
    else if (a == "--learn-lambda") learn_lambda = true;
    else if (a == "--no-decay") no_decay = true;
    else if (a == "--no-key-norm") no_key_norm = true;
    else if (a == "--no-mlp") no_mlp = true;
```

After the loop where `cfg` is built (`cfg.d = d; cfg.n_layers = n_layers; cfg.lambda = lambda;`), set the flags and record them:

```cpp
  cfg.use_gate = !no_gate;
  cfg.use_delta = !hebbian;
  cfg.learn_lambda = learn_lambda;
  cfg.no_decay = no_decay;
  cfg.normalize_keys = !no_key_norm;
  cfg.use_mlp = !no_mlp;
  config["use_gate"] = JsonValue::n(cfg.use_gate ? 1 : 0);
  config["use_delta"] = JsonValue::n(cfg.use_delta ? 1 : 0);
  config["learn_lambda"] = JsonValue::n(cfg.learn_lambda ? 1 : 0);
  config["no_decay"] = JsonValue::n(cfg.no_decay ? 1 : 0);
  config["normalize_keys"] = JsonValue::n(cfg.normalize_keys ? 1 : 0);
  config["use_mlp"] = JsonValue::n(cfg.use_mlp ? 1 : 0);
```

(The existing `config` map and `JsonValue` are already in scope; `cfg` is `dn::Config`.)

- [ ] **Step 4: Write the per-toggle test (failing first)**

In `arch/deltanet/deltanet_test.cpp`, add `#include <functional>` near the top includes, and add this test function before `main`:

```cpp
static void test_toggles_train_and_consistent() {
  struct Variant { const char* name; std::function<void(dn::Config&)> set; };
  std::vector<Variant> variants = {
    {"no_gate",      [](dn::Config& c) { c.use_gate = false; }},
    {"hebbian",      [](dn::Config& c) { c.use_delta = false; }},
    {"learn_lambda", [](dn::Config& c) { c.learn_lambda = true; }},
    {"no_decay",     [](dn::Config& c) { c.no_decay = true; }},
    {"no_key_norm",  [](dn::Config& c) { c.normalize_keys = false; }},
    {"no_mlp",       [](dn::Config& c) { c.use_mlp = false; }},
  };
  for (auto& var : variants) {
    torch::manual_seed(1);
    dn::Config c; c.d = 24; c.n_layers = 2; var.set(c);
    dn::DeltaNet model(c);
    // (a) trainability: overfit a period-3 sequence; loss must drop well below naive 8.0.
    const int T = 48;
    std::vector<int64_t> buf(T);
    for (int i = 0; i < T; ++i) buf[i] = 97 + (i % 3);
    auto x = torch::tensor(buf, torch::kLong).reshape({1, T});
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
    double last = 1e9;
    for (int s = 0; s < 300; ++s) {
      opt.zero_grad();
      auto loss = dn::bpb_loss(model, x);
      loss.backward();
      opt.step();
      last = loss.item<double>();
    }
    std::printf("    [toggle %s overfit bpb=%.4f]\n", var.name, last);
    CHECK(last < 4.0);
    // (b) consistency: batched forward bits vs online adapter bits must agree.
    model->eval();
    const int U = 40;
    std::vector<int64_t> buf2(U);
    for (int i = 0; i < U; ++i) buf2[i] = (i * 37 + 11) % 256;
    double train_bits = 0.0;
    {
      torch::NoGradGuard ng;
      auto xx = torch::tensor(buf2, torch::kLong).reshape({1, U});
      auto logp = torch::log_softmax(model->forward(xx).slice(1, 0, U - 1).reshape({-1, 256}), 1).contiguous();
      auto a = logp.accessor<float, 2>();
      for (int t = 0; t < U - 1; ++t) train_bits += -double(a[t][buf2[t + 1]]) / std::log(2.0);
    }
    double eval_bits = 0.0;
    dn::DeltaNetEval ev(model, c);
    float lg[256];
    for (int i = 0; i < U; ++i) {
      ev.predict(lg);
      if (i >= 1) eval_bits += seqbench::logit_bits(lg, static_cast<uint8_t>(buf2[i]));
      ev.observe(static_cast<uint8_t>(buf2[i]));
    }
    std::printf("    [toggle %s consistency train=%.4f eval=%.4f]\n", var.name, train_bits, eval_bits);
    CHECK_NEAR(train_bits, eval_bits, 0.05 * (U - 1));
  }
}
```

Register it in `main` with `RUN(test_toggles_train_and_consistent);` after the existing `RUN(...)` lines.

The `CHECK(last < 4.0)` is a generous trainability smoke (a period-3 pattern is trivial; the default config overfits to about 0.0003). It is not a precision bound, just "this variant learns." `CHECK_NEAR(..., 0.05 * 39)` is the same tolerance as the existing consistency test and is the real drift guard: it proves the online adapter matches the batched forward for every toggle. Do not loosen either; if a variant fails, the toggle wiring is wrong.

- [ ] **Step 5: Build and run the test to verify it FAILS before Step 2's wiring exists**

This task interleaves header/impl/test edits, so to get a clean TDD signal: after writing Step 4's test but before completing Step 2's toggle logic, the test would not compile (Config lacks the flags). That is the expected RED. Implement Steps 1-3 fully, then build:

```bash
cmake --build arch/deltanet/build -j --target deltanet_test train_deltanet
./arch/deltanet/build/deltanet_test
```
Expected: `OK`, with six `[toggle <name> overfit ...]` and `[toggle <name> consistency ...]` lines, each consistency pair matching within tolerance and each overfit below 4.0.

- [ ] **Step 6: Commit**

```bash
git add arch/deltanet/deltanet_model.hpp arch/deltanet/deltanet_model.cpp \
        arch/deltanet/train_deltanet.cpp arch/deltanet/deltanet_test.cpp
git commit  # message below, with the standard trailer
```
Message: `Add DeltaNet inductive-bias toggles with CLI flags and per-toggle tests`

---

## After the plan: run the battery

All at d=64, seq_len 128, about 6000 steps, batch 32, seed 1, on enwik8, each with `--ckpt-dir` for crash-safety. About 0.5 hours each on CPU.

```bash
B="--corpus data/enwik8 --d 64 --seq-len 128 --steps 6000 --seed 1"
./arch/deltanet/build/train_deltanet $B --ckpt-dir runs/ckpt/abl-baseline
./arch/deltanet/build/train_deltanet $B --no-gate     --ckpt-dir runs/ckpt/abl-nogate
./arch/deltanet/build/train_deltanet $B --hebbian     --ckpt-dir runs/ckpt/abl-hebbian
./arch/deltanet/build/train_deltanet $B --learn-lambda --ckpt-dir runs/ckpt/abl-learnlam
./arch/deltanet/build/train_deltanet $B --no-decay    --ckpt-dir runs/ckpt/abl-nodecay
./arch/deltanet/build/train_deltanet $B --no-key-norm --ckpt-dir runs/ckpt/abl-nokeynorm
./arch/deltanet/build/train_deltanet $B --no-mlp      --ckpt-dir runs/ckpt/abl-nomlp
./arch/deltanet/build/train_deltanet $B --n-layers 5  --ckpt-dir runs/ckpt/abl-L5
./arch/deltanet/build/train_deltanet $B --n-layers 6  --ckpt-dir runs/ckpt/abl-L6
./arch/deltanet/build/train_deltanet $B --n-layers 8  --ckpt-dir runs/ckpt/abl-L8
```

Each appends a record (with its toggle in `config`) to `runs/results.jsonl`. Compare `val_bpb` to the baseline: the toggle whose removal hurts most is the bias that matters most; the layer sweep shows depth scaling at d=64. Write the ranking into `arch/deltanet/README.md`, commit the records, and the best configuration becomes the candidate for the full d=128 asymptote run on the GPU.
