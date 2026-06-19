# GRU architecture + shared experiment runner

**Date:** 2026-06-19
**Status:** Design approved, pending spec review
**Author:** Alexander Towell (lex@metafunctor.com)

## Motivation

The learned fast-weights probes showed a clean split: fast-weights captures in-context
recall (induction fraction 0.32 vs the n-gram's 0.25) but fails state-tracking (parity
fraction 0.00, training loss never left the naive 1.0). That parity failure is an
expressivity wall: the delta rule is linear in the state `W`, and a linear recurrence
cannot track parity, which needs a non-linear state. The recommended next experiment is a
genuinely non-linear gated recurrence, a GRU, which theory says can track parity and which
gated RNNs reliably learn.

Adding a second libtorch architecture also exposes duplication: the ~180-line `train_fw`
runner (arg handling, data sampling, train loop, eval-via-bench, run-record assembly) is
model-independent. This spec factors that into a shared runner so the GRU, and every future
architecture, is a small model plus a thin caller.

## Win condition

By the end:

- A shared, model-agnostic experiment runner exists, and `train_fw` is migrated onto it
  (verified by `fw_test` and a `train_fw` smoke run, so no behavior changed).
- A GRU model (`embedding -> GRU -> readout`) trains via the shared runner and is scored
  through the existing bench on enwik8 bpb and the parity/induction capability tasks,
  recording `arch:"gru-rnn"` to `runs/results.jsonl`.
- The records are directly comparable to the `fast-weights-learned` and context-model
  numbers already on `main`, so we can answer: does a non-linear gated recurrence capture
  state-tracking (and recall) that the linear fast-weights and the n-gram cannot?

## Scope and isolation

In scope: the shared runner, the migration of `train_fw`, the GRU model + its runner +
tests + build. The bench core stays dependency-free and untouched; all libtorch stays
under `arch/`.

Deferred (later): the hybrid (fast-weights recall + non-linear state), multi-layer / size
tuning, and the actual capability runs (train GRU on the tasks and compare), which are the
follow-up experiment.

## Shared experiment runner (`arch/common/runner.hpp`, header-only)

A template that drives training and evaluation for any libtorch byte-model. It owns
everything model-independent and takes the model-specific behavior as callables.

```cpp
namespace archcommon {

struct RunConfig {
  int seq_len = 256, batch = 32, steps = 50000, eval_every = 1000;
  double lr = 1e-3;
  uint64_t seed = 1;
  std::string corpus = "data/enwik8", task = "enwik8", out = "runs/results.jsonl";
  int block_len = 16, dict_size = 16, key_len = 4;  // task generator params
};

// Drives: data sampling (corpus or task), Adam train loop, best-by-val checkpoint,
// final eval (corpus val-bpb, or task fraction-captured via score_diagnostic on a
// held-out test stream), and RunRecord assembly + append. Returns a process exit code.
template <class ModelT>
int run_experiment(const RunConfig& rc, const std::string& arch, const std::string& version,
                   std::map<std::string, seqbench::JsonValue> config, ModelT model,
                   std::function<torch::Tensor(ModelT&, torch::Tensor)> loss_fn,
                   std::function<std::unique_ptr<seqbench::Model>(ModelT&)> make_adapter);

}  // namespace archcommon
```

- The model-independent helpers `slice`, `sample_batch`, `sample_task_batch` move into this
  header.
- The runner builds the optimizer over `model->parameters()`, uses `loss_fn(model, batch)`
  for forward/loss, `torch::save/load(model, ...)` for the checkpoint, and
  `make_adapter(model)` to obtain a bench `Model` for task scoring.
- Final eval branches on `rc.task`: for a task it builds a held-out test `Diagnostic`
  (`make_parity` / `make_induction` with a salted seed and the task params), scores it with
  `score_diagnostic(*adapter, test)`, and records `test_bpb` / `fraction_captured` /
  `floor_bpb` / `naive_bpb`; otherwise it records `val_bpb` (batched val forward) and
  `train_bpb`. `config.task` and the passed `config` map are recorded.

`train_fw` is migrated to call `run_experiment<fw::FastWeights>(...)`, keeping only its arg
parsing, the `fw::Config`/`fw::FastWeights` construction, the `config` map, and the two
lambdas (`fw::bpb_loss` and a `fw::FastWeightsEval` factory). Its observable behavior and
record shape are unchanged.

## GRU model (`arch/gru/gru_model.hpp`, `arch/gru/gru_model.cpp`)

```cpp
namespace gru {
struct Config { int d = 128; int layers = 1; };

struct GruImpl : torch::nn::Module {
  torch::nn::Embedding emb{nullptr};
  torch::nn::GRU rnn{nullptr};
  torch::nn::Linear readout{nullptr};
  explicit GruImpl(const Config& c);
  torch::Tensor forward(torch::Tensor x_bt);  // [B,T] -> [B,T,256]
};
TORCH_MODULE(Gru);

torch::Tensor bpb_loss(Gru model, torch::Tensor x_bt);   // next-byte CE in bits, positions 1..T-1

class GruEval : public seqbench::Model {  // online adapter: carries the GRU hidden state
 public:
  GruEval(Gru model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;
 private:
  Gru model_;
  Config cfg_;
  torch::Tensor h_;   // [layers, 1, d]
  torch::Tensor last_;  // last GRU output (for readout), or zeros before any observe
  bool seen_ = false;
};
}  // namespace gru
```

Forward: `emb(x) -> GRU -> outputs [B,T,d] -> readout -> [B,T,256]`, causal (output at t
depends on inputs up to t), so `logits_t` predicts `b_{t+1}` exactly as the fast-weights
model. The eval adapter steps the GRU one byte per `observe` (feeding the hidden state
forward) and reads `readout` of the latest output in `predict`; before any byte it returns
`readout(zeros)`. Knobs `d`, `layers`; records `arch:"gru-rnn"`, `config.d`,
`config.layers`.

## Build (`arch/gru/CMakeLists.txt`)

Mirrors `arch/fast-weights/`: `find_package(Torch)`, compile `bench_core` from the bench
sources, build `train_gru` (from `train_gru.cpp` + `gru_model.cpp`) and `gru_test` (from
`gru_test.cpp` + `gru_model.cpp`), with `${BENCH_ROOT}/include`, `${BENCH_ROOT}/arch` (for
`common/runner.hpp`), and the gru dir on the include path. The fast-weights CMakeLists gains
`${BENCH_ROOT}/arch` on its include path so `train_fw` can include the shared runner. The
bench's plain Makefile and `make test` are untouched and libtorch-free.

## Tests (`arch/gru/gru_test.cpp`, built by CMake)

- Forward produces `[*, 256]` finite logits; deterministic given a seed.
- Overfit a tiny repeating sequence: loss collapses below 1.0, proving the GRU train loop
  learns.
- Train/eval consistency: the online adapter's per-byte bits match the batched-forward bits
  on a short sequence within a small tolerance (the GRU recurrence is reproduced online).
- No test asserts the capability result; whether GRU flips parity is the recorded
  experiment. Thresholds are verified-achievable; a mismatch is reported BLOCKED with the
  measured numbers, never relaxed.

## Non-goals

The hybrid architecture, multi-layer/size sweeps, and the capability *runs* (train GRU on
parity, induction, and enwik8 and compare to the fast-weights and context-model records).

## Future directions (context)

If the GRU flips parity (captures state-tracking where the linear fast-weights could not)
while the fast-weights still wins induction, that motivates the hybrid: a fast-weight memory
for recall plus a non-linear recurrent state for tracking, aiming to win both diagnostics.
