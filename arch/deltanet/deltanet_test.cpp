#include "test_util.hpp"
#include <torch/torch.h>
#include "seqbench/metric.hpp"
#include "deltanet_model.hpp"
#include "common/runner.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
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

static void test_run_experiment_resume() {
  // A tiny deltanet trained on the toy corpus, checkpointing to /tmp, then
  // resumed to a higher target step count. Verifies run_experiment wires
  // save + resume correctly (start_step advances, no error, record written).
  const std::string ckpt = "/tmp/sb_runexp_ckpt";
  if (std::system(("rm -rf " + ckpt).c_str()) != 0) { /* best-effort cleanup */ }
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

// Shared driver for the run_experiment coverage tests below. Trains a tiny
// deltanet on the toy corpus with seq_len 16 so the toy corpus fits. `d` varies
// the model width (and hence the fingerprint); `ckpt` "" exercises the /tmp
// fallback path.
static int run_toy(const std::string& ckpt, int d, int steps, bool resume,
                   const std::string& out) {
  archcommon::RunConfig rc;
  rc.corpus = "toy"; rc.seq_len = 16; rc.batch = 2; rc.steps = steps;
  rc.eval_every = 2; rc.ckpt_dir = ckpt; rc.ckpt_every = 2; rc.resume = resume;
  rc.out = out;
  dn::Config cfg; cfg.d = d; cfg.n_layers = 1;
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
}

static int read_latest_step(const std::string& ckpt) {
  archcommon::CkptMeta m; std::ifstream f(ckpt + "/latest.meta"); std::string k;
  while (f >> k) { if (k == "step") { f >> m.step; break; } }
  return m.step;
}

// A resume into a structurally different model (different fingerprint) must be
// rejected with exit code 2, leaving the existing checkpoint untouched.
static void test_run_experiment_fingerprint_mismatch() {
  const std::string ckpt = "/tmp/sb_runexp_fpmismatch";
  if (std::system(("rm -rf " + ckpt).c_str()) != 0) { /* best-effort cleanup */ }
  CHECK(run_toy(ckpt, 16, 4, false, "/tmp/sb_runexp_fpmismatch.jsonl") == 0);
  CHECK(archcommon::checkpoint_exists(ckpt, "latest"));
  // Same ckpt dir, resume requested, but d=32 makes a different fingerprint.
  CHECK(run_toy(ckpt, 32, 4, true, "/tmp/sb_runexp_fpmismatch.jsonl") == 2);
}

// Resuming a run whose target step count is already met runs the loop zero
// times but still completes (returns 0) and leaves latest.meta at the reached
// step.
static void test_run_experiment_noop_resume() {
  const std::string ckpt = "/tmp/sb_runexp_noop";
  if (std::system(("rm -rf " + ckpt).c_str()) != 0) { /* best-effort cleanup */ }
  CHECK(run_toy(ckpt, 16, 4, false, "/tmp/sb_runexp_noop.jsonl") == 0);
  CHECK(read_latest_step(ckpt) == 4);
  // Resume to the same target: nothing left to do.
  CHECK(run_toy(ckpt, 16, 4, true, "/tmp/sb_runexp_noop.jsonl") == 0);
  CHECK(read_latest_step(ckpt) == 4);
}

// With ckpt_dir empty, checkpointing is off and the best model is saved to and
// restored from /tmp/seqbench_best.pt. The run must still complete (returns 0).
static void test_run_experiment_tmp_fallback() {
  CHECK(run_toy("", 16, 4, false, "/tmp/sb_runexp_tmpfallback.jsonl") == 0);
}

int main() {
  RUN(test_forward_shape_finite);
  RUN(test_deterministic);
  RUN(test_overfit_tiny);
  RUN(test_train_eval_consistency);
  RUN(test_run_experiment_resume);
  RUN(test_run_experiment_fingerprint_mismatch);
  RUN(test_run_experiment_noop_resume);
  RUN(test_run_experiment_tmp_fallback);
  return test_summary();
}
