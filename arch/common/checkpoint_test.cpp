#include "test_util.hpp"
#include <torch/torch.h>
#include "common/checkpoint.hpp"
#include "common/runner.hpp"  // archcommon::sample_batch
#include "seqbench/byte_span.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
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

static void test_fingerprint_roundtrip() {
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
  // The stored fingerprint round-trips verbatim; the caller decides rejection by
  // comparing it (the real rejection path is covered end-to-end by
  // test_run_experiment_fingerprint_mismatch in deltanet_test.cpp).
  CHECK(got.fingerprint == "tiny|d=128");
  CHECK(got.fingerprint != "tiny|d=64");
}

// Regression: `best` starts as +infinity (its pre-first-eval sentinel). The
// decimal form emits "inf", which std::ifstream cannot parse, which would abort
// the parse loop and drop arch + fingerprint (both written after best). The bit
// serialization must round-trip inf exactly AND keep the later fields readable.
static void test_best_inf_roundtrip() {
  torch::manual_seed(13);
  Tiny model(8);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
  std::mt19937_64 rng(13);
  archcommon::CkptMeta meta;
  meta.best = std::numeric_limits<double>::infinity();
  meta.arch = "x"; meta.fingerprint = "fp|d=128";
  archcommon::save_checkpoint("/tmp/sb_ckpt_inf", "latest", model, opt, meta, rng);

  archcommon::CkptMeta got;
  std::mt19937_64 rng2(0);
  bool ok = archcommon::load_checkpoint("/tmp/sb_ckpt_inf", "latest", model, opt, got, rng2,
                                        torch::kCPU);
  CHECK(ok);
  CHECK(std::isinf(got.best));
  // These two prove the fields written AFTER best survived the parse:
  CHECK(got.arch == "x");
  CHECK(got.fingerprint == "fp|d=128");
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
  RUN(test_fingerprint_roundtrip);
  RUN(test_best_inf_roundtrip);
  RUN(test_resume_determinism);
  return test_summary();
}
