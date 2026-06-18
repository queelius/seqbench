#include "test_util.hpp"
#include <torch/torch.h>
#include "seqbench/corpus.hpp"
#include "seqbench/metric.hpp"
#include "fast_weights_model.hpp"
#include <cmath>
#include <cstdint>
#include <vector>

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
  std::printf("    [overfit_tiny final bpb=%.4f]\n", last);
  CHECK(last < 1.0);  // memorized the period-3 pattern
}

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
  std::printf("    [consistency train_bits=%.4f eval_bits=%.4f]\n", train_bits, eval_bits);
  CHECK_NEAR(train_bits, eval_bits, 0.05 * (T - 1));  // < 0.05 bits/byte average drift
}

int main() {
  RUN(test_torch_works);
  RUN(test_bench_links);
  RUN(test_forward_shape_finite);
  RUN(test_deterministic);
  RUN(test_overfit_tiny);
  RUN(test_train_eval_consistency);
  return test_summary();
}
