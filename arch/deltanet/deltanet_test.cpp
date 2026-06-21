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
