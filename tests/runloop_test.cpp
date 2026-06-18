#include "test_util.hpp"
#include "seqbench/metric.hpp"
#include <cmath>
#include <stdexcept>

using namespace seqbench;

// Always-uniform model: every byte costs exactly 8 bits.
struct UniformModel : Model {
  void predict(float logits[256]) override {
    for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
  }
  void observe(uint8_t) override {}
};

// Emits a NaN logit to exercise the finiteness guard.
struct NanModel : Model {
  void predict(float logits[256]) override {
    for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
    logits[3] = std::nanf("");
  }
  void observe(uint8_t) override {}
};

static void test_adaptive_uniform_is_eight() {
  const uint8_t buf[6] = {1, 2, 3, 4, 5, 6};
  UniformModel m;
  BpbResult r = run_adaptive(m, ByteSpan{buf, 6});
  CHECK(r.n_bytes == 6);
  CHECK_NEAR(r.bpb(), 8.0, 1e-9);
  CHECK_NEAR(r.total_bits, 48.0, 1e-7);
}

static void test_train_test_uniform_is_eight() {
  const uint8_t tr[3] = {9, 9, 9};
  const uint8_t va[4] = {1, 2, 3, 4};
  UniformModel m;
  BpbResult r = run_train_test(m, ByteSpan{tr, 3}, ByteSpan{va, 4});
  CHECK(r.n_bytes == 4);
  CHECK_NEAR(r.bpb(), 8.0, 1e-9);
}

static void test_guard_throws_on_nan() {
  const uint8_t buf[2] = {0, 1};
  NanModel m;
  bool threw = false;
  try { run_adaptive(m, ByteSpan{buf, 2}); }
  catch (const std::exception&) { threw = true; }
  CHECK(threw);
}

int main() {
  RUN(test_adaptive_uniform_is_eight);
  RUN(test_train_test_uniform_is_eight);
  RUN(test_guard_throws_on_nan);
  return test_summary();
}
