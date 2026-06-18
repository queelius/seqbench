#include "test_util.hpp"
#include "seqbench/metric.hpp"
#include <cmath>

using namespace seqbench;

// Uniform logits => every byte costs exactly log2(256) = 8 bits.
static void test_uniform_is_eight_bits() {
  float logits[256];
  for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
  CHECK_NEAR(logit_bits(logits, 0), 8.0, 1e-9);
  CHECK_NEAR(logit_bits(logits, 200), 8.0, 1e-9);
}

// Shift-invariance: adding a constant to all logits cannot change bits.
static void test_shift_invariance() {
  float a[256], b[256];
  for (int i = 0; i < 256; ++i) { a[i] = 0.1f * i; b[i] = 0.1f * i + 7.0f; }
  CHECK_NEAR(logit_bits(a, 42), logit_bits(b, 42), 1e-5);
}

// Hand golden: all logits 0 except index 7 = ln(255) gives p[7]=1/2,
// p[other]=1/510, so bits(7)=1.0 and bits(0)=log2(510).
static void test_hand_golden() {
  float logits[256];
  for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
  logits[7] = std::log(255.0f);
  CHECK_NEAR(logit_bits(logits, 7), 1.0, 1e-5);
  CHECK_NEAR(logit_bits(logits, 0), std::log2(510.0), 1e-4);
}

static void test_finiteness() {
  float ok[256]; for (int i = 0; i < 256; ++i) ok[i] = 1.0f;
  CHECK(logits_finite(ok));
  float bad[256]; for (int i = 0; i < 256; ++i) bad[i] = 1.0f;
  bad[3] = std::nanf("");
  CHECK(!logits_finite(bad));
}

// Kahan keeps a small running value exact across many tiny additions.
static void test_kahan() {
  Kahan k;
  for (int i = 0; i < 1000000; ++i) k.add(1e-6);
  CHECK_NEAR(k.value(), 1.0, 1e-9);
}

int main() {
  RUN(test_uniform_is_eight_bits);
  RUN(test_shift_invariance);
  RUN(test_hand_golden);
  RUN(test_finiteness);
  RUN(test_kahan);
  return test_summary();
}
