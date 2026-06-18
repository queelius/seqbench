#include "test_util.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/metric.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace seqbench;

// Golden: order-0, alpha=1, input "aaaa".
// Step bits: 8, log2(257/2), log2(258/3), log2(259/4); total ~= 27.4487.
static void test_order0_golden() {
  const char* s = "aaaa";
  ByteSpan data{reinterpret_cast<const uint8_t*>(s), 4};
  ContextModel m(0, 1.0);
  BpbResult r = run_adaptive(m, data);
  double expect = 8.0 + std::log2(257.0 / 2.0) + std::log2(258.0 / 3.0) +
                  std::log2(259.0 / 4.0);
  CHECK_NEAR(r.total_bits, expect, 1e-4);
  CHECK_NEAR(r.total_bits, 27.4487, 1e-2);
}

// On periodic data, a model with memory (order>=1) beats order-0.
static void test_higher_order_beats_order0_on_periodic() {
  std::string s;
  for (int i = 0; i < 2000; ++i) s += "ab";  // "abab..." length 4000
  ByteSpan data{reinterpret_cast<const uint8_t*>(s.data()), s.size()};
  ContextModel m0(0, 1.0);
  ContextModel m1(1, 1.0);
  double b0 = run_adaptive(m0, data).bpb();
  double b1 = run_adaptive(m1, data).bpb();
  CHECK(b1 < b0);
  CHECK(b1 < 0.75);  // order-1 nearly predicts the period (measured ~0.57 bpb)
}

static void test_factory() {
  auto m = make_context_model(2);
  CHECK(m != nullptr);
}

int main() {
  RUN(test_order0_golden);
  RUN(test_higher_order_beats_order0_on_periodic);
  RUN(test_factory);
  return test_summary();
}
