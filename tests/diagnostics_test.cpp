#include "test_util.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/context_model.hpp"

using namespace seqbench;

// Same seed gives an identical stream.
static void test_generators_deterministic() {
  Diagnostic a = make_parity(123, 100, 8);
  Diagnostic b = make_parity(123, 100, 8);
  CHECK(a.stream.size() == b.stream.size());
  CHECK(a.stream == b.stream);
}

// A finite-order context model captures essentially no parity structure.
static void test_context_fails_parity() {
  Diagnostic d = make_parity(7, 4000, 16);
  auto m = make_context_model(4);
  DiagResult r = score_diagnostic(*m, d);
  CHECK(r.observed_bpb > 0.95);       // near the naive 1.0
  CHECK(r.fraction_captured < 0.1);   // captured ~nothing
}

// An order>=1 context model captures a clear fraction of induction structure.
static void test_context_captures_induction() {
  Diagnostic d = make_induction(7, 8000, 16);
  auto m1 = make_context_model(1);
  DiagResult r1 = score_diagnostic(*m1, d);
  CHECK(r1.fraction_captured > 0.3);  // captures real structure
  auto m0 = make_context_model(0);
  DiagResult r0 = score_diagnostic(*m0, d);
  CHECK(r1.observed_bpb < r0.observed_bpb);  // memory helps
}

int main() {
  RUN(test_generators_deterministic);
  RUN(test_context_fails_parity);
  RUN(test_context_captures_induction);
  return test_summary();
}
