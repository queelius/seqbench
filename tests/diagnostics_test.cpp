#include "test_util.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/context_model.hpp"
#include <cstdint>
#include <vector>

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

// In-context induction (fresh per-sequence mapping, 4-byte keys): a finite-order count
// model cannot do it (it conflates the per-sequence mappings and cannot match full keys).
static void test_context_fails_incontext_induction() {
  Diagnostic d = make_induction(7, 400, 51, 16, 4);
  CHECK(d.naive_bpb >= d.floor_bpb);
  CHECK(d.floor_bpb > 0.0);
  auto m = make_context_model(3);
  DiagResult r = score_diagnostic(*m, d);
  std::printf("    [in-context induction: context o3 fraction=%.4f]\n", r.fraction_captured);
  CHECK(r.fraction_captured < 0.30);  // cannot do in-context recall of novel mappings
}

// Generators are deterministic and the fillers produce exactly T bytes.
static void test_fillers_deterministic() {
  std::vector<uint8_t> a(300), b(300), c(300), e(300);
  fill_parity(42, a.data(), 300, 16);
  fill_parity(42, b.data(), 300, 16);
  CHECK(a == b);
  fill_induction(9, c.data(), 300, 16, 4);
  fill_induction(9, e.data(), 300, 16, 4);
  CHECK(c == e);
  CHECK(c != a);  // different tasks differ
}

int main() {
  RUN(test_generators_deterministic);
  RUN(test_context_fails_parity);
  RUN(test_context_fails_incontext_induction);
  RUN(test_fillers_deterministic);
  return test_summary();
}
