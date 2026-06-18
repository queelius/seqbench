#include "test_util.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/metric.hpp"
#include <fstream>
#include <memory>
#include <string>

using namespace seqbench;

static void test_context_on_toy() {
  ByteSpan toy = toy_corpus();
  double b0 = run_adaptive(*make_context_model(0), toy).bpb();
  double b3 = run_adaptive(*make_context_model(3), toy).bpb();
  CHECK(b0 > 0.0);
  CHECK(b0 <= 8.0);
  CHECK(b3 < b0);  // memory helps on the repetitive toy text
}

// enwik8 is optional: skip if not fetched, so `make test` runs offline.
static void test_context_on_enwik8_if_present() {
  const std::string path = "data/enwik8";
  std::ifstream probe(path, std::ios::binary);
  if (!probe.good()) {
    std::printf("    (skipped: %s not present)\n", path.c_str());
    return;
  }
  probe.close();
  Corpus c(path);
  double b0 = run_adaptive(*make_context_model(0), c.bytes()).bpb();
  double b3 = run_adaptive(*make_context_model(3), c.bytes()).bpb();
  CHECK(b0 > 4.0);
  CHECK(b0 < 5.5);   // order-0 on English text
  CHECK(b3 < b0);    // higher order helps
  CHECK(b3 < 3.0);   // a few orders of context beat gzip-ish territory
}

int main() {
  RUN(test_context_on_toy);
  RUN(test_context_on_enwik8_if_present);
  return test_summary();
}
