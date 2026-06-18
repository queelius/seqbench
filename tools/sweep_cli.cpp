#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/sweep.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace seqbench;

static void usage() {
  std::fprintf(stderr,
               "usage: sweep <corpus> <out.csv> <order0> [order1 ...]\n"
               "  corpus: toy | <path>\n"
               "  sweeps context-model orders; writes knob,bpb,seconds CSV\n");
}

int main(int argc, char** argv) {
  if (argc < 4) { usage(); return 2; }
  std::string cspec = argv[1];
  std::string out = argv[2];
  std::vector<int> knobs;
  for (int i = 3; i < argc; ++i) knobs.push_back(std::atoi(argv[i]));

  std::unique_ptr<Corpus> corpus;
  ByteSpan span;
  if (cspec == "toy") {
    span = toy_corpus();
  } else {
    corpus = std::make_unique<Corpus>(cspec);
    span = corpus->bytes();
  }

  std::vector<SweepPoint> pts = run_sweep(make_context_model, knobs, span);
  write_csv(pts, out);
  for (const SweepPoint& p : pts)
    std::printf("order=%d bpb=%.4f seconds=%.3f\n", p.knob, p.bpb, p.seconds);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
