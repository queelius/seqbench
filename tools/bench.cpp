#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/metric.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace seqbench;

static void usage() {
  std::fprintf(stderr,
               "usage: bench <model> <corpus>\n"
               "  model:  ctx:N        (order-N context model, 0..8)\n"
               "  corpus: toy | <path> (path is a raw byte file)\n");
}

int main(int argc, char** argv) {
  if (argc != 3) { usage(); return 2; }
  std::string mspec = argv[1];
  std::string cspec = argv[2];

  std::unique_ptr<Model> model;
  if (mspec.rfind("ctx:", 0) == 0) {
    const std::string ord = mspec.substr(4);
    char* end = nullptr;
    long n = std::strtol(ord.c_str(), &end, 10);
    if (ord.empty() || *end != '\0') {
      std::fprintf(stderr, "invalid order in model spec: %s\n", mspec.c_str());
      return 2;
    }
    model = make_context_model(static_cast<int>(n));
  } else {
    std::fprintf(stderr, "unknown model: %s\n", mspec.c_str());
    usage();
    return 2;
  }

  std::unique_ptr<Corpus> corpus;
  ByteSpan span;
  if (cspec == "toy") {
    span = toy_corpus();
  } else {
    corpus = std::make_unique<Corpus>(cspec);
    span = corpus->bytes();
  }

  BpbResult r = run_adaptive(*model, span);
  std::printf("model=%s corpus=%s bytes=%zu bpb=%.4f throughput=%.2f MB/s\n",
              mspec.c_str(), cspec.c_str(), r.n_bytes, r.bpb(),
              r.bytes_per_sec() / 1e6);
  return 0;
}
