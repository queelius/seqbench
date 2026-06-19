#include <torch/torch.h>
#include "fast_weights_model.hpp"
#include "common/runner.hpp"
#include "seqbench/experiment.hpp"
#include "seqbench/model.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>

using namespace seqbench;

int main(int argc, char** argv) {
  archcommon::RunConfig rc;
  int d = 128;
  double beta = 1.0, lambda = 0.99;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--beta") beta = std::atof(need("--beta"));
    else if (a == "--lambda") lambda = std::atof(need("--lambda"));
    else if (a == "--seq-len") rc.seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") rc.batch = std::atoi(need("--batch"));
    else if (a == "--steps") rc.steps = std::atoi(need("--steps"));
    else if (a == "--lr") rc.lr = std::atof(need("--lr"));
    else if (a == "--eval-every") rc.eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") rc.seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") rc.corpus = need("--corpus");
    else if (a == "--task") rc.task = need("--task");
    else if (a == "--out") rc.out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  fw::Config cfg; cfg.d = d; cfg.beta = beta; cfg.lambda = lambda;
  std::map<std::string, JsonValue> config;
  config["d"] = JsonValue::n(d);
  config["beta"] = JsonValue::n(beta);
  config["lambda"] = JsonValue::n(lambda);

  fw::FastWeights model(cfg);
  return archcommon::run_experiment<fw::FastWeights>(
      rc, "fast-weights-learned", "v1-deltanet", config, model,
      [](fw::FastWeights& m, torch::Tensor x) { return fw::bpb_loss(m, x); },
      [cfg](fw::FastWeights& m) {
        return std::unique_ptr<seqbench::Model>(new fw::FastWeightsEval(m, cfg));
      });
}
