#include <torch/torch.h>
#include "gru_model.hpp"
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
  int d = 128, layers = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--layers") layers = std::atoi(need("--layers"));
    else if (a == "--seq-len") rc.seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") rc.batch = std::atoi(need("--batch"));
    else if (a == "--steps") rc.steps = std::atoi(need("--steps"));
    else if (a == "--lr") rc.lr = std::atof(need("--lr"));
    else if (a == "--eval-every") rc.eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") rc.seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") rc.corpus = need("--corpus");
    else if (a == "--task") rc.task = need("--task");
    else if (a == "--block-len") rc.block_len = std::atoi(need("--block-len"));
    else if (a == "--device") rc.device = need("--device");
    else if (a == "--ckpt-dir") rc.ckpt_dir = need("--ckpt-dir");
    else if (a == "--ckpt-every") rc.ckpt_every = std::atoi(need("--ckpt-every"));
    else if (a == "--resume") rc.resume = true;
    else if (a == "--out") rc.out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  gru::Config cfg; cfg.d = d; cfg.layers = layers;
  std::map<std::string, JsonValue> config;
  config["d"] = JsonValue::n(d);
  config["layers"] = JsonValue::n(layers);

  gru::Gru model(cfg);
  return archcommon::run_experiment<gru::Gru>(
      rc, "gru-rnn", "v1-gru", config, model,
      [](gru::Gru& m, torch::Tensor x) { return gru::bpb_loss(m, x); },
      [cfg](gru::Gru& m) {
        return std::unique_ptr<seqbench::Model>(new gru::GruEval(m, cfg));
      });
}
