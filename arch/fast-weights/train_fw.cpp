#include <torch/torch.h>
#include "fast_weights_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/experiment.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace seqbench;

static ByteSpan slice(ByteSpan s, double lo, double hi) {
  std::size_t a = static_cast<std::size_t>(s.len * lo);
  std::size_t b = static_cast<std::size_t>(s.len * hi);
  return ByteSpan{s.data + a, b - a};
}

static torch::Tensor sample_batch(ByteSpan span, int B, int T, std::mt19937_64& rng) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::uniform_int_distribution<std::size_t> start(0, span.len - T - 1);
  for (int b = 0; b < B; ++b) {
    std::size_t s = start(rng);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = span[s + t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

int main(int argc, char** argv) {
  int d = 128, seq_len = 256, batch = 32, steps = 50000, eval_every = 1000;
  double beta = 1.0, lambda = 0.99, lr = 1e-3;
  uint64_t seed = 1;
  std::string corpus = "data/enwik8", out = "runs/results.jsonl";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* n) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); std::exit(2); }
      return argv[++i];
    };
    if (a == "--d") d = std::atoi(need("--d"));
    else if (a == "--beta") beta = std::atof(need("--beta"));
    else if (a == "--lambda") lambda = std::atof(need("--lambda"));
    else if (a == "--seq-len") seq_len = std::atoi(need("--seq-len"));
    else if (a == "--batch") batch = std::atoi(need("--batch"));
    else if (a == "--steps") steps = std::atoi(need("--steps"));
    else if (a == "--lr") lr = std::atof(need("--lr"));
    else if (a == "--eval-every") eval_every = std::atoi(need("--eval-every"));
    else if (a == "--seed") seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") corpus = need("--corpus");
    else if (a == "--out") out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  torch::manual_seed(static_cast<int64_t>(seed));
  std::mt19937_64 rng(seed);

  std::unique_ptr<Corpus> cptr;
  ByteSpan full;
  if (corpus == "toy") full = toy_corpus();
  else { cptr = std::make_unique<Corpus>(corpus); full = cptr->bytes(); }
  ByteSpan train = slice(full, 0.0, 0.90);
  ByteSpan val = slice(full, 0.90, 0.95);
  if (train.len <= static_cast<std::size_t>(seq_len + 1) ||
      val.len <= static_cast<std::size_t>(seq_len + 1)) {
    std::fprintf(stderr, "corpus too small for seq_len=%d\n", seq_len);
    return 2;
  }
  std::fprintf(stderr, "corpus=%s train=%zu val=%zu\n", corpus.c_str(), train.len, val.len);

  fw::Config cfg; cfg.d = d; cfg.beta = beta; cfg.lambda = lambda;
  fw::FastWeights model(cfg);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(lr));

  std::vector<torch::Tensor> val_set;
  { std::mt19937_64 vr(seed ^ 0xdeadbeefULL);
    for (int i = 0; i < 8; ++i) val_set.push_back(sample_batch(val, batch, seq_len, vr)); }
  auto eval_val_bpb = [&]() -> double {
    torch::NoGradGuard ng; model->eval();
    double tot = 0.0;
    for (auto& vb : val_set) tot += fw::bpb_loss(model, vb).item<double>();
    model->train();
    return tot / val_set.size();
  };

  const std::string best_path = "/tmp/fw_best.pt";
  double best = std::numeric_limits<double>::infinity();
  for (int step = 1; step <= steps; ++step) {
    model->train();
    auto xb = sample_batch(train, batch, seq_len, rng);
    opt.zero_grad();
    auto loss = fw::bpb_loss(model, xb);
    loss.backward();
    opt.step();
    if (step % eval_every == 0 || step == steps) {
      double vbpb = eval_val_bpb();
      bool improved = vbpb < best;
      std::fprintf(stderr, "step %d train_bpb=%.4f val_bpb=%.4f%s\n",
                   step, loss.item<double>(), vbpb, improved ? " *" : "");
      if (improved) { best = vbpb; torch::save(model, best_path); }
    }
  }
  if (best < std::numeric_limits<double>::infinity()) torch::load(model, best_path);

  double val_bpb = eval_val_bpb();
  double train_bpb;
  { torch::NoGradGuard ng; model->eval();
    train_bpb = fw::bpb_loss(model, sample_batch(train, batch, seq_len, rng)).item<double>(); }

  Diagnostic ind = make_induction(7, 50000, 16);
  Diagnostic par = make_parity(7, 4000, 16);
  fw::FastWeightsEval ev_ind(model, cfg);
  double find = score_diagnostic(ev_ind, ind).fraction_captured;
  fw::FastWeightsEval ev_par(model, cfg);
  double fpar = score_diagnostic(ev_par, par).fraction_captured;

  std::printf("learned fast-weights d=%d beta=%.3g lambda=%.3g steps=%d\n", d, beta, lambda, steps);
  std::printf("  val_bpb=%.4f train_bpb=%.4f induction=%.4f parity=%.4f\n",
              val_bpb, train_bpb, find, fpar);

  RunRecord rec;
  rec.arch = "fast-weights-learned";
  rec.version = "v1-deltanet";
  rec.seed = static_cast<long>(seed);
  rec.config["d"] = JsonValue::n(d);
  rec.config["beta"] = JsonValue::n(beta);
  rec.config["lambda"] = JsonValue::n(lambda);
  rec.config["seq_len"] = JsonValue::n(seq_len);
  rec.config["batch"] = JsonValue::n(batch);
  rec.config["steps"] = JsonValue::n(steps);
  rec.config["lr"] = JsonValue::n(lr);
  rec.corpus_name = (corpus == "toy") ? "toy-val" : "enwik8-val";
  rec.corpus_bytes = val.len;
  rec.results["val_bpb"] = val_bpb;
  rec.results["train_bpb"] = train_bpb;
  rec.results["induction_fraction"] = find;
  rec.results["parity_fraction"] = fpar;
  append_record(rec, out);
  std::printf("  appended run record to %s\n", out.c_str());
  return 0;
}
