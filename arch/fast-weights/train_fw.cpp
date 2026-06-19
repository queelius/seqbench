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

// One batch of synthetic task sequences (each a fresh sample of length T).
static torch::Tensor sample_task_batch(const std::string& task, int B, int T,
                                       std::mt19937_64& rng, int block_len,
                                       int dict_size, int key_len) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::vector<uint8_t> seq(static_cast<std::size_t>(T));
  for (int b = 0; b < B; ++b) {
    uint64_t s = rng();
    if (task == "parity") fill_parity(s, seq.data(), T, block_len);
    else fill_induction(s, seq.data(), T, dict_size, key_len);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = seq[t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

int main(int argc, char** argv) {
  int d = 128, seq_len = 256, batch = 32, steps = 50000, eval_every = 1000;
  double beta = 1.0, lambda = 0.99, lr = 1e-3;
  uint64_t seed = 1;
  std::string corpus = "data/enwik8", out = "runs/results.jsonl", task = "enwik8";
  int block_len = 16, dict_size = 16, key_len = 4;

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
    else if (a == "--task") task = need("--task");
    else if (a == "--out") out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }

  torch::manual_seed(static_cast<int64_t>(seed));
  std::mt19937_64 rng(seed);

  const bool is_task = (task == "parity" || task == "induction");

  std::unique_ptr<Corpus> cptr;
  ByteSpan train, val;
  if (!is_task) {
    ByteSpan full;
    if (corpus == "toy") full = toy_corpus();
    else { cptr = std::make_unique<Corpus>(corpus); full = cptr->bytes(); }
    train = slice(full, 0.0, 0.90);
    val = slice(full, 0.90, 0.95);
    if (train.len <= static_cast<std::size_t>(seq_len + 1) ||
        val.len <= static_cast<std::size_t>(seq_len + 1)) {
      std::fprintf(stderr, "corpus too small for seq_len=%d\n", seq_len);
      return 2;
    }
    std::fprintf(stderr, "corpus=%s train=%zu val=%zu\n", corpus.c_str(), train.len, val.len);
  } else {
    std::fprintf(stderr, "task=%s seq_len=%d\n", task.c_str(), seq_len);
  }

  fw::Config cfg; cfg.d = d; cfg.beta = beta; cfg.lambda = lambda;
  fw::FastWeights model(cfg);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(lr));

  auto train_batch = [&](std::mt19937_64& r) {
    return is_task ? sample_task_batch(task, batch, seq_len, r, block_len, dict_size, key_len)
                   : sample_batch(train, batch, seq_len, r);
  };

  std::vector<torch::Tensor> val_set;
  { std::mt19937_64 vr(seed ^ 0xdeadbeefULL);
    for (int i = 0; i < 8; ++i) val_set.push_back(train_batch(vr)); }
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
    auto xb = train_batch(rng);
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

  double train_bpb;
  { torch::NoGradGuard ng; model->eval();
    train_bpb = fw::bpb_loss(model, train_batch(rng)).item<double>(); }

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
  rec.config["task"] = JsonValue::s(task);
  rec.results["train_bpb"] = train_bpb;

  if (is_task) {
    // Held-out test stream (different seed), scored via the online adapter.
    Diagnostic test = (task == "parity")
        ? make_parity(seed ^ 0x5eedULL, 4000, block_len)
        : make_induction(seed ^ 0x5eedULL, 400, 51, dict_size, key_len);
    fw::FastWeightsEval ev(model, cfg);
    DiagResult r = score_diagnostic(ev, test);
    rec.corpus_name = task + "-test";
    rec.corpus_bytes = test.stream.size();
    rec.results["test_bpb"] = r.observed_bpb;
    rec.results["fraction_captured"] = r.fraction_captured;
    rec.results["floor_bpb"] = test.floor_bpb;
    rec.results["naive_bpb"] = test.naive_bpb;
    std::printf("learned fast-weights task=%s d=%d steps=%d\n", task.c_str(), d, steps);
    std::printf("  test_bpb=%.4f fraction_captured=%.4f (floor=%.4f naive=%.4f) train_bpb=%.4f\n",
                r.observed_bpb, r.fraction_captured, test.floor_bpb, test.naive_bpb, train_bpb);
  } else {
    double val_bpb = eval_val_bpb();
    rec.corpus_name = (corpus == "toy") ? "toy-val" : "enwik8-val";
    rec.corpus_bytes = val.len;
    rec.results["val_bpb"] = val_bpb;
    std::printf("learned fast-weights task=%s d=%d steps=%d\n", task.c_str(), d, steps);
    std::printf("  val_bpb=%.4f train_bpb=%.4f\n", val_bpb, train_bpb);
  }
  append_record(rec, out);
  std::printf("  appended run record to %s\n", out.c_str());
  return 0;
}
