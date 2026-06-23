#pragma once
#include <torch/torch.h>
#include "seqbench/corpus.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/experiment.hpp"
#include "seqbench/model.hpp"
#include "common/checkpoint.hpp"
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace archcommon {

struct RunConfig {
  int seq_len = 256, batch = 32, steps = 50000, eval_every = 1000;
  double lr = 1e-3;
  uint64_t seed = 1;
  std::string corpus = "data/enwik8", task = "enwik8", out = "runs/results.jsonl";
  int block_len = 16, dict_size = 16, key_len = 4;  // task generator params
  std::string device = "cpu";  // "cpu" or "cuda"
  std::string ckpt_dir = "";   // "" disables checkpointing
  int ckpt_every = 0;          // 0 means "use eval_every"
  bool resume = false;
};

inline seqbench::ByteSpan slice(seqbench::ByteSpan s, double lo, double hi) {
  std::size_t a = static_cast<std::size_t>(s.len * lo);
  std::size_t b = static_cast<std::size_t>(s.len * hi);
  return seqbench::ByteSpan{s.data + a, b - a};
}

inline torch::Tensor sample_batch(seqbench::ByteSpan span, int B, int T, std::mt19937_64& rng) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::uniform_int_distribution<std::size_t> start(0, span.len - T - 1);
  for (int b = 0; b < B; ++b) {
    std::size_t s = start(rng);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = span[s + t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

inline torch::Tensor sample_task_batch(const std::string& task, int B, int T,
                                       std::mt19937_64& rng, int block_len,
                                       int dict_size, int key_len) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::vector<uint8_t> seq(static_cast<std::size_t>(T));
  for (int b = 0; b < B; ++b) {
    uint64_t s = rng();
    if (task == "parity") seqbench::fill_parity(s, seq.data(), T, block_len);
    else seqbench::fill_induction(s, seq.data(), T, dict_size, key_len);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = seq[t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}

template <class ModelT>
int run_experiment(const RunConfig& rc, const std::string& arch, const std::string& version,
                   std::map<std::string, seqbench::JsonValue> config, ModelT model,
                   std::function<torch::Tensor(ModelT&, torch::Tensor)> loss_fn,
                   std::function<std::unique_ptr<seqbench::Model>(ModelT&)> make_adapter) {
  using namespace seqbench;
  torch::manual_seed(static_cast<int64_t>(rc.seed));
  std::mt19937_64 rng(rc.seed);
  torch::Device dev(rc.device == "cuda" ? torch::kCUDA : torch::kCPU);
  model->to(dev);

  const bool is_task = (rc.task == "parity" || rc.task == "induction");

  std::unique_ptr<Corpus> cptr;
  ByteSpan train, val;
  if (!is_task) {
    ByteSpan full;
    if (rc.corpus == "toy") full = toy_corpus();
    else { cptr = std::make_unique<Corpus>(rc.corpus); full = cptr->bytes(); }
    train = slice(full, 0.0, 0.90);
    val = slice(full, 0.90, 0.95);
    if (train.len <= static_cast<std::size_t>(rc.seq_len + 1) ||
        val.len <= static_cast<std::size_t>(rc.seq_len + 1)) {
      std::fprintf(stderr, "corpus too small for seq_len=%d\n", rc.seq_len);
      return 2;
    }
    std::fprintf(stderr, "corpus=%s train=%zu val=%zu\n", rc.corpus.c_str(), train.len, val.len);
  } else {
    std::fprintf(stderr, "task=%s seq_len=%d\n", rc.task.c_str(), rc.seq_len);
  }

  auto train_batch = [&](std::mt19937_64& r) {
    return is_task ? sample_task_batch(rc.task, rc.batch, rc.seq_len, r, rc.block_len,
                                       rc.dict_size, rc.key_len)
                   : sample_batch(train, rc.batch, rc.seq_len, r);
  };

  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(rc.lr));

  std::vector<torch::Tensor> val_set;
  { std::mt19937_64 vr(rc.seed ^ 0xdeadbeefULL);
    for (int i = 0; i < 8; ++i) val_set.push_back(train_batch(vr)); }
  auto eval_val_bpb = [&]() -> double {
    torch::NoGradGuard ng; model->eval();
    double tot = 0.0;
    for (auto& vb : val_set) tot += loss_fn(model, vb.to(dev)).item().toDouble();
    model->train();
    return tot / val_set.size();
  };

  // Checkpoint plumbing. Fingerprint is arch + sorted structural config, so a
  // resume into a structurally different model is rejected.
  std::string fingerprint = arch;
  for (const auto& kv : config) {
    fingerprint += "|" + kv.first + "=";
    fingerprint += (kv.second.type == JsonValue::Number)
                       ? std::to_string(kv.second.num) : kv.second.str;
  }
  const bool ckpt_on = !rc.ckpt_dir.empty();
  const int ckpt_every = rc.ckpt_every > 0 ? rc.ckpt_every : rc.eval_every;
  const std::string fallback_best = "/tmp/seqbench_best.pt";

  double best = std::numeric_limits<double>::infinity();
  int start_step = 1;
  if (rc.resume && ckpt_on && archcommon::checkpoint_exists(rc.ckpt_dir, "latest")) {
    archcommon::CkptMeta meta;
    archcommon::load_checkpoint(rc.ckpt_dir, "latest", model, opt, meta, rng, dev);
    if (meta.fingerprint != fingerprint) {
      std::fprintf(stderr, "resume fingerprint mismatch:\n  ckpt:    %s\n  current: %s\n",
                   meta.fingerprint.c_str(), fingerprint.c_str());
      return 2;
    }
    start_step = meta.step + 1;
    best = meta.best;
    std::fprintf(stderr, "resumed from step %d (best_val_bpb=%.4f), target steps=%d\n",
                 meta.step, best, rc.steps);
  }

  auto save_latest = [&](int step) {
    if (!ckpt_on) return;
    archcommon::CkptMeta m; m.step = step; m.seed = rc.seed; m.best = best;
    m.arch = arch; m.fingerprint = fingerprint;
    archcommon::save_checkpoint(rc.ckpt_dir, "latest", model, opt, m, rng);
  };

  for (int step = start_step; step <= rc.steps; ++step) {
    model->train();
    auto xb = train_batch(rng).to(dev);
    opt.zero_grad();
    auto loss = loss_fn(model, xb);
    loss.backward();
    opt.step();
    if (step % rc.eval_every == 0 || step == rc.steps) {
      double vbpb = eval_val_bpb();
      bool improved = vbpb < best;
      std::fprintf(stderr, "step %d train_bpb=%.4f val_bpb=%.4f%s\n",
                   step, loss.item().toDouble(), vbpb, improved ? " *" : "");
      if (improved) {
        best = vbpb;
        if (ckpt_on) {
          archcommon::CkptMeta m; m.step = step; m.seed = rc.seed; m.best = best;
          m.arch = arch; m.fingerprint = fingerprint;
          archcommon::save_checkpoint(rc.ckpt_dir, "best", model, opt, m, rng);
        } else {
          torch::save(model, fallback_best);
        }
      }
    }
    if (ckpt_on && (step % ckpt_every == 0 || step == rc.steps)) save_latest(step);
  }
  // Restore best params for the final reported metrics.
  if (best < std::numeric_limits<double>::infinity()) {
    if (ckpt_on) {
      archcommon::CkptMeta m; std::mt19937_64 tmp;
      // `best` may be finite because it was carried over from latest.meta on a
      // resume, while the best.* files were never copied to this machine (e.g.
      // only latest.* was rsync'd). load_checkpoint then returns false and
      // loads nothing, so warn rather than silently report last-step weights.
      if (!archcommon::load_checkpoint(rc.ckpt_dir, "best", model, opt, m, tmp, dev))
        std::fprintf(stderr, "warning: best checkpoint absent in %s; reporting current weights\n",
                     rc.ckpt_dir.c_str());
    } else {
      torch::load(model, fallback_best);
    }
  }

  double train_bpb;
  { torch::NoGradGuard ng; model->eval();
    train_bpb = loss_fn(model, train_batch(rng).to(dev)).item().toDouble(); }

  RunRecord rec;
  rec.arch = arch;
  rec.version = version;
  rec.seed = static_cast<long>(rc.seed);
  for (const auto& kv : config) rec.config[kv.first] = kv.second;
  rec.config["seq_len"] = JsonValue::n(rc.seq_len);
  rec.config["batch"] = JsonValue::n(rc.batch);
  rec.config["steps"] = JsonValue::n(rc.steps);
  rec.config["lr"] = JsonValue::n(rc.lr);
  rec.config["task"] = JsonValue::s(rc.task);
  rec.results["train_bpb"] = train_bpb;

  if (is_task) {
    if (rc.task == "parity") {
      rec.config["block_len"] = JsonValue::n(rc.block_len);
    } else {
      rec.config["dict_size"] = JsonValue::n(rc.dict_size);
      rec.config["key_len"] = JsonValue::n(rc.key_len);
    }
    Diagnostic test = (rc.task == "parity")
        ? make_parity(rc.seed ^ 0x5eedULL, 4000, rc.block_len)
        : make_induction(rc.seed ^ 0x5eedULL, 400, 51, rc.dict_size, rc.key_len);
    auto adapter = make_adapter(model);
    DiagResult r = score_diagnostic(*adapter, test);
    rec.corpus_name = rc.task + "-test";
    rec.corpus_bytes = test.stream.size();
    rec.results["test_bpb"] = r.observed_bpb;
    rec.results["fraction_captured"] = r.fraction_captured;
    rec.results["floor_bpb"] = test.floor_bpb;
    rec.results["naive_bpb"] = test.naive_bpb;
    std::printf("%s task=%s steps=%d\n", arch.c_str(), rc.task.c_str(), rc.steps);
    std::printf("  test_bpb=%.4f fraction_captured=%.4f (floor=%.4f naive=%.4f) train_bpb=%.4f\n",
                r.observed_bpb, r.fraction_captured, test.floor_bpb, test.naive_bpb, train_bpb);
  } else {
    double val_bpb = eval_val_bpb();
    rec.corpus_name = (rc.corpus == "toy") ? "toy-val" : "enwik8-val";
    rec.corpus_bytes = val.len;
    rec.results["val_bpb"] = val_bpb;
    std::printf("%s task=%s steps=%d\n", arch.c_str(), rc.task.c_str(), rc.steps);
    std::printf("  val_bpb=%.4f train_bpb=%.4f\n", val_bpb, train_bpb);
  }
  append_record(rec, rc.out);
  std::printf("  appended run record to %s\n", rc.out.c_str());
  return 0;
}

}  // namespace archcommon
