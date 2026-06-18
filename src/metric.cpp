#include "seqbench/metric.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace seqbench {

double logit_bits(const float logits[256], uint8_t actual) {
  float m = logits[0];
  for (int i = 1; i < 256; ++i) if (logits[i] > m) m = logits[i];
  double sum = 0.0;
  for (int i = 0; i < 256; ++i) sum += std::exp(double(logits[i]) - double(m));
  double logZ = double(m) + std::log(sum);      // natural-log normalizer
  double log_p = double(logits[actual]) - logZ;  // ln p(actual)
  return -log_p / std::log(2.0);                 // convert to bits
}

bool logits_finite(const float logits[256]) {
  for (int i = 0; i < 256; ++i) if (!std::isfinite(logits[i])) return false;
  return true;
}

void Kahan::add(double x) {
  double y = x - c;
  double t = sum + y;
  c = (t - sum) - y;
  sum = t;
}

namespace {
BpbResult score_stream(Model& m, ByteSpan data) {
  Kahan bits;
  float logits[256];
  auto t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < data.len; ++i) {
    m.predict(logits);
    if (!logits_finite(logits))
      throw std::runtime_error("non-finite logits at position " +
                               std::to_string(i));
    bits.add(logit_bits(logits, data[i]));
    m.observe(data[i]);
  }
  auto t1 = std::chrono::steady_clock::now();
  BpbResult r;
  r.total_bits = bits.value();
  r.n_bytes = data.len;
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  return r;
}
}  // namespace

BpbResult run_adaptive(Model& m, ByteSpan data) { return score_stream(m, data); }

// NOTE: the "adaptive and train/test yield the same bpb" equivalence is only
// meaningful for a model whose observe() freezes parameters after train().
// The v1 ContextModel is adaptive-only (train() is a no-op, observe() always
// updates), so protocol-equivalence is verified once the first two-phase
// (neural) model that honors the freeze contract is added.
BpbResult run_train_test(Model& m, ByteSpan train, ByteSpan val) {
  m.train(train);
  return score_stream(m, val);
}

}  // namespace seqbench
