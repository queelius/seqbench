#pragma once
#include "seqbench/byte_span.hpp"
#include "seqbench/model.hpp"
#include <cstddef>
#include <cstdint>

namespace seqbench {

// Bits to code `actual` under softmax(logits), via a numerically stable
// log-sum-exp. This is the one canonical scoring path for every model.
double logit_bits(const float logits[256], uint8_t actual);

// True iff all 256 logits are finite (no inf/NaN).
bool logits_finite(const float logits[256]);

// Compensated (Kahan) summation for exact accumulation over ~1e8 terms.
struct Kahan {
  double sum = 0.0;
  double c = 0.0;
  void add(double x);
  double value() const { return sum; }
};

struct BpbResult {
  double total_bits = 0.0;
  std::size_t n_bytes = 0;
  double seconds = 0.0;
  double bpb() const { return n_bytes ? total_bits / double(n_bytes) : 0.0; }
  double bytes_per_sec() const {
    return seconds > 0.0 ? double(n_bytes) / seconds : 0.0;
  }
};

// Adaptive (one-pass) protocol: predict, score, observe(updates allowed).
BpbResult run_adaptive(Model& m, ByteSpan data);
// Train/test protocol: train(train_split), then score the val split.
// The model is responsible for freezing parameters in observe() after train().
BpbResult run_train_test(Model& m, ByteSpan train, ByteSpan val);

}  // namespace seqbench
