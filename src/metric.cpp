#include "seqbench/metric.hpp"
#include <cmath>

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

}  // namespace seqbench
