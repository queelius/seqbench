#include "seqbench/context_model.hpp"
#include <cmath>

namespace seqbench {

ContextModel::ContextModel(int order, double alpha)
    : order_(order), alpha_(alpha) {
  if (order_ < 0) order_ = 0;
  if (order_ > 8) order_ = 8;
  mask_ = (order_ >= 8) ? ~0ull : ((1ull << (8 * order_)) - 1);
}

void ContextModel::predict(float logits[256]) {
  auto it = tables_.find(ctx_);
  if (it == tables_.end()) {
    float v = static_cast<float>(std::log(alpha_));
    for (int b = 0; b < 256; ++b) logits[b] = v;
    return;
  }
  const std::array<uint32_t, 256>& counts = it->second;
  for (int b = 0; b < 256; ++b)
    logits[b] = static_cast<float>(std::log(double(counts[b]) + alpha_));
}

void ContextModel::observe(uint8_t b) {
  tables_[ctx_][b] += 1;
  ctx_ = ((ctx_ << 8) | uint64_t(b)) & mask_;
}

std::unique_ptr<Model> make_context_model(int order) {
  return std::make_unique<ContextModel>(order, 1.0);
}

}  // namespace seqbench
