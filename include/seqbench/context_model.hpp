#pragma once
#include "seqbench/model.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace seqbench {

// Adaptive order-N context model with add-alpha smoothing.
// Prediction for the current context is log(count_b + alpha) per byte b
// (unnormalized log-probs are valid logits). order in [0, 8].
class ContextModel : public Model {
 public:
  explicit ContextModel(int order, double alpha = 1.0);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  int order_;
  double alpha_;
  uint64_t ctx_ = 0;
  uint64_t mask_ = 0;
  std::unordered_map<uint64_t, std::array<uint32_t, 256>> tables_;
};

std::unique_ptr<Model> make_context_model(int order);

}  // namespace seqbench
