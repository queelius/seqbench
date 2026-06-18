#pragma once
#include <torch/torch.h>
#include "seqbench/model.hpp"
#include <cstdint>

namespace fw {

struct Config {
  int d = 128;
  double beta = 1.0;
  double lambda = 0.99;
};

struct FastWeightsImpl : torch::nn::Module {
  Config cfg;
  torch::nn::Embedding emb{nullptr};
  torch::nn::Linear wk{nullptr}, wv{nullptr}, wq{nullptr}, readout{nullptr};
  explicit FastWeightsImpl(const Config& c);
  // x_bt: [B, T] int64 byte ids -> logits [B, T, 256]
  torch::Tensor forward(torch::Tensor x_bt);
};
TORCH_MODULE(FastWeights);

// Mean next-byte cross-entropy in bits-per-byte for a [B,T] chunk (predicts positions 1..T-1).
torch::Tensor bpb_loss(FastWeights model, torch::Tensor x_bt);

// Bench Model adapter: runs the trained model online (no_grad), one byte at a time,
// replicating the training forward exactly so streaming and batched bits agree.
class FastWeightsEval : public seqbench::Model {
 public:
  FastWeightsEval(FastWeights model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  FastWeights model_;
  Config cfg_;
  torch::Tensor W_;  // [d, d]
  torch::Tensor o_;  // [d]
  bool seen_ = false;
};

}  // namespace fw
