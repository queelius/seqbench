#pragma once
#include <torch/torch.h>
#include "seqbench/model.hpp"
#include <cstdint>

namespace gru {

struct Config {
  int d = 128;
  int layers = 1;
};

struct GruImpl : torch::nn::Module {
  Config cfg;
  torch::nn::Embedding emb{nullptr};
  torch::nn::GRU rnn{nullptr};
  torch::nn::Linear readout{nullptr};
  explicit GruImpl(const Config& c);
  torch::Tensor forward(torch::Tensor x_bt);  // [B,T] int64 -> [B,T,256]
};
TORCH_MODULE(Gru);

torch::Tensor bpb_loss(Gru model, torch::Tensor x_bt);  // next-byte CE in bits, positions 1..T-1

// Online bench adapter: carries the GRU hidden state, one byte per observe.
class GruEval : public seqbench::Model {
 public:
  GruEval(Gru model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  Gru model_;
  Config cfg_;
  torch::Tensor h_;    // [layers, 1, d]
  torch::Tensor out_;  // [d] latest GRU output (input to readout)
  bool seen_ = false;
};

}  // namespace gru
