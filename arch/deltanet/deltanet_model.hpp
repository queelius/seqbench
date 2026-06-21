#pragma once
#include <torch/torch.h>
#include "seqbench/model.hpp"
#include <cstdint>
#include <vector>

namespace dn {

struct Config {
  int d = 128;
  int n_layers = 4;
  double lambda = 0.99;
};

// Fast-weights token mixing: delta-rule recurrence with a learned write gate.
struct FWMixImpl : torch::nn::Module {
  int dim;
  double lambda;
  torch::nn::Linear wk{nullptr}, wv{nullptr}, wq{nullptr}, wbeta{nullptr}, wo{nullptr};
  FWMixImpl(int d, double lam);
  torch::Tensor forward(torch::Tensor h);  // [B,T,d] -> [B,T,d]
};
TORCH_MODULE(FWMix);

// Pre-norm residual block: x += FWMix(LN(x)); x += MLP(LN(x)).
struct BlockImpl : torch::nn::Module {
  torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
  FWMix mix{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  BlockImpl(int d, double lam);
  torch::Tensor forward(torch::Tensor x);  // [B,T,d] -> [B,T,d]
};
TORCH_MODULE(Block);

struct DeltaNetImpl : torch::nn::Module {
  Config cfg;
  torch::nn::Embedding emb{nullptr};
  std::vector<Block> blocks;
  torch::nn::LayerNorm ln_f{nullptr};
  torch::nn::Linear readout{nullptr};
  explicit DeltaNetImpl(const Config& c);
  torch::Tensor forward(torch::Tensor x_bt);  // [B,T] int64 -> [B,T,256]
};
TORCH_MODULE(DeltaNet);

torch::Tensor bpb_loss(DeltaNet model, torch::Tensor x_bt);  // next-byte CE in bits, positions 1..T-1

// Online bench adapter: carries the N fast-weight matrices; everything else is position-wise.
class DeltaNetEval : public seqbench::Model {
 public:
  DeltaNetEval(DeltaNet model, const Config& c);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  DeltaNet model_;
  Config cfg_;
  std::vector<torch::Tensor> W_;  // n_layers x [d, d]
  torch::Tensor out_;             // [d] post-stack hidden
  bool seen_ = false;
};

}  // namespace dn
