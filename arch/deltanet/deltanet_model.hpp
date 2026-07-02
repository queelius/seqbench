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
  bool use_gate = true;        // false: beta = 1 (no learned write gate)
  bool use_delta = true;       // false: pure Hebbian, e = v (drop the -Wk term)
  bool learn_lambda = false;   // true: lambda = sigmoid(per-layer param), init from `lambda`
  bool no_decay = false;       // true: lambda = 1.0 (no forgetting; overrides learn_lambda)
  bool normalize_keys = true;  // false: raw keys
  bool use_mlp = true;         // false: block is mix-only (MLP sublayer skipped)
};

// Fast-weights token mixing: delta-rule recurrence with a learned write gate.
struct FWMixImpl : torch::nn::Module {
  int dim;
  double lambda;
  bool use_gate, use_delta, learn_lambda, no_decay, normalize_keys;
  torch::nn::Linear wk{nullptr}, wv{nullptr}, wq{nullptr}, wbeta{nullptr}, wo{nullptr};
  torch::Tensor lambda_logit;  // registered only when learn_lambda
  explicit FWMixImpl(const Config& c);
  torch::Tensor forward(torch::Tensor h);  // [B,T,d] -> [B,T,d]
  torch::Tensor step(torch::Tensor& W, torch::Tensor x);  // W:[B,d,d] mutated, x:[B,d] -> o:[B,d]
};
TORCH_MODULE(FWMix);

// Pre-norm residual block: x += FWMix(LN(x)); x += MLP(LN(x)).
struct BlockImpl : torch::nn::Module {
  bool use_mlp;
  torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
  FWMix mix{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  explicit BlockImpl(const Config& c);
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
  std::vector<torch::Tensor> W_;  // n_layers x [1,d,d]
  torch::Tensor out_;             // [1,d] post-stack hidden
  bool seen_ = false;
};

}  // namespace dn
