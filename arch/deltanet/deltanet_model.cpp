#include "deltanet_model.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace dn {

namespace F = torch::nn::functional;

FWMixImpl::FWMixImpl(int d, double lam) : dim(d), lambda(lam) {
  wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
  wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
  wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
  wbeta = register_module("wbeta", torch::nn::Linear(d, 1));
  wo = register_module("wo", torch::nn::Linear(torch::nn::LinearOptions(d, d).bias(false)));
}

torch::Tensor FWMixImpl::step(torch::Tensor& W, torch::Tensor x) {
  // x: [B,d], W: [B,d,d] mutated in place, returns o: [B,d].
  auto k = F::normalize(wk->forward(x), F::NormalizeFuncOptions().dim(1));   // [B,d]
  auto v = wv->forward(x);                                                   // [B,d]
  auto q = wq->forward(x);                                                   // [B,d]
  auto beta = torch::sigmoid(wbeta->forward(x));                            // [B,1]
  auto Wk = torch::bmm(W, k.unsqueeze(2)).squeeze(2);                        // [B,d]
  auto e = v - Wk;                                                           // [B,d]
  W = lambda * W + torch::bmm((beta * e).unsqueeze(2), k.unsqueeze(1));      // [B,d,d]
  auto o = torch::bmm(W, q.unsqueeze(2)).squeeze(2);                         // [B,d]
  return wo->forward(o);                                                     // [B,d]
}

torch::Tensor FWMixImpl::forward(torch::Tensor h) {
  auto B = h.size(0);
  auto T = h.size(1);
  int d = dim;
  auto W = torch::zeros({B, d, d}, h.options());
  std::vector<torch::Tensor> outs;
  outs.reserve(T);
  for (int64_t t = 0; t < T; ++t) outs.push_back(step(W, h.select(1, t)));
  return torch::stack(outs, 1);  // [B,T,d]
}

BlockImpl::BlockImpl(int d, double lam) {
  ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d})));
  mix = register_module("mix", FWMix(d, lam));
  ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d})));
  fc1 = register_module("fc1", torch::nn::Linear(d, 4 * d));
  fc2 = register_module("fc2", torch::nn::Linear(4 * d, d));
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
  x = x + mix->forward(ln1->forward(x));
  x = x + fc2->forward(torch::gelu(fc1->forward(ln2->forward(x))));
  return x;
}

DeltaNetImpl::DeltaNetImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  for (int i = 0; i < cfg.n_layers; ++i)
    blocks.push_back(register_module("block" + std::to_string(i), Block(cfg.d, cfg.lambda)));
  ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.d})));
  readout = register_module("readout", torch::nn::Linear(cfg.d, 256));
}

torch::Tensor DeltaNetImpl::forward(torch::Tensor x_bt) {
  auto x = emb->forward(x_bt);          // [B,T,d]
  for (auto& blk : blocks) x = blk->forward(x);
  x = ln_f->forward(x);                 // [B,T,d]
  return readout->forward(x);           // [B,T,256]
}

torch::Tensor bpb_loss(DeltaNet model, torch::Tensor x_bt) {
  auto T = x_bt.size(1);
  auto logits = model->forward(x_bt);                       // [B,T,256]
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});
  auto tgt = x_bt.slice(1, 1, T).reshape({-1});
  auto ce = F::cross_entropy(pred, tgt);
  return ce / std::log(2.0);
}

DeltaNetEval::DeltaNetEval(DeltaNet model, const Config& c) : model_(model), cfg_(c) {
  model_->eval();
  for (int i = 0; i < cfg_.n_layers; ++i) W_.push_back(torch::zeros({1, cfg_.d, cfg_.d}));
  out_ = torch::zeros({1, cfg_.d});
}

void DeltaNetEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto x = seen_ ? out_ : torch::zeros({1, cfg_.d});                              // [1,d]
  auto o = model_->readout->forward(model_->ln_f->forward(x)).contiguous().view({256});
  float* p = o.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void DeltaNetEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  int d = cfg_.d;
  auto x = model_->emb->forward(torch::tensor({static_cast<int64_t>(b)}, torch::kLong)).view({1, d});  // [1,d]
  for (int i = 0; i < cfg_.n_layers; ++i) {
    auto& blk = model_->blocks[i];
    auto h = blk->ln1->forward(x);              // [1,d]
    x = x + blk->mix->step(W_[i], h);           // step mutates W_[i] [1,d,d], returns [1,d]
    auto h2 = blk->ln2->forward(x);
    x = x + blk->fc2->forward(torch::gelu(blk->fc1->forward(h2)));
  }
  out_ = x;  // [1,d]
  seen_ = true;
}

}  // namespace dn
