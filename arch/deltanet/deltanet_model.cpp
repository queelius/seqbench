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

torch::Tensor FWMixImpl::forward(torch::Tensor h) {
  auto B = h.size(0);
  auto T = h.size(1);
  int d = dim;
  auto k = F::normalize(wk->forward(h), F::NormalizeFuncOptions().dim(2));  // [B,T,d]
  auto v = wv->forward(h);                                                  // [B,T,d]
  auto q = wq->forward(h);                                                  // [B,T,d]
  auto beta = torch::sigmoid(wbeta->forward(h));                            // [B,T,1]
  auto W = torch::zeros({B, d, d}, h.options());                           // [B,d,d]
  std::vector<torch::Tensor> outs;
  outs.reserve(T);
  for (int64_t t = 0; t < T; ++t) {
    auto kt = k.select(1, t);                                  // [B,d]
    auto vt = v.select(1, t);                                  // [B,d]
    auto qt = q.select(1, t);                                  // [B,d]
    auto bt = beta.select(1, t);                               // [B,1]
    auto Wk = torch::bmm(W, kt.unsqueeze(2)).squeeze(2);       // [B,d]
    auto e = vt - Wk;                                          // [B,d]
    W = lambda * W + torch::bmm((bt * e).unsqueeze(2), kt.unsqueeze(1));  // [B,d,d]
    auto ot = torch::bmm(W, qt.unsqueeze(2)).squeeze(2);       // [B,d]
    outs.push_back(ot);
  }
  auto o = torch::stack(outs, 1);                              // [B,T,d]
  return wo->forward(o);                                       // [B,T,d]
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
  for (int i = 0; i < cfg_.n_layers; ++i) W_.push_back(torch::zeros({cfg_.d, cfg_.d}));
  out_ = torch::zeros({cfg_.d});
}

void DeltaNetEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto x = seen_ ? out_ : torch::zeros({cfg_.d});
  auto o = model_->readout->forward(model_->ln_f->forward(x)).contiguous();  // [256]
  float* p = o.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void DeltaNetEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  int d = cfg_.d;
  auto x = model_->emb->forward(torch::tensor({static_cast<int64_t>(b)}, torch::kLong)).view({d});  // [d]
  for (int i = 0; i < cfg_.n_layers; ++i) {
    auto& blk = model_->blocks[i];
    // FWMix on ln1(x), updating W_[i].
    auto h = blk->ln1->forward(x);                                                      // [d]
    auto k = F::normalize(blk->mix->wk->forward(h), F::NormalizeFuncOptions().dim(0));  // [d]
    auto vv = blk->mix->wv->forward(h);                                                 // [d]
    auto qq = blk->mix->wq->forward(h);                                                 // [d]
    auto bt = torch::sigmoid(blk->mix->wbeta->forward(h));                              // [1]
    auto Wk = torch::mv(W_[i], k);                                                      // [d]
    auto e = vv - Wk;                                                                   // [d]
    W_[i] = cfg_.lambda * W_[i] + torch::outer(bt * e, k);                              // [d,d]
    auto o = torch::mv(W_[i], qq);                                                      // [d]
    x = x + blk->mix->wo->forward(o);                                                   // residual
    // MLP on ln2(x).
    auto h2 = blk->ln2->forward(x);
    x = x + blk->fc2->forward(torch::gelu(blk->fc1->forward(h2)));                      // residual
  }
  out_ = x;
  seen_ = true;
}

}  // namespace dn
