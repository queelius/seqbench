#include "fast_weights_model.hpp"
#include <cmath>
#include <vector>

namespace fw {

namespace F = torch::nn::functional;

FastWeightsImpl::FastWeightsImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(cfg.d, cfg.d).bias(false)));
  wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(cfg.d, cfg.d).bias(false)));
  wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(cfg.d, cfg.d).bias(false)));
  readout = register_module("readout", torch::nn::Linear(cfg.d, 256));
}

torch::Tensor FastWeightsImpl::forward(torch::Tensor x_bt) {
  auto B = x_bt.size(0);
  auto T = x_bt.size(1);
  int d = cfg.d;
  auto x = emb->forward(x_bt);                                  // [B,T,d]
  auto k = F::normalize(wk->forward(x), F::NormalizeFuncOptions().dim(2));  // [B,T,d]
  auto v = wv->forward(x);                                      // [B,T,d]
  auto q = wq->forward(x);                                      // [B,T,d]
  auto W = torch::zeros({B, d, d}, x.options());               // [B,d,d]
  std::vector<torch::Tensor> outs;
  outs.reserve(T);
  for (int64_t t = 0; t < T; ++t) {
    auto kt = k.select(1, t);                                  // [B,d]
    auto vt = v.select(1, t);                                  // [B,d]
    auto qt = q.select(1, t);                                  // [B,d]
    auto Wk = torch::bmm(W, kt.unsqueeze(2)).squeeze(2);       // [B,d]
    auto e = vt - Wk;                                          // [B,d]
    W = cfg.lambda * W + cfg.beta * torch::bmm(e.unsqueeze(2), kt.unsqueeze(1));  // [B,d,d]
    auto ot = torch::bmm(W, qt.unsqueeze(2)).squeeze(2);       // [B,d]
    outs.push_back(ot);
  }
  auto o = torch::stack(outs, 1);                              // [B,T,d]
  return readout->forward(o);                                  // [B,T,256]
}

torch::Tensor bpb_loss(FastWeights model, torch::Tensor x_bt) {
  auto T = x_bt.size(1);
  auto logits = model->forward(x_bt);                          // [B,T,256]
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});    // [B*(T-1),256]
  auto tgt = x_bt.slice(1, 1, T).reshape({-1});                // [B*(T-1)]
  auto ce = F::cross_entropy(pred, tgt);                       // natural-log mean CE
  return ce / std::log(2.0);                                   // bits per byte
}

}  // namespace fw
