#include "gru_model.hpp"
#include <cmath>

namespace gru {

namespace F = torch::nn::functional;

GruImpl::GruImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  rnn = register_module(
      "rnn", torch::nn::GRU(torch::nn::GRUOptions(cfg.d, cfg.d).num_layers(cfg.layers).batch_first(true)));
  readout = register_module("readout", torch::nn::Linear(cfg.d, 256));
}

torch::Tensor GruImpl::forward(torch::Tensor x_bt) {
  auto x = emb->forward(x_bt);                  // [B,T,d]
  auto out = std::get<0>(rnn->forward(x));      // [B,T,d] (batch_first)
  return readout->forward(out);                 // [B,T,256]
}

torch::Tensor bpb_loss(Gru model, torch::Tensor x_bt) {
  auto T = x_bt.size(1);
  auto logits = model->forward(x_bt);                       // [B,T,256]
  auto pred = logits.slice(1, 0, T - 1).reshape({-1, 256});
  auto tgt = x_bt.slice(1, 1, T).reshape({-1});
  auto ce = F::cross_entropy(pred, tgt);
  return ce / std::log(2.0);
}

GruEval::GruEval(Gru model, const Config& c) : model_(model), cfg_(c) {
  model_->eval();
  h_ = torch::zeros({cfg_.layers, 1, cfg_.d});
  out_ = torch::zeros({cfg_.d});
}

void GruEval::predict(float logits[256]) {
  torch::NoGradGuard ng;
  auto in = seen_ ? out_ : torch::zeros({cfg_.d});
  auto o = model_->readout->forward(in).contiguous();  // [256]
  float* p = o.data_ptr<float>();
  for (int c = 0; c < 256; ++c) logits[c] = p[c];
}

void GruEval::observe(uint8_t b) {
  torch::NoGradGuard ng;
  auto idx = torch::tensor({static_cast<int64_t>(b)}, torch::kLong);  // [1]
  auto x = model_->emb->forward(idx).view({1, 1, cfg_.d});           // [B=1,T=1,d]
  auto res = model_->rnn->forward(x, h_);
  out_ = std::get<0>(res).view({cfg_.d});  // latest output [d]
  h_ = std::get<1>(res);                   // [layers,1,d]
  seen_ = true;
}

}  // namespace gru
