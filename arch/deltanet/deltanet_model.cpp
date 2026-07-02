#include "deltanet_model.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace dn {

namespace F = torch::nn::functional;

FWMixImpl::FWMixImpl(const Config& c)
    : dim(c.d), lambda(c.lambda),
      use_gate(c.use_gate), use_delta(c.use_delta), learn_lambda(c.learn_lambda),
      no_decay(c.no_decay), normalize_keys(c.normalize_keys) {
  wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  wbeta = register_module("wbeta", torch::nn::Linear(dim, 1));
  wo = register_module("wo", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
  if (learn_lambda) {
    double logit = std::log(lambda / (1.0 - lambda));  // sigmoid(logit) == lambda
    // float32 to match the model params (a float64 tensor would error when multiplied with W).
    lambda_logit = register_parameter("lambda_logit", torch::full({1}, logit, torch::kFloat32));
  }
}

torch::Tensor FWMixImpl::step(torch::Tensor& W, torch::Tensor x) {
  auto k = wk->forward(x);
  if (normalize_keys) k = F::normalize(k, F::NormalizeFuncOptions().dim(1));
  auto v = wv->forward(x);
  auto q = wq->forward(x);
  torch::Tensor beta = use_gate ? torch::sigmoid(wbeta->forward(x))
                                : torch::ones({x.size(0), 1}, x.options());
  auto Wk = torch::bmm(W, k.unsqueeze(2)).squeeze(2);
  auto e = use_delta ? (v - Wk) : v;
  auto update = torch::bmm((beta * e).unsqueeze(2), k.unsqueeze(1));  // [B,d,d]
  if (no_decay) {
    W = W + update;                                  // lambda = 1
  } else if (learn_lambda) {
    W = torch::sigmoid(lambda_logit) * W + update;   // lambda_logit broadcasts [1] over [B,d,d]
  } else {
    W = lambda * W + update;                         // fixed lambda
  }
  auto o = torch::bmm(W, q.unsqueeze(2)).squeeze(2);
  return wo->forward(o);
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

BlockImpl::BlockImpl(const Config& c) : use_mlp(c.use_mlp) {
  ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.d})));
  mix = register_module("mix", FWMix(c));
  if (use_mlp) {
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.d})));
    fc1 = register_module("fc1", torch::nn::Linear(c.d, 4 * c.d));
    fc2 = register_module("fc2", torch::nn::Linear(4 * c.d, c.d));
  }
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
  x = x + mix->forward(ln1->forward(x));
  if (use_mlp) x = x + fc2->forward(torch::gelu(fc1->forward(ln2->forward(x))));
  return x;
}

DeltaNetImpl::DeltaNetImpl(const Config& c) : cfg(c) {
  emb = register_module("emb", torch::nn::Embedding(256, cfg.d));
  for (int i = 0; i < cfg.n_layers; ++i)
    blocks.push_back(register_module("block" + std::to_string(i), Block(cfg)));
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
    if (cfg_.use_mlp) {
      auto h2 = blk->ln2->forward(x);
      x = x + blk->fc2->forward(torch::gelu(blk->fc1->forward(h2)));
    }
  }
  out_ = x;  // [1,d]
  seen_ = true;
}

}  // namespace dn
