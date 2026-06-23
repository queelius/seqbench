#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <sys/stat.h>

namespace archcommon {

struct CkptMeta {
  int step = 0;
  uint64_t seed = 0;
  double best = 0.0;
  std::string arch;
  std::string fingerprint;
};

inline bool path_exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

// Create every component of a (possibly nested) directory path. Ignores
// already-exists. Best-effort: a real I/O failure surfaces later on write.
inline void make_dirs(const std::string& dir) {
  std::string acc;
  for (std::size_t i = 0; i < dir.size(); ++i) {
    acc.push_back(dir[i]);
    if (dir[i] == '/' || i + 1 == dir.size()) {
      if (!acc.empty() && acc != "/") ::mkdir(acc.c_str(), 0755);
    }
  }
}

inline bool checkpoint_exists(const std::string& dir, const std::string& prefix) {
  return path_exists(dir + "/" + prefix + ".meta");
}

// Write `src` to `final + ".tmp"` via `fn`, then rename over `final` so a crash
// mid-write cannot corrupt an existing good file.
template <class Fn>
inline void atomic_write(const std::string& final, Fn fn) {
  const std::string tmp = final + ".tmp";
  fn(tmp);
  std::rename(tmp.c_str(), final.c_str());
}

template <class ModelT>
void save_checkpoint(const std::string& dir, const std::string& prefix,
                     ModelT& model, torch::optim::Adam& opt,
                     const CkptMeta& meta, const std::mt19937_64& rng) {
  make_dirs(dir);
  const std::string base = dir + "/" + prefix;
  atomic_write(base + ".model.pt", [&](const std::string& p) { torch::save(model, p); });
  atomic_write(base + ".opt.pt", [&](const std::string& p) { torch::save(opt, p); });
  atomic_write(base + ".rng.pt", [&](const std::string& p) {
    // C++ libtorch has no torch::get_rng_state(); use the Generator API instead.
    // at::detail::getDefaultCPUGenerator().get_state() returns a ByteTensor
    // encoding the full mt19937 engine state, equivalent to torch.get_rng_state()
    // in Python.
    torch::save(at::detail::getDefaultCPUGenerator().get_state(), p);
  });
  atomic_write(base + ".mt", [&](const std::string& p) {
    std::ofstream f(p); f << rng;
  });
  // .meta written LAST: its presence marks the checkpoint set as valid.
  atomic_write(base + ".meta", [&](const std::string& p) {
    std::ofstream f(p);
    f << "step " << meta.step << "\n"
      << "seed " << meta.seed << "\n"
      << "best " << std::setprecision(17) << meta.best << "\n"
      << "arch " << meta.arch << "\n"
      << "fingerprint " << meta.fingerprint << "\n";
  });
}

// Returns true if a checkpoint set was present and loaded; false if absent.
// Fills `meta` (including the stored fingerprint, for the caller to validate).
template <class ModelT>
bool load_checkpoint(const std::string& dir, const std::string& prefix,
                     ModelT& model, torch::optim::Adam& opt,
                     CkptMeta& meta, std::mt19937_64& rng, torch::Device dev) {
  const std::string base = dir + "/" + prefix;
  if (!path_exists(base + ".meta")) return false;
  {
    std::ifstream f(base + ".meta");
    std::string key;
    while (f >> key) {
      if (key == "step") f >> meta.step;
      else if (key == "seed") f >> meta.seed;
      else if (key == "best") f >> meta.best;
      else if (key == "arch") f >> meta.arch;
      else if (key == "fingerprint") f >> meta.fingerprint;
    }
  }
  torch::load(model, base + ".model.pt", dev);
  model->to(dev);
  torch::load(opt, base + ".opt.pt");
  // GPU seam (untested on this box): Adam state tensors load CPU-resident. On a
  // CUDA resume they must be moved onto `dev` here before the first opt.step();
  // the CPU path needs no move. Verify and add the move when the GPU is online.
  {
    torch::Tensor rngst;
    torch::load(rngst, base + ".rng.pt");
    // C++ libtorch has no torch::set_rng_state(); use the Generator API.
    // Copying the const ref to a local non-const Generator shares the same
    // underlying GeneratorImpl (intrusive_ptr), so set_state modifies the
    // global default CPU generator in-place.
    auto gen = at::detail::getDefaultCPUGenerator();
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_state(rngst.to(torch::kCPU));
  }
  { std::ifstream f(base + ".mt"); f >> rng; }
  return true;
}

}  // namespace archcommon
