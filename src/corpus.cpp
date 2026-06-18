#include "seqbench/corpus.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

namespace seqbench {

Corpus::Corpus(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("Corpus: cannot open " + path);
  struct stat st;
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("Corpus: cannot stat " + path);
  }
  len_ = static_cast<std::size_t>(st.st_size);
  if (len_ == 0) {
    data_ = nullptr;
    return;
  }
  void* p = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) {
    ::close(fd_);
    throw std::runtime_error("Corpus: mmap failed for " + path);
  }
  data_ = static_cast<const uint8_t*>(p);
}

Corpus::~Corpus() {
  if (data_ != nullptr && len_ > 0)
    ::munmap(const_cast<uint8_t*>(data_), len_);
  if (fd_ >= 0) ::close(fd_);
}

ByteSpan Corpus::train() const {
  std::size_t end = (len_ * 90) / 100;
  return ByteSpan{data_, end};
}

ByteSpan Corpus::val() const {
  std::size_t a = (len_ * 90) / 100;
  std::size_t b = (len_ * 95) / 100;
  return ByteSpan{data_ + a, b - a};
}

ByteSpan Corpus::test() const {
  std::size_t a = (len_ * 95) / 100;
  return ByteSpan{data_ + a, len_ - a};
}

ByteSpan toy_corpus() {
  // Deterministic realistic ASCII, long enough that order-3 contexts warm up
  // and clearly beat order-0. A varied English block is repeated many times so
  // its trigrams recur; the static buffer is built once, so the returned
  // pointer and length are stable across calls.
  static const std::string text = [] {
    const std::string block =
        "the quick brown fox jumps over the lazy dog. "
        "pack my box with five dozen liquor jugs. "
        "how vexingly quick daft zebras jump! ";
    std::string s;
    s.reserve(block.size() * 40);
    for (int i = 0; i < 40; ++i) s += block;
    return s;
  }();
  return ByteSpan{reinterpret_cast<const uint8_t*>(text.data()), text.size()};
}

}  // namespace seqbench
