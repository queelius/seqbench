#pragma once
#include "seqbench/byte_span.hpp"
#include <cstddef>
#include <string>

namespace seqbench {

// Read-only, mmap-backed view of a byte corpus file.
class Corpus {
 public:
  explicit Corpus(const std::string& path);
  ~Corpus();
  Corpus(const Corpus&) = delete;
  Corpus& operator=(const Corpus&) = delete;

  ByteSpan bytes() const { return ByteSpan{data_, len_}; }
  ByteSpan train() const;  // [0, 90%)
  ByteSpan val() const;    // [90%, 95%)
  ByteSpan test() const;   // [95%, 100%)

 private:
  const uint8_t* data_ = nullptr;
  std::size_t len_ = 0;
  int fd_ = -1;
};

// Small, deterministic in-repo corpus for fast tests.
ByteSpan toy_corpus();

}  // namespace seqbench
