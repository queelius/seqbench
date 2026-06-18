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
  // Deterministic, structured text so order>0 can beat order-0.
  // The pattern "ABC" repeats many times. After context "AB", the next byte is
  // deterministically "C". After seeing enough repetitions, order-1 or order-2
  // beats order-0 because the context strongly predicts the next byte.
  static const char kText[] =
      "ABCABCABCABCABCABCABCABCABCABCABCABC"
      "ABCABCABCABCABCABCABCABCABCABCABCABC"
      "ABCABCABCABCABCABCABCABCABCABCABCABC"
      "ABCABCABCABCABCABCABCABCABCABCABCABC"
      "ABCABCABCABCABCABCABCABCABCABCABCABC"
      "ABCABCABCABCABCABCABCABCABCABCABCABC\n";
  return ByteSpan{reinterpret_cast<const uint8_t*>(kText),
                  sizeof(kText) - 1};
}

}  // namespace seqbench
