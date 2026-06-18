#pragma once
#include <cstddef>
#include <cstdint>

namespace seqbench {

struct ByteSpan {
  const uint8_t* data = nullptr;
  std::size_t len = 0;
  uint8_t operator[](std::size_t i) const { return data[i]; }
  ByteSpan subspan(std::size_t off, std::size_t n) const {
    return ByteSpan{data + off, n};
  }
};

}  // namespace seqbench
