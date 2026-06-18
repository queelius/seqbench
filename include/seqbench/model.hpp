#pragma once
#include "seqbench/byte_span.hpp"
#include <cstdint>

namespace seqbench {

// Every architecture, baseline or neural, satisfies this contract.
struct Model {
  // Fill 256 finite logits (unnormalized log-probabilities). The bench
  // applies one canonical log-softmax to score bits; models never normalize.
  virtual void predict(float logits[256]) = 0;
  // Reveal the actual next byte. Adaptive models update parameters here too.
  virtual void observe(uint8_t b) = 0;
  // Optional offline fit for two-phase (neural) models. Default: no-op.
  virtual void train(ByteSpan corpus) { (void)corpus; }
  virtual ~Model() = default;
};

}  // namespace seqbench
