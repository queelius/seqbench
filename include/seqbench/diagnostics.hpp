#pragma once
#include "seqbench/model.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace seqbench {

struct Diagnostic {
  std::vector<uint8_t> stream;
  double floor_bpb = 0.0;  // best achievable bpb if all structure is used
  double naive_bpb = 0.0;  // bpb of a marginal-only reference
};

// block_len random bits then their parity byte; needs unbounded state.
Diagnostic make_parity(uint64_t seed, std::size_t n_blocks, int block_len);

// (key, value) pairs over a dict_size alphabet; value = perm(key).
Diagnostic make_induction(uint64_t seed, std::size_t n_pairs, int dict_size);

struct DiagResult {
  double observed_bpb = 0.0;
  double floor_bpb = 0.0;
  double naive_bpb = 0.0;
  double fraction_captured = 0.0;  // (naive - observed) / (naive - floor)
};

DiagResult score_diagnostic(Model& m, const Diagnostic& d);

}  // namespace seqbench
