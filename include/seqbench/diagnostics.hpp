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
  // Positions counted in the metric (1 = scored). Empty means all positions are scored.
  // Used to score only the capability-bearing bytes (e.g. induction values, not keys).
  std::vector<char> scored;
};

// block_len random bits then their parity byte; needs unbounded state.
Diagnostic make_parity(uint64_t seed, std::size_t n_blocks, int block_len);

// In-context induction: each sequence draws a FRESH random mapping from `dict_size`
// distinct `key_len`-byte keys to single-byte values; emits `pairs_per_seq` (key, value)
// pairs. A value is recall-determined (free at the floor) once its key has appeared
// earlier in the same sequence. The floor is computed empirically from the stream.
Diagnostic make_induction(uint64_t seed, std::size_t n_sequences, int pairs_per_seq,
                          int dict_size, int key_len);

// Fill out[0..T) with one task sequence (fresh per seed), for batched training.
void fill_parity(uint64_t seed, uint8_t* out, int T, int block_len);
void fill_induction(uint64_t seed, uint8_t* out, int T, int dict_size, int key_len);

struct DiagResult {
  double observed_bpb = 0.0;
  double floor_bpb = 0.0;
  double naive_bpb = 0.0;
  double fraction_captured = 0.0;  // (naive - observed) / (naive - floor)
};

DiagResult score_diagnostic(Model& m, const Diagnostic& d);

}  // namespace seqbench
