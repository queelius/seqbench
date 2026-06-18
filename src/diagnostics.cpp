#include "seqbench/diagnostics.hpp"
#include "seqbench/metric.hpp"
#include <cmath>

namespace seqbench {

namespace {
// splitmix64: a tiny, deterministic PRNG (no external dependency).
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  // Assumes n is a power of two at all call sites (n=16): zero modulo bias.
  int below(int n) { return int(next() % uint64_t(n)); }
};
}  // namespace

Diagnostic make_parity(uint64_t seed, std::size_t n_blocks, int block_len) {
  Rng rng(seed);
  Diagnostic d;
  d.stream.reserve(n_blocks * std::size_t(block_len + 1));
  for (std::size_t b = 0; b < n_blocks; ++b) {
    uint8_t parity = 0;
    for (int i = 0; i < block_len; ++i) {
      uint8_t bit = uint8_t(rng.next() & 1ull);
      d.stream.push_back(bit);
      parity ^= bit;
    }
    d.stream.push_back(parity);
  }
  // Data bits are irreducible (1 bit each); the parity byte is determined.
  d.floor_bpb = double(block_len) / double(block_len + 1);
  d.naive_bpb = 1.0;  // to any finite model the stream looks like fair coins
  return d;
}

Diagnostic make_induction(uint64_t seed, std::size_t n_pairs, int dict_size) {
  Rng rng(seed);
  Diagnostic d;
  d.stream.reserve(n_pairs * 2);
  // Fixed permutation perm(k) = (k * 7 + 3) mod dict_size as the key->value map.
  for (std::size_t p = 0; p < n_pairs; ++p) {
    int k = rng.below(dict_size);
    int v = (k * 7 + 3) % dict_size;
    d.stream.push_back(uint8_t(k));
    d.stream.push_back(uint8_t(v));
  }
  double bits_per_symbol = std::log2(double(dict_size));
  d.floor_bpb = bits_per_symbol / 2.0;  // key costs log2(D), value costs 0
  d.naive_bpb = bits_per_symbol;        // both bytes look uniform over D
  return d;
}

DiagResult score_diagnostic(Model& m, const Diagnostic& d) {
  ByteSpan span{d.stream.data(), d.stream.size()};
  BpbResult r = run_adaptive(m, span);
  DiagResult out;
  out.observed_bpb = r.bpb();
  out.floor_bpb = d.floor_bpb;
  out.naive_bpb = d.naive_bpb;
  double denom = d.naive_bpb - d.floor_bpb;
  double frac = denom > 0.0 ? (d.naive_bpb - out.observed_bpb) / denom : 0.0;
  if (frac < 0.0) frac = 0.0;
  if (frac > 1.0) frac = 1.0;
  out.fraction_captured = frac;
  return out;
}

}  // namespace seqbench
