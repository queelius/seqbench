#include "seqbench/diagnostics.hpp"
#include "seqbench/metric.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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

// A fixed key dictionary shared across all induction sequences, so the same keys recur
// with DIFFERENT per-sequence values. A global count table conflates those conflicting
// mappings and cannot learn them; a decaying recurrence re-infers each sequence's mapping.
std::vector<std::vector<uint8_t>> induction_key_dict(int dict_size, int key_len) {
  Rng keyrng(0x123456789abcdef0ull ^ (uint64_t(dict_size) << 8) ^ uint64_t(key_len));
  std::vector<std::vector<uint8_t>> keys(dict_size, std::vector<uint8_t>(key_len));
  for (int k = 0; k < dict_size; ++k)
    for (int j = 0; j < key_len; ++j) keys[k][j] = static_cast<uint8_t>(keyrng.next() & 0xff);
  return keys;
}
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

Diagnostic make_induction(uint64_t seed, std::size_t n_sequences, int pairs_per_seq,
                          int dict_size, int key_len) {
  Rng rng(seed);
  auto keys = induction_key_dict(dict_size, key_len);  // fixed across all sequences
  Diagnostic d;
  std::vector<char> recall;  // 1 if this byte is a recall-determined value
  std::vector<char> isval;   // 1 if this byte is a value (the scored recall-target byte)
  for (std::size_t s = 0; s < n_sequences; ++s) {
    // fresh mapping per sequence (keys are fixed; only the key->value map changes).
    std::vector<uint8_t> vals(dict_size);
    for (int k = 0; k < dict_size; ++k) vals[k] = static_cast<uint8_t>(rng.next() & 0xff);
    std::vector<char> seen(dict_size, 0);
    for (int p = 0; p < pairs_per_seq; ++p) {
      int k = rng.below(dict_size);
      for (int j = 0; j < key_len; ++j) {
        d.stream.push_back(keys[k][j]); recall.push_back(0); isval.push_back(0);
      }
      d.stream.push_back(vals[k]);
      recall.push_back(seen[k]);  // recall-determined iff this key appeared earlier in-seq
      isval.push_back(1);
      seen[k] = 1;
    }
  }
  // Empirical marginal byte entropy. The metric scores ONLY value positions (the recall
  // target); keys are context the model reads but is not scored on, so a model that merely
  // compresses the fixed key dictionary gets no credit.
  std::array<double, 256> cnt;
  cnt.fill(0.0);
  for (uint8_t b : d.stream) cnt[b] += 1.0;
  double N = static_cast<double>(d.stream.size());
  std::array<double, 256> bits;
  for (int b = 0; b < 256; ++b) bits[b] = cnt[b] > 0.0 ? -std::log2(cnt[b] / N) : 0.0;
  double naive = 0.0, floor = 0.0;
  std::size_t nv = 0;
  for (std::size_t i = 0; i < d.stream.size(); ++i) {
    if (!isval[i]) continue;
    ++nv;
    double bi = bits[d.stream[i]];
    naive += bi;
    if (!recall[i]) floor += bi;  // recall-determined values cost 0 at the floor
  }
  d.naive_bpb = nv > 0 ? naive / static_cast<double>(nv) : 0.0;
  d.floor_bpb = nv > 0 ? floor / static_cast<double>(nv) : 0.0;
  d.scored = std::move(isval);
  return d;
}

void fill_parity(uint64_t seed, uint8_t* out, int T, int block_len) {
  Rng rng(seed);
  int pos = 0;
  while (pos < T) {
    uint8_t parity = 0;
    for (int i = 0; i < block_len && pos < T; ++i) {
      uint8_t bit = static_cast<uint8_t>(rng.next() & 1ull);
      out[pos++] = bit;
      parity ^= bit;
    }
    if (pos < T) out[pos++] = parity;
  }
}

void fill_induction(uint64_t seed, uint8_t* out, int T, int dict_size, int key_len) {
  Rng rng(seed);
  auto keys = induction_key_dict(dict_size, key_len);  // fixed across all sequences
  std::vector<uint8_t> vals(dict_size);
  for (int k = 0; k < dict_size; ++k) vals[k] = static_cast<uint8_t>(rng.next() & 0xff);
  int pos = 0;
  while (pos < T) {
    int k = rng.below(dict_size);
    for (int j = 0; j < key_len && pos < T; ++j) out[pos++] = keys[k][j];
    if (pos < T) out[pos++] = vals[k];
  }
}

DiagResult score_diagnostic(Model& m, const Diagnostic& d) {
  // Drive the model over the whole stream (it needs every byte as context), but accumulate
  // bits only at scored positions. An empty `scored` mask means every position is scored.
  const bool all = d.scored.empty();
  double bits = 0.0;
  std::size_t n = 0;
  float logits[256];
  for (std::size_t i = 0; i < d.stream.size(); ++i) {
    m.predict(logits);
    if (!logits_finite(logits))
      throw std::runtime_error("score_diagnostic: non-finite logits at position " +
                               std::to_string(i));
    if (all || d.scored[i]) {
      bits += logit_bits(logits, d.stream[i]);
      ++n;
    }
    m.observe(d.stream[i]);
  }
  DiagResult out;
  out.observed_bpb = n ? bits / static_cast<double>(n) : 0.0;
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
