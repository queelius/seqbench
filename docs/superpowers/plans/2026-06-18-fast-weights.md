# Gradient-free Fast-Weights + Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gradient-free fast-weights `Model` to the seqbench bench, plus a reusable experiment-tracking layer that emits provenance-complete JSONL run records, plus the `arch/fast-weights/` subdir with a runner and conceptual docs.

**Architecture:** Fast-weights is online associative memory: a fast-weight matrix `W` (the recurrent state) updated by the delta rule, with fixed unit-norm key/value embeddings and a tied readout (`logits = V*y`). It stores `key(context) -> V[next byte]` so it predicts the next byte by recall. It satisfies the existing `Model` contract, so the bench's `run_adaptive`, `score_diagnostic`, and `run_sweep` work unchanged. Each finished run appends one JSON line to a committed `runs/results.jsonl`.

**Tech Stack:** C++17, POSIX (`popen`, `mkdir`, `gmtime_r`), plain Makefile, the existing seqbench bench. No third-party libraries, no Python, no autodiff (deferred).

## Global Constraints

- C++17, compiled with `-std=c++17 -O2 -Wall -Wextra -Iinclude`.
- Dependency-free: C++ standard library + POSIX only. No third-party libs, no Python, no CMake, no JSON library (hand-write the serializer).
- All headers under `include/seqbench/`, included as `#include "seqbench/<name>.hpp"`. All library code in `namespace seqbench`.
- Model output is `float logits[256]`; logits must be finite.
- `runs/results.jsonl` is committed to git (the scientific record).
- NO em-dash characters in any committed file (a repo hook rejects them). Use commas, colons, parentheses, periods.
- NEVER weaken a test threshold/assertion to make it pass. If a test does not pass with the planned value, STOP and report BLOCKED with the measured numbers.
- Every commit message ends with these two trailer lines:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015zkyVS7cs6YxjoVn9HG1yH
  ```

### File map

| File | Responsibility |
|------|----------------|
| `include/seqbench/experiment.hpp` + `src/experiment.cpp` | `JsonValue`, `RunRecord`, `to_json`, `fill_provenance`, `append_record` |
| `include/seqbench/fast_weights.hpp` + `models/fast_weights.cpp` | `FastWeightsConfig`, `FastWeights : Model`, `make_fast_weights` |
| `arch/fast-weights/run_fw.cpp` | runner CLI: build model, run battery, append run record |
| `arch/fast-weights/README.md`, `arch/fast-weights/NOTES.md` | conceptual docs |
| `Makefile` | one-line change: also build `arch/*/*.cpp` as tools |
| `.gitignore` | ignore the compiled `arch/fast-weights/run_fw` binary |
| `runs/results.jsonl` | committed experiment log (first record produced by the runner) |
| `tests/experiment_test.cpp`, `tests/fast_weights_test.cpp` | tests |

---

## Task 1: Experiment-tracking layer

**Files:**
- Create: `include/seqbench/experiment.hpp`, `src/experiment.cpp`
- Test: `tests/experiment_test.cpp`

**Interfaces:**
- Produces: `struct JsonValue` with `static JsonValue n(double)` and `static JsonValue s(std::string)`
- Produces: `struct RunRecord { std::string arch, version, timestamp, git_sha; long seed; std::map<std::string,JsonValue> config; std::string corpus_name; std::size_t corpus_bytes; std::map<std::string,double> results; }`
- Produces: `std::string to_json(const RunRecord&)` (single-line JSON, no trailing newline)
- Produces: `void fill_provenance(RunRecord&)` (fills git_sha + timestamp if empty)
- Produces: `void append_record(RunRecord&, const std::string& path = "runs/results.jsonl")`

- [ ] **Step 1: Write the header**

`include/seqbench/experiment.hpp`:

```cpp
#pragma once
#include <cstddef>
#include <map>
#include <string>

namespace seqbench {

// A JSON scalar: number or string (enough for config values).
struct JsonValue {
  enum Type { Number, String };
  Type type = Number;
  double num = 0.0;
  std::string str;
  static JsonValue n(double v) { JsonValue j; j.type = Number; j.num = v; return j; }
  static JsonValue s(std::string v) { JsonValue j; j.type = String; j.str = std::move(v); return j; }
};

// A provenance-complete record of one experiment run, serialized as one JSON line.
struct RunRecord {
  std::string arch;
  std::string version;
  std::string timestamp;   // ISO-8601 UTC; auto-filled by fill_provenance if empty
  std::string git_sha;     // short SHA; auto-filled by fill_provenance if empty
  long seed = 0;
  std::map<std::string, JsonValue> config;
  std::string corpus_name;
  std::size_t corpus_bytes = 0;
  std::map<std::string, double> results;
};

// Serialize to a single-line JSON string (no trailing newline).
std::string to_json(const RunRecord& r);

// Fill git_sha (via `git rev-parse --short HEAD`) and timestamp (ISO-8601 UTC) if empty.
void fill_provenance(RunRecord& r);

// Append one record as a JSON line to `path` (creating the parent dir if needed).
// Fills provenance first. Throws std::runtime_error on I/O failure.
void append_record(RunRecord& r, const std::string& path = "runs/results.jsonl");

}  // namespace seqbench
```

- [ ] **Step 2: Write the failing test**

`tests/experiment_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/experiment.hpp"
#include <cstdio>
#include <fstream>
#include <string>

using namespace seqbench;

static bool has(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

static RunRecord sample() {
  RunRecord r;
  r.arch = "t";
  r.version = "v1";
  r.seed = 7;
  r.config["dim"] = JsonValue::n(64);
  r.config["rule"] = JsonValue::s("delta");
  r.corpus_name = "toy";
  r.corpus_bytes = 100;
  r.results["bpb"] = 3.5;
  return r;
}

static void test_to_json_shape() {
  RunRecord r = sample();
  r.timestamp = "2026-01-01T00:00:00Z";
  r.git_sha = "abc1234";
  std::string s = to_json(r);
  CHECK(s.front() == '{' && s.back() == '}');
  CHECK(has(s, "\"arch\":\"t\""));
  CHECK(has(s, "\"seed\":7"));
  CHECK(has(s, "\"git_sha\":\"abc1234\""));
  CHECK(has(s, "\"dim\":64"));          // numbers are unquoted
  CHECK(has(s, "\"rule\":\"delta\""));  // strings are quoted
  CHECK(has(s, "\"corpus\":{\"name\":\"toy\",\"bytes\":100}"));
  CHECK(has(s, "\"bpb\":3.5"));
  CHECK(s.find('\n') == std::string::npos);  // single line
}

static void test_string_escaping() {
  RunRecord r = sample();
  r.config["note"] = JsonValue::s("x\"y");  // a quote must be escaped
  std::string s = to_json(r);
  CHECK(has(s, "x\\\"y"));
}

static void test_fill_provenance() {
  RunRecord r = sample();
  fill_provenance(r);
  CHECK(!r.git_sha.empty());                 // real SHA or "unknown"
  CHECK(!r.timestamp.empty());
  CHECK(r.timestamp.back() == 'Z');          // ISO-8601 UTC
}

static void test_append_two_lines() {
  const std::string path = "/tmp/seqbench_exp_test.jsonl";
  std::remove(path.c_str());
  RunRecord a = sample();
  RunRecord b = sample();
  append_record(a, path);
  append_record(b, path);
  std::ifstream f(path);
  int lines = 0;
  std::string line, last;
  while (std::getline(f, line)) { ++lines; last = line; }
  f.close();
  CHECK(lines == 2);
  CHECK(has(last, "\"arch\":\"t\""));
  std::remove(path.c_str());
}

int main() {
  RUN(test_to_json_shape);
  RUN(test_string_escaping);
  RUN(test_fill_provenance);
  RUN(test_append_two_lines);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `to_json`, `fill_provenance`, `append_record`).

- [ ] **Step 4: Write the implementation**

`src/experiment.cpp`:

```cpp
#include "seqbench/experiment.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <ctime>
#include <stdexcept>

namespace seqbench {

namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned>(c));
          out += b;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string num_str(double v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.10g", v);
  return std::string(b);
}

std::string jval(const JsonValue& v) {
  if (v.type == JsonValue::Number) return num_str(v.num);
  return "\"" + json_escape(v.str) + "\"";
}

}  // namespace

std::string to_json(const RunRecord& r) {
  std::string o = "{";
  o += "\"arch\":\"" + json_escape(r.arch) + "\",";
  o += "\"version\":\"" + json_escape(r.version) + "\",";
  o += "\"timestamp\":\"" + json_escape(r.timestamp) + "\",";
  o += "\"git_sha\":\"" + json_escape(r.git_sha) + "\",";
  o += "\"seed\":" + std::to_string(r.seed) + ",";
  o += "\"config\":{";
  bool first = true;
  for (const auto& kv : r.config) {
    if (!first) o += ",";
    first = false;
    o += "\"" + json_escape(kv.first) + "\":" + jval(kv.second);
  }
  o += "},";
  o += "\"corpus\":{\"name\":\"" + json_escape(r.corpus_name) +
       "\",\"bytes\":" + std::to_string(r.corpus_bytes) + "},";
  o += "\"results\":{";
  first = true;
  for (const auto& kv : r.results) {
    if (!first) o += ",";
    first = false;
    o += "\"" + json_escape(kv.first) + "\":" + num_str(kv.second);
  }
  o += "}}";
  return o;
}

void fill_provenance(RunRecord& r) {
  if (r.git_sha.empty()) {
    r.git_sha = "unknown";
    std::FILE* p = ::popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (p) {
      char buf[64];
      if (std::fgets(buf, sizeof(buf), p)) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty()) r.git_sha = s;
      }
      ::pclose(p);
    }
  }
  if (r.timestamp.empty()) {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc;
    ::gmtime_r(&t, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    r.timestamp = buf;
  }
}

void append_record(RunRecord& r, const std::string& path) {
  fill_provenance(r);
  std::size_t slash = path.find_last_of('/');
  if (slash != std::string::npos) {
    std::string dir = path.substr(0, slash);
    ::mkdir(dir.c_str(), 0755);  // ignore errors (e.g., dir already exists)
  }
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) throw std::runtime_error("append_record: cannot open " + path);
  std::string line = to_json(r);
  if (std::fprintf(f, "%s\n", line.c_str()) < 0) {
    std::fclose(f);
    throw std::runtime_error("append_record: write failed for " + path);
  }
  if (std::fclose(f) != 0)
    throw std::runtime_error("append_record: close failed for " + path);
}

}  // namespace seqbench
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `experiment_test` prints `OK`; `ALL TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/experiment.hpp src/experiment.cpp tests/experiment_test.cpp
git commit -m "Add experiment-tracking layer: RunRecord, JSON serializer, JSONL appender"
```

---

## Task 2: Fast-weights model

**Files:**
- Create: `include/seqbench/fast_weights.hpp`, `models/fast_weights.cpp`
- Test: `tests/fast_weights_test.cpp`

**Interfaces:**
- Consumes: `Model`, `run_adaptive`, `BpbResult`, `make_context_model`, `Diagnostic`, `DiagResult`, `make_induction`, `make_parity`, `score_diagnostic`, `toy_corpus`
- Produces: `struct FastWeightsConfig { int dim=64; double beta=1.0; double lambda=0.99; int key_width=1; std::string update_rule="delta"; uint64_t seed=42; }`
- Produces: `class FastWeights : public Model` with `explicit FastWeights(const FastWeightsConfig&)`
- Produces: `std::unique_ptr<Model> make_fast_weights(const FastWeightsConfig&)`

- [ ] **Step 1: Write the header**

`include/seqbench/fast_weights.hpp`:

```cpp
#pragma once
#include "seqbench/model.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace seqbench {

struct FastWeightsConfig {
  int dim = 64;
  double beta = 1.0;
  double lambda = 0.99;
  int key_width = 1;
  std::string update_rule = "delta";  // "delta" or "hebbian"
  uint64_t seed = 42;
};

// Gradient-free fast-weights predictor: a delta-rule-updated fast-weight matrix W
// over fixed unit-norm key/value embeddings, with a tied readout (logits = V * y).
// Stores key(context) -> V[next byte] so prediction is next-byte recall.
class FastWeights : public Model {
 public:
  explicit FastWeights(const FastWeightsConfig& cfg);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  void key_from_history(std::vector<double>& out) const;  // writes a length-dim key

  FastWeightsConfig cfg_;
  int dim_;
  std::vector<double> E_;   // 256*dim key embeddings, row-major, unit-norm rows
  std::vector<double> V_;   // 256*dim value embeddings, row-major, unit-norm rows
  std::vector<double> W_;   // dim*dim fast weights (the recurrent state)
  std::vector<uint8_t> history_;  // last key_width observed bytes (most recent at back)
  std::vector<double> k_, q_, y_, Wk_;  // scratch, length dim
};

std::unique_ptr<Model> make_fast_weights(const FastWeightsConfig& cfg);

}  // namespace seqbench
```

- [ ] **Step 2: Write the failing test**

`tests/fast_weights_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/fast_weights.hpp"
#include "seqbench/metric.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/diagnostics.hpp"
#include <cmath>

using namespace seqbench;

static int argmax(const float logits[256]) {
  int best = 0;
  for (int c = 1; c < 256; ++c) if (logits[c] > logits[best]) best = c;
  return best;
}

// Unit: predict before any observe is uniform (all logits equal -> 8 bpb floor).
static void test_uniform_before_observe() {
  FastWeightsConfig cfg; cfg.dim = 32; cfg.seed = 1;
  FastWeights m(cfg);
  float logits[256];
  m.predict(logits);
  CHECK(logits[0] == logits[1]);
  CHECK(logits[0] == logits[255]);
}

// Unit: after A,B,A the model recalls that A is followed by B.
static void test_recall_bigram() {
  FastWeightsConfig cfg; cfg.dim = 64; cfg.beta = 1.0; cfg.lambda = 1.0;
  cfg.key_width = 1; cfg.seed = 1;
  FastWeights m(cfg);
  m.observe(10);  // A
  m.observe(20);  // B  (stores E[A] -> V[B])
  m.observe(10);  // A  (history is now [A])
  float logits[256];
  m.predict(logits);
  CHECK(argmax(logits) == 20);  // querying with A recalls B
}

// Unit: deterministic given the seed.
static void test_deterministic() {
  FastWeightsConfig cfg; cfg.dim = 32; cfg.seed = 5;
  FastWeights a(cfg), b(cfg);
  for (uint8_t x : {uint8_t(1), uint8_t(2), uint8_t(3), uint8_t(2)}) { a.observe(x); b.observe(x); }
  float la[256], lb[256];
  a.predict(la); b.predict(lb);
  for (int c = 0; c < 256; ++c) CHECK(la[c] == lb[c]);
}

// Unit: logits stay finite after a run.
static void test_finite() {
  FastWeightsConfig cfg; cfg.dim = 32; cfg.seed = 1;
  FastWeights m(cfg);
  ByteSpan toy = toy_corpus();
  float logits[256];
  for (std::size_t i = 0; i < toy.len; ++i) { m.predict(logits); m.observe(toy[i]); }
  m.predict(logits);
  for (int c = 0; c < 256; ++c) CHECK(std::isfinite(logits[c]));
}

// Integration (headline): fast-weights captures induction better than the context model.
static void test_induction_beats_context() {
  Diagnostic d = make_induction(7, 50000, 16);
  FastWeightsConfig cfg; cfg.dim = 64; cfg.beta = 1.0; cfg.lambda = 1.0;
  cfg.key_width = 1; cfg.seed = 1;
  FastWeights fw(cfg);
  DiagResult rfw = score_diagnostic(fw, d);
  ContextModel ctx(1);
  DiagResult rc = score_diagnostic(ctx, d);
  CHECK(rfw.fraction_captured > rc.fraction_captured);
  CHECK(rfw.fraction_captured > 0.3);
}

// Integration: fast-weights fails parity (no unbounded state), like the context model.
static void test_fails_parity() {
  Diagnostic p = make_parity(7, 4000, 16);
  FastWeightsConfig cfg; cfg.dim = 64; cfg.seed = 1;
  FastWeights fw(cfg);
  DiagResult r = score_diagnostic(fw, p);
  CHECK(r.fraction_captured < 0.1);
}

// Integration: beats the uniform floor on the toy corpus.
static void test_beats_uniform_on_toy() {
  FastWeightsConfig cfg; cfg.dim = 64; cfg.seed = 1;
  FastWeights fw(cfg);
  double bpb = run_adaptive(fw, toy_corpus()).bpb();
  CHECK(bpb > 0.0);
  CHECK(bpb < 8.0);
}

int main() {
  RUN(test_uniform_before_observe);
  RUN(test_recall_bigram);
  RUN(test_deterministic);
  RUN(test_finite);
  RUN(test_induction_beats_context);
  RUN(test_fails_parity);
  RUN(test_beats_uniform_on_toy);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `FastWeights::...`).

- [ ] **Step 4: Write the implementation**

`models/fast_weights.cpp`:

```cpp
#include "seqbench/fast_weights.hpp"
#include <algorithm>
#include <cmath>

namespace seqbench {

namespace {

const double kTwoPi = 6.283185307179586;

struct Sm64 {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
};

double u01(Sm64& r) { return (r.next() >> 11) * (1.0 / 9007199254740992.0); }

double gaussian(Sm64& r) {
  double u1 = u01(r);
  if (u1 < 1e-300) u1 = 1e-300;
  double u2 = u01(r);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

}  // namespace

FastWeights::FastWeights(const FastWeightsConfig& cfg) : cfg_(cfg), dim_(cfg.dim) {
  if (dim_ < 1) dim_ = 1;
  if (cfg_.key_width < 1) cfg_.key_width = 1;
  E_.assign(static_cast<std::size_t>(256) * dim_, 0.0);
  V_.assign(static_cast<std::size_t>(256) * dim_, 0.0);
  W_.assign(static_cast<std::size_t>(dim_) * dim_, 0.0);
  k_.assign(dim_, 0.0);
  q_.assign(dim_, 0.0);
  y_.assign(dim_, 0.0);
  Wk_.assign(dim_, 0.0);

  Sm64 rng{cfg_.seed ? cfg_.seed : 1};
  auto fill_unit_rows = [&](std::vector<double>& T) {
    for (int b = 0; b < 256; ++b) {
      double norm = 0.0;
      for (int j = 0; j < dim_; ++j) {
        double g = gaussian(rng);
        T[static_cast<std::size_t>(b) * dim_ + j] = g;
        norm += g * g;
      }
      norm = std::sqrt(norm);
      if (norm > 0.0)
        for (int j = 0; j < dim_; ++j) T[static_cast<std::size_t>(b) * dim_ + j] /= norm;
    }
  };
  fill_unit_rows(E_);
  fill_unit_rows(V_);
}

void FastWeights::key_from_history(std::vector<double>& out) const {
  std::fill(out.begin(), out.end(), 0.0);
  for (uint8_t b : history_)
    for (int j = 0; j < dim_; ++j) out[j] += E_[static_cast<std::size_t>(b) * dim_ + j];
  double norm = 0.0;
  for (int j = 0; j < dim_; ++j) norm += out[j] * out[j];
  norm = std::sqrt(norm);
  if (norm > 0.0)
    for (int j = 0; j < dim_; ++j) out[j] /= norm;
}

void FastWeights::predict(float logits[256]) {
  if (history_.empty()) {
    for (int c = 0; c < 256; ++c) logits[c] = 0.0f;
    return;
  }
  key_from_history(q_);  // q = key(context so far)
  for (int i = 0; i < dim_; ++i) {
    double acc = 0.0;
    const double* wrow = &W_[static_cast<std::size_t>(i) * dim_];
    for (int j = 0; j < dim_; ++j) acc += wrow[j] * q_[j];
    y_[i] = acc;
  }
  for (int c = 0; c < 256; ++c) {
    double acc = 0.0;
    const double* vrow = &V_[static_cast<std::size_t>(c) * dim_];
    for (int j = 0; j < dim_; ++j) acc += y_[j] * vrow[j];
    logits[c] = static_cast<float>(acc);
  }
}

void FastWeights::observe(uint8_t b) {
  if (!history_.empty()) {
    key_from_history(k_);  // k = key(context that PRECEDES b)
    for (int i = 0; i < dim_; ++i) {
      double acc = 0.0;
      const double* wrow = &W_[static_cast<std::size_t>(i) * dim_];
      for (int j = 0; j < dim_; ++j) acc += wrow[j] * k_[j];
      Wk_[i] = acc;
    }
    const double* v = &V_[static_cast<std::size_t>(b) * dim_];
    const bool hebb = (cfg_.update_rule == "hebbian");
    const double beta = cfg_.beta;
    const double lambda = cfg_.lambda;
    for (int i = 0; i < dim_; ++i) {
      double ei = hebb ? v[i] : (v[i] - Wk_[i]);
      double bei = beta * ei;
      double* wrow = &W_[static_cast<std::size_t>(i) * dim_];
      for (int j = 0; j < dim_; ++j) wrow[j] = lambda * wrow[j] + bei * k_[j];
    }
  }
  history_.push_back(b);
  if (static_cast<int>(history_.size()) > cfg_.key_width) history_.erase(history_.begin());
}

std::unique_ptr<Model> make_fast_weights(const FastWeightsConfig& cfg) {
  return std::make_unique<FastWeights>(cfg);
}

}  // namespace seqbench
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `fast_weights_test` prints `OK`; `ALL TESTS PASSED`. If `test_induction_beats_context` does not pass, STOP and report BLOCKED with the measured `rfw.fraction_captured` and `rc.fraction_captured` values (do not weaken the assertion).

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/fast_weights.hpp models/fast_weights.cpp tests/fast_weights_test.cpp
git commit -m "Add gradient-free fast-weights model with delta-rule associative memory"
```

---

## Task 3: Runner CLI, Makefile glob, first committed run record

**Files:**
- Create: `arch/fast-weights/run_fw.cpp`
- Modify: `Makefile` (extend the `TOOLS` glob to include `arch/*/*.cpp`)
- Modify: `.gitignore` (ignore the compiled `arch/fast-weights/run_fw`)
- Create (via running the tool): `runs/results.jsonl`

**Interfaces:**
- Consumes: `FastWeightsConfig`, `FastWeights`, `Corpus`, `toy_corpus`, `run_adaptive`, `BpbResult`, `make_induction`, `make_parity`, `score_diagnostic`, `Diagnostic`, `DiagResult`, `RunRecord`, `JsonValue`, `append_record`
- Produces: a `run_fw` executable under `arch/fast-weights/`

- [ ] **Step 1: Extend the Makefile TOOLS glob**

In `Makefile`, change the `TOOLS` line from:

```make
TOOLS   := $(patsubst %.cpp,%,$(wildcard tools/*.cpp))
```

to:

```make
TOOLS   := $(patsubst %.cpp,%,$(wildcard tools/*.cpp)) $(patsubst %.cpp,%,$(wildcard arch/*/*.cpp))
```

(The existing static pattern rule `$(TOOLS): %: %.cpp $(LIB_OBJ)` then builds `arch/fast-weights/run_fw.cpp` into `arch/fast-weights/run_fw`, linking the library objects which now include `src/experiment.o` and `models/fast_weights.o` via the existing wildcards.)

- [ ] **Step 2: Ignore the compiled runner binary**

In `.gitignore`, under the "Compiled test and tool binaries" section, add:

```
/arch/fast-weights/run_fw
```

- [ ] **Step 3: Write the runner**

`arch/fast-weights/run_fw.cpp`:

```cpp
#include "seqbench/corpus.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/experiment.hpp"
#include "seqbench/fast_weights.hpp"
#include "seqbench/metric.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace seqbench;

static void usage() {
  std::fprintf(stderr,
    "usage: run_fw [--dim N] [--beta F] [--lambda F] [--key-width N]\n"
    "              [--rule delta|hebbian] [--seed N] [--corpus toy|PATH]\n"
    "              [--out PATH]\n");
}

int main(int argc, char** argv) {
  FastWeightsConfig cfg;
  std::string corpus = "toy";
  std::string out = "runs/results.jsonl";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
      return argv[++i];
    };
    if (a == "--dim") cfg.dim = std::atoi(need("--dim"));
    else if (a == "--beta") cfg.beta = std::atof(need("--beta"));
    else if (a == "--lambda") cfg.lambda = std::atof(need("--lambda"));
    else if (a == "--key-width") cfg.key_width = std::atoi(need("--key-width"));
    else if (a == "--rule") cfg.update_rule = need("--rule");
    else if (a == "--seed") cfg.seed = std::strtoull(need("--seed"), nullptr, 10);
    else if (a == "--corpus") corpus = need("--corpus");
    else if (a == "--out") out = need("--out");
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(); return 2; }
  }

  std::unique_ptr<Corpus> c;
  ByteSpan span;
  if (corpus == "toy") {
    span = toy_corpus();
  } else {
    c = std::make_unique<Corpus>(corpus);
    span = c->bytes();
  }

  // The model is stateful/adaptive, so use a fresh one per evaluation.
  auto fresh = [&]() { return std::make_unique<FastWeights>(cfg); };

  BpbResult r = run_adaptive(*fresh(), span);
  Diagnostic ind = make_induction(7, 50000, 16);
  DiagResult dind = score_diagnostic(*fresh(), ind);
  Diagnostic par = make_parity(7, 4000, 16);
  DiagResult dpar = score_diagnostic(*fresh(), par);

  std::printf("fast-weights dim=%d beta=%.3g lambda=%.3g key_width=%d rule=%s seed=%llu\n",
              cfg.dim, cfg.beta, cfg.lambda, cfg.key_width, cfg.update_rule.c_str(),
              static_cast<unsigned long long>(cfg.seed));
  std::printf("  corpus=%s bytes=%zu bpb=%.4f (%.2f MB/s)\n",
              corpus.c_str(), r.n_bytes, r.bpb(), r.bytes_per_sec() / 1e6);
  std::printf("  induction fraction=%.4f (bpb %.4f)\n", dind.fraction_captured, dind.observed_bpb);
  std::printf("  parity    fraction=%.4f (bpb %.4f)\n", dpar.fraction_captured, dpar.observed_bpb);

  RunRecord rec;
  rec.arch = "fast-weights";
  rec.version = "v1-gradient-free";
  rec.seed = static_cast<long>(cfg.seed);
  rec.config["dim"] = JsonValue::n(cfg.dim);
  rec.config["beta"] = JsonValue::n(cfg.beta);
  rec.config["lambda"] = JsonValue::n(cfg.lambda);
  rec.config["key_width"] = JsonValue::n(cfg.key_width);
  rec.config["update_rule"] = JsonValue::s(cfg.update_rule);
  rec.corpus_name = corpus;
  rec.corpus_bytes = r.n_bytes;
  rec.results["bpb"] = r.bpb();
  rec.results["throughput_mbps"] = r.bytes_per_sec() / 1e6;
  rec.results["induction_fraction"] = dind.fraction_captured;
  rec.results["parity_fraction"] = dpar.fraction_captured;
  append_record(rec, out);
  std::printf("  appended run record to %s\n", out.c_str());
  return 0;
}
```

- [ ] **Step 4: Build and run on the toy corpus**

Run: `make clean && make all && make test`
Expected: builds `arch/fast-weights/run_fw`; `ALL TESTS PASSED`.

Run: `./arch/fast-weights/run_fw --corpus toy`
Expected: prints the config, corpus bpb (< 8), induction fraction (high), parity fraction (near 0), and `appended run record to runs/results.jsonl`.

- [ ] **Step 5: Verify the run record**

Run: `cat runs/results.jsonl`
Expected: one JSON line with `"arch":"fast-weights"`, a `config` object, `"corpus":{"name":"toy",...}`, and a `results` object containing `bpb`, `induction_fraction`, `parity_fraction`. Confirm it parses (for example, `cat runs/results.jsonl | tail -1 | python3 -c "import sys,json;json.loads(sys.stdin.read());print('valid json')"` if python is available, otherwise eyeball it).

- [ ] **Step 6: Commit (including the first run record)**

```bash
git add Makefile .gitignore arch/fast-weights/run_fw.cpp runs/results.jsonl
git commit -m "Add fast-weights runner CLI and first committed run record"
```

---

## Task 4: Conceptual docs

**Files:**
- Create: `arch/fast-weights/README.md`, `arch/fast-weights/NOTES.md`

This task has no tests (documentation). Verify each file renders as valid markdown and covers the listed points.

- [ ] **Step 1: Write the README**

`arch/fast-weights/README.md`:

````markdown
# Fast-weights (gradient-free)

Fast-weights is a sequence predictor whose recurrent state is a weight matrix
`W` updated by a local online learning rule, and read out by query. It is the
Schmidhuber-1992 fast-weights idea, the same object as modern linear attention,
DeltaNet, and test-time training. This is the gradient-free version: the
projections are fixed (random), so there is no backprop and no autodiff.

## The mechanism

State: a fast-weight matrix `W` (dim x dim), initialized to zero, plus the last
`key_width` observed bytes. Two fixed, unit-norm random embedding tables: keys
`E` (256 x dim) and values `V` (256 x dim).

- `observe(b)`: with `k = key(context before b)` and `v = V[b]`, apply the delta
  rule `W <- lambda*W + beta*(v - W k) k^T`. This stores the association
  "this context is followed by b".
- `predict`: with `q = key(context so far)`, read `y = W q` and score each byte
  by `logits[c] = <y, V[c]>`.

For `key_width = 1` the key is just the previous byte's embedding, so the model
is a soft bigram memory in embedding space. Larger `key_width` sums (and
renormalizes) the embeddings of the last few bytes.

## Why the readout is tied

The readout `logits[c] = <y, V[c]>` is the value table itself ("tied"). The
prediction is "which byte's value vector does the recalled vector most
resemble." Without this tie, a random readout would not favor the correct byte
and the model would sit at the 8-bits-per-byte uniform floor. The tie is what
makes a gradient-free model produce sensible logits.

## What it should and should not do

- It should excel at the bench's induction diagnostic, because induction is
  key-to-value recall, which is exactly what this object does.
- It should fail the parity diagnostic, because parity needs unbounded state
  and `W` is a fixed-size linear memory.
- On real text it is a lossy soft bigram (random features lose information), so
  it will not necessarily beat the count-based context-model baseline. Beating
  strong baselines on text is the job of the later learned-projection version.

## Running it

```
make all
./arch/fast-weights/run_fw --corpus toy
./arch/fast-weights/run_fw --dim 128 --beta 1.0 --lambda 1.0 --corpus data/enwik8
```

Each run appends a provenance-complete record to `runs/results.jsonl`.

## Knobs

- `dim`: feature dimension (cost is O(dim^2) per byte). Default 64.
- `beta`: delta-rule step size. Default 1.0.
- `lambda`: decay (1.0 keeps all associations; < 1 forgets stale ones). Default 0.99.
- `key_width`: bytes forming the key. Default 1.
- `rule`: `delta` (default) or `hebbian` (comparison only).
- `seed`: seeds the fixed embeddings.

## Deferred

Autodiff and learned projections (backprop through the fast-weight recurrence),
order-sensitive keys, and multi-head structure are future specs.
````

- [ ] **Step 2: Write the implementation notes**

`arch/fast-weights/NOTES.md`:

````markdown
# Fast-weights implementation notes

## Choices

- **Fixed unit-norm random embeddings.** `E` and `V` are seeded Gaussian rows
  normalized to unit L2 norm. Unit norms keep the delta-rule scale predictable
  (with unit keys and beta = 1, one observation stores an association nearly
  exactly).
- **Context-to-next-byte association.** In `observe`, the key is computed from
  the history BEFORE the new byte is pushed, and the value is the new byte. This
  is what makes the model predict the next byte rather than re-predict the
  current one. The first byte has no preceding context and triggers no update.
- **Tied readout.** `logits = V * y`, so no separate (untrained) output matrix
  is needed.
- **Delta rule over Hebbian.** The delta rule is self-correcting (online ridge
  regression) and does not saturate; Hebbian is kept only as a comparison knob.
- **Cost.** O(dim^2) per byte for the update and read, plus O(256 * dim) for the
  readout. dim is the natural sweep axis.
- **Divergence is surfaced, not hidden.** If hyperparameters blow `W` up, the
  bench's finiteness guard throws, which correctly flags a bad configuration.

## Results log

Record runs here as they are produced (the machine-readable source of truth is
`runs/results.jsonl`; this is the human interpretation).

- (first toy run) see `runs/results.jsonl`: fast-weights beats the uniform floor
  on toy, strongly captures induction, and fails parity, as expected.
````

- [ ] **Step 3: Commit**

```bash
git add arch/fast-weights/README.md arch/fast-weights/NOTES.md
git commit -m "Add fast-weights conceptual docs and implementation notes"
```

---

## Final verification (after all tasks)

- [ ] `make clean && make all && make test` -> `ALL TESTS PASSED`.
- [ ] `./arch/fast-weights/run_fw --corpus toy` prints sensible numbers and appends a record.
- [ ] `runs/results.jsonl` is committed and contains at least one valid `fast-weights` record.
- [ ] (Optional, needs enwik8) `./arch/fast-weights/run_fw --dim 128 --lambda 1.0 --corpus data/enwik8` to record a real-text run.

## Plan self-review notes

- **Spec coverage:** the gradient-free fast-weights model with delta rule, fixed tied embeddings, and the context-to-next-byte association (Task 2); the reusable experiment-tracking layer with provenance-complete JSONL records (Task 1); the `arch/fast-weights/` subdir with runner (Task 3) and conceptual docs (Task 4); committed `runs/results.jsonl` (Task 3); the headline behaviors (beats context on induction, fails parity) as tests (Task 2). All spec sections covered.
- **Deferred per spec (not in this plan):** autodiff, learned projections, order-sensitive keys, multi-head, a leaderboard tool, GPU.
- **Type consistency:** `FastWeightsConfig`, `FastWeights`, `make_fast_weights`, `RunRecord`, `JsonValue`, `to_json`, `append_record` are used with identical signatures across Tasks 1, 2, and 3. The runner builds `JsonValue::n`/`JsonValue::s` exactly as defined in Task 1.
- **Anti-fudging:** the induction threshold is the one place reality could differ from expectation; Task 2 Step 5 instructs BLOCKED-with-numbers rather than weakening it.
