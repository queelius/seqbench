# seqbench Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free C++ bench that measures any byte-level sequence-prediction model in bits-per-byte, with a context-model baseline, two discriminating synthetic diagnostics, and a scaling-sweep runner.

**Architecture:** Every model satisfies one small polymorphic contract (`predict(float logits[256])` / `observe(uint8_t)`). The bench drives a model over a byte stream, scores each step with one canonical log-sum-exp (so all models are compared by identical code), and accumulates total bits with Kahan summation. Corpus is `mmap`ed; baselines, diagnostics, and the sweep runner all sit on top of the same contract and metric.

**Tech Stack:** C++17, POSIX `mmap`, a plain Makefile, a tiny home-grown test harness. No third-party libraries, no Python, CPU only.

## Global Constraints

- Language: **C++17** (`-std=c++17`), compiled with `-O2 -Wall -Wextra -Iinclude`.
- **Dependency-free**: standard library + POSIX only. No BLAS/Eigen/CUDA, no test framework, no Python.
- Build: a single plain **Makefile**. No CMake.
- All headers live under `include/seqbench/` and are included as `#include "seqbench/<name>.hpp"`.
- All code lives in `namespace seqbench`.
- Model output is `float logits[256]`; the metric accumulates total bits in `double`.
- The metric is the **only** place exp/log runs in the eval path.
- Logits must be **finite** (no inf/NaN). The run loop guards this and throws.
- **No em-dashes** in any committed file (a repo hook rejects them). Use commas, colons, parentheses, or periods.
- Every commit message ends with these two trailer lines (shown once here, assumed on every commit step):

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015zkyVS7cs6YxjoVn9HG1yH
  ```

### File map

| File | Responsibility |
|------|----------------|
| `Makefile` | Build lib objects, tools, and per-file test binaries; `make test`; `make data/enwik8` |
| `tests/test_util.hpp` | Home-grown assertion macros + summary |
| `include/seqbench/byte_span.hpp` | `ByteSpan` value type (pointer + length) |
| `include/seqbench/model.hpp` | The `Model` contract |
| `include/seqbench/metric.hpp` + `src/metric.cpp` | `logit_bits`, `logits_finite`, `Kahan`, `BpbResult`, `run_adaptive`, `run_train_test` |
| `include/seqbench/corpus.hpp` + `src/corpus.cpp` | `Corpus` (mmap + 90/5/5 splits), `toy_corpus()` |
| `include/seqbench/context_model.hpp` + `models/context_model.cpp` | order-N adaptive context model + `make_context_model` |
| `include/seqbench/diagnostics.hpp` + `src/diagnostics.cpp` | `make_parity`, `make_induction`, `score_diagnostic` |
| `include/seqbench/sweep.hpp` + `src/sweep.cpp` | `run_sweep`, `write_csv` |
| `tools/bench.cpp` | CLI: model + corpus to bpb |
| `tools/sweep_cli.cpp` | CLI: sweep context-model order to CSV |
| `tests/*.cpp` | One executable per area |

---

## Task 1: Project skeleton, build, test harness, Model contract

**Files:**
- Create: `Makefile`, `tests/test_util.hpp`, `include/seqbench/byte_span.hpp`, `include/seqbench/model.hpp`
- Test: `tests/byte_span_test.cpp`

**Interfaces:**
- Produces: `seqbench::ByteSpan { const uint8_t* data; size_t len; uint8_t operator[](size_t) const; ByteSpan subspan(size_t off, size_t n) const; }`
- Produces: `struct seqbench::Model { virtual void predict(float logits[256])=0; virtual void observe(uint8_t)=0; virtual void train(ByteSpan){}; virtual ~Model()=default; }`
- Produces (test harness): `CHECK(cond)`, `CHECK_NEAR(a,b,eps)`, `RUN(fn)`, `int test_summary()`, global `int g_test_failures`

- [ ] **Step 1: Write the Makefile**

```make
CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude

LIB_SRC := $(wildcard src/*.cpp) $(wildcard models/*.cpp)
LIB_OBJ := $(LIB_SRC:.cpp=.o)
TOOLS   := $(patsubst %.cpp,%,$(wildcard tools/*.cpp))
TESTS   := $(patsubst %.cpp,%,$(wildcard tests/*.cpp))

all: $(TOOLS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TOOLS): %: %.cpp $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJ) -o $@

$(TESTS): %: %.cpp $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJ) -o $@

test: $(TESTS)
	@fail=0; for t in $(TESTS); do echo "== $$t =="; ./$$t || fail=1; done; \
	 if [ $$fail -ne 0 ]; then echo "TESTS FAILED"; exit 1; fi; echo "ALL TESTS PASSED"

data/enwik8:
	mkdir -p data
	cd data && curl -L -o enwik8.zip http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip && rm -f enwik8.zip

clean:
	rm -f $(LIB_OBJ) $(TOOLS) $(TESTS)

.PHONY: all test clean
```

- [ ] **Step 2: Write the test harness header**

`tests/test_util.hpp`:

```cpp
#pragma once
#include <cstdio>
#include <cmath>

inline int g_test_failures = 0;

#define CHECK(cond) \
  do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
    ++g_test_failures; } } while (0)

#define CHECK_NEAR(a, b, eps) \
  do { double _a = (a), _b = (b); \
    if (std::fabs(_a - _b) > (eps)) { \
      std::fprintf(stderr, "FAIL %s:%d: CHECK_NEAR(%s, %s): %.10g vs %.10g\n", \
                   __FILE__, __LINE__, #a, #b, _a, _b); \
      ++g_test_failures; } } while (0)

#define RUN(fn) do { std::printf("  - %s\n", #fn); fn(); } while (0)

inline int test_summary() {
  if (g_test_failures) {
    std::fprintf(stderr, "%d failure(s)\n", g_test_failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
```

- [ ] **Step 3: Write `ByteSpan` and `Model`**

`include/seqbench/byte_span.hpp`:

```cpp
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
```

`include/seqbench/model.hpp`:

```cpp
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
```

- [ ] **Step 4: Write the failing smoke test**

`tests/byte_span_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/byte_span.hpp"

using namespace seqbench;

static void test_byte_span_basics() {
  const uint8_t buf[5] = {10, 20, 30, 40, 50};
  ByteSpan s{buf, 5};
  CHECK(s.len == 5);
  CHECK(s[0] == 10);
  CHECK(s[4] == 50);
  ByteSpan sub = s.subspan(1, 3);
  CHECK(sub.len == 3);
  CHECK(sub[0] == 20);
  CHECK(sub[2] == 40);
}

int main() {
  RUN(test_byte_span_basics);
  return test_summary();
}
```

- [ ] **Step 5: Build and run; verify it passes**

Run: `make test`
Expected: compiles cleanly, prints `== tests/byte_span_test ==`, `OK`, then `ALL TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add Makefile tests/test_util.hpp tests/byte_span_test.cpp include/seqbench/
git commit -m "Add build skeleton, test harness, ByteSpan and Model contract"
```

---

## Task 2: The bits metric (log-sum-exp + Kahan)

**Files:**
- Create: `include/seqbench/metric.hpp`, `src/metric.cpp`
- Test: `tests/metric_test.cpp`

**Interfaces:**
- Consumes: `seqbench::Model`, `seqbench::ByteSpan`
- Produces: `double logit_bits(const float logits[256], uint8_t actual)` (bits to code `actual` under the softmax of `logits`)
- Produces: `bool logits_finite(const float logits[256])`
- Produces: `struct Kahan { double sum=0; double c=0; void add(double x); double value() const; }`
- Produces: `struct BpbResult { double total_bits; size_t n_bytes; double seconds; double bpb() const; double bytes_per_sec() const; }`
- Produces (declared now, implemented in Task 3): `BpbResult run_adaptive(Model&, ByteSpan)`, `BpbResult run_train_test(Model&, ByteSpan, ByteSpan)`

- [ ] **Step 1: Write the header**

`include/seqbench/metric.hpp`:

```cpp
#pragma once
#include "seqbench/byte_span.hpp"
#include "seqbench/model.hpp"
#include <cstddef>
#include <cstdint>

namespace seqbench {

// Bits to code `actual` under softmax(logits), via a numerically stable
// log-sum-exp. This is the one canonical scoring path for every model.
double logit_bits(const float logits[256], uint8_t actual);

// True iff all 256 logits are finite (no inf/NaN).
bool logits_finite(const float logits[256]);

// Compensated (Kahan) summation for exact accumulation over ~1e8 terms.
struct Kahan {
  double sum = 0.0;
  double c = 0.0;
  void add(double x);
  double value() const { return sum; }
};

struct BpbResult {
  double total_bits = 0.0;
  std::size_t n_bytes = 0;
  double seconds = 0.0;
  double bpb() const { return n_bytes ? total_bits / double(n_bytes) : 0.0; }
  double bytes_per_sec() const {
    return seconds > 0.0 ? double(n_bytes) / seconds : 0.0;
  }
};

// Adaptive (one-pass) protocol: predict, score, observe(updates allowed).
BpbResult run_adaptive(Model& m, ByteSpan data);
// Train/test protocol: train(train_split), then score the val split.
// The model is responsible for freezing parameters in observe() after train().
BpbResult run_train_test(Model& m, ByteSpan train, ByteSpan val);

}  // namespace seqbench
```

- [ ] **Step 2: Write the failing test**

`tests/metric_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/metric.hpp"
#include <cmath>

using namespace seqbench;

// Uniform logits => every byte costs exactly log2(256) = 8 bits.
static void test_uniform_is_eight_bits() {
  float logits[256];
  for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
  CHECK_NEAR(logit_bits(logits, 0), 8.0, 1e-9);
  CHECK_NEAR(logit_bits(logits, 200), 8.0, 1e-9);
}

// Shift-invariance: adding a constant to all logits cannot change bits.
static void test_shift_invariance() {
  float a[256], b[256];
  for (int i = 0; i < 256; ++i) { a[i] = 0.1f * i; b[i] = 0.1f * i + 7.0f; }
  CHECK_NEAR(logit_bits(a, 42), logit_bits(b, 42), 1e-5);
}

// Hand golden: all logits 0 except index 7 = ln(255) gives p[7]=1/2,
// p[other]=1/510, so bits(7)=1.0 and bits(0)=log2(510).
static void test_hand_golden() {
  float logits[256];
  for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
  logits[7] = std::log(255.0f);
  CHECK_NEAR(logit_bits(logits, 7), 1.0, 1e-5);
  CHECK_NEAR(logit_bits(logits, 0), std::log2(510.0), 1e-4);
}

static void test_finiteness() {
  float ok[256]; for (int i = 0; i < 256; ++i) ok[i] = 1.0f;
  CHECK(logits_finite(ok));
  float bad[256]; for (int i = 0; i < 256; ++i) bad[i] = 1.0f;
  bad[3] = std::nanf("");
  CHECK(!logits_finite(bad));
}

// Kahan keeps a small running value exact across many tiny additions.
static void test_kahan() {
  Kahan k;
  for (int i = 0; i < 1000000; ++i) k.add(1e-6);
  CHECK_NEAR(k.value(), 1.0, 1e-9);
}

int main() {
  RUN(test_uniform_is_eight_bits);
  RUN(test_shift_invariance);
  RUN(test_hand_golden);
  RUN(test_finiteness);
  RUN(test_kahan);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `logit_bits`, `logits_finite`, `Kahan::add`), because `src/metric.cpp` does not exist yet.

- [ ] **Step 4: Write the implementation**

`src/metric.cpp` (only the parts needed for this task; the run loops come in Task 3, but include their definitions now as declared, or stub them. To keep `make test` green here, implement everything except the loops as below and add the loops in Task 3.):

```cpp
#include "seqbench/metric.hpp"
#include <cmath>

namespace seqbench {

double logit_bits(const float logits[256], uint8_t actual) {
  float m = logits[0];
  for (int i = 1; i < 256; ++i) if (logits[i] > m) m = logits[i];
  double sum = 0.0;
  for (int i = 0; i < 256; ++i) sum += std::exp(double(logits[i]) - double(m));
  double logZ = double(m) + std::log(sum);      // natural-log normalizer
  double log_p = double(logits[actual]) - logZ;  // ln p(actual)
  return -log_p / std::log(2.0);                 // convert to bits
}

bool logits_finite(const float logits[256]) {
  for (int i = 0; i < 256; ++i) if (!std::isfinite(logits[i])) return false;
  return true;
}

void Kahan::add(double x) {
  double y = x - c;
  double t = sum + y;
  c = (t - sum) - y;
  sum = t;
}

}  // namespace seqbench
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `metric_test` prints `OK`; `ALL TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/metric.hpp src/metric.cpp tests/metric_test.cpp
git commit -m "Add bits metric: logit_bits, logits_finite, Kahan accumulation"
```

---

## Task 3: The run loops (adaptive + train/test)

**Files:**
- Modify: `src/metric.cpp` (add `run_adaptive`, `run_train_test`)
- Test: `tests/runloop_test.cpp`

**Interfaces:**
- Consumes: `logit_bits`, `logits_finite`, `Kahan`, `BpbResult`, `Model`, `ByteSpan`
- Produces: definitions of `run_adaptive` and `run_train_test` (signatures already declared in Task 2)

- [ ] **Step 1: Write the failing test**

`tests/runloop_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/metric.hpp"
#include <cmath>
#include <stdexcept>

using namespace seqbench;

// Always-uniform model: every byte costs exactly 8 bits.
struct UniformModel : Model {
  void predict(float logits[256]) override {
    for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
  }
  void observe(uint8_t) override {}
};

// Emits a NaN logit to exercise the finiteness guard.
struct NanModel : Model {
  void predict(float logits[256]) override {
    for (int i = 0; i < 256; ++i) logits[i] = 0.0f;
    logits[3] = std::nanf("");
  }
  void observe(uint8_t) override {}
};

static void test_adaptive_uniform_is_eight() {
  const uint8_t buf[6] = {1, 2, 3, 4, 5, 6};
  UniformModel m;
  BpbResult r = run_adaptive(m, ByteSpan{buf, 6});
  CHECK(r.n_bytes == 6);
  CHECK_NEAR(r.bpb(), 8.0, 1e-9);
  CHECK_NEAR(r.total_bits, 48.0, 1e-7);
}

static void test_train_test_uniform_is_eight() {
  const uint8_t tr[3] = {9, 9, 9};
  const uint8_t va[4] = {1, 2, 3, 4};
  UniformModel m;
  BpbResult r = run_train_test(m, ByteSpan{tr, 3}, ByteSpan{va, 4});
  CHECK(r.n_bytes == 4);
  CHECK_NEAR(r.bpb(), 8.0, 1e-9);
}

static void test_guard_throws_on_nan() {
  const uint8_t buf[2] = {0, 1};
  NanModel m;
  bool threw = false;
  try { run_adaptive(m, ByteSpan{buf, 2}); }
  catch (const std::exception&) { threw = true; }
  CHECK(threw);
}

int main() {
  RUN(test_adaptive_uniform_is_eight);
  RUN(test_train_test_uniform_is_eight);
  RUN(test_guard_throws_on_nan);
  return test_summary();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `run_adaptive` / `run_train_test`).

- [ ] **Step 3: Add the run loops to `src/metric.cpp`**

Add these includes at the top of `src/metric.cpp`:

```cpp
#include <chrono>
#include <stdexcept>
#include <string>
```

Append inside `namespace seqbench`:

```cpp
namespace {
BpbResult score_stream(Model& m, ByteSpan data) {
  Kahan bits;
  float logits[256];
  auto t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < data.len; ++i) {
    m.predict(logits);
    if (!logits_finite(logits))
      throw std::runtime_error("non-finite logits at position " +
                               std::to_string(i));
    bits.add(logit_bits(logits, data[i]));
    m.observe(data[i]);
  }
  auto t1 = std::chrono::steady_clock::now();
  BpbResult r;
  r.total_bits = bits.value();
  r.n_bytes = data.len;
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  return r;
}
}  // namespace

BpbResult run_adaptive(Model& m, ByteSpan data) { return score_stream(m, data); }

BpbResult run_train_test(Model& m, ByteSpan train, ByteSpan val) {
  m.train(train);
  return score_stream(m, val);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make test`
Expected: `runloop_test` prints `OK`; `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/metric.cpp tests/runloop_test.cpp
git commit -m "Add adaptive and train/test run loops with finiteness guard"
```

---

## Task 4: Order-N context model baseline

**Files:**
- Create: `include/seqbench/context_model.hpp`, `models/context_model.cpp`
- Test: `tests/context_model_test.cpp`

**Interfaces:**
- Consumes: `Model`, `run_adaptive`, `ByteSpan`
- Produces: `class ContextModel : public Model` with `ContextModel(int order, double alpha = 1.0)`
- Produces: `std::unique_ptr<Model> make_context_model(int order)` (alpha defaults to 1.0)
- Note: `order` is in `[0, 8]`; the last `order` bytes are packed into a `uint64_t` context key. `order == 0` is the single-table sanity floor.

- [ ] **Step 1: Write the header**

`include/seqbench/context_model.hpp`:

```cpp
#pragma once
#include "seqbench/model.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace seqbench {

// Adaptive order-N context model with add-alpha smoothing.
// Prediction for the current context is log(count_b + alpha) per byte b
// (unnormalized log-probs are valid logits). order in [0, 8].
class ContextModel : public Model {
 public:
  explicit ContextModel(int order, double alpha = 1.0);
  void predict(float logits[256]) override;
  void observe(uint8_t b) override;

 private:
  int order_;
  double alpha_;
  uint64_t ctx_ = 0;
  uint64_t mask_ = 0;
  std::unordered_map<uint64_t, std::array<uint32_t, 256>> tables_;
};

std::unique_ptr<Model> make_context_model(int order);

}  // namespace seqbench
```

- [ ] **Step 2: Write the failing test**

`tests/context_model_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/metric.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace seqbench;

// Golden: order-0, alpha=1, input "aaaa".
// Step bits: 8, log2(257/2), log2(258/3), log2(259/4); total ~= 27.4487.
static void test_order0_golden() {
  const char* s = "aaaa";
  ByteSpan data{reinterpret_cast<const uint8_t*>(s), 4};
  ContextModel m(0, 1.0);
  BpbResult r = run_adaptive(m, data);
  double expect = 8.0 + std::log2(257.0 / 2.0) + std::log2(258.0 / 3.0) +
                  std::log2(259.0 / 4.0);
  CHECK_NEAR(r.total_bits, expect, 1e-4);
  CHECK_NEAR(r.total_bits, 27.4487, 1e-2);
}

// On periodic data, a model with memory (order>=1) beats order-0.
static void test_higher_order_beats_order0_on_periodic() {
  std::string s;
  for (int i = 0; i < 2000; ++i) s += "ab";  // "abab..." length 4000
  ByteSpan data{reinterpret_cast<const uint8_t*>(s.data()), s.size()};
  ContextModel m0(0, 1.0);
  ContextModel m1(1, 1.0);
  double b0 = run_adaptive(m0, data).bpb();
  double b1 = run_adaptive(m1, data).bpb();
  CHECK(b1 < b0);
  CHECK(b1 < 0.5);  // order-1 nearly predicts the period
}

static void test_factory() {
  auto m = make_context_model(2);
  CHECK(m != nullptr);
}

int main() {
  RUN(test_order0_golden);
  RUN(test_higher_order_beats_order0_on_periodic);
  RUN(test_factory);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `ContextModel` / `make_context_model`).

- [ ] **Step 4: Write the implementation**

`models/context_model.cpp`:

```cpp
#include "seqbench/context_model.hpp"
#include <cmath>

namespace seqbench {

ContextModel::ContextModel(int order, double alpha)
    : order_(order), alpha_(alpha) {
  if (order_ < 0) order_ = 0;
  if (order_ > 8) order_ = 8;
  mask_ = (order_ >= 8) ? ~0ull : ((1ull << (8 * order_)) - 1);
}

void ContextModel::predict(float logits[256]) {
  auto it = tables_.find(ctx_);
  if (it == tables_.end()) {
    float v = static_cast<float>(std::log(alpha_));
    for (int b = 0; b < 256; ++b) logits[b] = v;
    return;
  }
  const std::array<uint32_t, 256>& counts = it->second;
  for (int b = 0; b < 256; ++b)
    logits[b] = static_cast<float>(std::log(double(counts[b]) + alpha_));
}

void ContextModel::observe(uint8_t b) {
  tables_[ctx_][b] += 1;
  ctx_ = ((ctx_ << 8) | uint64_t(b)) & mask_;
}

std::unique_ptr<Model> make_context_model(int order) {
  return std::make_unique<ContextModel>(order, 1.0);
}

}  // namespace seqbench
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `context_model_test` prints `OK`; `ALL TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/context_model.hpp models/context_model.cpp tests/context_model_test.cpp
git commit -m "Add order-N adaptive context-model baseline"
```

---

## Task 5: Corpus layer (mmap + splits + toy)

**Files:**
- Create: `include/seqbench/corpus.hpp`, `src/corpus.cpp`
- Test: `tests/corpus_test.cpp`

**Interfaces:**
- Consumes: `ByteSpan`
- Produces: `class Corpus` with `explicit Corpus(const std::string& path)`, `ByteSpan bytes() const`, `ByteSpan train() const`, `ByteSpan val() const`, `ByteSpan test() const` (splits are 90/5/5 of the full length)
- Produces: `ByteSpan toy_corpus()` returning a deterministic built-in byte stream
- Note: the `make data/enwik8` target (already in the Makefile from Task 1) fetches the corpus; it requires network and is never invoked by `make test`.

- [ ] **Step 1: Write the header**

`include/seqbench/corpus.hpp`:

```cpp
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
```

- [ ] **Step 2: Write the failing test**

`tests/corpus_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/corpus.hpp"
#include <cstdio>
#include <fstream>
#include <string>

using namespace seqbench;

static void test_toy_corpus_deterministic() {
  ByteSpan a = toy_corpus();
  ByteSpan b = toy_corpus();
  CHECK(a.len > 0);
  CHECK(a.data == b.data);
  CHECK(a.len == b.len);
}

static void test_mmap_roundtrip_and_splits() {
  const std::string path = "/tmp/seqbench_corpus_test.bin";
  {
    std::ofstream f(path, std::ios::binary);
    for (int i = 0; i < 1000; ++i) f.put(char(i % 256));
  }
  Corpus c(path);
  CHECK(c.bytes().len == 1000);
  CHECK(c.bytes()[0] == 0);
  CHECK(c.bytes()[255] == 255);
  CHECK(c.train().len == 900);
  CHECK(c.val().len == 50);
  CHECK(c.test().len == 50);
  // val starts right after train.
  CHECK(c.val()[0] == c.bytes()[900]);
  std::remove(path.c_str());
}

int main() {
  RUN(test_toy_corpus_deterministic);
  RUN(test_mmap_roundtrip_and_splits);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `Corpus::Corpus`, `toy_corpus`, etc.).

- [ ] **Step 4: Write the implementation**

`src/corpus.cpp`:

```cpp
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
  // Deterministic, mildly structured ASCII so order>0 can beat order-0.
  static const char kText[] =
      "the quick brown fox jumps over the lazy dog. "
      "the quick brown fox jumps over the lazy dog. "
      "pack my box with five dozen liquor jugs. "
      "pack my box with five dozen liquor jugs. "
      "the quick brown fox jumps over the lazy dog.\n";
  return ByteSpan{reinterpret_cast<const uint8_t*>(kText),
                  sizeof(kText) - 1};
}

}  // namespace seqbench
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `corpus_test` prints `OK`; `ALL TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/corpus.hpp src/corpus.cpp tests/corpus_test.cpp
git commit -m "Add mmap corpus layer with 90/5/5 splits and toy corpus"
```

---

## Task 6: Bench CLI + integration

**Files:**
- Create: `tools/bench.cpp`
- Test: `tests/integration_test.cpp`

**Interfaces:**
- Consumes: `make_context_model`, `Corpus`, `toy_corpus`, `run_adaptive`, `BpbResult`
- Produces: a `bench` executable. Usage: `bench <model> <corpus>` where `<model>` is `ctx:N` and `<corpus>` is `toy` or a file path.

- [ ] **Step 1: Write the failing integration test**

`tests/integration_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/metric.hpp"
#include <fstream>
#include <memory>
#include <string>

using namespace seqbench;

static void test_context_on_toy() {
  ByteSpan toy = toy_corpus();
  double b0 = run_adaptive(*make_context_model(0), toy).bpb();
  double b3 = run_adaptive(*make_context_model(3), toy).bpb();
  CHECK(b0 > 0.0);
  CHECK(b0 <= 8.0);
  CHECK(b3 < b0);  // memory helps on the repetitive toy text
}

// enwik8 is optional: skip if not fetched, so `make test` runs offline.
static void test_context_on_enwik8_if_present() {
  const std::string path = "data/enwik8";
  std::ifstream probe(path, std::ios::binary);
  if (!probe.good()) {
    std::printf("    (skipped: %s not present)\n", path.c_str());
    return;
  }
  probe.close();
  Corpus c(path);
  double b0 = run_adaptive(*make_context_model(0), c.bytes()).bpb();
  double b3 = run_adaptive(*make_context_model(3), c.bytes()).bpb();
  CHECK(b0 > 4.0);
  CHECK(b0 < 5.5);   // order-0 on English text
  CHECK(b3 < b0);    // higher order helps
  CHECK(b3 < 3.0);   // a few orders of context beat gzip-ish territory
}

int main() {
  RUN(test_context_on_toy);
  RUN(test_context_on_enwik8_if_present);
  return test_summary();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: `integration_test` compiles and runs but `test_context_on_toy` FAILS at `b3 < b0` only if the context model is broken; if Tasks 4 and 5 are correct it should already PASS. The genuinely new artifact in this task is the CLI, so the failing check is the CLI build in Step 3. Proceed to build the CLI.

(If `test_context_on_toy` fails here, fix the context model or toy corpus before continuing.)

- [ ] **Step 3: Write the CLI**

`tools/bench.cpp`:

```cpp
#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/metric.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace seqbench;

static void usage() {
  std::fprintf(stderr,
               "usage: bench <model> <corpus>\n"
               "  model:  ctx:N        (order-N context model, 0..8)\n"
               "  corpus: toy | <path> (path is a raw byte file)\n");
}

int main(int argc, char** argv) {
  if (argc != 3) { usage(); return 2; }
  std::string mspec = argv[1];
  std::string cspec = argv[2];

  std::unique_ptr<Model> model;
  if (mspec.rfind("ctx:", 0) == 0) {
    model = make_context_model(std::atoi(mspec.c_str() + 4));
  } else {
    std::fprintf(stderr, "unknown model: %s\n", mspec.c_str());
    usage();
    return 2;
  }

  std::unique_ptr<Corpus> corpus;
  ByteSpan span;
  if (cspec == "toy") {
    span = toy_corpus();
  } else {
    corpus = std::make_unique<Corpus>(cspec);
    span = corpus->bytes();
  }

  BpbResult r = run_adaptive(*model, span);
  std::printf("model=%s corpus=%s bytes=%zu bpb=%.4f throughput=%.2f MB/s\n",
              mspec.c_str(), cspec.c_str(), r.n_bytes, r.bpb(),
              r.bytes_per_sec() / 1e6);
  return 0;
}
```

- [ ] **Step 4: Build everything and run the CLI**

Run: `make all && make test`
Expected: `bench` builds; `make test` prints `ALL TESTS PASSED`.

Run: `./tools/bench ctx:0 toy`
Expected: a line like `model=ctx:0 corpus=toy bytes=229 bpb=4.7xxx throughput=... MB/s` (exact bpb may vary; it must be between 0 and 8).

Run: `./tools/bench ctx:3 toy`
Expected: a similar line with a clearly lower bpb than `ctx:0`.

- [ ] **Step 5: Commit**

```bash
git add tools/bench.cpp tests/integration_test.cpp
git commit -m "Add bench CLI and context-model integration tests"
```

---

## Task 7: Diagnostics (parity + induction)

**Files:**
- Create: `include/seqbench/diagnostics.hpp`, `src/diagnostics.cpp`
- Test: `tests/diagnostics_test.cpp`

**Interfaces:**
- Consumes: `Model`, `run_adaptive`, `ByteSpan`
- Produces: `struct Diagnostic { std::vector<uint8_t> stream; double floor_bpb; double naive_bpb; }`
- Produces: `Diagnostic make_parity(uint64_t seed, std::size_t n_blocks, int block_len)`
- Produces: `Diagnostic make_induction(uint64_t seed, std::size_t n_pairs, int dict_size)`
- Produces: `struct DiagResult { double observed_bpb; double floor_bpb; double naive_bpb; double fraction_captured; }`
- Produces: `DiagResult score_diagnostic(Model& m, const Diagnostic& d)`
- Note: parity uses alphabet {0,1}; each block is `block_len` random bits followed by their XOR parity byte. A finite-order model cannot beat `naive_bpb = 1.0`. Induction uses single-byte keys over `dict_size` symbols, each mapped by a fixed permutation to a value byte; an order>=1 adaptive model captures a clear fraction.

- [ ] **Step 1: Write the header**

`include/seqbench/diagnostics.hpp`:

```cpp
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
```

- [ ] **Step 2: Write the failing test**

`tests/diagnostics_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/diagnostics.hpp"
#include "seqbench/context_model.hpp"

using namespace seqbench;

// Same seed gives an identical stream.
static void test_generators_deterministic() {
  Diagnostic a = make_parity(123, 100, 8);
  Diagnostic b = make_parity(123, 100, 8);
  CHECK(a.stream.size() == b.stream.size());
  CHECK(a.stream == b.stream);
}

// A finite-order context model captures essentially no parity structure.
static void test_context_fails_parity() {
  Diagnostic d = make_parity(7, 4000, 16);
  auto m = make_context_model(4);
  DiagResult r = score_diagnostic(*m, d);
  CHECK(r.observed_bpb > 0.95);       // near the naive 1.0
  CHECK(r.fraction_captured < 0.1);   // captured ~nothing
}

// An order>=1 context model captures a clear fraction of induction structure.
static void test_context_captures_induction() {
  Diagnostic d = make_induction(7, 8000, 16);
  auto m1 = make_context_model(1);
  DiagResult r1 = score_diagnostic(*m1, d);
  CHECK(r1.fraction_captured > 0.3);  // captures real structure
  auto m0 = make_context_model(0);
  DiagResult r0 = score_diagnostic(*m0, d);
  CHECK(r1.observed_bpb < r0.observed_bpb);  // memory helps
}

int main() {
  RUN(test_generators_deterministic);
  RUN(test_context_fails_parity);
  RUN(test_context_captures_induction);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `make_parity` / `make_induction` / `score_diagnostic`).

- [ ] **Step 4: Write the implementation**

`src/diagnostics.cpp`:

```cpp
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `diagnostics_test` prints `OK`; `ALL TESTS PASSED`. This encodes the "diagnostics discriminate" property: the context model fails parity and captures induction.

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/diagnostics.hpp src/diagnostics.cpp tests/diagnostics_test.cpp
git commit -m "Add parity and induction diagnostics with entropy floors"
```

---

## Task 8: Scaling-sweep runner + CLI

**Files:**
- Create: `include/seqbench/sweep.hpp`, `src/sweep.cpp`, `tools/sweep_cli.cpp`
- Test: `tests/sweep_test.cpp`

**Interfaces:**
- Consumes: `Model`, `run_adaptive`, `BpbResult`, `ByteSpan`, `make_context_model`
- Produces: `struct SweepPoint { int knob; double bpb; double seconds; }`
- Produces: `std::vector<SweepPoint> run_sweep(const std::function<std::unique_ptr<Model>(int)>& factory, const std::vector<int>& knobs, ByteSpan data)`
- Produces: `void write_csv(const std::vector<SweepPoint>& pts, const std::string& path)` (header `knob,bpb,seconds`)
- Produces: a `sweep` executable. Usage: `sweep <corpus> <out.csv> <k0> [k1 ...]` sweeping context-model orders. The compute axis is wall-clock `seconds`.

- [ ] **Step 1: Write the header**

`include/seqbench/sweep.hpp`:

```cpp
#pragma once
#include "seqbench/byte_span.hpp"
#include "seqbench/model.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace seqbench {

struct SweepPoint {
  int knob = 0;
  double bpb = 0.0;
  double seconds = 0.0;  // wall-clock compute proxy for v1
};

std::vector<SweepPoint> run_sweep(
    const std::function<std::unique_ptr<Model>(int)>& factory,
    const std::vector<int>& knobs, ByteSpan data);

void write_csv(const std::vector<SweepPoint>& pts, const std::string& path);

}  // namespace seqbench
```

- [ ] **Step 2: Write the failing test**

`tests/sweep_test.cpp`:

```cpp
#include "test_util.hpp"
#include "seqbench/sweep.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace seqbench;

static void test_sweep_monotone_on_toy() {
  std::vector<int> knobs = {0, 1, 2, 3};
  std::vector<SweepPoint> pts =
      run_sweep(make_context_model, knobs, toy_corpus());
  CHECK(pts.size() == 4);
  CHECK(pts[0].knob == 0);
  CHECK(pts[3].knob == 3);
  // bpb improves (drops) as order rises on the repetitive toy corpus.
  CHECK(pts[3].bpb < pts[0].bpb);
}

static void test_write_csv() {
  std::vector<SweepPoint> pts = {{0, 4.5, 0.01}, {1, 3.0, 0.02}};
  const std::string path = "/tmp/seqbench_sweep_test.csv";
  write_csv(pts, path);
  std::ifstream f(path);
  std::string header;
  std::getline(f, header);
  CHECK(header == "knob,bpb,seconds");
  std::string line;
  std::getline(f, line);
  CHECK(line.rfind("0,", 0) == 0);
  f.close();
  std::remove(path.c_str());
}

int main() {
  RUN(test_sweep_monotone_on_toy);
  RUN(test_write_csv);
  return test_summary();
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `make test`
Expected: link error (undefined reference to `run_sweep` / `write_csv`).

- [ ] **Step 4: Write the implementation**

`src/sweep.cpp`:

```cpp
#include "seqbench/sweep.hpp"
#include "seqbench/metric.hpp"
#include <cstdio>
#include <stdexcept>

namespace seqbench {

std::vector<SweepPoint> run_sweep(
    const std::function<std::unique_ptr<Model>(int)>& factory,
    const std::vector<int>& knobs, ByteSpan data) {
  std::vector<SweepPoint> pts;
  pts.reserve(knobs.size());
  for (int k : knobs) {
    std::unique_ptr<Model> m = factory(k);
    BpbResult r = run_adaptive(*m, data);
    pts.push_back(SweepPoint{k, r.bpb(), r.seconds});
  }
  return pts;
}

void write_csv(const std::vector<SweepPoint>& pts, const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("write_csv: cannot open " + path);
  std::fprintf(f, "knob,bpb,seconds\n");
  for (const SweepPoint& p : pts)
    std::fprintf(f, "%d,%.6f,%.6f\n", p.knob, p.bpb, p.seconds);
  std::fclose(f);
}

}  // namespace seqbench
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: `sweep_test` prints `OK`; `ALL TESTS PASSED`.

- [ ] **Step 6: Write the sweep CLI**

`tools/sweep_cli.cpp`:

```cpp
#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include "seqbench/sweep.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace seqbench;

static void usage() {
  std::fprintf(stderr,
               "usage: sweep <corpus> <out.csv> <order0> [order1 ...]\n"
               "  corpus: toy | <path>\n"
               "  sweeps context-model orders; writes knob,bpb,seconds CSV\n");
}

int main(int argc, char** argv) {
  if (argc < 4) { usage(); return 2; }
  std::string cspec = argv[1];
  std::string out = argv[2];
  std::vector<int> knobs;
  for (int i = 3; i < argc; ++i) knobs.push_back(std::atoi(argv[i]));

  std::unique_ptr<Corpus> corpus;
  ByteSpan span;
  if (cspec == "toy") {
    span = toy_corpus();
  } else {
    corpus = std::make_unique<Corpus>(cspec);
    span = corpus->bytes();
  }

  std::vector<SweepPoint> pts = run_sweep(make_context_model, knobs, span);
  write_csv(pts, out);
  for (const SweepPoint& p : pts)
    std::printf("order=%d bpb=%.4f seconds=%.3f\n", p.knob, p.bpb, p.seconds);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
```

- [ ] **Step 7: Build and run the sweep CLI**

Run: `make all && make test`
Expected: `sweep` builds; `ALL TESTS PASSED`.

Run: `./tools/sweep toy /tmp/toy_sweep.csv 0 1 2 3 4`
Expected: five lines with bpb dropping as order rises, then `wrote /tmp/toy_sweep.csv`.

- [ ] **Step 8: Commit**

```bash
git add include/seqbench/sweep.hpp src/sweep.cpp tools/sweep_cli.cpp tests/sweep_test.cpp
git commit -m "Add scaling-sweep runner and sweep CLI"
```

---

## Final verification (after all tasks)

- [ ] Run the full suite: `make clean && make test` -> `ALL TESTS PASSED`.
- [ ] Build tools: `make all` -> `bench` and `sweep` exist under `tools/`.
- [ ] (Optional, needs network) `make data/enwik8`, then `./tools/bench ctx:0 data/enwik8` and `./tools/bench ctx:3 data/enwik8`; confirm order-0 is ~4.5 to 5.5 bpb and order-3 is clearly lower. Re-run `make test` so the enwik8 integration assertions execute instead of skipping.

## Plan self-review notes

- **Spec coverage:** Model contract (Task 1), bits-per-byte metric with log-sum-exp and Kahan (Task 2), the two measurement protocols (Task 3), order-N context baseline with order-0 as N=0 and `log(count+alpha)` logits (Task 4), mmap corpus + 90/5/5 + toy + `make data/enwik8` fetch (Tasks 1 and 5), induction + parity diagnostics with entropy floors and the "diagnostics discriminate" property as a test (Task 7), scaling-sweep runner emitting CSV with wall-clock as the compute axis (Task 8), bench CLI (Task 6), tests at every task. All spec components are covered.
- **Deferred per spec (not in this plan):** autodiff, neural architectures, BLAS/GPU, real arithmetic coder, context-mixing baseline, MQAR/Dyck diagnostics, in-C++ plotting.
- **Refinement vs spec:** the spec said the context model "partially succeeds on induction." This plan makes induction a positive control where an order>=1 model captures a clear fraction (assertion: fraction > 0.3) while parity is the negative control (fraction < 0.1). Same intent (the diagnostics discriminate architecture classes), with robust thresholds rather than fragile exact floors.
- **Type consistency:** `ByteSpan`, `Model`, `BpbResult`, `logit_bits`, `run_adaptive`, `make_context_model`, `Diagnostic`, `DiagResult`, `SweepPoint` are used with identical signatures across all tasks.
