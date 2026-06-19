# Train-on-task Capability Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the bench's diagnostics as train-on-task capability probes: per-sequence in-context induction and state-tracking parity with exact entropy floors, plus a `train_fw --task` mode that fits the learned model on a task and records its fraction-captured.

**Architecture:** The bench gets per-sequence sequence-fillers (`fill_parity`, `fill_induction`) for batched training and a redefined `make_induction` (fresh per-sequence mapping, multi-byte keys, empirically-computed floor) for held-out test streams. `train_fw` gains `--task` to sample training batches from a generator and score the trained model on a held-out test stream via the existing `score_diagnostic`. Scoring machinery is unchanged.

**Tech Stack:** C++17 (bench core, dependency-free), libtorch (the `train_fw` runner), the existing bench.

## Global Constraints

- C++17. The bench core stays dependency-free; its plain Makefile and `make test` never link libtorch.
- libtorch stays confined to `arch/fast-weights/` (CMake build, venv torch prefix `/home/spinoza/venv/lib/python3.12/site-packages/torch/share/cmake`).
- All bench headers under `include/seqbench/`, code in `namespace seqbench`.
- `runs/results.jsonl` is committed; records carry `arch` and (new) `config.task`.
- NO em-dash characters in any committed file (a repo hook rejects them).
- NEVER weaken a test threshold to pass. If a test does not pass with the planned value, STOP and report BLOCKED with the measured numbers.
- The libtorch C++ API targets torch 2.8; adapt a call to the installed API if it does not compile, preserving semantics.
- Every commit message ends with these two trailer lines:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015zkyVS7cs6YxjoVn9HG1yH
  ```

### File map

| File | Change |
|------|--------|
| `include/seqbench/diagnostics.hpp` | redefine `make_induction` signature; add `fill_parity`, `fill_induction` |
| `src/diagnostics.cpp` | redefine `make_induction` (per-sequence mapping, multi-byte keys, empirical floor); add `fill_parity`, `fill_induction`; `make_parity` unchanged |
| `tests/diagnostics_test.cpp` | update the induction test to in-context semantics; add fill + floor tests |
| `arch/fast-weights/train_fw.cpp` | add `--task {enwik8\|parity\|induction}`; task batch sampling; task eval/record; drop the OOV diagnostics from the enwik8 path |

---

## Task 1: Train-on-task generators (bench core)

**Files:**
- Modify: `include/seqbench/diagnostics.hpp`, `src/diagnostics.cpp`, `tests/diagnostics_test.cpp`

**Interfaces:**
- Consumes: `Diagnostic`, `DiagResult`, `score_diagnostic`, `make_context_model`, `Model`
- Produces: `Diagnostic make_induction(uint64_t seed, std::size_t n_sequences, int pairs_per_seq, int dict_size, int key_len)` (per-sequence fresh mapping; exact empirical floor)
- Produces: `void fill_parity(uint64_t seed, uint8_t* out, int T, int block_len)` (one parity sequence of exactly T bytes)
- Produces: `void fill_induction(uint64_t seed, uint8_t* out, int T, int dict_size, int key_len)` (one fresh-mapping induction sequence of exactly T bytes)
- `make_parity` is unchanged.

- [ ] **Step 1: Update the header**

In `include/seqbench/diagnostics.hpp`, replace the `make_induction` declaration line with the new signature and add the fillers. The block currently reads:

```cpp
// (key, value) pairs over a dict_size alphabet; value = perm(key).
Diagnostic make_induction(uint64_t seed, std::size_t n_pairs, int dict_size);
```

Replace it with:

```cpp
// In-context induction: each sequence draws a FRESH random mapping from `dict_size`
// distinct `key_len`-byte keys to single-byte values; emits `pairs_per_seq` (key, value)
// pairs. A value is recall-determined (free at the floor) once its key has appeared
// earlier in the same sequence. The floor is computed empirically from the stream.
Diagnostic make_induction(uint64_t seed, std::size_t n_sequences, int pairs_per_seq,
                          int dict_size, int key_len);

// Fill out[0..T) with one task sequence (fresh per seed), for batched training.
void fill_parity(uint64_t seed, uint8_t* out, int T, int block_len);
void fill_induction(uint64_t seed, uint8_t* out, int T, int dict_size, int key_len);
```

- [ ] **Step 2: Write the failing tests**

Replace the body of `test_context_captures_induction` (and its `RUN` line) in `tests/diagnostics_test.cpp`. Open the file; ensure these includes are present at the top (add any missing): `#include <cstdint>`, `#include <vector>`. Replace the function `test_context_captures_induction` entirely with:

```cpp
// In-context induction (fresh per-sequence mapping, 4-byte keys): a finite-order count
// model cannot do it (it conflates the per-sequence mappings and cannot match full keys).
static void test_context_fails_incontext_induction() {
  Diagnostic d = make_induction(7, 400, 51, 16, 4);
  CHECK(d.naive_bpb >= d.floor_bpb);
  CHECK(d.floor_bpb > 0.0);
  auto m = make_context_model(3);
  DiagResult r = score_diagnostic(*m, d);
  std::printf("    [in-context induction: context o3 fraction=%.4f]\n", r.fraction_captured);
  CHECK(r.fraction_captured < 0.30);  // cannot do in-context recall of novel mappings
}

// Generators are deterministic and the fillers produce exactly T bytes.
static void test_fillers_deterministic() {
  std::vector<uint8_t> a(300), b(300), c(300);
  fill_parity(42, a.data(), 300, 16);
  fill_parity(42, b.data(), 300, 16);
  CHECK(a == b);
  fill_induction(9, c.data(), 300, 16, 4);
  std::vector<uint8_t> e(300);
  fill_induction(9, e.data(), 300, 16, 4);
  CHECK(c == e);
  CHECK(c != a);  // different tasks differ
}
```

Update `main` in `tests/diagnostics_test.cpp`: replace `RUN(test_context_captures_induction);` with:

```cpp
  RUN(test_context_fails_incontext_induction);
  RUN(test_fillers_deterministic);
```

- [ ] **Step 3: Run to verify failure**

Run: `make test`
Expected: `diagnostics_test` fails to compile (the old `make_induction(7, 50000, 16)` is gone; new symbols `fill_parity`/`fill_induction` and the 5-arg `make_induction` are undefined).

- [ ] **Step 4: Implement in `src/diagnostics.cpp`**

Ensure these includes are at the top of `src/diagnostics.cpp` (add any missing): `#include <array>`, `#include <cmath>`, `#include <vector>`. The file already defines an anonymous-namespace `Rng` (splitmix64) with `next()` and `below(int)`; reuse it.

Replace the entire existing `make_induction` function with:

```cpp
Diagnostic make_induction(uint64_t seed, std::size_t n_sequences, int pairs_per_seq,
                          int dict_size, int key_len) {
  Rng rng(seed);
  Diagnostic d;
  std::vector<char> recall;  // parallel to stream: 1 if this byte is a recall-determined value
  for (std::size_t s = 0; s < n_sequences; ++s) {
    // fresh mapping: dict_size keys of key_len bytes, each mapped to a value byte.
    std::vector<std::vector<uint8_t>> keys(dict_size, std::vector<uint8_t>(key_len));
    std::vector<uint8_t> vals(dict_size);
    for (int k = 0; k < dict_size; ++k) {
      for (int j = 0; j < key_len; ++j) keys[k][j] = static_cast<uint8_t>(rng.next() & 0xff);
      vals[k] = static_cast<uint8_t>(rng.next() & 0xff);
    }
    std::vector<char> seen(dict_size, 0);
    for (int p = 0; p < pairs_per_seq; ++p) {
      int k = rng.below(dict_size);
      for (int j = 0; j < key_len; ++j) { d.stream.push_back(keys[k][j]); recall.push_back(0); }
      d.stream.push_back(vals[k]);
      recall.push_back(seen[k]);  // recall-determined iff this key appeared earlier in-seq
      seen[k] = 1;
    }
  }
  // Empirical marginal byte entropy -> exact naive and floor.
  std::array<double, 256> cnt;
  cnt.fill(0.0);
  for (uint8_t b : d.stream) cnt[b] += 1.0;
  double N = static_cast<double>(d.stream.size());
  std::array<double, 256> bits;
  for (int b = 0; b < 256; ++b) bits[b] = cnt[b] > 0.0 ? -std::log2(cnt[b] / N) : 0.0;
  double naive = 0.0, floor = 0.0;
  for (std::size_t i = 0; i < d.stream.size(); ++i) {
    double bi = bits[d.stream[i]];
    naive += bi;
    if (!recall[i]) floor += bi;  // recall-determined values cost 0 at the floor
  }
  d.naive_bpb = N > 0.0 ? naive / N : 0.0;
  d.floor_bpb = N > 0.0 ? floor / N : 0.0;
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
  std::vector<std::vector<uint8_t>> keys(dict_size, std::vector<uint8_t>(key_len));
  std::vector<uint8_t> vals(dict_size);
  for (int k = 0; k < dict_size; ++k) {
    for (int j = 0; j < key_len; ++j) keys[k][j] = static_cast<uint8_t>(rng.next() & 0xff);
    vals[k] = static_cast<uint8_t>(rng.next() & 0xff);
  }
  int pos = 0;
  while (pos < T) {
    int k = rng.below(dict_size);
    for (int j = 0; j < key_len && pos < T; ++j) out[pos++] = keys[k][j];
    if (pos < T) out[pos++] = vals[k];
  }
}
```

- [ ] **Step 5: Run to verify pass**

Run: `make test`
Expected: `diagnostics_test` prints `OK` (including the printed `context o3 fraction`). If `test_context_fails_incontext_induction` does not satisfy `< 0.30`, STOP and report BLOCKED with the printed fraction (do not loosen the threshold).

- [ ] **Step 6: Commit**

```bash
git add include/seqbench/diagnostics.hpp src/diagnostics.cpp tests/diagnostics_test.cpp
git commit -m "Redefine induction as in-context capability task; add task fillers"
```

---

## Task 2: train_fw --task (libtorch) + smoke

**Files:**
- Modify: `arch/fast-weights/train_fw.cpp`

**Interfaces:**
- Consumes: `fill_parity`, `fill_induction`, `make_parity`, `make_induction` (new signature), `score_diagnostic`, `Diagnostic`, `DiagResult`, `fw::FastWeights`, `fw::bpb_loss`, `fw::FastWeightsEval`, `Corpus`, `toy_corpus`, `RunRecord`, `JsonValue`, `append_record`
- Produces: `train_fw` accepting `--task {enwik8|parity|induction}` (default `enwik8`); for tasks it trains on generated batches and records `config.task`, `fraction_captured`.

- [ ] **Step 1: Add the task sampler and arg, restructure the data + eval paths**

In `arch/fast-weights/train_fw.cpp`, add the include `#include "seqbench/diagnostics.hpp"` if not already present (it is, from the old diagnostics eval). Add a task batch sampler near `sample_batch`:

```cpp
static torch::Tensor sample_task_batch(const std::string& task, int B, int T,
                                       std::mt19937_64& rng, int block_len,
                                       int dict_size, int key_len) {
  std::vector<int64_t> buf(static_cast<std::size_t>(B) * T);
  std::vector<uint8_t> seq(static_cast<std::size_t>(T));
  for (int b = 0; b < B; ++b) {
    uint64_t s = rng();
    if (task == "parity") fill_parity(s, seq.data(), T, block_len);
    else fill_induction(s, seq.data(), T, dict_size, key_len);
    for (int t = 0; t < T; ++t) buf[static_cast<std::size_t>(b) * T + t] = seq[t];
  }
  return torch::tensor(buf, torch::kLong).reshape({B, T});
}
```

Add `--task` parsing (default `"enwik8"`) alongside the other args, plus task params with defaults:

```cpp
  std::string task = "enwik8";
  int block_len = 16, dict_size = 16, key_len = 4;
```

In the arg loop add:

```cpp
    else if (a == "--task") task = need("--task");
```

- [ ] **Step 2: Branch the data source and final evaluation on `task`**

Replace the corpus-loading + training-data + final-eval section so that, for a task, data comes from the generator and the final eval scores fraction-captured. Concretely, after arg parsing and `torch::manual_seed(...)`, structure it as:

```cpp
  fw::Config cfg; cfg.d = d; cfg.beta = beta; cfg.lambda = lambda;
  fw::FastWeights model(cfg);
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(lr));

  const bool is_task = (task == "parity" || task == "induction");

  // Data sources.
  std::unique_ptr<Corpus> cptr;
  ByteSpan train, val;
  if (!is_task) {
    ByteSpan full;
    if (corpus == "toy") full = toy_corpus();
    else { cptr = std::make_unique<Corpus>(corpus); full = cptr->bytes(); }
    train = slice(full, 0.0, 0.90);
    val = slice(full, 0.90, 0.95);
    if (train.len <= static_cast<std::size_t>(seq_len + 1) ||
        val.len <= static_cast<std::size_t>(seq_len + 1)) {
      std::fprintf(stderr, "corpus too small for seq_len=%d\n", seq_len);
      return 2;
    }
  }

  auto train_batch = [&](std::mt19937_64& r) {
    return is_task ? sample_task_batch(task, batch, seq_len, r, block_len, dict_size, key_len)
                   : sample_batch(train, batch, seq_len, r);
  };

  std::vector<torch::Tensor> val_set;
  { std::mt19937_64 vr(seed ^ 0xdeadbeefULL);
    for (int i = 0; i < 8; ++i) val_set.push_back(train_batch(vr)); }
  auto eval_val_bpb = [&]() -> double {
    torch::NoGradGuard ng; model->eval();
    double tot = 0.0;
    for (auto& vb : val_set) tot += fw::bpb_loss(model, vb).item<double>();
    model->train();
    return tot / val_set.size();
  };
```

Keep the existing training loop (it uses `sample_batch(train, ...)`) but change the per-step batch to `train_batch(rng)`:

```cpp
    auto xb = train_batch(rng);
```

Replace the final evaluation + record block (everything from `if (best < ...) torch::load(...)` onward) with:

```cpp
  if (best < std::numeric_limits<double>::infinity()) torch::load(model, best_path);

  double train_bpb;
  { torch::NoGradGuard ng; model->eval();
    train_bpb = fw::bpb_loss(model, train_batch(rng)).item<double>(); }

  RunRecord rec;
  rec.arch = "fast-weights-learned";
  rec.version = "v1-deltanet";
  rec.seed = static_cast<long>(seed);
  rec.config["d"] = JsonValue::n(d);
  rec.config["beta"] = JsonValue::n(beta);
  rec.config["lambda"] = JsonValue::n(lambda);
  rec.config["seq_len"] = JsonValue::n(seq_len);
  rec.config["batch"] = JsonValue::n(batch);
  rec.config["steps"] = JsonValue::n(steps);
  rec.config["lr"] = JsonValue::n(lr);
  rec.config["task"] = JsonValue::s(task);
  rec.results["train_bpb"] = train_bpb;

  if (is_task) {
    // Held-out test stream (different seed), scored via the online adapter.
    Diagnostic test = (task == "parity")
        ? make_parity(seed ^ 0x5eedULL, 4000, block_len)
        : make_induction(seed ^ 0x5eedULL, 400, 51, dict_size, key_len);
    fw::FastWeightsEval ev(model, cfg);
    DiagResult r = score_diagnostic(ev, test);
    rec.corpus_name = task + "-test";
    rec.corpus_bytes = test.stream.size();
    rec.results["test_bpb"] = r.observed_bpb;
    rec.results["fraction_captured"] = r.fraction_captured;
    rec.results["floor_bpb"] = test.floor_bpb;
    rec.results["naive_bpb"] = test.naive_bpb;
    std::printf("learned fast-weights task=%s d=%d steps=%d\n", task.c_str(), d, steps);
    std::printf("  test_bpb=%.4f fraction_captured=%.4f (floor=%.4f naive=%.4f) train_bpb=%.4f\n",
                r.observed_bpb, r.fraction_captured, test.floor_bpb, test.naive_bpb, train_bpb);
  } else {
    double val_bpb = eval_val_bpb();
    rec.corpus_name = (corpus == "toy") ? "toy-val" : "enwik8-val";
    rec.corpus_bytes = val.len;
    rec.results["val_bpb"] = val_bpb;
    std::printf("learned fast-weights task=%s d=%d steps=%d\n", task.c_str(), d, steps);
    std::printf("  val_bpb=%.4f train_bpb=%.4f\n", val_bpb, train_bpb);
  }
  append_record(rec, out);
  std::printf("  appended run record to %s\n", out.c_str());
  return 0;
}
```

Remove the old enwik8-path diagnostics block (the lines that created `make_induction(7, 50000, 16)` / `make_parity(7, 4000, 16)` and `ev_ind`/`ev_par` and printed `induction=/parity=`), since it is replaced by the task path above and used the removed `make_induction` signature.

- [ ] **Step 3: Build**

Run:

```bash
cmake --build arch/fast-weights/build -j 2>&1 | tail -5
```

Expected: `train_fw` and `fw_test` build. (`fw_test` does not use the diagnostics, so it is unaffected.) If a compile error references the old `make_induction` call, ensure it was fully removed.

- [ ] **Step 4: Smoke-run the task path (offline, fast)**

Run: `./arch/fast-weights/build/train_fw --task parity --d 32 --seq-len 64 --batch 8 --steps 300 --eval-every 100 --out /tmp/task_smoke.jsonl`
Expected: prints decreasing `train_bpb`, then a summary line with `test_bpb`, `fraction_captured`, `floor`, `naive`, and `appended run record to /tmp/task_smoke.jsonl`.

Run: `tail -1 /tmp/task_smoke.jsonl | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('ok task=',d['config']['task'],'results=',sorted(d['results']))" && rm -f /tmp/task_smoke.jsonl`
Expected: `ok task= parity results= ['floor_bpb', 'fraction_captured', 'naive_bpb', 'test_bpb', 'train_bpb']`.

(This is a wiring smoke at a tiny budget; whether fast-weights actually captures parity is the real experiment, not asserted here.)

- [ ] **Step 5: Commit**

```bash
git add arch/fast-weights/train_fw.cpp
git commit -m "Add train_fw --task mode for capability probes (parity, induction)"
```

---

## Final verification (after all tasks)

- [ ] `make test` (bench suite) -> `ALL TESTS PASSED`, libtorch-free.
- [ ] `cmake --build arch/fast-weights/build -j && ./arch/fast-weights/build/fw_test` -> `OK`.
- [ ] `train_fw --task parity --steps 300 --d 32 --seq-len 64 --batch 8` produces a `parity-test` record with a `fraction_captured`.

## Plan self-review notes

- **Spec coverage:** the fit-then-score protocol via the existing scoring machinery (Task 2 reuses `score_diagnostic`); per-sequence in-context induction with multi-byte keys and an exact empirical floor (Task 1 `make_induction`); per-sequence fillers for batched training (Task 1 `fill_parity`/`fill_induction`); parity kept (`make_parity` unchanged); `train_fw --task` with `config.task` and `fraction_captured` (Task 2); the existing induction test updated to in-context semantics with the context model failing (Task 1). All spec sections covered.
- **Deferred per spec:** no change to the enwik8 bpb path beyond dropping the OOV diagnostics from its record; the actual capability runs (train fast-weights on parity/induction, context baselines) are the follow-up experiment.
- **Callers updated:** the two `make_induction` callers found by grep (`tests/diagnostics_test.cpp`, `arch/fast-weights/train_fw.cpp`) are both updated; `make_parity` signature is unchanged so its callers are untouched.
- **Anti-fudging:** the reality-could-differ threshold (context o3 in-context induction `< 0.30`) carries a BLOCKED-with-numbers instruction and prints the measured value.
