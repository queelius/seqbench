# Resumable, device-aware checkpointing (shared runner) design

## Goal

Make long training runs in seqbench crash-safe and resumable, and make the
checkpoint format device-portable so a run started on CPU can be continued on a
GPU later. The capability lives once in the shared `run_experiment` so every
architecture (deltanet, fast-weights, gru) inherits it.

## Motivation

The next experiment is a ~44k-step asymptote run of the stacked DeltaNet at
~2.93 s/step on CPU, about 36 hours of wall-clock. With no checkpointing, a
reboot, an OOM, or an accidental kill loses the entire run. The user also has a
GPU available later, so a checkpoint written on CPU must load and continue on
CUDA without conversion. Both needs are one design problem: capture all mutable
training state and serialize it device-neutrally.

## Scope

This spec covers ONLY the checkpoint/resume/device capability in the shared
runner plus the CLI flags to drive it. It does NOT cover the ablation and
layer-scale battery (write-gate, delta-vs-Hebb, lambda decay, key-norm, MLP,
multi-head), which is a separate sub-project (sub-project 2) brainstormed after
this lands.

## What must be captured for an exact resume

A resume is "exact" when continuing from a checkpoint produces bit-identical
parameters to an uninterrupted run of the same length. That requires four pieces
of mutable state, not just the weights:

1. Model parameters (`model->parameters()`).
2. Optimizer state: Adam's per-parameter first and second moment estimates and
   per-parameter step counts. Dropping these makes the first post-resume steps
   lurch because Adam's bias correction and moment history restart.
3. The step counter (where to resume the loop).
4. RNG state, both:
   - the `std::mt19937_64 rng` that drives `sample_batch` / `sample_task_batch`
     (so the post-resume batch stream continues as if uninterrupted), and
   - torch's global RNG (`torch::get_rng_state()`), so any future stochastic op
     (e.g. dropout in a later architecture) stays on the same trajectory. The
     current DeltaNet forward is deterministic, but restoring it costs nothing
     and keeps the guarantee architecture-independent.

The validation set is rebuilt deterministically from `seed` on every run
(`vr(seed ^ 0xdeadbeef)`), so it does not need to be checkpointed. The best-bpb
scalar IS checkpointed so "best so far" survives a resume.

## RunConfig additions

```cpp
struct RunConfig {
  // existing fields unchanged ...
  std::string ckpt_dir = "";   // "" means checkpointing OFF (default)
  int ckpt_every = 0;          // 0 means "use eval_every"
  bool resume = false;         // --resume: continue from ckpt_dir if present
  std::string device = "cpu";  // "cpu" or "cuda"
};
```

Default behavior is unchanged: with no `--ckpt-dir`, no checkpoints are written
and short ablation runs stay lean. Checkpointing is opt-in per run.

## Checkpoint file layout

A checkpoint is a set of files under `ckpt_dir`, with two prefixes kept by
rolling overwrite: `latest` (most recent) and `best` (lowest val_bpb seen).

```
<ckpt_dir>/latest.model.pt   torch::save(model)        params
<ckpt_dir>/latest.opt.pt     torch::save(opt)          Adam moment + step state
<ckpt_dir>/latest.rng.pt     torch::save(rng_tensor)   torch global RNG state
<ckpt_dir>/latest.mt         (text) mt19937_64 state via operator<<
<ckpt_dir>/latest.meta       (text) scalars: step seed best arch fingerprint
<ckpt_dir>/best.model.pt , best.meta   (only what the final-eval restore needs)
```

Rationale for multiple files over one archive: libtorch's native
`torch::save` / `torch::load` cleanly handle a Module and an Optimizer
individually, and `std::mt19937_64` serializes via its stream operators. Packing
a Module, an Optimizer, scalars, and RNG into a single `serialize::OutputArchive`
forces fragile key namespacing across those writers. The project already favors
structured-over-stringly state; the `.meta` sidecar is a small fixed set of
`key value` lines parsed with `ifstream >>` (no JSON parser exists in the repo,
and none is added here).

The `.meta` fingerprint is a short string of the architecture-shaping config
(for DeltaNet: `d`, `n_layers`, `lambda`), used to reject a resume into a
structurally different model.

## Save logic

A free function in a new header `arch/common/checkpoint.hpp`:

```cpp
template <class ModelT>
void save_checkpoint(const std::string& dir, const std::string& prefix,
                     ModelT& model, torch::optim::Adam& opt,
                     int step, uint64_t seed, double best,
                     const std::string& arch, const std::string& fingerprint,
                     const std::mt19937_64& rng);
```

- Writes each file to `<path>.tmp` then `rename()` over the final name, so a
  crash mid-write cannot corrupt an existing good checkpoint.
- Renames `.meta` LAST. Load treats a checkpoint as valid only if `.meta` is
  present and every file it implies is present; a torn write (meta missing)
  reads as "no checkpoint", falling back to a fresh start rather than crashing.
- For device portability, tensors are snapshotted to CPU before writing, so the
  checkpoint is always CPU-resident and loadable onto any device. On the current
  CPU-only path this is a no-op copy.

Cadence in the loop: save `latest` every `effective_ckpt_every` steps (where
`effective_ckpt_every = ckpt_every > 0 ? ckpt_every : eval_every`) and at the
final step; save/refresh `best` whenever val_bpb improves (replacing today's
`/tmp/seqbench_best.pt` mechanism).

## Resume logic

```cpp
template <class ModelT>
bool load_checkpoint(const std::string& dir, const std::string& prefix,
                     ModelT& model, torch::optim::Adam& opt,
                     int& step, double& best, const std::string& fingerprint,
                     std::mt19937_64& rng, torch::Device dev);
```

In `run_experiment`, after the model, optimizer, and RNGs are constructed:

1. If `rc.resume` and `<ckpt_dir>/latest.meta` exists: load model params,
   optimizer state, `rng`, torch global RNG, `step`, and `best`. Move loaded
   tensors to `dev`.
2. Verify the loaded fingerprint equals the current run's fingerprint. On
   mismatch, print both and exit non-zero rather than silently training a
   different model into the same directory.
3. Set the loop's `start_step = loaded_step + 1`. The loop becomes
   `for (int step = start_step; step <= rc.steps; ++step)`, so `--steps` is the
   TARGET TOTAL: resuming a 10k checkpoint with `--steps 44000` trains 34k more.
   If `loaded_step >= rc.steps`, the loop body runs zero times and the run goes
   straight to final eval and record (a safe no-op resume).

Construction order matters and is preserved: `torch::manual_seed(seed)` then
model init then optimizer construction happen first (as today); the resume load
then overwrites params, optimizer state, and both RNGs with the checkpoint's
post-step-N state, reproducing the uninterrupted trajectory.

## Device handling

- `torch::Device dev = (rc.device == "cuda") ? torch::kCUDA : torch::kCPU;`
- `model->to(dev)` after construction (and after any resume load).
- Each training and eval batch is moved with `.to(dev)` before the forward.
- Checkpoints are written CPU-resident (see Save logic), and `load_checkpoint`
  maps onto `dev`, so a CPU checkpoint resumes on CUDA and vice versa.
- Default `dev` is CPU, so present behavior is unchanged.

Honesty note: there is no GPU on the development box, so the CUDA path is wired
and compiles but is exercised for real only when the GPU is available. The CPU
path is fully tested. The one seam most likely to need attention on first GPU
use is moving Adam's loaded state tensors onto the device; the plan calls this
out explicitly.

## CLI flags (each train_*.cpp arg parser)

Add, alongside the existing flags, one line each:
`--ckpt-dir <dir>`, `--ckpt-every <int>`, `--resume` (no value), `--device <str>`.
These set the corresponding RunConfig fields. All three architectures get the
same four flags since the parsing lives in each `train_*` but sets shared fields.

## Testing

New `arch/common/checkpoint_test.cpp` (libtorch, like the other arch tests):

1. Round-trip: build a tiny module + Adam, run a few steps, `save_checkpoint`,
   perturb the live params and optimizer, `load_checkpoint`, and assert params
   and Adam moment tensors equal the saved values (max-abs-diff 0).
2. Fingerprint rejection: `load_checkpoint` with a mismatched fingerprint returns
   false / signals rejection (so the runner can exit).
3. mt19937_64 and torch-RNG restore: after load, the next N draws from `rng` and
   the next torch random tensor equal those of an uninterrupted reference.

New runner-level determinism test (toy corpus, tiny model, deterministic):

4. Train 2k steps continuously to params P. Separately train 1k steps with
   checkpointing, `load_checkpoint`, train 1k more to params P'. Assert
   `max(abs(P - P')) == 0`. This single equality proves params, optimizer, step,
   and both RNGs are all captured correctly.

The existing dependency-free bench tests are untouched and still never link
libtorch.

## Out of scope

- The ablation / layer-scale battery (sub-project 2).
- Learning-rate schedules (the runner uses fixed Adam lr; unchanged here).
- Keeping a full history of checkpoints (rolling latest + best only, per the
  approved design).
- A general JSON reader (the `.meta` sidecar is line-based key/value).

## GPU enablement (deferred, verify on real hardware)

The CPU path is implemented and tested. The CUDA path is wired (`--device cuda`,
device-mapped model and batches) but cannot be runtime-verified on the
development box (no GPU). A whole-branch adversarial review confirmed three
CUDA-only items to fix and verify the first time a GPU is online. None affects
the CPU path, and none affects a fresh enwik8 asymptote run on GPU EXCEPT where
noted:

1. `load_checkpoint` (checkpoint.hpp) loads Adam optimizer state CPU-resident;
   on a CUDA resume the moment tensors must be moved onto the device before the
   first `opt.step()`. Needed for resuming a run on GPU.
2. `save_checkpoint` does not snapshot tensors to CPU before writing, so a
   checkpoint written on CUDA is not loadable on a CPU-only machine. Add a CPU
   snapshot on save (or a device arg on the optimizer load) for cross-device
   portability. Needed for CPU-started, GPU-resumed runs.
3. The eval adapters (`DeltaNetEval`, `GRUEval`, `FastWeightsEval`) build their
   recurrent-state and per-step index tensors on CPU with no device awareness,
   so `--device cuda` combined with `--task parity|induction` crashes at the
   first adapter forward (device mismatch). Thread `dev` into `make_adapter` and
   build adapter tensors on `dev`. NOTE: this path is only used for the synthetic
   task diagnostics; the enwik8 asymptote run uses `eval_val_bpb` (forward-based,
   no adapter) and is unaffected.

A fresh enwik8 asymptote run on GPU therefore needs only item 1 if it will be
resumed, and item 2 only for cross-device portability; item 3 is for GPU runs of
the parity/induction probes.

## After this lands

Launch the resumable asymptote run (on the GPU box after the items above, or on
CPU now since the CPU path is complete):
`train_deltanet --corpus data/enwik8 --d 128 --n-layers 4 --steps 44000 \
  --ckpt-dir runs/ckpt/deltanet-asymptote --ckpt-every 2000`
Watch the streaming eval bpb, stop at plateau, then move to sub-project 2.
