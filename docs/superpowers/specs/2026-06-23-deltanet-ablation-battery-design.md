# DeltaNet ablation and layer battery design (sub-project 2)

## Goal

Find which inductive biases carry the stacked-DeltaNet win, cheaply, by
ablating one bias at a time and ranking the bits-per-byte effect. The winning
configuration is what later gets the full 44k-step asymptote run on the GPU.
This follows the project's cheap-signal-first method: screen many variants at a
small budget, then scale the winner.

## Scope

In scope:
- A refactor of the fast-weights mixing into a single shared per-step primitive,
  so the batched training forward and the online eval adapter run the SAME
  recurrence code (no train/eval drift as toggles are added).
- Five inductive-bias toggles, each a `dn::Config` flag defaulting to the
  current behavior: write gate, delta-rule-vs-Hebbian, learned lambda, no-decay
  (lambda=1), key normalization, and the per-block MLP.
- CLI flags and run-record provenance for each toggle.
- The screening battery: baseline plus one run per toggle plus a 5/6/8 layer
  sweep, all at d=64, seq_len 128, about 6000 steps, single seed.

Out of scope (deferred to separate follow-ups):
- Multi-head mixing (a structural reshape of the fast-weight matrix).
- Width sweep (d=192/256).
- The deferred GPU-enablement items (tracked in the checkpointing spec).

## Architecture: shared mix step primitive

The drift-prone part is the stateful delta-rule recurrence, currently written
twice: batched in `FWMixImpl::forward` (a loop over T) and per-byte in
`DeltaNetEval::observe`. Unify it into one method:

```
torch::Tensor FWMixImpl::step(torch::Tensor& W, torch::Tensor x);
// W: [B, d, d], mutated in place. x: [B, d]. returns o: [B, d] (post-wo).
```

- `forward(h /*[B,T,d]*/)`: `W = zeros({B,d,d}); for t: out_t = step(W, h.select(1,t)); stack -> [B,T,d]`.
- `DeltaNetEval` holds each layer's `W` as `[1,d,d]` and calls the SAME `step`
  per byte with `x` shaped `[1,d]`, then squeezes. One code path.

Inside `step` (this is where the gate / delta / lambda / key-norm toggles live):
```
k = wk(x); if (normalize_keys) k = F::normalize(k, dim=-1)
v = wv(x); q = wq(x)
beta   = use_gate ? sigmoid(wbeta(x)) : ones({B,1})
lambda = no_decay ? 1.0 : (learn_lambda ? sigmoid(lambda_logit) : fixed_lambda)
Wk = bmm(W, k.unsqueeze(2)).squeeze(2)
e  = use_delta ? (v - Wk) : v
W  = lambda * W + bmm((beta * e).unsqueeze(2), k.unsqueeze(1))
o  = bmm(W, q.unsqueeze(2)).squeeze(2)
return wo(o)
```

The MLP sublayer is stateless and stays batched in `BlockImpl::forward` (its
per-position application in the eval adapter is unchanged); it only gains a
`use_mlp` guard in both places. So the shared primitive covers the recurrence
(the part with state), and the MLP stays efficient (batched at train time)
rather than being pulled into the per-timestep loop.

`lambda_logit` is a per-`FWMix` scalar parameter, constant-initialized to
`logit(fixed_lambda)` so it consumes no RNG at construction (the existing
Linear parameter draws are unchanged), and it is only read when `learn_lambda`
is set.

## Toggles

`dn::Config` gains six bools, each defaulting to the current behavior so the
default config reproduces today's DeltaNet:

| Flag | default | ablation effect | lives in |
|------|---------|-----------------|----------|
| `use_gate` | true | false: beta = 1 (no learned write gate) | step() |
| `use_delta` | true | false: pure Hebbian, e = v (drop the -Wk term) | step() |
| `learn_lambda` | false | true: lambda = sigmoid(per-layer param), init 0.99 | step() |
| `no_decay` | false | true: lambda = 1.0 (no forgetting; overrides the others) | step() |
| `normalize_keys` | true | false: raw keys | step() |
| `use_mlp` | true | false: block is mix-only (MLP sublayer skipped) | Block |

When `use_mlp` is false, the MLP submodules (`ln2`, `fc1`, `fc2`) are not
constructed, and neither `BlockImpl::forward` nor the eval adapter applies them.

## Defaults preserve current behavior

With all flags at their defaults, the model is the current stacked DeltaNet. The
guard is the existing `arch/deltanet/deltanet_test.cpp` staying green after the
refactor: `test_overfit_tiny` (loss to about 0.0003), `test_train_eval_consistency`
(train_bits and eval_bits match), and `test_deterministic`. The screening
baseline is a fresh d=64 run, so exact bit-for-bit equality with the merged
d=128 result is not required; behavioral correctness (trains, train==eval) is.

## CLI and provenance

`train_deltanet` gains one flag per toggle:
`--no-gate`, `--hebbian`, `--learn-lambda`, `--no-decay`, `--no-key-norm`,
`--no-mlp`. Each sets the corresponding `Config` bool. Each toggle is written
into the run's `config` map (so `runs/results.jsonl` records exactly which
ablation produced each number), which also means the runner's config-derived
fingerprint distinguishes ablation configs and a resume cannot cross configs.

The fast-weights and gru trainers are not changed; these toggles are DeltaNet
specific.

## Tests

- Defaults unchanged: `deltanet_test` passes as-is after the refactor.
- Per-toggle correctness: a parametrized test iterates the single-flag configs
  (`no_gate`, `hebbian`, `learn_lambda`, `no_decay`, `no_key_norm`, `no_mlp`)
  and for each asserts (a) trainability: a short overfit run drives the loss
  clearly below the naive 8.0 bits (the variant learns), and (b) consistency:
  `train_bits` and `eval_bits` agree within the existing tolerance (the shared
  `step` keeps the online adapter faithful for every toggle). This is the test
  that protects against eval-adapter drift across the whole battery.

These remain `arch/deltanet` libtorch tests; the dependency-free bench core and
`make test` are untouched.

## The battery

After the build, run at d=64, seq_len 128, about 6000 steps, batch 32, seed 1,
on enwik8 (the same corpus split the runner uses). About 0.5 hours each on CPU,
about 5 hours for the full set, run serially. Use `--ckpt-dir` so each run is
crash-safe.

Runs:
1. baseline (all defaults)
2. `--no-gate`
3. `--hebbian`
4. `--learn-lambda`
5. `--no-decay`
6. `--no-key-norm`
7. `--no-mlp`
8. layer sweep: `--n-layers 5`, `--n-layers 6`, `--n-layers 8` (defaults otherwise)

Analysis: each run appends a record to `runs/results.jsonl` with its toggle in
`config`. Compare `val_bpb` against the baseline; the toggle whose removal hurts
most is the bias that matters most, and the layer sweep shows depth scaling at
d=64. The best configuration found is the candidate for the full d=128 asymptote
run on the GPU. A short markdown summary of the ranking goes in
`arch/deltanet/README.md`.

## After this lands

Rank the biases from the battery, pick the winning configuration, and run the
resumable 44k-step asymptote on the GPU on that configuration. Multi-head and
the width sweep remain available as later follow-ups.
