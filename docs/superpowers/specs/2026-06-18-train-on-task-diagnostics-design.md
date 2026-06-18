# Train-on-task capability diagnostics

**Date:** 2026-06-18
**Status:** Design approved, pending spec review
**Author:** Alexander Towell (lex@metafunctor.com)

## Motivation

The first enwik8 run of the learned fast-weights model returned `induction=0` and
`parity=0`, but those were artifacts, not findings: the diagnostics use bytes 0-15 and 0/1
(about 1.1% of enwik8), so a text-trained model meets them as out-of-vocabulary, and more
fundamentally, running a foreign-trained model over a synthetic stream tests *transfer*,
not whether the architecture has the *inductive bias* for the capability.

This spec reframes a diagnostic as a **capability probe**: fit the architecture on samples
from a task generator, then score it on a held-out test stream. A finite-order count model
cannot learn parity (no unbounded state) or in-context recall of a fresh per-sequence
mapping (its global counts conflate sequences and it cannot forget); a recurrent
fast-weights model can. That difference is the discriminating signal the bench exists to
surface, and fitting each model on the task's own bytes makes the OOV problem vanish.

## Win condition

By the end of this project:

- The induction and parity diagnostics are genuine capability tasks with per-sequence
  samplers, held-out test streams, and exact entropy floors.
- `train_fw --task {enwik8|parity|induction}` trains the learned model on a task and records
  its fraction-captured on a held-out test stream.
- The context-model baseline can be scored on the same test streams (adaptively).
- We can answer cleanly: does fast-weights capture in-context induction and state-tracking
  that a finite-order n-gram cannot?

## Protocol

A diagnostic is a **fit-then-score** capability probe:

- **Fit:** adaptive models (the context model) fit online by running over the stream;
  trained models (fast-weights) train on task samples by BPTT, then freeze.
- **Score:** run the (fitted) model over a held-out test stream with `run_adaptive` /
  `score_diagnostic` and report bits-per-byte plus fraction-captured against the entropy
  floor. A trained recurrent model's state still updates online during this scoring pass
  with frozen parameters, and that online state is exactly how it performs in-context
  inference. So the existing scoring machinery is reused unchanged; the fix is what the
  model is fit on, and the generator design.

Because each model is fit on the task's own bytes, the out-of-vocabulary confound is gone.

## Task generators (bench core, dependency-free)

Both generators live in the diagnostics module and expose two entry points: sample one
sequence of length T (for batched training) and build a held-out test stream plus its
`floor_bpb` and `naive_bpb` (for scoring). Determinism is seeded; train and test use
different seeds so the test set is genuinely held out.

### Parity (state-tracking)
Keep the existing structure: blocks of L random bits (bytes 0/1) followed by their XOR
parity byte. Add a per-sequence sampler so training can batch it. Floor: data bits are
irreducible (1 bit each), the parity byte is free given perfect state; `floor_bpb =
L/(L+1)`, `naive_bpb = 1.0`. A finite-order count model cannot beat naive; a state-tracking
recurrence can approach the floor.

### In-context induction (recall), redefined in place
Each sequence draws a **fresh random injective mapping** from `dict_size` distinct
`key_len`-byte keys (default `key_len = 4`, `dict_size = 16`) to single-byte values from a
small value alphabet. The sequence emits pairs: a randomly chosen key (its `key_len` bytes)
followed by its mapped value byte. The mapping is fixed within a sequence and fresh across
sequences. To predict a value the model must recall the value bound to this key earlier in
*this* sequence.

This redefinition (per-sequence mapping, multi-byte keys) replaces the old fixed-global
single-byte mapping, which was learnable n-gram structure, not in-context recall. Two
properties make it discriminating: a finite-order count model below `key_len` cannot match
the full key, and a global count table conflates the conflicting per-sequence mappings and
cannot forget, whereas a decaying fast-weight memory adapts within each sequence.

Scoring runs continuously over the concatenated test stream with no per-sequence state
reset, and this is intentional: a persistent count table cannot forget the previous
sequence's mapping (so it conflates and fails), while a decaying recurrence forgets across
boundaries and re-infers each sequence's mapping. Test-stream sequences are sized so the
recurrence's effective memory horizon (about `1/(1-lambda)`) fits within a sequence, so
cross-boundary bleed is bounded. The entropy floor is computed per independent sequence
(the ideal); fraction-captured measures how close a model gets to it.

Entropy floor (computed exactly by the generator, since it knows the structure):
`naive_bpb` is the marginal byte entropy of the stream; `floor_bpb` is `naive_bpb` minus
the value-recall savings, where a value byte costs its marginal bits on the first
occurrence of its key in the sequence and zero on subsequent occurrences (key bytes are
treated identically under naive and floor, so they cancel). `fraction_captured =
(naive_bpb - observed_bpb) / (naive_bpb - floor_bpb)`, clamped to [0,1], measures how much
of the in-context recall the model achieved.

## `train_fw --task {enwik8|parity|induction}` (libtorch)

Generalize the runner's data source. For `enwik8` (default) nothing changes. For a task,
training batches are sampled from the task generator instead of enwik8 chunks; after
training, the model is scored on a freshly-seeded held-out test stream via
`score_diagnostic`, and the runner records `val_bpb` (on the task test stream),
`fraction_captured`, and `config.task`. The model byte-vocabulary is still the full 256, so
no model change is needed.

## Update the existing induction diagnostic test

The current bench test asserts the order-1 context model *captures* (fixed-mapping)
induction. Under the corrected in-context definition the context model **cannot** capture
per-sequence novel mappings with multi-byte keys, so the test is updated to assert the
context model *fails* in-context induction (low fraction-captured). This is the correct
discrimination and a more honest test. The parity test (context model fails parity) is
unchanged in spirit.

## Testing

- Generators are deterministic per seed; train and test streams are disjoint (different
  seeds).
- Entropy floors are correct: a hand-constructed tiny parity and induction sequence yields
  the floor the generator reports; `naive_bpb >= floor_bpb`; fraction-captured of a uniform
  model is about 0 and of an oracle is about 1.
- The redefined induction diagnostic: a finite-order context model scores low
  fraction-captured (it cannot do per-sequence multi-byte-key recall); this replaces the
  old assertion.
- A fast train-on-task smoke (`train_fw --task parity`, few steps, tiny config) trains and
  records a well-formed run record (machinery correctness, not the research result).
- No test asserts that fast-weights beats the context model on the tasks; that is the
  recorded capability experiment that follows.

## Non-goals

No change to the enwik8 bits-per-byte path; no new model architecture; no change to the
scoring machinery (`run_adaptive` / `score_diagnostic` reused as-is). The actual capability
*runs* (train fast-weights on parity and on induction, score the context model on the same
tasks, compare) are the follow-up experiment, recorded to `runs/results.jsonl`.

## Future directions (context, not in scope)

If fast-weights cleanly wins the capability probes where n-grams fail, that is the
evidence that motivates scaling the architecture (capacity, gates, layers) on the bpb
metric, and adding further capability tasks (copying, sorting, modular arithmetic, long
-range needle-in-haystack) on the same fit-then-score footing.
