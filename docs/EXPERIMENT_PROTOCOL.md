# ZeBRa experimental contract

Status: development protocol, frozen before the full prospective grid.

This repository has two deliberately separate tracks. Numbers from one track
must never be described as numbers from the other.

## Primary track: corrected temporal evaluation

The primary experiment uses the released RPL CSV corpus at commit
`8fa707f14c318495e7550fcdb410544cc0c4ad51`, but corrects several problems in
the upstream training scripts.

- A sample is a 10-minute by 14-feature temporal window, not a flattened
  140-feature vector presented to an LSTM as one timestep.
- Windows use stride 3 and are labelled by their final timestep.
- Simulation runs, never overlapping windows, are the split unit.
- Runs are deterministically partitioned into training, validation, and test
  sets. Validation alone controls early stopping. Test results are reporting
  outputs and never affect training, calibration, or ordering.
- One min--max map is fitted on training runs of the first domain in each
  ordering, then frozen for the whole stream. This avoids a domain-identity
  oracle and per-domain preprocessing state. Values outside its range are not
  clipped. The 28 float64 extrema occupy 224 common bytes.
- Every strategy uses the same 14-input, 10-hidden-unit, single-layer LSTM and
  two-class linear head (1,062 parameters).
- The 60 domains are all reported. Version-number results are also shown as a
  separate stratum because those simulations were generated separately.
- Four fixed random permutations replace the upstream best/worst orderings,
  whose construction depends on measured generalization. The permutation
  seeds are part of every artifact.
- Model seeds are 7, 19, and 43. Comparisons are paired by ordering and seed.

The primary baselines are sequential fine-tuning, EWC, SI, LwF, experience
replay, and temporal generative replay. ZeBRa, ZeBRa plus replay, single-rule
ablations, and a shuffled-rule negative control test the mechanism. NFL is not
named as a baseline until a faithful implementation passes its own parity
tests; a simplified freezing method must not be called NFL.

## Protocol-rule contract

The three available quantities are protocol-guided *constraint signals*, not
formal invariants proven to hold for every deployment. Their names must state
what is measured: solicitation pressure, DIO temporal regularity, and DIO
cross-node dispersion. Rank stability, parent optimality, and version
monotonicity are not observable in the released aggregate features.

Raw signals are calibrated once using benign training windows from the first
domain in each ordering; no future domain, validation run, or test run may affect
them. Calibration values are then frozen, written into the run artifact with the
source domain, and reported as fixed metadata bytes. They are not silently
counted as zero.
Constraint loss is one-sided: strong violations may support an attack verdict,
but absence of a violation never implies benign traffic. This is essential for
blackhole traffic, which is a pre-specified negative control.

## Evidence gates

A result may enter the manuscript only when all of the following hold:

1. The run records corpus commit, configuration, ordering, seed, and code commit.
2. Training, validation, and test run identifiers are disjoint.
3. Repeating a smoke run with the same seed reproduces its metrics.
4. Every continual-learning strategy changes either its loss or its training
   batch after the first domain; renamed fine-tuning is a failed test.
5. Rule calibration and the fixed scaler use only first-domain training runs.
6. The complete stage-by-domain metric matrix is finite where defined.
7. Memory values come from the actual strategy object used by that run.
8. Generated manuscript macros fail on missing cells rather than emitting a
   placeholder or silently reusing an old value.

## Memory terminology

The paper reports model tensor bytes, retained continual-learning tensor bytes,
optimizer tensors, and worst-case gradient payload separately rather than calling
all of them “memory.” Python/container overhead, the common 224-byte scaler, and
static rule metadata are also separate. Activations, allocator workspace, and runtime peak memory are not yet
measured. RFC 7228 values are nominal envelopes, not a claim that an entire
detector fits after the operating system and RPL stack.

## Prospective model evaluation versus rule confirmation

The existing corpus-wide standalone rule AUCs are exploratory because they
influenced rule formulation. The frozen grid is prospective and its model fitting
is run-level leakage-safe, but the held-out runs are not independent confirmation
of rule discovery: the exploratory probe already inspected this corpus. Blackhole
is expected not to benefit. Unexpected version-number gains are treated as
possible shortcut behavior until counterfactual, out-of-distribution, and new-data
checks support a causal protocol interpretation.
