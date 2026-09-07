# Experimental protocol

Frozen before the prospective grid was run, and unchanged since. The repository
keeps two separate tracks; results from one are not reported as results from the
other.

## Primary track: corrected temporal evaluation

The primary experiment uses the released RPL CSV corpus at commit
`8fa707f14c318495e7550fcdb410544cc0c4ad51` and corrects several problems in the
upstream training scripts.

- A sample is a 10-minute by 14-feature temporal window, not a flattened
  140-feature vector presented to an LSTM as a single timestep.
- Windows use stride 3 and take the label of their final timestep.
- Simulation runs are the split unit, not overlapping windows.
- Runs are partitioned deterministically into training, validation and test
  sets. Validation alone controls early stopping; test results are reporting
  outputs and do not affect training, calibration or ordering.
- One min--max map is fitted on the training runs of the first domain in each
  ordering and frozen for the stream, which avoids a domain-identity oracle and
  per-domain preprocessing state. Values outside its range are left unclipped.
  The 28 float64 extrema occupy 224 bytes, common to every strategy.
- Every strategy uses the same 14-input, 10-hidden-unit, single-layer LSTM with
  a two-class linear head: 1,062 parameters.
- All 60 domains are reported. Version number is also shown as its own stratum,
  since those simulations were generated separately.
- Four fixed random permutations replace the upstream best/worst orderings,
  whose construction depends on measured generalisation. Permutation seeds are
  recorded in every artifact.
- Model seeds are 7, 19 and 43. Comparisons are paired by ordering and seed.

Baselines are sequential fine-tuning, EWC, SI, LwF, experience replay and
temporal generative replay. ZeBRa, ZeBRa with replay, single-rule ablations and
a shuffled-rule control test the mechanism.

## Protocol-rule contract

The three available quantities are protocol-guided constraint signals rather
than formal invariants proven to hold for every deployment, and are named for
what they measure: solicitation pressure, DIO temporal regularity, and DIO
cross-node dispersion. Rank stability, parent optimality and version
monotonicity are not observable in the released aggregate features.

Raw signals are calibrated once on benign training windows from the first domain
in each ordering; no future domain, validation run or test run affects them.
Calibration values are then frozen, written into the run artifact with their
source domain, and reported as fixed metadata bytes rather than as zero.

The constraint loss is one-sided. A strong violation may support an attack
verdict; the absence of one does not imply benign traffic. That asymmetry
matters for blackhole, which is a pre-specified negative control.

## Checks applied to every run

1. The run records corpus commit, configuration, ordering, seed and code commit.
2. Training, validation and test run identifiers are disjoint.
3. A smoke run repeated under the same seed reproduces its metrics.
4. Each continual-learning strategy changes either its loss or its training
   batch after the first domain.
5. Rule calibration and the fixed scaler use first-domain training runs only.
6. The stage-by-domain metric matrix is finite wherever it is defined.
7. Memory values come from the strategy object that run actually used.
8. Generated manuscript macros fail on a missing cell rather than emitting a
   placeholder or reusing an earlier value.

## Memory terminology

Model tensor bytes, retained continual-learning tensor bytes, optimiser tensors
and worst-case gradient payload are reported separately rather than pooled as
"memory". Python and container overhead, the common 224-byte scaler, and static
rule metadata are also separate. Activations, allocator workspace and runtime
peak are excluded: they scale with batch size rather than with strategy. RFC 7228
values are nominal envelopes, not a claim that a detector fits alongside the
operating system and RPL stack.

## Prospective evaluation versus rule confirmation

The corpus-wide standalone rule AUCs are exploratory, since they informed rule
formulation. The frozen grid is prospective and its model fitting is run-level
leakage-safe, but the held-out runs are not independent confirmation of rule
discovery, because the exploratory probe had already inspected this corpus.
Blackhole was expected not to benefit. Confirming that the rules generalise
requires simulations or a corpus they were not developed on.
