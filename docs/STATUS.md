# Implementation and evidence status

Last updated 2026-09-06. “Implemented” below means exercised by tests or smoke
runs; it does not mean the complete 84-run primary grid has finished.

## What the audit changed

- The primary model now receives actual `(10, 14)` sequences. The released
  benchmark reshapes each window to `(1, 140)` and uses a different head; that is
  a reproduction track, not the architecture used for the manuscript's edge-cost
  argument.
- Runs are split 12 training / 4 validation / remainder test. Test files no longer
  control early stopping. The two known 19/21-run anomalies are explicit.
- A single 224-byte min--max map is fitted on first-domain training runs and
  frozen; held-out and future values remain unclipped. This removes the hidden
  domain-identity oracle and uncounted state required by per-domain scaling.
- Domain orderings are fixed random permutations of identifiers (seeds 11, 29,
  47, 83), never “best/worst” sequences chosen from model performance.
- EWC, SI, LwF, reservoir replay, and temporal VAE replay now execute genuine
  update mechanisms through one tested lifecycle. NFL is excluded until a faithful
  implementation and parity test exist.
- The original conservation, Trickle-decay, and rank-stability claims were not
  observable from the aggregate columns. They were replaced by honestly named
  solicitation-pressure, DIO-regularity, and DIO-dispersion signals.
- ZeBRa uses only `violation -> attack`; rule compliance never asserts benignness.
  Quantile calibration uses benign training windows from the first domain only,
  so future domains cannot leak into its thresholds, and is frozen into each artifact.

## Current byte measurements

These are tensor payloads from instantiated current implementations. Python
object overhead is emitted separately in `generated/memory.json` and must not be
confused with allocator or interpreter peak memory.

| strategy | model B | retained CL B | AdamW B | gradient B | scaler B | rule B |
|---|---:|---:|---:|---:|---:|---:|
| no-CL | 4,248 | 0 | 8,520 | 4,248 | 224 | 0 |
| EWC | 4,248 | 8,496 | 8,520 | 4,248 | 224 | 0 |
| SI | 4,248 | 16,992 | 8,520 | 4,248 | 224 | 0 |
| LwF | 4,248 | 4,248 | 8,520 | 4,248 | 224 | 0 |
| temporal generative replay | 4,248 | 61,648 | 8,520 | 4,248 | 224 | 0 |
| experience replay | 4,248 | 1,136,000 | 8,520 | 4,248 | 224 | 0 |
| ZeBRa | 4,248 | 0 | 8,520 | 4,248 | 224 | 48 |

A 2,000-window replay reservoir stores 1,120,000 feature bytes plus 16,000 label
bytes. Relative to **this repository's** 4,248-byte detector, retained replay is
267.4× larger. This is not the released benchmark model: its executable
architecture has 6,212 parameters, so a paper must never attribute the 267× ratio
to that published detector.

RFC 7228 Class 0 has no numerical RAM threshold and is reported as unknown.
Class 1 and Class 2 are nominal envelopes only; operating system, network stack,
allocator behavior, activations, and deployment framework are not included.

## Evidence produced so far

- All 60 domain folders and disjoint run partitions validate on the local corpus.
- Synthetic rule tests exercise controlled pass/violate cases and NaN behavior;
  they are not selected extrema from the evaluation corpus.
- Paired three-domain no-CL/ZeBRa and two-domain baseline smokes finish and emit
  matrices, checkpoints, provenance, and memory rows. With the fixed first-domain
  scaler and calibration, the latest three-domain ZeBRa smoke has higher final AUC
  and stability but lower plasticity and PS than no-CL. These are execution
  diagnostics, not scientific evidence; they show why a single final metric would
  support an overstated narrative.
- The report layer accepts complete paired artifacts only and computes final AUC,
  F1, final and stage-average BWT, plasticity, stability, PS, attack strata, and
  paired replay-gap recovery.

## Required before filling manuscript results

1. Freeze baseline and ZeBRa hyperparameters using validation-only development
   runs and record the rationale; defaults are currently explicit but not tuned.
2. Commit a clean code state, run all 7 primary methods × 4 orders × 3 seeds, and
   verify deterministic repeats on a sampled pair.
3. Run all five mechanism controls, the 20-node OOD analysis, and counterfactual
   rule probes before making causal claims.
4. Obtain independent simulations or another dataset for confirmatory rule
   validation; current corpus-wide rule exploration consumed that independence.
5. Generate manuscript macros only from the strict complete-run aggregator.
