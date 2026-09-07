# Artifacts

The aggregated summaries are committed here — about 100 kB, enough to check
every number in the paper without re-running anything. The per-run artifacts
and the corpus are not; they are large and regenerable.

## Committed

| File | What it holds |
|---|---|
| `results_summary.{json,csv}` | AUC, F1, BWT, plasticity, stability, PS per strategy over the grid |
| `contrasts.{json,csv}` | paired differences with Wilcoxon and paired-t p-values |
| `memory.{json,csv}` | resident-byte accounting per strategy, with RFC 7228 pass/fail |
| `memory_probe.json` | the same walk on a single smoke run, as a sanity check |
| `rule_probe.{json,csv}` | standalone AUC per rule per attack (exploratory) |
| `invariant_probe.{json,csv}` | per-domain invariant scores |
| `ood_summary.json` | development-stream AUC, 5/10/15-node domains |
| `ood_heldout.json` | held-out 20-node AUC |
| `*.tex` | the macro files the manuscript reads; every reported value comes from these |

## Not committed, and where to get it

**`runs/` — 432 files, 109 MB.** One JSON per (strategy, ordering, seed) with
the full stage-by-domain AUC and F1 matrices, run partitions, calibration,
provenance and per-domain memory. To regenerate:

```bash
make corpus
python -m zebra.cli experiment      # resumable; the full grid takes hours on one GPU
python -m zebra.cli summarize       # rebuilds everything above from runs/
```

**`ood/` — 45 files, 9 MB.** Network-size shift sweep, same runner with the
20-node domains held out.

**`smoke*/` — 1.7 MB.** Throwaway execution checks, not results.

**`corpus/` — 1200 CSVs, 164 MB.** Third-party and not ours to redistribute:
`uu-core/IoT-Attacks-IDS`, BSD-3, pinned at
`8fa707f14c318495e7550fcdb410544cc0c4ad51`. `make corpus` clones it; the loader
expects `src/attack_data` underneath.

## Caveat

`summarize` rebuilds these summaries from `runs/`. Given a different seed set or
an incomplete grid it refuses rather than writing partial numbers, so a mismatch
against the files here means the inputs differed, not that aggregation drifted.
