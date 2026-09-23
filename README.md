# ZeBRa: Byte-Level Memory Accounting for Continual Intrusion Detection

ZeBRa measures what a continual-learning strategy actually keeps resident
between domains, in bytes, and re-runs a released RPL intrusion-detection
benchmark under a corrected protocol. Seven strategies are compared on one
shared detector across 60 domains, four orderings and three paired seeds.

## What this provides

- Resident-byte accounting that walks the tensors a strategy holds, rather than
  multiplying a sample count by an assumed element size.
- A corrected temporal protocol: run-disjoint splits, validation-only early
  stopping, true `(10, 14)` windows, and orderings drawn from domain
  identifiers instead of observed performance.
- Seven strategies behind one interface — no-CL, EWC, SI, LwF, reservoir
  replay, temporal generative replay, and ZeBRa — plus a shuffled-rule control.
- Paired contrasts over the grid, with Wilcoxon signed-rank alongside a paired
  t so that disagreement between them is visible.

Every number in the paper comes from an artifact under `generated/`.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+. Torch, NumPy, pandas, scikit-learn, SciPy.

## Data

The RPL corpus is third-party (`uu-core/IoT-Attacks-IDS`, BSD-3) and is not
vendored. Pinned to `8fa707f14c318495e7550fcdb410544cc0c4ad51`; `make corpus`
fetches it, or place `src/attack_data` at
`corpus/IoT-Attacks-IDS/src/attack_data` yourself.

## Artifacts

The aggregated results are committed under `generated/` — roughly 100 kB, which
is enough to check every number in the paper without re-running the grid. The
828 per-run files (225 MB) and the corpus are not; `generated/README.md` says
what they are and how to regenerate them.

## Quick start

```bash
make lint typecheck test        # test-integration additionally needs the corpus
make icc-report                 # memory accounting, rule probes, paired grid
make sweep sweep-report         # replay buffer-capacity sweep (needs the corpus)
python -m zebra.cli experiment  # runs or resumes the full grid
python -m zebra.cli summarize
```

`summarize` refuses incomplete artifacts, missing primary strategies and
unpaired ordering–seed cells, so a partial grid cannot reach the manuscript.
Everything under `generated/` is derived and gitignored.

## Layout

```text
src/zebra/data/          run splits, normalisation, temporal windows
src/zebra/invariants/    raw signals and training-only calibration
src/zebra/strategies/    the seven arms, the byte-efficient ones, and the controls
src/zebra/eval/          orderings, runner, metrics, controls, aggregation
src/zebra/memory/        tensor residency and RFC 7228 envelopes
docs/                    protocol, decisions, status
```

## Notes

`docs/EXPERIMENT_PROTOCOL.md` has the details. The protocol differs from the
upstream scripts, so these numbers will not line up with the released ones.

ZeBRa keeps no past samples, though it does carry six calibration scalars (48
bytes), and every strategy carries the 224-byte scaler.

The standalone rule AUCs from `probe` are exploratory: the rules were tuned on
the same corpus they are measured on. `smoke` just checks that a run goes
through end to end.

## License

MIT, see [LICENSE](LICENSE). The RPL corpus is third-party and carries its own
licence (BSD-3, `uu-core/IoT-Attacks-IDS`).
