"""Aggregate the replay buffer-capacity sweep.

The primary grid fixes replay at the released capacity of 2000 windows, which
makes the headline retention ratio rest on a number we inherited rather than
one we varied. This module reads the sweep produced by ``zebra sweep-capacity``
and pairs each capacity against the same LwF and no-CL cells the primary grid
already ran, so the frontier becomes a measured curve instead of one point.

Capacity 2000 is not re-run; it is read back from ``generated/runs`` and is the
sweep's own consistency check.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from zebra.eval.contrasts import Contrast, paired_contrast
from zebra.eval.report import (
    CONTRAST_METRICS,
    _interval,
    _macro_tag,
    load_complete_runs,
)

WINDOW_BYTES = 568
"""One stored window and its label: 10 x 14 float32 features plus an int64."""

BASELINES: tuple[str, ...] = ("lwf", "no-cl")

# Reported per arm. Contrasts stay on the two metrics the primary grid tests;
# PS is descriptive here and is carried so the manuscript table needs no gaps.
ARM_METRICS: tuple[str, ...] = (*CONTRAST_METRICS, "ps_auc")


def _cells(runs: list[dict[str, Any]], strategy: str) -> dict[tuple[str, int], Any]:
    return {
        (str(run["ordering"]), int(run["seed"])): run["summary"]
        for run in runs
        if str(run["strategy"]) == strategy
    }


def collect(primary: Path, sweep: Path) -> dict[str, Any]:
    """Read the primary grid and every cap-N directory into one payload."""
    primary_runs = load_complete_runs(primary)
    baseline_cells = {name: _cells(primary_runs, name) for name in BASELINES}
    arms: dict[int, dict[tuple[str, int], Any]] = {2000: _cells(primary_runs, "replay")}

    for directory in sorted(sweep.glob("cap-*")):
        capacity = int(directory.name.removeprefix("cap-"))
        if capacity in arms:
            raise ValueError(f"capacity {capacity} is already in the primary grid")
        arms[capacity] = _cells(load_complete_runs(directory), "replay")

    rows: list[dict[str, Any]] = []
    contrasts: list[Contrast] = []
    for capacity in sorted(arms):
        cells = arms[capacity]
        retained = capacity * WINDOW_BYTES
        row: dict[str, Any] = {
            "capacity": capacity,
            "retained_bytes": retained,
            "n_runs": len(cells),
        }
        for metric in ARM_METRICS:
            row[metric] = _interval([float(s[metric]) for s in cells.values()])
        rows.append(row)

        for name in BASELINES:
            shared = sorted(set(cells) & set(baseline_cells[name]))
            if len(shared) < 3:
                continue
            for metric in CONTRAST_METRICS:
                contrasts.append(
                    paired_contrast(
                        np.array([cells[k][metric] for k in shared], dtype=np.float64),
                        np.array(
                            [baseline_cells[name][k][metric] for k in shared],
                            dtype=np.float64,
                        ),
                        left=f"cap-{capacity}",
                        right=name,
                        metric=metric,
                    )
                )

    baselines = {
        name: {
            metric: _interval([float(s[metric]) for s in baseline_cells[name].values()])
            for metric in CONTRAST_METRICS
        }
        for name in BASELINES
    }
    return {
        "window_bytes": WINDOW_BYTES,
        "arms": rows,
        "baselines": baselines,
        "contrasts": contrasts,
    }


def write(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "window_bytes": payload["window_bytes"],
        "arms": payload["arms"],
        "baselines": payload["baselines"],
        "contrasts": [
            {
                "left": c.left,
                "right": c.right,
                "metric": c.metric,
                "mean_difference": c.mean_difference,
                "low": c.low,
                "high": c.high,
                "wilcoxon_p": c.wilcoxon_p,
                "ttest_p": c.ttest_p,
                "n_pairs": c.n_pairs,
            }
            for c in payload["contrasts"]
        ],
    }
    (output / "capacity_sweep.json").write_text(json.dumps(serialisable, indent=2))

    with (output / "capacity_sweep.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("capacity", "retained_bytes", "n_runs", "metric", "mean", "low", "high")
        )
        for row in payload["arms"]:
            for metric in ARM_METRICS:
                estimate = row[metric]
                writer.writerow(
                    (
                        row["capacity"],
                        row["retained_bytes"],
                        row["n_runs"],
                        metric,
                        estimate["mean"],
                        estimate["low"],
                        estimate["high"],
                    )
                )

    lines = ["% Capacity sweep generated from complete run artifacts; do not edit."]
    for row in payload["arms"]:
        tag = _macro_tag(f"cap-{row['capacity']}")
        for metric in ARM_METRICS:
            lines.append(rf"\def\ZBC{tag}{_macro_tag(metric)}{{{row[metric]['mean']:.3f}}}")
        # Plain comma, matching the other emitters. The manuscript adds any
        # math-mode grouping it needs; emitting {,} here nests braces and breaks
        # the macro-legality check that already caught one stray-brace bug.
        lines.append(rf"\def\ZBC{tag}Bytes{{{row['retained_bytes']:,}}}")
    for c in payload["contrasts"]:
        tag = f"{_macro_tag(c.left)}Vs{_macro_tag(c.right)}{_macro_tag(c.metric)}"
        lines.append(rf"\def\ZBCD{tag}{{{c.mean_difference:+.4f}}}")
        p = c.wilcoxon_p
        lines.append(rf"\def\ZBCD{tag}P{{{'<0.001' if p < 0.001 else f'={p:.3f}'}}}")
    (output / "capacity_results.tex").write_text("\n".join(lines) + "\n")


def main(primary: Path, sweep: Path, output: Path) -> int:
    payload = collect(primary, sweep)
    write(payload, output)
    print(f"aggregated {len(payload['arms'])} capacity arms into {output}")
    return 0
