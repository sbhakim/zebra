"""Aggregate only complete run artifacts into machine-readable manuscript evidence."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from zebra.eval.contrasts import Contrast, paired_contrast
from zebra.eval.controls import PRIMARY_STRATEGIES
from zebra.eval.metrics import recovery

SCALAR_METRICS = (
    "auc",
    "f1",
    "final_bwt_auc",
    "mean_stage_bwt_auc",
    "plasticity_auc",
    "stability_auc",
    "ps_auc",
)


def _interval(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    mean = float(np.mean(array))
    if n < 2:
        return {"mean": mean, "low": mean, "high": mean, "n": n}
    half = float(t.ppf(0.975, n - 1) * np.std(array, ddof=1) / math.sqrt(n))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": n}


def load_complete_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        payload = json.loads(path.read_text())
        complete = payload.get("completed_domains") == len(payload.get("order", []))
        if "summary" not in payload or not complete:
            raise ValueError(f"incomplete or malformed run artifact: {path}")
        payload["_path"] = str(path)
        runs.append(payload)
    if not runs:
        raise FileNotFoundError(f"no complete run artifacts in {root}")
    return runs


def aggregate(
    runs: list[dict[str, Any]], *, require_primary: bool = True
) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for run in runs:
        strategy = str(run["strategy"])
        by_strategy[strategy].append(run)
        key = (str(run["ordering"]), int(run["seed"]), strategy)
        if key in by_key:
            raise ValueError(f"duplicate paired run: {key}")
        by_key[key] = run
    if require_primary:
        missing = sorted(set(PRIMARY_STRATEGIES) - set(by_strategy))
        if missing:
            raise ValueError(f"primary evidence incomplete; missing strategies: {missing}")
        pair_prefixes = sorted(
            {(str(run["ordering"]), int(run["seed"])) for run in runs}
        )
        for ordering, seed in pair_prefixes:
            absent = [
                strategy
                for strategy in PRIMARY_STRATEGIES
                if (ordering, seed, strategy) not in by_key
            ]
            if absent:
                raise ValueError(
                    "unpaired primary evidence for "
                    f"{ordering}/seed-{seed}; missing strategies: {absent}"
                )

    summary: dict[str, Any] = {}
    for strategy, strategy_runs in sorted(by_strategy.items()):
        summary[strategy] = {
            metric: _interval([float(run["summary"][metric]) for run in strategy_runs])
            for metric in SCALAR_METRICS
        }
        attacks = sorted(strategy_runs[0]["summary"]["auc_by_attack"])
        if any(
            sorted(run["summary"]["auc_by_attack"]) != attacks
            for run in strategy_runs
        ):
            raise ValueError(f"inconsistent attack strata for strategy {strategy}")
        summary[strategy]["auc_by_attack"] = {
            attack: _interval(
                [float(run["summary"]["auc_by_attack"][attack]) for run in strategy_runs]
            )
            for attack in attacks
        }

    recovery_values: list[float] = []
    recovery_failures = 0
    recovery_by_attack: dict[str, list[float]] = defaultdict(list)
    recovery_attack_failures: dict[str, int] = defaultdict(int)
    pair_prefixes = sorted({(str(run["ordering"]), int(run["seed"])) for run in runs})
    for ordering, seed in pair_prefixes:
        needed = [
            by_key.get((ordering, seed, name)) for name in ("no-cl", "replay", "zebra")
        ]
        if all(run is not None for run in needed):
            sequential, replay_run, zebra = needed
            assert sequential is not None and replay_run is not None and zebra is not None
            try:
                recovery_values.append(
                    recovery(
                        float(zebra["summary"]["auc"]),
                        float(sequential["summary"]["auc"]),
                        float(replay_run["summary"]["auc"]),
                    )
                )
            except ValueError:
                recovery_failures += 1
            attacks = sorted(sequential["summary"]["auc_by_attack"])
            if any(
                sorted(run["summary"]["auc_by_attack"]) != attacks
                for run in (replay_run, zebra)
            ):
                raise ValueError(f"paired attack strata differ for {ordering}/seed-{seed}")
            for attack in attacks:
                try:
                    value = recovery(
                        float(zebra["summary"]["auc_by_attack"][attack]),
                        float(sequential["summary"]["auc_by_attack"][attack]),
                        float(replay_run["summary"]["auc_by_attack"][attack]),
                    )
                    recovery_by_attack[attack].append(value)
                except ValueError:
                    recovery_attack_failures[attack] += 1

    ordering_summary: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for ordering, _ in pair_prefixes:
        if ordering in ordering_summary:
            continue
        ordering_summary[ordering] = {}
        for strategy in sorted(by_strategy):
            selected = [
                run
                for run in by_strategy[strategy]
                if str(run["ordering"]) == ordering
            ]
            if selected:
                ordering_summary[ordering][strategy] = {
                    metric: _interval(
                        [float(run["summary"][metric]) for run in selected]
                    )
                    for metric in SCALAR_METRICS
                }

    attack_recovery_summary = {
        attack: {
            "estimate": (
                _interval(values)
                if values and recovery_attack_failures[attack] == 0
                else None
            ),
            "valid_only": _interval(values) if values else None,
            "reference_failures": recovery_attack_failures[attack],
        }
        for attack, values in sorted(recovery_by_attack.items())
    }
    for attack, count in recovery_attack_failures.items():
        attack_recovery_summary.setdefault(
            attack,
            {"estimate": None, "valid_only": None, "reference_failures": count},
        )
    return {
        "schema": 1,
        "runs": len(runs),
        "strategies": summary,
        "by_ordering": ordering_summary,
        "zebra_recovery": (
            _interval(recovery_values)
            if recovery_values and recovery_failures == 0
            else None
        ),
        "zebra_recovery_valid_only": (
            _interval(recovery_values) if recovery_values else None
        ),
        "zebra_recovery_reference_failures": recovery_failures,
        "zebra_recovery_by_attack": attack_recovery_summary,
    }


_DIGIT_WORDS = str.maketrans(
    {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
     "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
)


REPORTED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("replay", "lwf"),
    ("replay", "si"),
    ("replay", "no-cl"),
    ("lwf", "no-cl"),
    ("si", "no-cl"),
    ("zebra", "no-cl"),
    ("zebra", "zebra-shuffled"),
    ("zebra-replay", "replay"),
)
CONTRAST_METRICS: tuple[str, ...] = ("auc", "final_bwt_auc")


def build_contrasts(runs: list[dict[str, Any]]) -> list[Contrast]:
    """Every reported contrast, over the cells both strategies completed.

    Cells are keyed by (ordering, seed); a pair is dropped only if one side is
    missing, and a contrast is skipped entirely rather than computed on a
    partially overlapping set.
    """
    cells: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        key = (str(run["ordering"]), int(run["seed"]))
        cells.setdefault(key, {})[str(run["strategy"])] = run["summary"]

    out: list[Contrast] = []
    for left, right in REPORTED_CONTRASTS:
        shared = sorted(k for k, v in cells.items() if left in v and right in v)
        if len(shared) < 3:
            continue
        for metric in CONTRAST_METRICS:
            out.append(
                paired_contrast(
                    np.array([cells[k][left][metric] for k in shared], dtype=np.float64),
                    np.array([cells[k][right][metric] for k in shared], dtype=np.float64),
                    left=left,
                    right=right,
                    metric=metric,
                )
            )
    return out


def _macro_tag(value: str) -> str:
    """Strategy or metric name to a legal TeX control-sequence fragment.

    Digits are not letters in TeX, so ``zebra-r1`` would silently define
    ``\\ZBZebraR`` and leave ``1Auc`` as stray body text -- a macro that looks
    defined, renders wrong, and never errors. Digits become words.
    """
    return re.sub(r"[^A-Za-z0-9]", "", value.title()).translate(_DIGIT_WORDS)


def write_report(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results_summary.json").write_text(json.dumps(payload, indent=2))
    with (output / "results_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("strategy", "metric", "mean", "low", "high", "n"))
        for strategy, metrics in payload["strategies"].items():
            for metric in SCALAR_METRICS:
                estimate = metrics[metric]
                writer.writerow(
                    (
                        strategy,
                        metric,
                        estimate["mean"],
                        estimate["low"],
                        estimate["high"],
                        estimate["n"],
                    )
                )
    lines = ["% Generated from complete run artifacts; do not edit."]
    for strategy, metrics in payload["strategies"].items():
        tag = _macro_tag(strategy)
        for metric in SCALAR_METRICS:
            metric_tag = _macro_tag(metric)
            lines.append(
                rf"\def\ZB{tag}{metric_tag}{{{metrics[metric]['mean']:.3f}}}"
            )
    if payload["zebra_recovery"] is not None:
        lines.append(
            rf"\def\ZBZebraRecovery{{{100 * payload['zebra_recovery']['mean']:.1f}}}"
        )
    for attack, recovery_row in payload["zebra_recovery_by_attack"].items():
        estimate = recovery_row["estimate"]
        if estimate is not None:
            lines.append(
                rf"\def\ZBZebraRecovery{_macro_tag(attack)}"
                rf"{{{100 * estimate['mean']:.1f}}}"
            )
    (output / "experiment_results.tex").write_text("\n".join(lines) + "\n")


def write_contrasts(contrasts: list[Contrast], output: Path) -> None:
    """Emit paired contrasts as JSON, CSV, and TeX macros.

    Each contrast yields the effect size, its paired interval, and both tests,
    so a reader can see that a claimed difference and a claimed equivalence rest
    on the same procedure rather than on a threshold chosen after the fact.
    """
    rows = [asdict(c) for c in contrasts]
    (output / "contrasts.json").write_text(json.dumps(rows, indent=2))
    with (output / "contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["left"])
        writer.writeheader()
        writer.writerows(rows)

    lines = ["% Paired contrasts generated from complete run artifacts; do not edit."]
    for c in contrasts:
        tag = f"{_macro_tag(c.left)}Vs{_macro_tag(c.right)}{_macro_tag(c.metric)}"
        lines.append(rf"\def\ZBD{tag}{{{c.mean_difference:+.4f}}}")
        lines.append(rf"\def\ZBD{tag}Low{{{c.low:+.4f}}}")
        lines.append(rf"\def\ZBD{tag}High{{{c.high:+.4f}}}")
        p = c.wilcoxon_p
        shown = "<0.001" if p < 0.001 else f"={p:.3f}"
        lines.append(rf"\def\ZBD{tag}P{{{shown}}}")
    (output / "contrast_results.tex").write_text("\n".join(lines) + "\n")


def main(run_dir: Path, output: Path, *, require_primary: bool = True) -> int:
    runs = load_complete_runs(run_dir)
    payload = aggregate(runs, require_primary=require_primary)
    write_report(payload, output)
    write_contrasts(build_contrasts(runs), output)
    print(f"aggregated {payload['runs']} complete runs into {output}")
    return 0
