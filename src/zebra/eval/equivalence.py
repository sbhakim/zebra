"""Two one-sided tests for the contrasts we report as null results.

A large signed-rank p-value says the grid found no evidence of a difference.
It does not say the difference is small, and a reader is right to refuse to
read it that way. TOST asks the question we actually mean: is the paired
difference inside a margin we fixed in advance? Rejecting both one-sided nulls
licenses the positive claim; failing to reject leaves the comparison simply
underpowered, which is also worth saying plainly.

Margins are the manuscript's own yardstick, not thresholds picked after seeing
the intervals: one AUC point is the figure the abstract already uses to call
LwF close to replay, and the same bound is applied to backward transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

MARGINS: dict[str, float] = {"auc": 0.01, "final_bwt_auc": 0.01}


@dataclass(frozen=True, slots=True)
class Equivalence:
    """One TOST result, reported with the margin it was tested against."""

    left: str
    right: str
    metric: str
    margin: float
    mean_difference: float
    low: float
    high: float
    p_value: float
    n_pairs: int

    @property
    def equivalent(self) -> bool:
        """The 90% interval lies inside the margin, which is the same test."""
        return self.p_value < 0.05


def equivalence(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    left: str,
    right: str,
    metric: str,
    margin: float,
) -> Equivalence:
    """Test whether the paired difference is bounded by +/- margin."""
    if left_values.shape != right_values.shape:
        raise ValueError("paired equivalence requires matching cell vectors")
    if len(left_values) < 3:
        raise ValueError(f"too few pairs for an equivalence test: {len(left_values)}")
    if margin <= 0:
        raise ValueError("equivalence margin must be positive")

    difference = np.asarray(left_values, dtype=np.float64) - np.asarray(
        right_values, dtype=np.float64
    )
    n = len(difference)
    mean = float(difference.mean())
    standard_error = float(difference.std(ddof=1)) / float(np.sqrt(n))

    # A degenerate difference vector has no scale to test against. Treat it as
    # equivalent only because every pair is literally identical, and say so with
    # a zero p-value rather than dividing by zero.
    if standard_error == 0.0:
        p_value = 0.0 if abs(mean) < margin else 1.0
    else:
        lower = float(t.sf((mean + margin) / standard_error, n - 1))
        upper = float(t.cdf((mean - margin) / standard_error, n - 1))
        p_value = max(lower, upper)

    # The 90% interval is the one that corresponds to two 5% one-sided tests.
    half = float(t.ppf(0.95, n - 1)) * standard_error
    return Equivalence(
        left=left,
        right=right,
        metric=metric,
        margin=margin,
        mean_difference=mean,
        low=mean - half,
        high=mean + half,
        p_value=p_value,
        n_pairs=n,
    )


def build_equivalences(runs: list[dict[str, Any]]) -> list[Equivalence]:
    """Run TOST over the same contrasts and cells the signed-rank tests use."""
    from zebra.eval.report import CONTRAST_METRICS, REPORTED_CONTRASTS

    cells: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for run in runs:
        key = (str(run["ordering"]), int(run["seed"]))
        cells.setdefault(key, {})[str(run["strategy"])] = run["summary"]

    out: list[Equivalence] = []
    for left, right in REPORTED_CONTRASTS:
        shared = sorted(k for k, v in cells.items() if left in v and right in v)
        if len(shared) < 3:
            continue
        for metric in CONTRAST_METRICS:
            out.append(
                equivalence(
                    np.array([cells[k][left][metric] for k in shared], dtype=np.float64),
                    np.array([cells[k][right][metric] for k in shared], dtype=np.float64),
                    left=left,
                    right=right,
                    metric=metric,
                    margin=MARGINS[metric],
                )
            )
    return out


def write_equivalences(results: list[Equivalence], output: Path) -> None:
    """Emit TOST results as JSON and TeX macros alongside the paired contrasts."""
    import csv
    import json
    from dataclasses import asdict

    from zebra.eval.report import _macro_tag

    rows = [asdict(e) | {"equivalent": e.equivalent} for e in results]
    (output / "equivalence.json").write_text(json.dumps(rows, indent=2))
    with (output / "equivalence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["left"])
        writer.writeheader()
        writer.writerows(rows)

    lines = ["% TOST equivalence generated from complete run artifacts; do not edit."]
    for e in results:
        tag = f"{_macro_tag(e.left)}Vs{_macro_tag(e.right)}{_macro_tag(e.metric)}"
        lines.append(rf"\def\ZBE{tag}Low{{{e.low:+.4f}}}")
        lines.append(rf"\def\ZBE{tag}High{{{e.high:+.4f}}}")
        lines.append(
            rf"\def\ZBE{tag}P{{{'<0.001' if e.p_value < 0.001 else f'={e.p_value:.3f}'}}}"
        )
    (output / "equivalence_results.tex").write_text("\n".join(lines) + "\n")
