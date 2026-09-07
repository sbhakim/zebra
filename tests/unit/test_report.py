"""Evidence aggregation rejects incomplete or unpaired experiment grids."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zebra.eval.controls import PRIMARY_STRATEGIES
from zebra.eval.report import aggregate


def _run(strategy: str, ordering: str = "random-11", seed: int = 7) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "ordering": ordering,
        "seed": seed,
        "summary": {
            "auc": 0.7,
            "f1": 0.6,
            "final_bwt_auc": -0.1,
            "mean_stage_bwt_auc": -0.05,
            "plasticity_auc": 0.8,
            "stability_auc": 0.9,
            "ps_auc": 0.847,
            "auc_by_attack": {"blackhole": 0.7},
        },
    }


def test_strict_aggregation_requires_every_primary_strategy() -> None:
    with pytest.raises(ValueError, match="missing strategies"):
        aggregate([_run("no-cl")])


def test_paired_recovery_uses_matching_order_and_seed() -> None:
    runs = [_run(name) for name in PRIMARY_STRATEGIES]
    for run in runs:
        if run["strategy"] == "no-cl":
            run["summary"]["auc"] = 0.5
        elif run["strategy"] == "replay":
            run["summary"]["auc"] = 0.9
        elif run["strategy"] == "zebra":
            run["summary"]["auc"] = 0.7
    result = aggregate(runs)
    assert result["zebra_recovery"]["mean"] == pytest.approx(0.5)


def test_strict_aggregation_rejects_an_unpaired_second_seed() -> None:
    runs = [_run(name) for name in PRIMARY_STRATEGIES]
    runs.append(_run("no-cl", seed=19))
    with pytest.raises(ValueError, match="unpaired primary evidence"):
        aggregate(runs)


def test_recovery_is_withheld_when_replay_is_not_an_upper_reference() -> None:
    runs = [_run(name) for name in PRIMARY_STRATEGIES]
    for run in runs:
        if run["strategy"] == "no-cl":
            run["summary"]["auc"] = 0.8
        elif run["strategy"] == "replay":
            run["summary"]["auc"] = 0.7
    result = aggregate(runs)
    assert result["zebra_recovery"] is None
    assert result["zebra_recovery_reference_failures"] == 1


def test_every_emitted_macro_is_a_legal_tex_definition(tmp_path) -> None:
    """Regression: three emitters produced `\\def\\Name}{value}` with a stray
    brace, and `_macro_tag` produced digits, which TeX cannot put in a control
    sequence -- `\\ZBZebraR1Auc` silently defines `\\ZBZebraR` and leaves `1Auc`
    as body text. Neither fails loudly in LaTeX, so it is checked here."""
    import re
    import subprocess
    import sys

    for module in ("zebra.eval.memory_report", "zebra.eval.probe"):
        subprocess.run(  # noqa: S603
            (sys.executable, "-m", module), check=True, capture_output=True
        )
    pattern = re.compile(r"\\def\\[A-Za-z]+\{[^{}]*\}")
    root = Path(__file__).resolve().parents[2] / "generated"
    checked = 0
    for path in root.glob("*.tex"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line or line.startswith("%"):
                continue
            assert pattern.fullmatch(line), f"{path.name}:{number}: {line}"
            checked += 1
    assert checked > 0, "no macros were emitted to check"
