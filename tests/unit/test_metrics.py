"""Exact continual-learning metric definitions and ordering contracts."""

from __future__ import annotations

import numpy as np
import pytest

from zebra.eval.metrics import (
    final_average,
    final_bwt,
    mean_stage_bwt,
    plasticity,
    plasticity_stability,
    recovery,
    stability,
)
from zebra.eval.orderings import fixed_random_orderings
from zebra.model import DomainId


def test_metrics_on_known_matrix() -> None:
    matrix = np.array(
        [
            [0.80, np.nan, np.nan],
            [0.70, 0.90, np.nan],
            [0.60, 0.85, 0.75],
        ]
    )
    pre = np.array([np.nan, 0.50, 0.25])
    assert final_average(matrix) == pytest.approx(0.7333333)
    assert final_bwt(matrix) == pytest.approx(-0.125)
    assert mean_stage_bwt(matrix) == pytest.approx((-0.10 - 0.20 - 0.05) / 3)
    assert plasticity(matrix, pre) == pytest.approx((0.8 + 2 / 3) / 2)
    assert stability(matrix) == pytest.approx(0.875)
    p = plasticity(matrix, pre)
    assert plasticity_stability(matrix, pre) == pytest.approx(2 * p * 0.875 / (p + 0.875))
    assert recovery(0.7, 0.5, 0.9) == pytest.approx(0.5)


def test_recovery_rejects_nonpositive_reference_gap() -> None:
    with pytest.raises(ValueError, match="outperform"):
        recovery(0.7, 0.5, 0.5)
    with pytest.raises(ValueError, match="outperform"):
        recovery(0.7, 0.8, 0.6)


def test_orderings_are_deterministic_permutations() -> None:
    domains = [
        DomainId.parse("blackhole_var5_base"),
        DomainId.parse("disflooding_var10_oo"),
        DomainId.parse("localrepair_var15_dec"),
    ]
    first = fixed_random_orderings(domains, (11, 29))
    second = fixed_random_orderings(list(reversed(domains)), (11, 29))
    assert first == second
    for order in first.values():
        assert set(order) == set(domains)


def test_recovery_refuses_a_noise_scale_denominator() -> None:
    """Regression: a 0.0004 AUC gap on local repair produced '73.4% recovery'
    under the old 1e-8 floor. A ratio is only meaningful over a material gap."""
    import pytest

    from zebra.eval.metrics import MIN_RECOVERY_DENOMINATOR, recovery

    with pytest.raises(ValueError, match="at least"):
        recovery(0.9874, 0.9870, 0.9874)  # denominator 0.0004
    with pytest.raises(ValueError):
        recovery(0.97, 0.972, 0.969)  # reversed reference
    assert recovery(0.90, 0.85, 0.95) == pytest.approx(0.5)  # 0.10 gap is material
    assert MIN_RECOVERY_DENOMINATOR >= 0.01


def test_paired_contrast_recovers_an_effect_that_unpaired_analysis_hides() -> None:
    """The grid is paired by (ordering, seed) and ordering variance dwarfs the
    effects, so unpaired intervals overlap for every strategy. This is the
    property the whole contrast module exists for."""
    import numpy as np

    from zebra.eval.contrasts import paired_contrast

    # A large per-cell offset (ordering) plus a small constant effect.
    ordering = np.array([0.83, 0.86, 0.89, 0.84, 0.87, 0.90, 0.85, 0.88, 0.86, 0.87, 0.84, 0.89])
    left = ordering + 0.007
    right = ordering

    c = paired_contrast(left, right, left="a", right="b", metric="auc")
    assert c.mean_difference == pytest.approx(0.007, abs=1e-6)
    assert c.low > 0.0, "paired interval must exclude zero despite huge shared variance"
    assert c.significant

    # Unpaired, the same data is indistinguishable.
    pooled = 1.96 * np.sqrt(left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right))
    assert pooled > abs(c.mean_difference), "unpaired comparison should not resolve it"


def test_paired_contrast_reports_equivalence_without_erroring() -> None:
    """An identical pair is degenerate for the signed-rank statistic; it must
    read as no evidence of a difference, not as a crash."""
    import numpy as np

    from zebra.eval.contrasts import paired_contrast

    values = np.linspace(0.80, 0.90, 12)
    c = paired_contrast(values, values.copy(), left="a", right="a", metric="auc")
    assert c.mean_difference == pytest.approx(0.0)
    assert c.wilcoxon_p == 1.0
    assert not c.significant


def test_paired_contrast_refuses_too_few_pairs() -> None:
    import numpy as np
    import pytest as _pytest

    from zebra.eval.contrasts import paired_contrast

    with _pytest.raises(ValueError, match="too few pairs"):
        paired_contrast(
            np.array([0.1, 0.2]), np.array([0.1, 0.2]), left="a", right="b", metric="auc"
        )
