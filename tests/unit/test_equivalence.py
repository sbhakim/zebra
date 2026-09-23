"""TOST behaviour on cases whose answer is known before running the test."""

from __future__ import annotations

import numpy as np
import pytest

from zebra.eval.equivalence import equivalence


def _pair(differences: list[float]) -> tuple[np.ndarray, np.ndarray]:
    right = np.zeros(len(differences), dtype=np.float64)
    return np.asarray(differences, dtype=np.float64), right


def test_tight_differences_inside_the_margin_are_equivalent() -> None:
    left, right = _pair([0.001, -0.001, 0.0005, -0.0005, 0.0, 0.001])
    result = equivalence(left, right, left="a", right="b", metric="auc", margin=0.01)
    assert result.equivalent
    assert -0.01 < result.low and result.high < 0.01


def test_a_difference_beyond_the_margin_is_not_equivalent() -> None:
    left, right = _pair([0.05, 0.048, 0.052, 0.049, 0.051, 0.05])
    result = equivalence(left, right, left="a", right="b", metric="auc", margin=0.01)
    assert not result.equivalent


def test_a_noisy_null_is_inconclusive_rather_than_equivalent() -> None:
    """A large signed-rank p-value must not become an equivalence claim."""
    left, right = _pair([0.03, -0.03, 0.02, -0.02, 0.04, -0.04])
    result = equivalence(left, right, left="a", right="b", metric="auc", margin=0.01)
    assert abs(result.mean_difference) < 0.01
    assert not result.equivalent


def test_identical_arms_are_equivalent_without_dividing_by_zero() -> None:
    left, right = _pair([0.0] * 6)
    result = equivalence(left, right, left="a", right="b", metric="auc", margin=0.01)
    assert result.equivalent
    assert result.p_value == 0.0


def test_margin_and_pair_count_are_validated() -> None:
    left, right = _pair([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="margin"):
        equivalence(left, right, left="a", right="b", metric="auc", margin=0.0)
    with pytest.raises(ValueError, match="too few pairs"):
        equivalence(left[:2], right[:2], left="a", right="b", metric="auc", margin=0.01)
