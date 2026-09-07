"""Protocol-rule semantics, calibration, bounds, and statelessness."""

from __future__ import annotations

import numpy as np
import pytest

from zebra.invariants.calibration import RuleCalibration, benign_fire_rate
from zebra.invariants.registry import ORDER, REGISTRY, violation_scores
from zebra.invariants.spec import assert_stateless


def _controlled_windows() -> tuple[np.ndarray, np.ndarray]:
    passing = np.zeros((4, 10, 14), dtype=np.float64)
    violating = passing.copy()
    # R1: solicitation dominates advertisement.
    passing[:, :, 4] = 10.0
    violating[:, :, 2] = 10.0
    # R2: passing DIO emission is irregular; violating emission is uniform.
    passing[:, :, 4] = np.array([0, 0, 0, 1, 0, 0, 3, 0, 0, 6])
    violating[:, :, 4] = 1.0
    # R3: passing nodes agree; violating windows have high cross-node spread.
    passing[:, :, 11] = 0.0
    violating[:, :, 11] = 10.0
    return passing, violating


@pytest.mark.parametrize("ident", ORDER)
def test_controlled_violation_scores_above_controlled_passing(ident) -> None:
    inv = REGISTRY[ident]
    passing, violating = _controlled_windows()
    assert inv.score(violating).mean() > inv.score(passing).mean()


@pytest.mark.parametrize("ident", ORDER)
def test_bounded_and_stateless(ident) -> None:
    inv = REGISTRY[ident]
    passing, _ = _controlled_windows()
    s = inv.score(passing)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert_stateless(inv, passing)


def test_nan_input_yields_finite_neutral_scores() -> None:
    """An unobservable rule makes no positive assertion."""
    w = np.full((4, 10, 14), np.nan)
    s = violation_scores(w)
    assert np.isfinite(s).all()
    assert np.array_equal(s, np.zeros_like(s))


def test_scores_do_not_depend_on_batch_order() -> None:
    rng = np.random.default_rng(0)
    w = rng.random((16, 10, 14)) * 10
    a = violation_scores(w)
    perm = rng.permutation(16)
    b = violation_scores(w[perm])
    assert np.allclose(a[perm], b)


def test_training_only_calibration_is_conservative_and_accounted() -> None:
    rng = np.random.default_rng(3)
    benign = rng.uniform(0.0, 0.5, size=(2000, 3))
    attack = rng.uniform(0.5, 1.0, size=(500, 3))
    signals = np.concatenate([benign, attack])
    labels = np.concatenate([np.zeros(len(benign)), np.ones(len(attack))])
    calibration = RuleCalibration.fit(signals, labels)
    calibrated = calibration.transform(signals)
    assert np.all(benign_fire_rate(calibrated, labels) <= 0.011)
    assert calibrated[labels == 1].mean() > calibrated[labels == 0].mean()
    assert calibration.metadata_bytes == 48
