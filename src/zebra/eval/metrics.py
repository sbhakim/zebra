"""Continual-learning metrics computed from explicit stage-by-domain matrices."""

from __future__ import annotations

import numpy as np


def _square(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square performance matrix, got {values.shape}")
    return values


def final_average(matrix: np.ndarray) -> float:
    values = _square(matrix)
    if np.isnan(values[-1]).any():
        raise ValueError("final performance row is incomplete")
    return float(np.mean(values[-1]))


def final_bwt(matrix: np.ndarray) -> float:
    """Final backward transfer: final minus immediate post-training performance."""
    values = _square(matrix)
    if len(values) < 2:
        return 0.0
    learned = np.diag(values)[:-1]
    final = values[-1, :-1]
    if np.isnan(learned).any() or np.isnan(final).any():
        raise ValueError("BWT requires the diagonal and final prior-domain scores")
    return float(np.mean(final - learned))


def mean_stage_bwt(matrix: np.ndarray) -> float:
    """Mean forgetting across every stage and previously learned domain."""
    values = _square(matrix)
    terms: list[float] = []
    for stage in range(1, len(values)):
        for domain in range(stage):
            terms.append(float(values[stage, domain] - values[domain, domain]))
    if not terms or np.isnan(terms).any():
        return 0.0
    return float(np.mean(terms))


def plasticity(matrix: np.ndarray, pre_scores: np.ndarray) -> float:
    """NFL-normalized fraction of the incoming-domain performance gap closed."""
    values = _square(matrix)
    pre = np.asarray(pre_scores, dtype=np.float64)
    if len(pre) != len(values):
        raise ValueError("one pre-training score is required per domain")
    terms = []
    for domain in range(1, len(values)):
        before = pre[domain]
        after = values[domain, domain]
        if np.isnan(before) or np.isnan(after):
            raise ValueError("plasticity inputs are incomplete")
        terms.append(np.clip((after - before) / max(1.0 - before, 1e-8), 0.0, 1.0))
    return float(np.mean(terms)) if terms else 1.0


def stability(matrix: np.ndarray) -> float:
    """One minus non-negative final forgetting, following the PS definition."""
    values = _square(matrix)
    if len(values) < 2:
        return 1.0
    forgetting = np.maximum(np.diag(values)[:-1] - values[-1, :-1], 0.0)
    if np.isnan(forgetting).any():
        raise ValueError("stability inputs are incomplete")
    return float(np.clip(1.0 - np.mean(forgetting), 0.0, 1.0))


def plasticity_stability(matrix: np.ndarray, pre_scores: np.ndarray) -> float:
    p = plasticity(matrix, pre_scores)
    s = stability(matrix)
    return 0.0 if p + s == 0.0 else float(2.0 * p * s / (p + s))


# Recovery divides by the replay-minus-sequential gap, so a small gap amplifies
# noise: local repair's 0.0004 gap turned a 0.0003 difference into "73.4%
# recovery" under the old 1e-8 floor. Below one AUC point the ratio says nothing,
# so those strata are reported as suppressed rather than quietly rescaled.
MIN_RECOVERY_DENOMINATOR = 0.01


def recovery(method: float, sequential: float, replay: float) -> float:
    """Fraction of a material sequential-to-replay gap closed.

    Replay is a meaningful upper reference only when it beats sequential
    training by more than `MIN_RECOVERY_DENOMINATOR`. A reversed, negligible, or
    noise-scale denominator produces a misleading percentage rather than a
    useful normalized comparison, so it raises instead.
    """
    denominator = replay - sequential
    if denominator <= MIN_RECOVERY_DENOMINATOR:
        raise ValueError(
            "recovery requires replay to outperform sequential training by at least "
            f"{MIN_RECOVERY_DENOMINATOR}; got {denominator:.5f}"
        )
    return float((method - sequential) / denominator)
