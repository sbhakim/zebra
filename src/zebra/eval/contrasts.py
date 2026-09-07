"""Paired contrasts between strategies.

Every arm sees the same orderings and the same seeds, so comparisons are paired
by construction. Unpaired they say very little: ordering alone moves AUC by
0.036--0.060, several times the effects under study, and every interval overlaps.

Wilcoxon signed-rank is what we report -- twelve pairs is too few to lean on
normality. A paired t runs alongside; the two disagreeing is reason to distrust
both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


@dataclass(frozen=True, slots=True)
class Contrast:
    """One paired comparison, reported with its effect size and both tests."""

    left: str
    right: str
    metric: str
    mean_difference: float
    low: float
    high: float
    wilcoxon_p: float
    ttest_p: float
    n_pairs: int

    @property
    def significant(self) -> bool:
        """Both tests agree at the 5% level. Deliberately conservative."""
        return self.wilcoxon_p < 0.05 and self.ttest_p < 0.05


def paired_contrast(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    left: str,
    right: str,
    metric: str,
) -> Contrast:
    """Compare two strategies over their common (ordering, seed) cells."""
    if left_values.shape != right_values.shape:
        raise ValueError("paired contrast requires matching cell vectors")
    if len(left_values) < 3:
        raise ValueError(f"too few pairs for a paired contrast: {len(left_values)}")

    difference = np.asarray(left_values, dtype=np.float64) - np.asarray(
        right_values, dtype=np.float64
    )
    half_width = 1.96 * float(difference.std(ddof=1)) / float(np.sqrt(len(difference)))
    mean = float(difference.mean())

    # An all-zero difference vector is degenerate for the signed-rank statistic;
    # report it as no evidence of a difference rather than propagating an error.
    if np.allclose(difference, 0.0):
        w_p = 1.0
    else:
        w_p = float(wilcoxon(difference).pvalue)

    return Contrast(
        left=left,
        right=right,
        metric=metric,
        mean_difference=mean,
        low=mean - half_width,
        high=mean + half_width,
        wilcoxon_p=w_p,
        ttest_p=float(ttest_rel(left_values, right_values).pvalue),
        n_pairs=len(difference),
    )
