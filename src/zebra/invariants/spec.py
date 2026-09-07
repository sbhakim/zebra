"""Typed contract for a registered protocol-guided signal.

A rule is a pure function from a batch of *raw* windows to a signal in [0, 1]
(ADR-001). It carries no state and no parameters. A separately reported,
training-only calibration determines when that signal is strong enough to make
a one-sided violation assertion.

Controlled synthetic tests vary each signal's intended inputs independently.
Corpus-selected extrema are deliberately forbidden because they would make the
test circular.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from zebra.model import InvariantSpec

# (B, L, 14) raw windows -> (B,) violation scores in [0, 1]
ScoreFn = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class Invariant:
    """A registered rule specification and its pure scoring function."""

    spec: InvariantSpec
    score: ScoreFn

def assert_stateless(inv: Invariant, window: np.ndarray) -> None:
    """Two calls on the same input must agree exactly.

    A scorer that accumulated hidden history between calls would drift here.
    """
    a = inv.score(window)
    b = inv.score(window)
    if not np.array_equal(a, b):
        raise AssertionError(f"{inv.spec.ident.value} is not stateless")
