"""The frozen protocol-guided signal registry.

Registration is explicit and ordered. Adding an entry changes the tested
mechanism, so it is a reviewed protocol change rather than an import side effect.

`violation_scores` returns separate columns so ablations and controls can select
signals before the ZeBRa loss combines them as a fuzzy OR.
"""

from __future__ import annotations

import numpy as np

from zebra.invariants import conservation, rank, trickle
from zebra.invariants.spec import Invariant
from zebra.model import InvariantId

REGISTRY: dict[InvariantId, Invariant] = {
    conservation.SPEC.ident: Invariant(conservation.SPEC, conservation.score),
    trickle.SPEC.ident: Invariant(trickle.SPEC, trickle.score),
    rank.SPEC.ident: Invariant(rank.SPEC, rank.score),
}

ORDER: tuple[InvariantId, ...] = (
    InvariantId.SOLICITATION,
    InvariantId.TRICKLE_REGULARITY,
    InvariantId.DIO_DISPERSION,
)


def violation_scores(window: np.ndarray) -> np.ndarray:
    """Score a batch of raw windows against every registered signal.

    Returns (B, 3) in ``ORDER``. Calibration and aggregation happen later.
    """
    if window.ndim != 3:
        raise ValueError(f"expected (B, L, F) raw windows, got shape {window.shape}")
    cols = [REGISTRY[i].score(window) for i in ORDER]
    return np.stack(cols, axis=1)
