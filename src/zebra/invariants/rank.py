"""R3: cross-node DIO-dispersion signal.

The released features cannot evaluate rank stability: they omit per-node rank
and parent relationships. This signal instead measures DIO dispersion and is
treated as a heuristic whose semantics must be confirmed on held-out benign
data.

Scored as the dispersion-to-level ratio, which is scale-free and needs no
reference topology.
"""

from __future__ import annotations

import warnings

import numpy as np

from zebra.model import Attack, InvariantId, InvariantSpec

SPEC = InvariantSpec(
    ident=InvariantId.DIO_DISPERSION,
    title="DIO cross-node dispersion",
    rationale=(
        "Nodes in a converged DODAG run the same Trickle state machine, so per-node "
        "DIO emission is comparable; spread without a rise in level indicates a "
        "subtree advertising inconsistently with the rest."
    ),
    targets=(Attack.LOCALREPAIR,),
)

_DIOS_MEAN, _DIOS_SD = 4, 11
def score(window: np.ndarray) -> np.ndarray:
    """Cross-node dispersion of DIO emission, relative to its level."""
    # An all-NaN window is unobservable, not compliant-by-accident; nanmean
    # warns there, so we silence the warning and handle the value explicitly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(window[:, :, _DIOS_MEAN], axis=1)
        sd = np.nanmean(window[:, :, _DIOS_SD], axis=1)
    ratio = np.maximum(sd, 0.0) / (np.maximum(mu, 0.0) + 1e-6)
    bounded = ratio / (1.0 + ratio)
    return np.asarray(
        np.clip(np.nan_to_num(bounded, nan=0.0), 0.0, 1.0), dtype=np.float64
    )
