"""R2: DIO temporal regularity, motivated by Trickle resets.

RFC 6206 doubles the interval while the DODAG stays consistent, so a converged
network emits DIO unevenly and the coefficient of variation is high. Repeated
inconsistency pins the interval near Imin and emission goes uniform. Uniformity
is the signal -- not a universal invariant, since suppression, topology and the
stochastic timer move the CV too.

The first formulation scored the rising slope instead and got AUC 0.42--0.46:
a ten-minute window is far too short to see the doubling as a trend.
"""

from __future__ import annotations

import warnings

import numpy as np

from zebra.model import Attack, InvariantId, InvariantSpec

SPEC = InvariantSpec(
    ident=InvariantId.TRICKLE_REGULARITY,
    title="DIO temporal regularity",
    rationale=(
        "A consistent DODAG doubles its Trickle interval, so DIO emission is uneven "
        "within a window. Uniform emission means the timer is being reset continually."
    ),
    targets=(Attack.DISFLOODING, Attack.LOCALREPAIR),
)

_DIOS = 4
def score(window: np.ndarray) -> np.ndarray:
    """One minus the coefficient of variation of per-minute DIO emission."""
    d = window[:, :, _DIOS]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(d, axis=1)
        sd = np.nanstd(d, axis=1)
    cv = np.nan_to_num(sd / (mu + 1e-6), nan=1.0)
    # Silent windows emit nothing to be regular about; treat as compliant.
    silent = mu <= 1e-9
    return np.clip(np.where(silent, 0.0, 1.0 - cv), 0.0, 1.0).astype(np.float64)
