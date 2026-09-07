"""R1: solicitation-pressure signal.

A node emits DIS because it lacks a usable DODAG and the network answers with
DIO, so solicitation is rare in a converged graph. A sustained solicitation
share means nodes keep failing to acquire or hold a parent.

Motivated by the protocol rather than packet conservation: broadcast
sent/received counts depend on topology and will not support a domain-universal
equality at sink aggregation.
"""

from __future__ import annotations

import warnings

import numpy as np

from zebra.model import Attack, InvariantId, InvariantSpec

SPEC = InvariantSpec(
    ident=InvariantId.SOLICITATION,
    title="Solicitation pressure",
    rationale=(
        "DIS requests an advertisement and DIO supplies it. A converged DODAG needs "
        "few requests, so a sustained solicitation share indicates repeated failure "
        "to acquire or retain a parent."
    ),
    targets=(Attack.DISFLOODING,),
)

_DISR, _DISS, _DIOR, _DIOS = 1, 2, 3, 4
def score(window: np.ndarray) -> np.ndarray:
    """Share of control traffic that is solicitation, over the window."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dis = np.nanmean(window[:, :, _DISS] + window[:, :, _DISR], axis=1)
        dio = np.nanmean(window[:, :, _DIOR] + window[:, :, _DIOS], axis=1)
    share = dis / (dis + dio + 1e-6)
    # A window with no observed control traffic asserts nothing.
    return np.asarray(np.clip(np.nan_to_num(share, nan=0.0), 0.0, 1.0), dtype=np.float64)
