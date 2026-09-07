"""Mechanism-focused strata and out-of-distribution partitions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from zebra.model import DomainId


def network_size_ood(
    domains: list[DomainId], held_out_size: int = 20
) -> tuple[list[DomainId], list[DomainId]]:
    """Partition identities before loading any labels or performance values."""
    development = [domain for domain in domains if domain.size != held_out_size]
    held_out = [domain for domain in domains if domain.size == held_out_size]
    if not development or not held_out:
        raise ValueError(f"network-size split {held_out_size} produced an empty side")
    return development, held_out


def final_attack_strata(matrix: np.ndarray, order: tuple[DomainId, ...]) -> dict[str, float]:
    """Macro-average final performance by attack family."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(order), len(order)) or np.isnan(values[-1]).any():
        raise ValueError("complete square matrix and matching order required")
    grouped: dict[str, list[float]] = defaultdict(list)
    for score, domain in zip(values[-1], order, strict=True):
        grouped[domain.attack.value].append(float(score))
    return {attack: float(np.mean(scores)) for attack, scores in sorted(grouped.items())}


PRIMARY_STRATEGIES: tuple[str, ...] = (
    "no-cl",
    "ewc",
    "si",
    "lwf",
    "gen-replay",
    "replay",
    "zebra",
)

MECHANISM_CONTROLS: tuple[str, ...] = (
    "zebra-r1",
    "zebra-r2",
    "zebra-r3",
    "zebra-shuffled",
    "zebra-replay",
)
