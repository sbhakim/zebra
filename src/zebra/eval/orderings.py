"""Leakage-free, deterministic domain permutations."""

from __future__ import annotations

import random

from zebra.model import DomainId

ORDERING_SEEDS: tuple[int, ...] = (11, 29, 47, 83)


def fixed_random_orderings(
    domains: list[DomainId], seeds: tuple[int, ...] = ORDERING_SEEDS
) -> dict[str, tuple[DomainId, ...]]:
    """Return fixed permutations using identities only, never measured performance."""
    canonical = sorted(domains, key=lambda domain: domain.name)
    result: dict[str, tuple[DomainId, ...]] = {}
    for seed in seeds:
        order = canonical.copy()
        random.Random(seed).shuffle(order)  # noqa: S311 -- reproducible experiment permutation
        result[f"perm-{seed}"] = tuple(order)
    return result
