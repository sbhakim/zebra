"""ZeBRa: the invariant penalty. Retains nothing.

A function of the window's precomputed invariant scores and the detector
output. Nothing is held between domains, so ``retained`` is ``None`` and the
accounting reports zero -- not a small number, zero.

Rules combine as a probabilistic OR and the loss encodes only
``violation -> attack``. The converse is deliberately absent: blackhole traffic
satisfies every observable control-plane rule and is still malicious.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zebra.strategies.base import Strategy


class ZeBRa(Strategy):
    name = "zebra"

    def __init__(
        self, model: nn.Module, lam: float = 0.1, rule_indices: tuple[int, ...] = (0, 1, 2)
    ) -> None:
        super().__init__(model)
        self.lam = lam
        self.rule_indices = rule_indices

    @property
    def retained(self) -> Any:
        return None

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        """One-sided differentiable implication from violation to attack.

        ``nu`` is (B, 3), precomputed from raw windows at load time (ADR-001),
        so no gradient flows into the invariants and none can.
        """
        return one_sided_constraint_loss(logits, nu[:, self.rule_indices], self.lam)


def one_sided_constraint_loss(
    logits: torch.Tensor, nu: torch.Tensor, lam: float
) -> torch.Tensor:
    """Fuzzy OR followed by the implication ``rule violation -> attack``."""
    support = 1.0 - torch.prod(1.0 - nu.detach().clamp(0.0, 1.0), dim=1)
    attack_margin = logits[:, 1] - logits[:, 0]
    return lam * torch.mean(support * torch.nn.functional.softplus(-attack_margin))


class ShuffledZeBRa(ZeBRa):
    """Negative control preserving rule marginals while breaking alignment."""

    name = "zebra-shuffled"

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        shuffled = torch.roll(nu, shifts=1, dims=0)
        return one_sided_constraint_loss(logits, shuffled[:, self.rule_indices], self.lam)
