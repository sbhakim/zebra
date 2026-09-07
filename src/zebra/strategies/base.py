"""Unified lifecycle shared by every continual-learning strategy.

Every strategy declares its retained state through one property, ``retained``.
That is the single hook the memory accounting reads, so a strategy cannot
quietly hold something the byte table does not see -- the property *is* the
declaration, and a test asserts the two agree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class Strategy(ABC):
    """A continual-learning strategy over a shared detector."""

    name: str = "abstract"

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.domain: str | None = None

    @property
    @abstractmethod
    def retained(self) -> Any:
        """Everything carried between domains. ``None`` means nothing."""

    def start_domain(self, name: str) -> None:
        self.domain = name

    def augment_batch(
        self, x: torch.Tensor, y: torch.Tensor, nu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optionally add replay samples before the forward pass."""
        return x, y, nu

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        """Extra loss term. Zero unless the strategy defines one."""
        return torch.zeros((), device=logits.device)

    def capture_task_gradients(self, task_loss: torch.Tensor) -> None:
        """Capture pure-task gradients when an algorithm needs them (SI)."""
        return None

    def before_optimizer_step(self) -> None:
        return None

    def after_optimizer_step(self) -> None:
        return None

    def after_domain(self, loader: Any, device: torch.device) -> None:
        """Hook to update retained state once a domain is finished."""
        return None
