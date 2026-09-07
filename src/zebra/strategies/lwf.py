"""Learning without Forgetting with a frozen previous-domain teacher."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from zebra.strategies.base import Strategy


class LearningWithoutForgetting(Strategy):
    name = "lwf"

    def __init__(self, model: nn.Module, alpha: float = 0.5, temperature: float = 2.0) -> None:
        super().__init__(model)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher: nn.Module | None = None

    @property
    def retained(self) -> Any:
        return {"teacher": self.teacher}

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        if self.teacher is None:
            return torch.zeros((), device=logits.device)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        temperature = self.temperature
        student_log = torch.nn.functional.log_softmax(logits / temperature, dim=1)
        teacher_prob = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
        return self.alpha * temperature**2 * torch.nn.functional.kl_div(
            student_log, teacher_prob, reduction="batchmean"
        )

    def after_domain(self, loader: Any, device: torch.device) -> None:
        self.teacher = deepcopy(self.model).to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
