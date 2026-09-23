"""Per-window uint8 replay: more history inside the same byte envelope.

A stored window is 568 B as float32; uint8 cuts it to 156 B, so a Class 2 node
holds 217 of them instead of 59.

The scale is per window, not global. The frozen scaler leaves held-out values
well outside [0, 1] and corpus.py keeps them on purpose, so a shared range would
clip exactly the tail that signals domain shift.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zebra.strategies.replay import ExperienceReplay

LEVELS = 255.0


def quantise(window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map one window to uint8 plus the float32 range needed to invert it."""
    lo = window.min()
    hi = window.max()
    span = hi - lo
    # A constant window has no range to rescale; store it at zero and recover it
    # from `lo` alone, rather than dividing by zero.
    scale = span if span > 0 else torch.ones_like(span)
    codes = torch.round((window - lo) / scale * LEVELS).clamp(0, LEVELS).to(torch.uint8)
    return codes, lo.reshape(1).float(), hi.reshape(1).float()


def dequantise(codes: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Invert `quantise` for a batch of stored windows."""
    span = (hi - lo).reshape(-1, 1, 1)
    span = torch.where(span > 0, span, torch.ones_like(span))
    return codes.float() / LEVELS * span + lo.reshape(-1, 1, 1)


class QuantizedExperienceReplay(ExperienceReplay):
    """Reservoir replay whose buffer is stored as per-window uint8."""

    name = "replay-q8"

    def __init__(
        self,
        model: nn.Module,
        capacity: int = 2000,
        replay_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(model, capacity=capacity, replay_ratio=replay_ratio, seed=seed)
        self.buf_q: torch.Tensor | None = None
        self.buf_lo: torch.Tensor | None = None
        self.buf_hi: torch.Tensor | None = None

    @property
    def retained(self) -> Any:
        # buf_x/buf_y from the float32 parent are never populated here; the
        # quantised tensors are the whole of what this strategy carries.
        return {
            "buffer_codes": self.buf_q,
            "buffer_lo": self.buf_lo,
            "buffer_hi": self.buf_hi,
            "buffer_y": self.buf_y,
        }

    def absorb(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for xi, yi in zip(x.detach().cpu(), y.detach().cpu(), strict=True):
            codes, lo, hi = quantise(xi)
            self.n_seen += 1
            if self.buf_q is None:
                self.buf_q = codes.unsqueeze(0).clone()
                self.buf_lo = lo.clone()
                self.buf_hi = hi.clone()
                self.buf_y = yi.reshape(1).clone()
            elif len(self.buf_q) < self.capacity:
                assert self.buf_lo is not None and self.buf_hi is not None
                assert self.buf_y is not None
                self.buf_q = torch.cat((self.buf_q, codes.unsqueeze(0)))
                self.buf_lo = torch.cat((self.buf_lo, lo))
                self.buf_hi = torch.cat((self.buf_hi, hi))
                self.buf_y = torch.cat((self.buf_y, yi.reshape(1)))
            else:
                assert self.buf_lo is not None and self.buf_hi is not None
                assert self.buf_y is not None
                j = int(torch.randint(self.n_seen, (1,), generator=self.generator).item())
                if j < self.capacity:
                    self.buf_q[j].copy_(codes)
                    self.buf_lo[j].copy_(lo[0])
                    self.buf_hi[j].copy_(hi[0])
                    self.buf_y[j].copy_(yi)

    def sample(
        self, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.buf_q is None or self.buf_y is None or n <= 0:
            return None
        assert self.buf_lo is not None and self.buf_hi is not None
        idx = torch.randperm(len(self.buf_q), generator=self.generator)[
            : min(n, len(self.buf_q))
        ]
        windows = dequantise(self.buf_q[idx], self.buf_lo[idx], self.buf_hi[idx])
        return windows.to(device), self.buf_y[idx].to(device)


class DistillQuantizedReplay(QuantizedExperienceReplay):
    """Distillation plus a quantised buffer: the largest hybrid that fits Class 2.

    The budget picks the shape: a 4,248 B teacher leaves room for 190 quantised
    windows, against 52 float32 ones. Coefficients are LwF's and replay's
    defaults, so this arm gets no tuning the others did not.
    """

    name = "lwf-q8replay"

    def __init__(
        self,
        model: nn.Module,
        capacity: int = 190,
        replay_ratio: float = 0.5,
        alpha: float = 0.5,
        temperature: float = 2.0,
        seed: int = 42,
    ) -> None:
        super().__init__(model, capacity=capacity, replay_ratio=replay_ratio, seed=seed)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher: nn.Module | None = None

    @property
    def retained(self) -> Any:
        state = dict(super().retained)
        state["teacher"] = self.teacher
        return state

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
        # Absorb first, then snapshot: the teacher should reflect the model that
        # finished this domain, and the buffer the data it finished on.
        super().after_domain(loader, device)
        from copy import deepcopy

        self.teacher = deepcopy(self.model).to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)


class DistillReplay(ExperienceReplay):
    """float32 control for `DistillQuantizedReplay`: same teacher, same budget.

    Separates what the teacher buys from what the extra history buys: same
    4,248 B teacher, same envelope, but 52 uncompressed windows instead of 190
    quantised ones.
    """

    name = "lwf-replay"

    def __init__(
        self,
        model: nn.Module,
        capacity: int = 52,
        replay_ratio: float = 0.5,
        alpha: float = 0.5,
        temperature: float = 2.0,
        seed: int = 42,
    ) -> None:
        super().__init__(model, capacity=capacity, replay_ratio=replay_ratio, seed=seed)
        self.alpha = alpha
        self.temperature = temperature
        self.teacher: nn.Module | None = None

    @property
    def retained(self) -> Any:
        return {"buffer_x": self.buf_x, "buffer_y": self.buf_y, "teacher": self.teacher}

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
        super().after_domain(loader, device)
        from copy import deepcopy

        self.teacher = deepcopy(self.model).to(device).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
