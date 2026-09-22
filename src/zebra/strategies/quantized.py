"""Per-window uint8 replay: more history inside the same byte envelope.

A stored window is 10x14 float32 plus an int64 label, 568 B. Under the RFC 7228
Class 2 envelope only 59 of them fit beside the model, optimiser, gradients and
scaler. Quantising the window to uint8 cuts that to 156 B and raises the
feasible buffer to 217.

The scale is *per window*, not global, and that is the whole design. A frozen
first-domain scaler leaves held-out values outside [0, 1] -- measured range on
this corpus is -0.201 to 30.127 -- and `corpus.py` keeps them on purpose,
because clipping would conceal genuine domain shift. A global uint8 range over
[0, 1] clips that tail with a max error of 29.1. Storing each window's own
min and max costs 8 B and drops the mean error to 0.00038, which is 0.15% of a
feature standard deviation.
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

    The byte accounting picks the shape of this arm rather than the other way
    round. Class 2 leaves 33,960 B for retained state; the frozen teacher takes
    4,248 B, and the remaining 29,712 B holds 190 quantised windows against 52
    float32 ones. Nothing here is tuned -- alpha and temperature are LwF's
    defaults and the ratio is replay's, so the arm carries no budget the others
    did not get.
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
