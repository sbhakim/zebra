"""Reservoir experience replay with explicit sampling and byte-visible state.

Buffer size follows the released default (``memory_size=2000``). One stored
item is a full window, not a row, which is the whole reason the footprint is
megabyte-scale: 10 timesteps x 14 features x 4 bytes = 560 bytes each.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zebra.strategies.base import Strategy


class ExperienceReplay(Strategy):
    name = "replay"

    def __init__(
        self,
        model: nn.Module,
        capacity: int = 2000,
        replay_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(model)
        self.capacity = capacity
        self.replay_ratio = replay_ratio
        self.generator = torch.Generator().manual_seed(seed)
        self.n_seen = 0
        self.buf_x: torch.Tensor | None = None
        self.buf_y: torch.Tensor | None = None

    @property
    def retained(self) -> Any:
        return {"buffer_x": self.buf_x, "buffer_y": self.buf_y}

    def absorb(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add each sample once using a global reservoir."""
        for xi, yi in zip(x.detach().cpu(), y.detach().cpu(), strict=True):
            self.n_seen += 1
            if self.buf_x is None:
                self.buf_x = xi.unsqueeze(0).clone()
                self.buf_y = yi.reshape(1).clone()
            elif len(self.buf_x) < self.capacity:
                assert self.buf_y is not None
                self.buf_x = torch.cat((self.buf_x, xi.unsqueeze(0).clone()))
                self.buf_y = torch.cat((self.buf_y, yi.reshape(1).clone()))
            else:
                assert self.buf_y is not None
                j = int(torch.randint(self.n_seen, (1,), generator=self.generator).item())
                if j < self.capacity:
                    self.buf_x[j].copy_(xi)
                    self.buf_y[j].copy_(yi)

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.buf_x is None or self.buf_y is None or n <= 0:
            return None
        idx = torch.randperm(len(self.buf_x), generator=self.generator)[: min(n, len(self.buf_x))]
        return self.buf_x[idx].to(device), self.buf_y[idx].to(device)

    def augment_batch(
        self, x: torch.Tensor, y: torch.Tensor, nu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        replay = self.sample(round(self.replay_ratio * len(x)), x.device)
        if replay is None:
            return x, y, nu
        rx, ry = replay
        # Replay itself needs no rule tensor. A composite ZeBRa+replay strategy
        # applies rule support only to current examples unless raw replay is kept.
        rnu = torch.zeros((len(rx), nu.shape[1]), device=nu.device, dtype=nu.dtype)
        return torch.cat((x, rx)), torch.cat((y, ry)), torch.cat((nu, rnu))

    def after_domain(self, loader: Any, device: torch.device) -> None:
        for batch in loader:
            x, y = batch[0], batch[1]
            self.absorb(x, y)
