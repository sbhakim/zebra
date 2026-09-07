"""EWC and SI: regularisation strategies.

Both keep per-parameter statistics rather than data, which is why the
sample-count view calls them memory-free and the byte view does not -- roughly
8.5 kB each here against replay's 1.14 MB.

An earlier revision left ``penalty`` inherited from the base class, quietly
making both of these fine-tuning under another name. A test now asserts a
non-zero penalty once consolidated.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from zebra.strategies.base import Strategy


class EWC(Strategy):
    """Elastic weight consolidation: quadratic pull toward the previous optimum."""

    name = "ewc"

    def __init__(self, model: nn.Module, lam: float = 1.0) -> None:
        super().__init__(model)
        self.lam = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

    @property
    def retained(self) -> Any:
        return {"fisher": self.fisher, "anchor": self.anchor}

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=logits.device)
        total = torch.zeros((), device=logits.device)
        for n, p in self.model.named_parameters():
            f, a = self.fisher.get(n), self.anchor.get(n)
            if f is None or a is None:
                continue
            total = total + (f * (p - a) ** 2).sum()
        return 0.5 * self.lam * total

    def consolidate(self, grads: dict[str, torch.Tensor | None]) -> None:
        """Snapshot the optimum and the diagonal Fisher after a domain."""
        for n, p in self.model.named_parameters():
            self.anchor[n] = p.detach().clone()
            g = grads.get(n)
            self.fisher[n] = (g.detach() ** 2).clone() if g is not None else torch.zeros_like(p)

    def after_domain(self, loader: Any, device: torch.device) -> None:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        count = 0
        # Fisher wants eval-mode semantics: dropout off, so the curvature is
        # that of the deployed model rather than of a randomly thinned one.
        # cuDNN's fused RNN kernel refuses backward in eval mode, so this pass
        # runs with cuDNN off: unfused kernel, same gradients, and the estimate
        # then matches on CPU and GPU.
        self.model.eval()
        with torch.backends.cudnn.flags(enabled=False):
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                self.model.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(self.model(x), y.long())
                loss.backward()  # type: ignore[no-untyped-call]
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.detach().square()
                count += 1
        if count == 0:
            raise ValueError("cannot estimate Fisher from an empty loader")
        self.anchor = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.fisher = {n: value / count for n, value in fisher.items()}


class SI(Strategy):
    """Synaptic intelligence: path-integral importance accumulated online."""

    name = "si"

    def __init__(self, model: nn.Module, lam: float = 1.0, damping: float = 0.1) -> None:
        super().__init__(model)
        self.lam = lam
        self.damping = damping
        self.omega: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
        }
        self.anchor: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }
        self._w: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
        }
        self._prev: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }
        self._task_grads: dict[str, torch.Tensor | None] = {}

    @property
    def retained(self) -> Any:
        # The running path integral is retained state too, not scratch: it
        # survives between domains and must appear in the byte table.
        return {
            "omega": self.omega,
            "anchor": self.anchor,
            "path": self._w,
            "previous": self._prev,
        }

    def penalty(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        total = torch.zeros((), device=logits.device)
        for n, p in self.model.named_parameters():
            total = total + (self.omega[n] * (p - self.anchor[n]) ** 2).sum()
        return self.lam * total

    def capture_task_gradients(self, task_loss: torch.Tensor) -> None:
        grads = torch.autograd.grad(
            task_loss,
            tuple(self.model.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        self._task_grads = {
            n: None if g is None else g.detach().clone()
            for (n, _), g in zip(self.model.named_parameters(), grads, strict=True)
        }

    def before_optimizer_step(self) -> None:
        for n, p in self.model.named_parameters():
            self._prev[n].copy_(p.detach())

    def after_optimizer_step(self) -> None:
        """Add this step's contribution to the path integral. Call after step()."""
        for n, p in self.model.named_parameters():
            grad = self._task_grads.get(n)
            if grad is None:
                continue
            delta = p.detach() - self._prev[n]
            self._w[n] -= grad * delta
            self._prev[n].copy_(p.detach())

    def consolidate(self) -> None:
        """Fold the path integral into omega at a domain boundary."""
        for n, p in self.model.named_parameters():
            delta = p.detach() - self.anchor[n]
            self.omega[n] += self._w[n] / (delta**2 + self.damping)
            self.omega[n].clamp_(min=0.0)
            self.anchor[n] = p.detach().clone()
            self._w[n] = torch.zeros_like(p)

    def after_domain(self, loader: Any, device: torch.device) -> None:
        self.consolidate()
