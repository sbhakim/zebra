"""Single strategy factory used by runners and memory reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from zebra.strategies.base import Strategy
from zebra.strategies.ewc import EWC, SI
from zebra.strategies.generative import GenerativeReplay
from zebra.strategies.lwf import LearningWithoutForgetting
from zebra.strategies.naive import Naive
from zebra.strategies.replay import ExperienceReplay
from zebra.strategies.zebra import ShuffledZeBRa, ZeBRa, one_sided_constraint_loss

STRATEGY_DEFAULTS: dict[str, dict[str, float | int]] = {
    "no-cl": {},
    "ewc": {"lam": 1.0},
    "si": {"lam": 1.0, "damping": 0.1},
    "lwf": {"alpha": 0.5, "temperature": 2.0},
    "replay": {"capacity": 2000, "replay_ratio": 0.5},
    "gen-replay": {
        "replay_ratio": 0.5,
        "temperature": 2.0,
        "vae_epochs": 5,
        "vae_lr": 1e-3,
    },
    "zebra": {"lam": 0.1},
    "zebra-r1": {"lam": 0.1},
    "zebra-r2": {"lam": 0.1},
    "zebra-r3": {"lam": 0.1},
    "zebra-shuffled": {"lam": 0.1},
    "zebra-replay": {"lam": 0.1, "capacity": 2000, "replay_ratio": 0.5},
}


class ZeBRaReplay(ExperienceReplay):
    name = "zebra-replay"

    def __init__(self, model: nn.Module, *, lam: float = 0.1, **kwargs: object) -> None:
        super().__init__(model, **kwargs)  # type: ignore[arg-type]
        self.lam = lam
        self._current_count = 0

    def augment_batch(
        self, x: torch.Tensor, y: torch.Tensor, nu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._current_count = len(x)
        return super().augment_batch(x, y, nu)

    def penalty(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        # Replay rows deliberately have no stored raw-rule tensor. Restricting
        # the rule mean to current rows prevents batch augmentation from
        # silently reducing lambda and confounding this complementarity control.
        count = self._current_count or len(logits)
        return one_sided_constraint_loss(logits[:count], nu[:count], self.lam)


def strategy_parameters(
    name: str, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the complete, artifact-ready hyperparameter mapping."""
    if name not in STRATEGY_DEFAULTS:
        raise ValueError(f"unknown strategy {name!r}")
    return {**STRATEGY_DEFAULTS[name], **(overrides or {})}


def build_strategy(
    name: str,
    model: nn.Module,
    *,
    seed: int = 42,
    overrides: Mapping[str, Any] | None = None,
) -> Strategy:
    parameters = strategy_parameters(name, overrides)
    if name == "no-cl":
        if parameters:
            raise ValueError("no-cl accepts no strategy hyperparameters")
        return Naive(model)
    if name == "ewc":
        return EWC(model, **parameters)
    if name == "si":
        return SI(model, **parameters)
    if name == "lwf":
        return LearningWithoutForgetting(model, **parameters)
    if name == "replay":
        return ExperienceReplay(model, seed=seed, **parameters)
    if name == "gen-replay":
        return GenerativeReplay(model, **parameters)
    if name == "zebra":
        return ZeBRa(model, **parameters)
    if name in {"zebra-r1", "zebra-r2", "zebra-r3"}:
        index = {"zebra-r1": 0, "zebra-r2": 1, "zebra-r3": 2}[name]
        return ZeBRa(model, rule_indices=(index,), **parameters)
    if name == "zebra-shuffled":
        return ShuffledZeBRa(model, **parameters)
    if name == "zebra-replay":
        return ZeBRaReplay(model, seed=seed, **parameters)
    raise AssertionError("strategy registry and factory branches diverged")
