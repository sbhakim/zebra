"""Sequential fine-tuning: the no-CL floor. Retains nothing."""

from __future__ import annotations

from typing import Any

from zebra.strategies.base import Strategy


class Naive(Strategy):
    name = "no-cl"

    @property
    def retained(self) -> Any:
        return None
