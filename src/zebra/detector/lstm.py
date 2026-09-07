"""The corrected lightweight temporal LSTM used by every primary-track method.

The upstream implementation flattens a window into one 140-feature timestep
and uses a two-layer head. The primary protocol intentionally uses ten real
timesteps over fourteen features and a linear binary head. Its 1,062-parameter
size is therefore our lightweight-detector configuration, not an upstream
measurement. Every strategy receives the same module factory.
"""

from __future__ import annotations

import torch
from torch import nn

from zebra.model import N_FEATURES


class Detector(nn.Module):
    """Single-layer LSTM over a window, linear head on the final state."""

    def __init__(self, hidden: int = 10, n_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_FEATURES, hidden_size=hidden, num_layers=1, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    @property
    def configuration(self) -> dict[str, int | float]:
        """Serializable architecture metadata written into every run artifact."""
        return {
            "features": N_FEATURES,
            "hidden": self.lstm.hidden_size,
            "layers": self.lstm.num_layers,
            "classes": self.head.out_features,
            "dropout": self.drop.p,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits: torch.Tensor = self.head(self.drop(out[:, -1, :]))
        return logits
