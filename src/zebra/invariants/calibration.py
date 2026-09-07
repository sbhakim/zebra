"""Training-only calibration for fixed protocol-guided rule signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RuleCalibration:
    """Map raw rule signals to conservative one-sided violation strengths.

    Thresholds and scales are estimated from benign *training* windows before
    sequential learning. A score below the benign 99th percentile asserts
    nothing. This avoids treating rule compliance as proof of benign behavior.
    """

    thresholds: tuple[float, float, float]
    scales: tuple[float, float, float]
    quantile: float
    n_benign: int
    source_domains: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        signals: np.ndarray,
        labels: np.ndarray,
        *,
        quantile: float = 0.99,
        source_domains: tuple[str, ...] = (),
    ) -> RuleCalibration:
        if signals.ndim != 2 or signals.shape[1] != 3:
            raise ValueError(f"expected (N, 3) rule signals, got {signals.shape}")
        if len(signals) != len(labels):
            raise ValueError("signals and labels are not aligned")
        benign = np.asarray(signals[np.asarray(labels) == 0], dtype=np.float64)
        if len(benign) < 100:
            raise ValueError("at least 100 benign training windows are required for calibration")
        thresholds = np.quantile(benign, quantile, axis=0)
        upper = np.quantile(benign, min(0.999, (1.0 + quantile) / 2.0), axis=0)
        scales = np.maximum(upper - thresholds, 1e-3)
        return cls(
            thresholds=tuple(float(v) for v in thresholds),  # type: ignore[arg-type]
            scales=tuple(float(v) for v in scales),  # type: ignore[arg-type]
            quantile=quantile,
            n_benign=len(benign),
            source_domains=source_domains,
        )

    def transform(self, signals: np.ndarray) -> np.ndarray:
        values = np.asarray(signals, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError(f"expected final rule axis of size 3, got {values.shape}")
        threshold = np.asarray(self.thresholds)
        scale = np.asarray(self.scales)
        return np.asarray(np.clip((values - threshold) / scale, 0.0, 1.0), dtype=np.float64)

    @property
    def metadata_bytes(self) -> int:
        """Resident numeric metadata, reported separately from CL state."""
        return 6 * np.dtype(np.float64).itemsize

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def benign_fire_rate(calibrated: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fraction of benign windows on which each rule makes a positive assertion."""
    benign = calibrated[np.asarray(labels) == 0]
    if not len(benign):
        raise ValueError("no benign windows")
    return np.asarray(np.mean(benign > 0.0, axis=0), dtype=np.float64)
