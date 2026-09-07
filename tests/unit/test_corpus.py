"""Preprocessing contracts whose failure would invalidate every result."""

from __future__ import annotations

import numpy as np

from zebra.data.corpus import FeatureScaler, _windows


def test_windows_preserve_time_and_use_final_label() -> None:
    x = np.arange(12 * 14, dtype=np.float64).reshape(12, 14)
    y = np.arange(12, dtype=np.int64) % 2
    wx, wy = _windows(x, y, length=10, step=1)
    assert wx.shape == (3, 10, 14)
    assert np.array_equal(wx[0], x[:10])
    assert np.array_equal(wx[-1], x[2:12])
    assert np.array_equal(wy, y[[9, 10, 11]])


def test_window_stride_is_explicit() -> None:
    x = np.zeros((16, 14), dtype=np.float64)
    y = np.zeros(16, dtype=np.int64)
    wx, _ = _windows(x, y, length=10, step=3)
    assert len(wx) == 3  # starts 0, 3, 6; the final valid window is included


def test_fixed_scaler_metadata_is_byte_explicit() -> None:
    scaler = FeatureScaler((0.0,) * 14, (1.0,) * 14, "first-domain")
    assert scaler.metadata_bytes == 28 * 8 == 224
