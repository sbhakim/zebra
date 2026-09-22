"""Byte-efficient replay: fidelity, residency, and the accounting it reports."""

from __future__ import annotations

import torch

from zebra.detector.lstm import Detector
from zebra.memory.accounting import account, assert_materialised
from zebra.strategies.factory import build_strategy
from zebra.strategies.quantized import dequantise, quantise

WINDOW_BYTES_F32 = 568
WINDOW_BYTES_Q8 = 156


def test_round_trip_preserves_the_out_of_range_tail() -> None:
    """The frozen scaler leaves values far outside [0, 1]; they must survive.

    corpus.py keeps them deliberately -- clipping would conceal domain shift --
    so a global uint8 range is wrong here and the extremes are what prove it.
    """
    window = torch.rand(10, 14)
    window[3, 5] = 30.127
    window[7, 2] = -0.201
    codes, lo, hi = quantise(window)
    back = dequantise(codes.unsqueeze(0), lo, hi)[0]
    assert torch.isclose(back.max(), window.max(), atol=1e-4)
    assert torch.isclose(back.min(), window.min(), atol=1e-4)


def test_quantisation_error_is_small_relative_to_the_signal() -> None:
    torch.manual_seed(0)
    window = torch.rand(10, 14)
    codes, lo, hi = quantise(window)
    back = dequantise(codes.unsqueeze(0), lo, hi)[0]
    assert (back - window).abs().mean() < 0.01 * window.std()


def test_constant_window_does_not_divide_by_zero() -> None:
    window = torch.full((10, 14), 0.7)
    codes, lo, hi = quantise(window)
    back = dequantise(codes.unsqueeze(0), lo, hi)[0]
    assert torch.allclose(back, window, atol=1e-6)


def test_buffer_costs_156_bytes_per_window() -> None:
    """The whole claim is bytes, so the accounting -- not arithmetic -- checks it."""
    capacity = 32
    model = Detector()
    strategy = build_strategy("replay-q8", model, seed=7, overrides={"capacity": capacity})
    strategy.absorb(torch.rand(capacity, 10, 14), torch.randint(0, 2, (capacity,)))
    report = account("replay-q8", model, strategy.retained, None)
    assert report.retained_bytes == capacity * WINDOW_BYTES_Q8
    assert report.retained_bytes < capacity * WINDOW_BYTES_F32


def test_retained_state_is_materialised_not_a_view() -> None:
    model = Detector()
    strategy = build_strategy("replay-q8", model, seed=7, overrides={"capacity": 8})
    strategy.absorb(torch.rand(20, 10, 14), torch.randint(0, 2, (20,)))
    for name, value in strategy.retained.items():
        if isinstance(value, torch.Tensor):
            assert_materialised(name, value)


def test_reservoir_respects_capacity() -> None:
    strategy = build_strategy("replay-q8", Detector(), seed=7, overrides={"capacity": 5})
    strategy.absorb(torch.rand(40, 10, 14), torch.randint(0, 2, (40,)))
    assert len(strategy.retained["buffer_codes"]) == 5
    assert len(strategy.retained["buffer_y"]) == 5


def test_sampled_windows_have_the_detector_input_shape() -> None:
    strategy = build_strategy("replay-q8", Detector(), seed=7, overrides={"capacity": 16})
    strategy.absorb(torch.rand(16, 10, 14), torch.randint(0, 2, (16,)))
    sampled = strategy.sample(4, torch.device("cpu"))
    assert sampled is not None
    windows, labels = sampled
    assert windows.shape == (4, 10, 14) and windows.dtype == torch.float32
    assert labels.shape == (4,)


def test_hybrid_fits_the_class_2_envelope() -> None:
    """190 quantised windows plus a teacher is the largest hybrid that fits."""
    model = Detector()
    strategy = build_strategy("lwf-q8replay", model, seed=7, overrides={"capacity": 190})
    strategy.absorb(torch.rand(190, 10, 14), torch.randint(0, 2, (190,)))
    strategy.teacher = Detector().eval()
    report = account("lwf-q8replay", model, strategy.retained, None)
    budget = 50 * 1024 - (4248 + 8520 + 4248 + 224)  # C2 minus fixed adaptation cost
    assert report.retained_bytes <= budget


def test_hybrid_penalty_is_zero_before_the_first_teacher() -> None:
    model = Detector()
    strategy = build_strategy("lwf-q8replay", model, seed=7)
    logits = model(torch.rand(4, 10, 14))
    zero = strategy.penalty(torch.rand(4, 10, 14), torch.zeros(4), logits, torch.zeros(4, 3))
    assert float(zero) == 0.0
