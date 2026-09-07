"""Every named strategy must have a measurable effect after domain one."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from zebra.detector.lstm import Detector
from zebra.strategies.ewc import EWC
from zebra.strategies.factory import ZeBRaReplay
from zebra.strategies.generative import TemporalVAE
from zebra.strategies.lwf import LearningWithoutForgetting
from zebra.strategies.replay import ExperienceReplay
from zebra.strategies.zebra import ZeBRa


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(32, 10, 14), torch.randint(0, 2, (32,)), torch.zeros(32, 3)


def test_zebra_is_one_sided_and_has_gradient_when_supported() -> None:
    model = Detector(dropout=0.0)
    x, y, nu = _batch()
    logits = model(x)
    zebra = ZeBRa(model)
    assert float(zebra.penalty(x, y, logits, nu).detach()) == 0.0
    supported = torch.ones_like(nu)
    loss = zebra.penalty(x, y, logits, supported)
    assert float(loss.detach()) > 0.0
    loss.backward()  # type: ignore[no-untyped-call]
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_replay_uses_prior_samples_and_respects_capacity() -> None:
    model = Detector()
    replay = ExperienceReplay(model, capacity=20, replay_ratio=0.5, seed=4)
    old_x = torch.full((40, 10, 14), 7.0)
    old_y = torch.ones(40, dtype=torch.long)
    replay.absorb(old_x, old_y)
    assert replay.buf_x is not None and len(replay.buf_x) == 20
    x, y, nu = _batch()
    ax, ay, anu = replay.augment_batch(x, y, nu)
    assert len(ax) == len(x) + 16
    assert torch.all(ax[len(x) :] == 7.0)
    assert torch.all(ay[len(y) :] == 1)
    assert torch.all(anu[len(nu) :] == 0)


def test_ewc_fisher_and_lwf_teacher_are_data_derived() -> None:
    model = Detector(dropout=0.0)
    x, y, nu = _batch()
    loader = DataLoader(TensorDataset(x, y, nu), batch_size=16)
    ewc = EWC(model)
    ewc.after_domain(loader, torch.device("cpu"))
    assert sum(float(value.sum()) for value in ewc.fisher.values()) > 0.0

    lwf = LearningWithoutForgetting(model)
    lwf.after_domain(loader, torch.device("cpu"))
    assert lwf.teacher is not None
    for parameter in model.parameters():
        parameter.data.add_(0.2)
    logits = model(x)
    assert float(lwf.penalty(x, y, logits, nu).detach()) > 0.0


def test_temporal_vae_preserves_window_shape() -> None:
    vae = TemporalVAE(hidden=8, latent=4)
    x = torch.randn(5, 10, 14)
    reconstructed, mu, logvar = vae(x)
    assert reconstructed.shape == x.shape
    assert mu.shape == logvar.shape == (5, 4)


def test_zebra_replay_does_not_dilute_rule_loss_with_replay_rows() -> None:
    model = Detector(dropout=0.0)
    combo = ZeBRaReplay(model, capacity=20, replay_ratio=0.5, seed=4)
    combo.absorb(torch.randn(20, 10, 14), torch.zeros(20, dtype=torch.long))
    x, y, _ = _batch()
    nu = torch.ones(len(x), 3)
    ax, ay, anu = combo.augment_batch(x, y, nu)
    logits = model(ax)
    combined = combo.penalty(ax, ay, logits, anu)
    expected = ZeBRa(model).penalty(x, y, logits[: len(x)], nu)
    assert torch.allclose(combined, expected)


def test_ewc_fisher_pass_survives_eval_mode_on_any_backend() -> None:
    """Regression: cuDNN refuses RNN backward in eval mode, so the Fisher pass
    disables cuDNN. Only CUDA reproduced this; CPU smoke tests never did."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from zebra.detector.lstm import Detector
    from zebra.strategies.ewc import EWC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector().to(device)
    strategy = EWC(model)
    loader = DataLoader(
        TensorDataset(torch.randn(32, 10, 14), torch.randint(0, 2, (32,))), batch_size=8
    )
    strategy.after_domain(loader, device)
    assert strategy.fisher, "Fisher was not populated"
    assert all(torch.isfinite(v).all() for v in strategy.fisher.values())
