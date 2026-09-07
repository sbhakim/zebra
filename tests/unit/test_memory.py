"""The byte accounting is the paper's first contribution; pin it."""

from __future__ import annotations

import torch

from zebra.detector.lstm import Detector
from zebra.memory.accounting import account, module_bytes
from zebra.strategies.replay import ExperienceReplay
from zebra.strategies.zebra import ZeBRa


def test_standardized_detector_size_is_pinned() -> None:
    m = Detector()
    assert sum(p.numel() for p in m.parameters()) == 1062
    assert module_bytes(m) == 1062 * 4


def test_zebra_retains_exactly_zero() -> None:
    m = Detector()
    r = account("zebra", m, ZeBRa(m).retained)
    assert r.retained_bytes == 0, "the zero-history claim requires no retained examples"


def test_replay_window_is_560_bytes() -> None:
    m = Detector()
    rep = ExperienceReplay(m, capacity=1000)
    rep.absorb(torch.zeros(1000, 10, 14), torch.zeros(1000, dtype=torch.long))
    x_bytes = rep.buf_x.untyped_storage().nbytes()
    assert x_bytes // 1000 == 10 * 14 * 4 == 560


def test_replay_dwarfs_the_model_it_protects() -> None:
    m = Detector()
    rep = ExperienceReplay(m)
    for _ in range(8):
        rep.absorb(torch.zeros(500, 10, 14), torch.zeros(500, dtype=torch.long))
    r = account("replay", m, rep.retained)
    assert r.retained_bytes / r.model_bytes > 200


def test_ewc_and_si_actually_penalise_once_consolidated() -> None:
    """Regression: both were inherited no-ops, i.e. fine-tuning in disguise."""
    from zebra.strategies.ewc import EWC, SI

    m = Detector()
    x = torch.randn(8, 10, 14)
    logits = m(x)
    torch.nn.functional.cross_entropy(logits, torch.zeros(8, dtype=torch.long)).backward()

    ewc = EWC(m)
    y = torch.zeros(8, dtype=torch.long)
    nu = torch.zeros(8, 3)
    assert float(ewc.penalty(x, y, logits, nu)) == 0.0  # nothing learned yet
    ewc.consolidate({n: p.grad for n, p in m.named_parameters()})
    for p in m.parameters():
        p.data.add_(0.5)
    assert float(ewc.penalty(x, y, logits, nu).detach()) > 0.0

    si = SI(m)
    logits = m(x)
    task_loss = torch.nn.functional.cross_entropy(logits, y)
    si.capture_task_gradients(task_loss)
    si.before_optimizer_step()
    for p in m.parameters():
        p.data.add_(0.01)
    si.after_optimizer_step()
    si.consolidate()
    si.omega["head.bias"] += 1.0
    for p in m.parameters():
        p.data.add_(0.5)
    assert float(si.penalty(x, y, logits, nu).detach()) > 0.0


def test_retained_state_is_materialised_not_a_view() -> None:
    """A retained view would pin its whole base and silently inflate the table."""
    from zebra.memory.accounting import assert_materialised

    big = torch.zeros(1000, 10, 14)
    assert_materialised("clone", big[:10].clone())
    try:
        assert_materialised("view", big[:10])
    except AssertionError:
        return
    raise AssertionError("a retained view should have been rejected")
