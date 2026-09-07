"""Resident-byte accounting for continual-learning strategies.

Walks what a strategy actually holds between domains and sums storage, rather
than multiplying a sample count by an assumed element size.

Three boundaries, each of which changes the answer: activations are excluded
(they scale with batch, not with strategy), the optimiser is reported
separately (Adam holds two moments per parameter for everyone alike), and only
resident state counts -- a generator kept around for sampling does, a
checkpoint written to disk does not.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn

from zebra.model import RFC7228_CLASSES, DeviceClass, MemoryReport


def tensor_bytes(t: torch.Tensor) -> int:
    """Storage actually pinned by this tensor.

    A view keeps its whole base storage alive, so charging the storage rather
    than the logical extent is the honest answer for a residency question: a
    strategy that retains ``big[:10]`` really does hold all of ``big``.

    The consequence is that an accidental view inflates the table by orders of
    magnitude, which is why `assert_materialised` exists and why every strategy
    clones what it keeps.
    """
    return int(t.untyped_storage().nbytes())


def assert_materialised(name: str, t: torch.Tensor) -> None:
    """Fail loudly if retained state is a view into something larger.

    Without this, a strategy could appear to hold a megabyte because it sliced
    one, or -- worse -- genuinely pin a megabyte and have nobody notice.
    """
    logical = t.element_size() * t.nelement()
    pinned = tensor_bytes(t)
    if pinned > logical:
        raise AssertionError(
            f"{name}: retains a view pinning {pinned} B for {logical} B of data; "
            "clone it so the accounting is interpretable"
        )


def module_bytes(m: nn.Module) -> int:
    """Parameters plus persistent buffers of a module."""
    return sum(tensor_bytes(p) for p in m.parameters()) + sum(
        tensor_bytes(b) for b in m.buffers()
    )


def optimiser_bytes(opt: torch.optim.Optimizer | None) -> int:
    """Resident optimiser state (Adam moments, momentum buffers, ...)."""
    if opt is None:
        return 0
    total = 0
    for state in opt.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += tensor_bytes(v)
    return total


def gradient_bytes(model: nn.Module) -> int:
    """Worst-case gradient tensor payload during adaptation."""
    return sum(tensor_bytes(parameter) for parameter in model.parameters())


def python_overhead(obj: Any, seen: set[int] | None = None) -> int:
    """Shallow CPython container/tensor-object overhead, separate from payload."""
    visited = set() if seen is None else seen
    if id(obj) in visited:
        return 0
    visited.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        total += sum(
            python_overhead(k, visited) + python_overhead(v, visited)
            for k, v in obj.items()
        )
    elif isinstance(obj, list | tuple | set):
        total += sum(python_overhead(value, visited) for value in obj)
    return total


def _walk(obj: Any, seen: set[int]) -> int:
    """Sum tensor storage reachable from a strategy's retained state.

    Deliberately shallow-but-general: tensors, modules, and the containers we
    actually use. An unrecognised object contributes zero and is the caller's
    responsibility to declare -- silently guessing at an unknown container is
    how an accounting table becomes fiction.
    """
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return tensor_bytes(obj)
    if isinstance(obj, nn.Module):
        return module_bytes(obj)
    if isinstance(obj, dict):
        return sum(_walk(v, seen) for v in obj.values())
    if isinstance(obj, list | tuple | set):
        return sum(_walk(v, seen) for v in obj)
    return 0


def account(
    strategy: str,
    model: nn.Module,
    retained: Any,
    optimiser: torch.optim.Optimizer | None = None,
    *,
    preprocessor_bytes: int = 0,
    static_rule_bytes: int = 0,
    adapting: bool = True,
) -> MemoryReport:
    """Measure one strategy's resident footprint.

    ``retained`` is whatever the strategy carries between domains: a replay
    tensor, a dict of importance weights, a teacher module, a generator, or
    ``None`` for a strategy that carries nothing.
    """
    detail: dict[str, int] = {}
    if isinstance(retained, dict):
        for k, v in retained.items():
            detail[str(k)] = _walk(v, set())
    return MemoryReport(
        strategy=strategy,
        model_bytes=module_bytes(model),
        retained_bytes=_walk(retained, set()),
        optimiser_bytes=optimiser_bytes(optimiser),
        gradient_bytes=gradient_bytes(model) if adapting else 0,
        preprocessor_bytes=preprocessor_bytes,
        static_rule_bytes=static_rule_bytes,
        python_overhead_bytes=python_overhead(retained),
        detail=detail,
    )


def fits_class(report: MemoryReport, cls: DeviceClass, *, adapting: bool = True) -> bool | None:
    """Does the strategy fit this class's data memory (RFC 7228)?

    Two regimes, and the paper reports the second as primary:

    * ``adapting=False`` -- inference only. Model plus retained state.
    * ``adapting=True`` -- on-device continual adaptation, which additionally
      needs the optimiser resident. This is the regime the paper is about: a
      detector that cannot update on the device is not doing continual learning
      there, and the optimiser is 8.5 kB against Class 1's 10 KiB, so including
      it is decisive rather than cosmetic.
    """
    if cls.ram_bytes is None:
        return None
    total = (
        report.model_bytes
        + report.retained_bytes
        + report.preprocessor_bytes
        + report.static_rule_bytes
    )
    if adapting:
        total += report.optimiser_bytes + report.gradient_bytes
    return total <= cls.ram_bytes


def class_table(report: MemoryReport, *, adapting: bool = True) -> dict[str, bool | None]:
    return {c.name: fits_class(report, c, adapting=adapting) for c in RFC7228_CLASSES}
