"""Released per-minute CSVs -> temporal train/val/test tensors.

One CSV per simulation run; each row is a minute of sink-aggregated control
stats (7 means, 7 sds, label). Runs shuffle under a local seed and split
12/4/rest, min--max fitted on training files only. The runner freezes one
scaler from the first domain -- a per-domain scaler leaks domain identity.

Upstream early-stops on its test split and flattens the window, so this is not
bit-for-bit parity; see docs/EXPERIMENT_PROTOCOL.md. Invariant scores are taken
from the same pre-normalisation windows (ADR-001).
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zebra.model import FEATURE_STEMS, N_FEATURES, DomainId

_RUN_RE = re.compile(r"^(\d+)_features_timeseries_60_sec\.csv$")
LABEL_COL = "label"
N_RUNS = 20
N_TRAIN = 12
N_VALIDATION = 4
SPLIT_SEED = 42
SCALER_METADATA_BYTES = 2 * N_FEATURES * np.dtype(np.float64).itemsize


@dataclass(frozen=True)
class Split:
    """One side of a domain: normalised windows, labels, and invariant scores."""

    x: torch.Tensor  # (N, L, 14) normalised, model input
    y: torch.Tensor  # (N,)
    nu: torch.Tensor  # (N, 3) invariant violation, from raw units


@dataclass(frozen=True)
class LoadedDomain:
    """A domain as the training loop consumes it."""

    ident: DomainId
    train: Split
    validation: Split
    test: Split
    source: Path


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    """Fixed min--max map fitted from named training runs only."""

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    source_domain: str

    @property
    def metadata_bytes(self) -> int:
        return SCALER_METADATA_BYTES

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The 14 feature columns in canonical order: seven means, seven sigmas.

    The released CSVs name sigmas by pandas' duplicate-column suffix
    (``rank.1``), so the layout is positional in practice. We assert the stems
    rather than trusting position.
    """
    cols = [*FEATURE_STEMS, *(f"{s}.1" for s in FEATURE_STEMS)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"corpus CSV missing expected columns: {missing}")
    return cols


def discover_domains(root: Path) -> list[DomainId]:
    """Every domain folder under the corpus root, sorted for determinism."""
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root not found: {root}")
    idents = [DomainId.parse(p.name) for p in sorted(root.iterdir()) if p.is_dir()]
    if not idents:
        raise FileNotFoundError(f"no domain folders under {root}")
    return idents


def _run_index(path: Path) -> int:
    m = _RUN_RE.match(path.name)
    if m is None:
        raise ValueError(f"not a run file: {path}")
    return int(m.group(1))


# Two domains deviate from the nominal 20 runs, and this is upstream reality
# rather than a checkout artefact: disflooding_var10_oo ships 21 and
# disflooding_var15_base ships 19. Upstream truncates with [:20], so the extra
# run is dropped and the short domain gets a three-file test split. We match
# that and record the deviation instead of raising, because raising would drop
# two domains the published baselines include.
KNOWN_RUN_COUNTS: dict[str, int] = {
    "disflooding_var10_oo": 21,
    "disflooding_var15_base": 19,
}


def split_runs(folder: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Create deterministic, disjoint run-level train/validation/test splits."""
    runs = sorted((p for p in folder.iterdir() if _RUN_RE.match(p.name)), key=_run_index)
    if len(runs) != N_RUNS and KNOWN_RUN_COUNTS.get(folder.name) != len(runs):
        raise ValueError(
            f"{folder.name}: {len(runs)} runs, expected {N_RUNS} and not a known deviation"
        )
    runs = runs[:N_RUNS]
    rng = random.Random(SPLIT_SEED)  # noqa: S311 -- reproducing upstream's split, not crypto
    rng.shuffle(runs)
    train = runs[:N_TRAIN]
    validation = runs[N_TRAIN : N_TRAIN + N_VALIDATION]
    test = runs[N_TRAIN + N_VALIDATION :]
    if not train or not validation or not test:
        raise ValueError(f"{folder.name}: split produced an empty partition")
    return train, validation, test


def _windows(a: np.ndarray, y: np.ndarray, length: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows labelled by their final timestep."""
    xs, ys = [], []
    for i in range(0, len(a) - length + 1, step):
        xs.append(a[i : i + length])
        ys.append(y[i + length - 1])
    if not xs:
        return np.empty((0, length, N_FEATURES), np.float64), np.empty((0,), np.int64)
    return np.stack(xs), np.asarray(ys, np.int64)


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    a = df[feature_columns(df)].to_numpy(np.float64)
    y = df[LABEL_COL].to_numpy(np.float64).round().astype(np.int64)
    return a, y


def _build(files: list[Path], scaler: FeatureScaler, length: int, step: int) -> Split:
    from zebra.invariants.registry import violation_scores

    lo = np.asarray(scaler.lo, dtype=np.float64)
    hi = np.asarray(scaler.hi, dtype=np.float64)
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    xs, ys, nus = [], [], []
    for p in files:
        a, y = _read(p)
        wx_raw, wy = _windows(a, y, length, step)
        if not len(wx_raw):
            continue
        nus.append(violation_scores(wx_raw))  # raw units, ADR-001
        # Retain held-out values outside the training range. Clipping would
        # conceal genuine domain shift.
        wx = (wx_raw - lo) / span
        # Model inputs must be finite; invariant scores use raw NaN-aware data.
        xs.append(np.nan_to_num(wx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        ys.append(wy)
    if not xs:
        raise ValueError("every run was shorter than the window length")
    return Split(
        x=torch.from_numpy(np.concatenate(xs)),
        y=torch.from_numpy(np.concatenate(ys)),
        nu=torch.from_numpy(np.concatenate(nus).astype(np.float32)),
    )


def fit_feature_scaler(root: Path, ident: DomainId) -> FeatureScaler:
    """Fit the stream's fixed scaler from one domain's training runs."""
    train_files, _, _ = split_runs(root / ident.name)
    mins, maxs = [], []
    for path in train_files:
        values, _ = _read(path)
        mins.append(np.nanmin(values, axis=0))
        maxs.append(np.nanmax(values, axis=0))
    lo = np.nanmin(np.stack(mins), axis=0)
    hi = np.nanmax(np.stack(maxs), axis=0)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError(f"{ident.name}: training runs cannot define a finite scaler")
    return FeatureScaler(
        tuple(float(value) for value in lo),
        tuple(float(value) for value in hi),
        ident.name,
    )


def load_domain(
    root: Path,
    ident: DomainId,
    *,
    length: int = 10,
    step: int = 3,
    scaler: FeatureScaler | None = None,
) -> LoadedDomain:
    """Load one domain under the corrected temporal evaluation protocol."""
    folder = root / ident.name
    train_files, validation_files, test_files = split_runs(folder)

    active_scaler = scaler or fit_feature_scaler(root, ident)

    return LoadedDomain(
        ident=ident,
        train=_build(train_files, active_scaler, length, step),
        validation=_build(validation_files, active_scaler, length, step),
        test=_build(test_files, active_scaler, length, step),
        source=folder,
    )


def raw_windows(
    root: Path, ident: DomainId, *, length: int = 10, step: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """All windows of a domain in raw units, for probes and fixture building.

    Deliberately separate from `load_domain`: probes characterise an invariant
    over the whole domain, while training must respect the split.
    """
    folder = root / ident.name
    runs = sorted((p for p in folder.iterdir() if _RUN_RE.match(p.name)), key=_run_index)
    xs, ys = [], []
    for p in runs:
        a, y = _read(p)
        wx, wy = _windows(a, y, length, step)
        if len(wx):
            xs.append(wx)
            ys.append(wy)
    return np.concatenate(xs), np.concatenate(ys)
