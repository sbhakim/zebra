"""Typed records exchanged between components.

Boundaries pass dataclasses, never raw dicts or bare tensors with an implied
layout, so a mislabelled axis is a type error rather than a wrong number in a
table. ``MemoryReport`` and ``Estimate`` are frozen on purpose: the first is
the byte accounting, the second is the one type every reported proportion goes
through, so no two tables can mix interval conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------- corpus ----


class Attack(StrEnum):
    """The five attack families present in the released corpus.

    VERSIONNUMBER is in the archive but not in the source study's analysis; we
    report it as a separate stratum rather than folding it into the pooled
    result.
    """

    BLACKHOLE = "blackhole"
    DISFLOODING = "disflooding"
    LOCALREPAIR = "localrepair"
    WORSTPARENT = "worstparent"
    VERSIONNUMBER = "versionnumber"


class Variant(StrEnum):
    """Behavioural variant of an attack within a domain.

    The release spells the gradual-decrease variant two ways: the four attacks
    the source study analyses use ``dec``, while ``versionnumber`` -- released
    but not analysed there -- uses ``gc``. They denote the same variant, so we
    alias rather than fork the enum, but the alias is recorded here and in
    docs/STATUS.md because it is evidence that versionnumber was generated in a
    separate pass. That matters for whether it may be pooled with the rest, and
    we keep it as its own stratum for exactly this reason.
    """

    BASE = "base"
    ONOFF = "oo"
    DECREASE = "dec"

    @classmethod
    def _missing_(cls, value: object) -> Variant | None:
        return cls.DECREASE if value == "gc" else None


@dataclass(frozen=True, slots=True)
class DomainId:
    """Identity of one domain: attack family, network size, variant.

    ``folder`` keeps the on-disk spelling while ``variant`` is canonical. The
    two differ only for ``versionnumber``, whose gradual-decrease folders are
    named ``gc`` rather than ``dec``. Deriving the path from the canonical
    variant instead would make those twelve domains unreadable, so the record
    carries both and never reconstructs a path from the enum.
    """

    attack: Attack
    size: int
    variant: Variant
    folder: str

    @property
    def name(self) -> str:
        return self.folder

    @property
    def canonical(self) -> str:
        return f"{self.attack.value}_var{self.size}_{self.variant.value}"

    @classmethod
    def parse(cls, folder: str) -> DomainId:
        """Parse ``blackhole_var10_base`` into a DomainId.

        Raises rather than guessing: an unrecognised folder is a corpus problem
        we want to see, not a domain to silently skip.
        """
        try:
            attack, rest = folder.split("_var", 1)
            size_s, variant = rest.split("_", 1)
            return cls(Attack(attack), int(size_s), Variant(variant), folder)
        except (ValueError, KeyError) as exc:  # noqa: B904
            raise ValueError(f"unrecognised domain folder: {folder!r}") from exc


# The seven aggregated control-plane statistics, in the column order the
# released CSVs use. Means come first, then the matching standard deviations.
FEATURE_STEMS: tuple[str, ...] = ("rank", "disr", "diss", "dior", "dios", "diar", "tots")
N_FEATURES: int = 2 * len(FEATURE_STEMS)  # 14


@dataclass(frozen=True, slots=True)
class Domain:
    """One loaded domain: windowed features, labels, and provenance."""

    ident: DomainId
    x: object  # torch.Tensor (N, L, 14); typed loosely to keep torch out of this module
    y: object  # torch.Tensor (N,)
    n_runs: int
    source: Path


# ------------------------------------------------------------ invariants ----


class InvariantId(StrEnum):
    """Frozen rule identifiers. Legacy names remain aliases for old artifacts."""

    SOLICITATION = "R1-SOL"
    TRICKLE_REGULARITY = "R2-REG"
    DIO_DISPERSION = "R3-DISP"
    CONSERVATION = SOLICITATION
    TRICKLE = TRICKLE_REGULARITY
    RANK = DIO_DISPERSION


@dataclass(frozen=True, slots=True)
class InvariantSpec:
    """Metadata every protocol-guided signal must carry to be registered.

    ``observable`` records whether the named quantity can be evaluated from
    sink-aggregated features. The historical class name is retained for artifact
    compatibility; no formal-invariant claim follows from it.
    """

    ident: InvariantId
    title: str
    rationale: str
    targets: tuple[Attack, ...]
    observable: bool = True


# --------------------------------------------------------------- memory ----


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """Resident bytes for one strategy, split by what holds them.

    Transient activations are excluded on purpose: they scale with batch size
    rather than with strategy, so including them would mask the differences the
    paper is about. The optimiser is reported separately for the same reason --
    Adam costs two floats per parameter for every strategy alike.
    """

    strategy: str
    model_bytes: int
    retained_bytes: int
    optimiser_bytes: int
    gradient_bytes: int = 0
    preprocessor_bytes: int = 0
    static_rule_bytes: int = 0
    python_overhead_bytes: int = 0
    detail: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return (
            self.model_bytes
            + self.retained_bytes
            + self.optimiser_bytes
            + self.gradient_bytes
            + self.preprocessor_bytes
            + self.static_rule_bytes
            + self.python_overhead_bytes
        )


@dataclass(frozen=True, slots=True)
class DeviceClass:
    """A constrained-node class from RFC 7228, Section 3."""

    name: str
    ram_bytes: int | None
    flash_bytes: int | None


# RFC 7228 Table 1. KiB = 1024 bytes.
RFC7228_CLASSES: tuple[DeviceClass, ...] = (
    # RFC 7228 describes Class 0 as "much less than" these values rather
    # than defining a numerical capacity. Treating 10 KiB as a pass/fail
    # boundary would incorrectly make it identical to Class 1.
    DeviceClass("Class 0", None, None),
    DeviceClass("Class 1", 10 * 1024, 100 * 1024),
    DeviceClass("Class 2", 50 * 1024, 250 * 1024),
)


# -------------------------------------------------------------- results ----


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with an interval.

    Every proportion the paper reports is constructed here so that no two
    tables can disagree about what their intervals mean.
    """

    value: float
    low: float
    high: float
    n: int

    def as_pct(self) -> str:
        return f"{100 * self.value:.1f}"


@dataclass(frozen=True, slots=True)
class DomainResult:
    """Performance of one model state on one domain."""

    trained_upto: int
    evaluated_on: int
    auc: float
    f1: float


@dataclass(frozen=True, slots=True)
class RunResult:
    """One (strategy, ordering, seed) run: the full accuracy matrix plus cost."""

    strategy: str
    ordering: str
    seed: int
    order: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]  # A[i][j]: perf on domain j after training i
    memory: MemoryReport
    wall_seconds: float
