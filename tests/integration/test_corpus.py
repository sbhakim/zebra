"""Corpus-backed split, shape, and leakage checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from zebra.data.corpus import discover_domains, load_domain, split_runs

ROOT = Path("corpus/IoT-Attacks-IDS/src/attack_data")


@pytest.mark.integration
def test_all_domains_have_disjoint_run_splits() -> None:
    domains = discover_domains(ROOT)
    assert len(domains) == 60
    for ident in domains:
        train, validation, test = split_runs(ROOT / ident.name)
        names = [set(p.name for p in group) for group in (train, validation, test)]
        assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
        assert len(train) == 12
        assert len(validation) == 4
        assert len(test) in {3, 4}


@pytest.mark.integration
def test_loaded_domain_is_temporal_aligned_and_finite() -> None:
    ident = next(d for d in discover_domains(ROOT) if d.name == "localrepair_var5_oo")
    domain = load_domain(ROOT, ident)
    for split in (domain.train, domain.validation, domain.test):
        assert split.x.ndim == 3 and split.x.shape[1:] == (10, 14)
        assert len(split.x) == len(split.y) == len(split.nu)
        assert torch.isfinite(split.x).all()
        assert torch.isfinite(split.nu).all()
