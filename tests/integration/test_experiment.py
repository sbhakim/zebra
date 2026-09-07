"""Small corpus-backed run proving the evidence pipeline is connected."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zebra.data.corpus import discover_domains
from zebra.eval.experiment import ExperimentConfig, fit_rule_calibration, run_one

ROOT = Path("corpus/IoT-Attacks-IDS/src/attack_data")


@pytest.mark.integration
def test_two_domain_run_writes_complete_auditable_artifact(tmp_path: Path) -> None:
    domains = discover_domains(ROOT)[:2]
    config = ExperimentConfig(
        corpus=ROOT,
        output=tmp_path,
        epochs=1,
        patience=1,
        batch_size=512,
        max_domains=2,
    )
    calibration = fit_rule_calibration(config, domains[:1])
    result = run_one(config, "no-cl", "integration", tuple(domains), 7, calibration)

    artifact = tmp_path / "no-cl__integration__seed-7.json"
    checkpoint = tmp_path / "no-cl__integration__seed-7.checkpoint.pt"
    assert artifact.is_file() and checkpoint.is_file()
    saved = json.loads(artifact.read_text())
    assert saved["completed_domains"] == 2
    assert saved["provenance"]["corpus"]["commit"] == config.expected_corpus_commit
    assert saved["calibration"]["source_domains"] == [domains[0].name]
    assert saved["preprocessor"]["source_domain"] == domains[0].name
    assert saved["memory"][-1]["preprocessor_bytes"] == 224
    assert set(saved["run_partitions"][domains[0].name]) == {
        "train",
        "validation",
        "test",
    }
    assert len(saved["auc_matrix"][-1]) == 2
    assert result["summary"] == saved["summary"]

    # A completed checkpoint is restart-safe and resolves to the same artifact.
    resumed = run_one(config, "no-cl", "integration", tuple(domains), 7, calibration)
    assert resumed["summary"] == saved["summary"]

    repeat_config = ExperimentConfig(
        corpus=ROOT,
        output=tmp_path / "independent-repeat",
        epochs=1,
        patience=1,
        batch_size=512,
        max_domains=2,
    )
    repeated = run_one(
        repeat_config,
        "no-cl",
        "integration",
        tuple(domains),
        7,
        calibration,
    )
    assert repeated["auc_matrix"] == saved["auc_matrix"]
    assert repeated["f1_matrix"] == saved["f1_matrix"]


def test_resume_restores_rng_state_on_the_active_device(tmp_path: Path) -> None:
    """Regression: map_location moved the saved CPU RNG state onto the GPU,
    and set_rng_state requires a CPU ByteTensor. GPU-only failure."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.random.get_rng_state()
    path = tmp_path / "ckpt.pt"
    torch.save({"torch_state": state}, path)
    loaded = torch.load(path, map_location=device, weights_only=False)
    torch.random.set_rng_state(loaded["torch_state"].cpu().to(torch.uint8))
    assert torch.equal(torch.random.get_rng_state(), state)


def test_checkpoint_round_trips_on_the_active_device(tmp_path: Path) -> None:
    """Regression: checkpoints pickle a live strategy, and a torch.Generator
    nested in one cannot be rehydrated onto CUDA. map_location=device raised
    SystemError from Generator.__setstate__; only CPU-then-move works."""
    import torch

    from zebra.detector.lstm import Detector
    from zebra.strategies.lwf import LearningWithoutForgetting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector().to(device)
    strategy = LearningWithoutForgetting(model)
    path = tmp_path / "ck.pt"
    torch.save({"strategy": strategy, "torch_state": torch.random.get_rng_state()}, path)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = loaded["strategy"]
    restored.model = restored.model.to(device)
    x = torch.randn(4, 10, 14, device=device)
    assert restored.model(x).shape == (4, 2)
