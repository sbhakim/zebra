"""Leakage-safe continual-learning runner for the corrected temporal protocol."""

from __future__ import annotations

import json
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from zebra.data.corpus import (
    LoadedDomain,
    Split,
    discover_domains,
    fit_feature_scaler,
    load_domain,
    split_runs,
)
from zebra.detector.lstm import Detector
from zebra.eval.controls import final_attack_strata
from zebra.eval.metrics import (
    final_average,
    final_bwt,
    mean_stage_bwt,
    plasticity,
    plasticity_stability,
    stability,
)
from zebra.eval.orderings import fixed_random_orderings
from zebra.invariants.calibration import RuleCalibration
from zebra.memory.accounting import account
from zebra.strategies.base import Strategy
from zebra.strategies.factory import build_strategy, strategy_parameters

EXPECTED_CORPUS_COMMIT = "8fa707f14c318495e7550fcdb410544cc0c4ad51"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    corpus: Path = Path("corpus/IoT-Attacks-IDS/src/attack_data")
    output: Path = Path("generated/runs")
    epochs: int = 20
    patience: int = 4
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    window_length: int = 10
    window_step: int = 3
    calibration_quantile: float = 0.99
    max_calibration_per_domain: int = 2000
    max_domains: int | None = None
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    expected_corpus_commit: str = EXPECTED_CORPUS_COMMIT
    strategy_options: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, and batch size must be positive")
        if not 0.5 < self.calibration_quantile < 1.0:
            raise ValueError("calibration quantile must be in (0.5, 1)")
        if not self.corpus.is_dir():
            raise FileNotFoundError(self.corpus)


def _git_state(path: Path) -> dict[str, str | bool]:
    """Resolve an auditable revision and dirty flag without invoking a shell."""
    revision = subprocess.run(  # noqa: S603 -- fixed git argv; path is never a command
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # noqa: S603 -- fixed git argv; path is never a command
            ("git", "-C", str(path), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": revision, "dirty": dirty}


def provenance(config: ExperimentConfig) -> dict[str, object]:
    """Record code and corpus identity, rejecting the wrong corpus revision."""
    code_root = Path(__file__).resolve().parents[3]
    corpus_root = config.corpus.resolve().parent.parent
    corpus_state = _git_state(corpus_root)
    if corpus_state["commit"] != config.expected_corpus_commit:
        raise ValueError(
            "corpus revision mismatch: "
            f"expected {config.expected_corpus_commit}, got {corpus_state['commit']}"
        )
    return {
        "protocol": "corrected-temporal-v1",
        "code": _git_state(code_root),
        "corpus": corpus_state,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(
    split: Split,
    calibration: RuleCalibration,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    calibrated = torch.from_numpy(
        calibration.transform(split.nu.numpy()).astype(np.float32)
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(split.x, split.y, calibrated),  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def fit_rule_calibration(config: ExperimentConfig, domains: list[Any]) -> RuleCalibration:
    """Fit from named available domains; the grid passes only its first domain."""
    signals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for ident in domains:
        loaded = load_domain(
            config.corpus, ident, length=config.window_length, step=config.window_step
        )
        y = loaded.train.y.numpy()
        benign = np.flatnonzero(y == 0)[: config.max_calibration_per_domain]
        signals.append(loaded.train.nu.numpy()[benign])
        labels.append(y[benign])
    return RuleCalibration.fit(
        np.concatenate(signals),
        np.concatenate(labels),
        quantile=config.calibration_quantile,
        source_domains=tuple(ident.name for ident in domains),
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(device))
            probability = torch.softmax(logits, dim=1)[:, 1]
            labels.append(y.numpy())
            probabilities.append(probability.cpu().numpy())
            predictions.append((probability >= 0.5).long().cpu().numpy())
    truth = np.concatenate(labels)
    probability_values = np.concatenate(probabilities)
    prediction_values = np.concatenate(predictions)
    return float(roc_auc_score(truth, probability_values)), float(
        f1_score(truth, prediction_values)
    )


def _json_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [[None if np.isnan(value) else float(value) for value in row] for row in values]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "corpus": str(config.corpus),
        "output": str(config.output),
    }


def run_one(
    config: ExperimentConfig,
    strategy_name: str,
    ordering_name: str,
    order: tuple[Any, ...],
    seed: int,
    calibration: RuleCalibration,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one paired (strategy, ordering, seed) experiment."""
    config.validate()
    seed_everything(seed)
    device = torch.device(config.device)
    domains = order[: config.max_domains] if config.max_domains else order
    n_domains = len(domains)
    auc_matrix = np.full((n_domains, n_domains), np.nan)
    f1_matrix = np.full((n_domains, n_domains), np.nan)
    pre_auc = np.full(n_domains, np.nan)
    pre_f1 = np.full(n_domains, np.nan)
    memory_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    config.output.mkdir(parents=True, exist_ok=True)
    stem = f"{strategy_name}__{ordering_name}__seed-{seed}"
    final_path = config.output / f"{stem}.json"
    partial_path = config.output / f"{stem}.partial.json"
    checkpoint_path = config.output / f"{stem}.checkpoint.pt"
    run_provenance = provenance(config)
    run_config = _config_dict(config)
    scaler = fit_feature_scaler(config.corpus, domains[0])
    partitions = {
        domain.name: {
            name: [path.name for path in files]
            for name, files in zip(
                ("train", "validation", "test"),
                split_runs(config.corpus / domain.name),
                strict=True,
            )
        }
        for domain in domains
    }

    model = Detector().to(device)
    strategy = build_strategy(
        strategy_name,
        model,
        seed=seed,
        overrides=config.strategy_options.get(strategy_name),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_stage = 0
    partial: dict[str, Any]
    if resume and checkpoint_path.exists() and partial_path.exists():
        # Always unpickle onto the CPU. The checkpoint carries a live strategy
        # object, and a torch.Generator nested inside one cannot be rehydrated
        # onto CUDA -- map_location=device raises SystemError from
        # Generator.__setstate__. Devices are restored explicitly below, which
        # is the only ordering that works on both backends.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = checkpoint["partial"]
        expected_order = [domain.name for domain in domains]
        if (
            saved["strategy"] != strategy_name
            or saved["ordering"] != ordering_name
            or saved["seed"] != seed
            or saved["order"] != expected_order
            or saved["config"] != run_config
            or saved["calibration"] != calibration.as_dict()
            or saved.get("preprocessor") != scaler.as_dict()
            or saved["provenance"]["corpus"] != run_provenance["corpus"]
        ):
            raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
        strategy = checkpoint["strategy"]
        model = cast(Detector, strategy.model).to(device)
        strategy.model = model
        # A strategy may hold further modules (LwF's teacher, generative
        # replay's VAE) and tensors (EWC's Fisher, SI's path integral) that were
        # just loaded onto the CPU; move every one or the first backward pass
        # fails on a device mismatch.
        for attribute, value in vars(strategy).items():
            if isinstance(value, torch.nn.Module):
                setattr(strategy, attribute, value.to(device))
            elif isinstance(value, torch.Tensor):
                setattr(strategy, attribute, value.to(device))
            elif isinstance(value, dict):
                setattr(
                    strategy,
                    attribute,
                    {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    },
                )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        auc_matrix = np.asarray(saved["auc_matrix"], dtype=np.float64)
        f1_matrix = np.asarray(saved["f1_matrix"], dtype=np.float64)
        pre_auc = np.asarray(saved["pre_auc"], dtype=np.float64)
        pre_f1 = np.asarray(saved["pre_f1"], dtype=np.float64)
        memory_rows = list(saved["memory"])
        partial = saved
        start_stage = int(checkpoint["stage"]) + 1
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        # torch.load(map_location=device) moves every tensor in the checkpoint,
        # including the CPU RNG state, so it must come back to the CPU before
        # restoring. This only ever failed on GPU, which is why the CPU smoke
        # runs never caught it.
        torch.random.set_rng_state(checkpoint["torch_state"].cpu().to(torch.uint8))
        if start_stage >= n_domains and final_path.exists():
            return cast(dict[str, Any], json.loads(final_path.read_text()))

    loaded_cache: dict[int, LoadedDomain] = {}

    def loaded(index: int) -> LoadedDomain:
        if index not in loaded_cache:
            loaded_cache[index] = load_domain(
                config.corpus,
                domains[index],
                length=config.window_length,
                step=config.window_step,
                scaler=scaler,
            )
        return loaded_cache[index]

    for stage in range(start_stage, n_domains):
        ident = domains[stage]
        current = loaded(stage)
        strategy.start_domain(ident.name)
        train_loader = _loader(
            current.train, calibration, config.batch_size, shuffle=True, seed=seed + stage
        )
        validation_loader = _loader(
            current.validation, calibration, config.batch_size, shuffle=False, seed=seed
        )
        test_loader = _loader(
            current.test, calibration, config.batch_size, shuffle=False, seed=seed
        )
        if stage > 0:
            pre_auc[stage], pre_f1[stage] = evaluate(model, test_loader, device)

        best_auc = -np.inf
        best_strategy: Strategy | None = None
        best_optimizer_state: dict[str, Any] | None = None
        stale = 0
        for _ in range(config.epochs):
            model.train()
            for x, y, nu in train_loader:
                x, y, nu = x.to(device), y.to(device), nu.to(device)
                x, y, nu = strategy.augment_batch(x, y, nu)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                task_loss = torch.nn.functional.cross_entropy(logits, y.long())
                strategy.capture_task_gradients(task_loss)
                extra_loss = strategy.penalty(x, y, logits, nu)
                (task_loss + extra_loss).backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                strategy.before_optimizer_step()
                optimizer.step()
                strategy.after_optimizer_step()
            validation_auc, _ = evaluate(model, validation_loader, device)
            if validation_auc > best_auc + 1e-6:
                best_auc = validation_auc
                # SI and other online methods mutate state during an epoch.
                # Restore the whole lifecycle state, not only neural weights.
                best_strategy = deepcopy(strategy)
                best_optimizer_state = deepcopy(optimizer.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best_strategy is None or best_optimizer_state is None:
            raise RuntimeError(f"{ident.name}: validation never produced a model")
        strategy = best_strategy
        model = cast(Detector, strategy.model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        optimizer.load_state_dict(best_optimizer_state)
        strategy.after_domain(train_loader, device)

        # Test scores are reporting-only. Evaluate every learned domain to form
        # the lower triangle and current diagonal of the CL matrix.
        for prior in range(stage + 1):
            prior_test = _loader(
                loaded(prior).test,
                calibration,
                config.batch_size,
                shuffle=False,
                seed=seed,
            )
            auc_matrix[stage, prior], f1_matrix[stage, prior] = evaluate(
                model, prior_test, device
            )
        report = account(
            strategy.name,
            model,
            strategy.retained,
            optimizer,
            preprocessor_bytes=scaler.metadata_bytes,
            static_rule_bytes=(
                calibration.metadata_bytes if "zebra" in strategy_name else 0
            ),
        )
        memory_rows.append(asdict(report))

        partial = {
            "schema": 1,
            "strategy": strategy_name,
            "ordering": ordering_name,
            "seed": seed,
            "completed_domains": stage + 1,
            "order": [domain.name for domain in domains],
            "config": run_config,
            "provenance": run_provenance,
            "run_partitions": partitions,
            "detector": model.configuration,
            "preprocessor": scaler.as_dict(),
            "strategy_hyperparameters": strategy_parameters(
                strategy_name, config.strategy_options.get(strategy_name)
            ),
            "calibration": calibration.as_dict(),
            "auc_matrix": _json_matrix(auc_matrix),
            "f1_matrix": _json_matrix(f1_matrix),
            "pre_auc": [None if np.isnan(v) else float(v) for v in pre_auc],
            "pre_f1": [None if np.isnan(v) else float(v) for v in pre_f1],
            "memory": memory_rows,
        }
        _write_json_atomic(partial_path, partial)
        checkpoint_temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "stage": stage,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "strategy": strategy,
                "partial": partial,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.random.get_rng_state(),
            },
            checkpoint_temporary,
        )
        checkpoint_temporary.replace(checkpoint_path)

    summary = {
        "auc": final_average(auc_matrix),
        "f1": final_average(f1_matrix),
        "final_bwt_auc": final_bwt(auc_matrix),
        "mean_stage_bwt_auc": mean_stage_bwt(auc_matrix),
        "plasticity_auc": plasticity(auc_matrix, pre_auc),
        "stability_auc": stability(auc_matrix),
        "ps_auc": plasticity_stability(auc_matrix, pre_auc),
        "auc_by_attack": final_attack_strata(auc_matrix, domains),
        "f1_by_attack": final_attack_strata(f1_matrix, domains),
    }
    result = {**partial, "wall_seconds": time.perf_counter() - started, "summary": summary}
    _write_json_atomic(final_path, result)
    return result


def run_grid(
    config: ExperimentConfig,
    strategies: tuple[str, ...],
    model_seeds: tuple[int, ...] = (7, 19, 43),
    ordering_seeds: tuple[int, ...] = (11, 29, 47, 83),
) -> list[dict[str, Any]]:
    config.validate()
    domains = discover_domains(config.corpus)
    orderings = fixed_random_orderings(domains, ordering_seeds)
    outputs = []
    for name, order in orderings.items():
        # Calibration is part of online initialization: future domains cannot
        # influence its thresholds. Every paired strategy sees the same map.
        calibration = fit_rule_calibration(config, [order[0]])
        for seed in model_seeds:
            for strategy in strategies:
                outputs.append(run_one(config, strategy, name, order, seed, calibration))
    return outputs
