"""Command-line entry points for reproducible ZeBRa experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command("status")
def status() -> None:
    """Show which evidence-producing components are available."""
    typer.echo("ZeBRa corrected temporal protocol")
    typer.echo("protocol: docs/EXPERIMENT_PROTOCOL.md")
    typer.echo("commands: validate-data, memory-report, probe-rules, experiment, summarize")


@app.command("validate-data")
def validate_data(
    corpus: Annotated[
        Path, typer.Option(exists=True, file_okay=False, readable=True)
    ] = Path("corpus/IoT-Attacks-IDS/src/attack_data"),
) -> None:
    """Validate domain discovery and run-level split integrity."""
    from zebra.data.corpus import discover_domains, fit_feature_scaler, split_runs

    domains = discover_domains(corpus)
    for ident in domains:
        split = split_runs(corpus / ident.name)
        groups = [set(p.name for p in part) for part in split]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise typer.BadParameter(f"run leakage in {ident.name}")
        fit_feature_scaler(corpus, ident)
    typer.echo(f"validated {len(domains)} domains with disjoint train/validation/test runs")


@app.command("memory-report")
def memory_report() -> None:
    """Generate memory artifacts from instantiated strategies."""
    from zebra.eval.memory_report import main

    raise typer.Exit(main())


@app.command("probe-rules")
def probe_rules() -> None:
    """Generate the exploratory standalone rule probe."""
    from zebra.eval.probe import main

    raise typer.Exit(main())


@app.command("smoke")
def smoke(
    strategy: Annotated[str, typer.Option()] = "zebra",
    domains: Annotated[int, typer.Option(min=2, max=6)] = 3,
    epochs: Annotated[int, typer.Option(min=1)] = 2,
    corpus: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "corpus/IoT-Attacks-IDS/src/attack_data"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("generated/smoke"),
) -> None:
    """Run a small end-to-end experiment that cannot be used as paper evidence."""
    from zebra.eval.experiment import ExperimentConfig, run_grid

    config = ExperimentConfig(
        corpus=corpus,
        output=output,
        epochs=epochs,
        patience=1,
        max_domains=domains,
    )
    outputs = run_grid(config, (strategy,), model_seeds=(7,), ordering_seeds=(11,))
    typer.echo(json.dumps(outputs[0]["summary"], indent=2))


@app.command("experiment")
def experiment(
    strategies: Annotated[
        str,
        typer.Option(help="Comma-separated paired strategy names."),
    ] = "no-cl,ewc,si,lwf,gen-replay,replay,zebra",
    epochs: Annotated[int, typer.Option(min=1)] = 20,
    patience: Annotated[int, typer.Option(min=1)] = 4,
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
    corpus: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "corpus/IoT-Attacks-IDS/src/attack_data"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("generated/runs"),
) -> None:
    """Run or safely resume the frozen paired grid."""
    from zebra.eval.experiment import ExperimentConfig, run_grid

    names = tuple(name.strip() for name in strategies.split(",") if name.strip())
    config = ExperimentConfig(
        corpus=corpus,
        output=output,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
    )
    outputs = run_grid(config, names)
    typer.echo(f"completed {len(outputs)} paired runs in {output}")


@app.command("sweep-capacity")
def sweep_capacity(
    capacities: Annotated[
        str, typer.Option(help="Comma-separated replay buffer capacities, in windows.")
    ] = "59,250,500,1000,4000",
    epochs: Annotated[int, typer.Option(min=1)] = 20,
    patience: Annotated[int, typer.Option(min=1)] = 4,
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
    corpus: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "corpus/IoT-Attacks-IDS/src/attack_data"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("generated/capacity"),
) -> None:
    """Re-run the replay column at other buffer capacities.

    Each capacity gets its own directory. Run identity is (strategy, ordering,
    seed), so two capacities would otherwise write the same filename and the
    checkpoint identity check would reject the second one.
    """
    from zebra.eval.experiment import ExperimentConfig, run_grid

    for capacity in (int(value) for value in capacities.split(",") if value.strip()):
        config = ExperimentConfig(
            corpus=corpus,
            output=output / f"cap-{capacity}",
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            strategy_options={"replay": {"capacity": capacity, "replay_ratio": 0.5}},
        )
        run_grid(config, ("replay",))
        typer.echo(f"capacity {capacity}: 12 paired runs in {config.output}")


@app.command("summarize")
def summarize(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "generated/runs"
    ),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("generated"),
    allow_incomplete: Annotated[bool, typer.Option()] = False,
) -> None:
    """Aggregate completed paired runs; strict mode requires every primary method."""
    from zebra.eval.report import main

    raise typer.Exit(main(run_dir, output, require_primary=not allow_incomplete))


@app.command("sweep-report")
def sweep_report(
    primary: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = Path("generated/runs"),
    sweep: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = Path("generated/capacity"),
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("generated"),
) -> None:
    """Aggregate the capacity sweep against the primary grid's paired cells."""
    from zebra.eval.capacity import main

    raise typer.Exit(main(primary, sweep, output))


if __name__ == "__main__":
    app()
