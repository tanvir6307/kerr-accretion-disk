"""Run the pre-Phase-14 joint multi-epoch comparison."""

from pathlib import Path
from typing import Annotated

import typer

from kerrdisk.multi_epoch import (
    load_multi_epoch_config,
    run_multi_epoch_campaign,
    write_multi_epoch_outputs,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Versioned multi-epoch comparison configuration.",
        ),
    ] = Path("configs/production/phase13p5_multi_epoch.yaml"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for machine-readable multi-epoch outputs.",
        ),
    ] = Path("data/processed/multi_epoch"),
) -> None:
    """Run shared-spin multi-epoch fits and write artifacts."""

    multi_epoch_config = load_multi_epoch_config(config)
    result = run_multi_epoch_campaign(multi_epoch_config)
    paths = write_multi_epoch_outputs(result, output_dir)
    completed = sum(summary.status == "COMPLETED" for summary in result.summaries)
    narrowed = sum(
        summary.width_68_reduction_fraction
        > multi_epoch_config.width_reduction_required_fraction
        for summary in result.summaries
    )
    typer.echo(f"Multi-epoch groups: {len(result.groups)}")
    typer.echo(f"Completed groups: {completed}/{len(result.summaries)}")
    typer.echo(f"68pct-width reduction groups: {narrowed}/{len(result.summaries)}")
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
