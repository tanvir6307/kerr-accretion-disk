"""Run the Phase 11 coarse screening campaign."""

from pathlib import Path
from typing import Annotated

import typer

from kerrdisk.screening import (
    load_screening_config,
    run_screening_campaign,
    write_screening_outputs,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Versioned Phase 11 screening configuration.",
        ),
    ] = Path("configs/production/phase11_screening.yaml"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for machine-readable Phase 11 outputs.",
        ),
    ] = Path("data/processed/screening"),
) -> None:
    """Run screening and write status, replicate, and protocol artifacts."""

    screening_config = load_screening_config(config)
    result = run_screening_campaign(screening_config)
    paths = write_screening_outputs(result, output_dir)
    print(f"Screening conditions: {len(result.conditions)}")
    print(f"Confirmatory candidates: {len(result.confirmatory_conditions)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    app()
