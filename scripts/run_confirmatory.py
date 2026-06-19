"""Run the Phase 12 confirmatory campaign."""

from pathlib import Path
from typing import Annotated

import typer

from kerrdisk.confirmatory import (
    load_confirmatory_config,
    run_confirmatory_campaign,
    write_confirmatory_outputs,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Versioned Phase 12 confirmatory configuration.",
        ),
    ] = Path("configs/production/phase12_confirmatory.yaml"),
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for machine-readable Phase 12 outputs.",
        ),
    ] = Path("data/processed/confirmatory"),
) -> None:
    """Run locked confirmatory conditions and write Phase 12 artifacts."""

    confirmatory_config = load_confirmatory_config(config)
    result = run_confirmatory_campaign(confirmatory_config)
    paths = write_confirmatory_outputs(result, output_dir)
    stable = sum(summary.stable for summary in result.resolution_summaries)
    print(f"Confirmatory conditions: {len(result.conditions)}")
    print(f"Base replicates: {len(result.base_replicates)}")
    print(f"Resolution-stable conditions: {stable}/{len(result.resolution_summaries)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    app()
