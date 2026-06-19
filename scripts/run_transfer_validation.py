"""Run Phase 12.5 transfer-map validation artifacts."""

from pathlib import Path
from typing import Annotated

import typer

from kerrdisk.transfer_validation import (
    run_capture_returning_diagnostics,
    run_external_transfer_comparison,
    run_transfer_convergence,
    write_rows,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for Phase 12.5 validation artifacts."),
    ] = Path("data/processed/transfer_validation"),
    external_transfer_csv: Annotated[
        Path | None,
        typer.Option(
            "--external-transfer-csv",
            help="Optional external ray-tracer transfer-map CSV.",
        ),
    ] = None,
) -> None:
    """Generate transfer-map convergence and transport-diagnostic artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    convergence = [row.as_dict() for row in run_transfer_convergence()]
    capture = run_capture_returning_diagnostics()
    external = [
        row.as_dict() for row in run_external_transfer_comparison(external_transfer_csv)
    ]

    convergence_path = output_dir / "phase12p5_transfer_convergence.csv"
    capture_path = output_dir / "phase12p5_capture_returning_diagnostics.csv"
    external_path = output_dir / "phase12p5_external_transfer_comparison.csv"
    write_rows(convergence_path, convergence)
    write_rows(capture_path, capture)
    write_rows(external_path, external)

    typer.echo(f"transfer_convergence: {convergence_path}")
    typer.echo(f"capture_returning_diagnostics: {capture_path}")
    typer.echo(f"external_transfer_comparison: {external_path}")


if __name__ == "__main__":
    app()
