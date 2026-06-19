"""Run validation checks for the current phase."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from kerrdisk.diagnostics import run_independent_validation, write_validation_summary
from kerrdisk.disk_flux import luminosity_at_infinity, page_thorne_flux_profile
from kerrdisk.isco import isco_radius

app = typer.Typer(help="Generate validation artifacts.")


@app.command()
def page_thorne_flux(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for validation artifacts."),
    ] = Path("data/processed/validation"),
) -> None:
    """Generate Phase 3 Page-Thorne flux validation data and plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    radii = np.geomspace(isco_radius(0.0), 1.0e5, 10_000, dtype=np.float64)
    profile = page_thorne_flux_profile(0.0, radii)
    scaled_flux = profile.flux * 8.0 * np.pi * profile.radii**3 / 3.0
    luminosity = luminosity_at_infinity(profile)

    csv_path = output_dir / "phase3_page_thorne_flux_schwarzschild.csv"
    header = (
        "radius_gm_c2,one_face_flux_geometric,"
        "newtonian_scaled_flux,luminosity_at_infinity"
    )
    data = np.column_stack(
        [
            profile.radii,
            profile.flux,
            scaled_flux,
            np.full_like(profile.radii, luminosity),
        ]
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    figure_path = output_dir / "phase3_page_thorne_flux_schwarzschild.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(profile.radii, profile.flux, label="Page-Thorne flux")
    ax.set_xlabel(r"Radius $r/(GM/c^2)$")
    ax.set_ylabel("One-face flux, geometric units")
    ax.set_title("Phase 3 validation: Schwarzschild zero-torque disk")
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    typer.echo(f"Wrote {csv_path}")
    typer.echo(f"Wrote {figure_path}")


@app.command()
def independent(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for validation summary artifacts."),
    ] = Path("data/processed"),
) -> None:
    """Generate Phase 7 independent validation summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_independent_validation()
    csv_path = output_dir / "validation_summary.csv"
    write_validation_summary(csv_path, rows)

    residual_path = output_dir / "validation_residuals.png"
    labels = [row.check_id for row in rows]
    normalized = [
        abs(row.residual) / row.tolerance if row.tolerance > 0.0 else abs(row.residual)
        for row in rows
    ]
    colors = ["#3b7a57" if row.status == "PASS" else "#b23b3b" for row in rows]
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.bar(np.arange(len(rows)), normalized, color=colors)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("abs(residual) / tolerance")
    ax.set_title("Phase 7 independent validation residuals")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(residual_path, dpi=160)
    plt.close(fig)

    failed = [row.check_id for row in rows if row.status != "PASS"]
    typer.echo(f"Wrote {csv_path}")
    typer.echo(f"Wrote {residual_path}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def bootstrap() -> None:
    """Report the current validation scope."""

    print("Phase 1 bootstrap validation is handled by pytest, ruff, and mypy.")


if __name__ == "__main__":
    app()
