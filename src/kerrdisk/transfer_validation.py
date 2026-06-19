"""Phase 12.5 transfer-map validation campaign helpers."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, radians
from pathlib import Path

import numpy as np

from kerrdisk.radiation_diagnostics import sample_emission_outcomes
from kerrdisk.spectrum import TransferMap, build_transfer_map
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import (
    KerrThinDiskSettings,
    ray_traced_kerr_thin_disk_energy_flux,
)


@dataclass(frozen=True)
class TransferConvergenceRow:
    """One screen-resolution convergence comparison."""

    a_star: float
    inclination_deg: float
    screen_size: int
    disk_hits: int
    reference_screen_size: int
    relative_l1_spectrum_delta: float

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExternalComparisonRow:
    """One external transfer-map comparison status row."""

    status: str
    external_path: str
    matched_records: int
    max_relative_radius_delta: float
    max_relative_redshift_delta: float
    max_abs_mu_delta: float
    notes: str

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def run_transfer_convergence(
    *,
    screen_sizes: Sequence[int] = (3, 5, 7),
) -> list[TransferConvergenceRow]:
    """Run a small deterministic screen-resolution convergence campaign."""

    cases = ((0.0, 40.0), (0.9, 70.0))
    bins = make_log_energy_bins(
        energy_min_kev=0.1,
        energy_max_kev=20.0,
        bin_count=24,
    )
    settings = KerrThinDiskSettings(
        radial_grid_count=72,
        disk_outer_radius_rg=80.0,
        temperature_scale_kev=20.0,
    )
    rows: list[TransferConvergenceRow] = []
    for a_star, inclination_deg in cases:
        spectra: dict[int, np.ndarray] = {}
        hits: dict[int, int] = {}
        for size in screen_sizes:
            transfer_map = _validation_transfer_map(a_star, inclination_deg, size)
            hits[size] = int(transfer_map.emission_radius.size)
            spectra[size] = ray_traced_kerr_thin_disk_energy_flux(
                transfer_map=transfer_map,
                a_star=a_star,
                eddington_ratio=0.1,
                f_col=1.7,
                delta_eta=0.02,
                energy_bins=bins,
                settings=settings,
                limb_darkening="electron_scattering",
            )
        reference_size = max(screen_sizes)
        reference = spectra[reference_size]
        reference_norm = float(np.sum(np.abs(reference)))
        for size in screen_sizes:
            delta = float(np.sum(np.abs(spectra[size] - reference)) / reference_norm)
            rows.append(
                TransferConvergenceRow(
                    a_star=a_star,
                    inclination_deg=inclination_deg,
                    screen_size=size,
                    disk_hits=hits[size],
                    reference_screen_size=reference_size,
                    relative_l1_spectrum_delta=delta,
                )
            )
    return rows


def run_external_transfer_comparison(
    external_path: Path | None,
) -> list[ExternalComparisonRow]:
    """Compare against an externally generated transfer-map CSV when supplied."""

    if external_path is None or not external_path.exists():
        return [
            ExternalComparisonRow(
                status="SKIPPED",
                external_path="" if external_path is None else str(external_path),
                matched_records=0,
                max_relative_radius_delta=float("nan"),
                max_relative_redshift_delta=float("nan"),
                max_abs_mu_delta=float("nan"),
                notes=(
                    "No external ray-tracer CSV was supplied. Provide a CSV with "
                    "columns alpha,beta,emission_radius,redshift,emission_mu."
                ),
            )
        ]
    external_rows = _read_external_rows(external_path)
    alpha = sorted({row["alpha"] for row in external_rows})
    beta = sorted({row["beta"] for row in external_rows})
    production = build_transfer_map(
        0.0,
        np.array(alpha, dtype=np.float64),
        np.array(beta, dtype=np.float64),
        observer_radius=100.0,
        observer_theta=radians(40.0),
        disk_outer_radius=80.0,
        observer_distance=100.0,
        step_size=0.1,
        max_steps=3_000,
        escape_radius=160.0,
    )
    production_rows = {
        (float(a), float(b)): (float(r), float(g), float(m))
        for a, b, r, g, m in zip(
            production.alpha,
            production.beta,
            production.emission_radius,
            production.redshift,
            production.emission_mu,
            strict=True,
        )
    }
    radius_delta: list[float] = []
    redshift_delta: list[float] = []
    mu_delta: list[float] = []
    for row in external_rows:
        key = (row["alpha"], row["beta"])
        if key not in production_rows:
            continue
        radius, redshift, emission_mu = production_rows[key]
        radius_delta.append(
            abs(radius - row["emission_radius"]) / row["emission_radius"]
        )
        redshift_delta.append(abs(redshift - row["redshift"]) / row["redshift"])
        mu_delta.append(abs(emission_mu - row["emission_mu"]))
    matched = len(radius_delta)
    max_radius = max(radius_delta) if radius_delta else float("inf")
    max_redshift = max(redshift_delta) if redshift_delta else float("inf")
    max_mu = max(mu_delta) if mu_delta else float("inf")
    status = (
        "PASS"
        if matched
        and max_radius <= 5.0e-2
        and max_redshift <= 5.0e-2
        and max_mu <= 1.0e-1
        else "FAIL"
    )
    return [
        ExternalComparisonRow(
            status=status,
            external_path=str(external_path),
            matched_records=matched,
            max_relative_radius_delta=max_radius,
            max_relative_redshift_delta=max_redshift,
            max_abs_mu_delta=max_mu,
            notes="Comparison against supplied external ray-tracer transfer-map CSV.",
        )
    ]


def run_capture_returning_diagnostics() -> list[dict[str, object]]:
    """Run a small photon-capture and returning-radiation diagnostic grid."""

    mu_values = (0.25, 0.5, 0.75)
    azimuth_values = tuple(np.linspace(0.0, 2.0 * pi, 8, endpoint=False))
    rows: list[dict[str, object]] = []
    for a_star in (0.0, 0.9):
        for radius_factor in (1.2, 2.0):
            radius = isco_radius_for_validation(a_star) * radius_factor
            summary = sample_emission_outcomes(
                a_star,
                radius,
                mu_values=mu_values,
                azimuth_values=azimuth_values,
                disk_outer_radius=80.0,
                step_size=0.1,
                max_steps=5_000,
                escape_radius=160.0,
            )
            rows.append(summary.as_dict())
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def isco_radius_for_validation(a_star: float) -> float:
    """Import-local wrapper to keep campaign row construction readable."""

    from kerrdisk.isco import isco_radius

    return isco_radius(a_star)


def _validation_transfer_map(
    a_star: float,
    inclination_deg: float,
    screen_size: int,
) -> TransferMap:
    return build_transfer_map(
        a_star,
        _screen_centers(-8.0, 8.0, screen_size),
        _screen_centers(35.0, 65.0, screen_size),
        observer_radius=100.0,
        observer_theta=radians(inclination_deg),
        disk_outer_radius=80.0,
        observer_distance=100.0,
        step_size=0.1,
        max_steps=3_000,
        escape_radius=160.0,
    )


def _screen_centers(lower: float, upper: float, count: int) -> np.ndarray:
    width = (upper - lower) / count
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * width


def _read_external_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            {
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "emission_radius": float(row["emission_radius"]),
                "redshift": float(row["redshift"]),
                "emission_mu": float(row["emission_mu"]),
            }
            for row in csv.DictReader(stream)
        ]
    if not rows:
        msg = "external transfer-map CSV is empty"
        raise ValueError(msg)
    return rows
