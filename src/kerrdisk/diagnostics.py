"""Independent validation diagnostics."""

import csv
from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kerrdisk.circular_orbits import nt_efficiency
from kerrdisk.disk_flux import page_thorne_flux_profile
from kerrdisk.geodesics import carter_constant_null, hamiltonian
from kerrdisk.isco import isco_radius
from kerrdisk.raytrace import (
    RayOutcome,
    initial_photon_covector,
    schwarzschild_shadow_radius,
    trace_ray,
)
from kerrdisk.spectrum import TransferMap, observed_flux_density

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ValidationRow:
    """One independent validation comparison."""

    check_id: str
    category: str
    quantity: str
    production_value: float
    reference_value: float
    residual: float
    tolerance: float
    status: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        """Return a CSV-ready representation."""

        return {
            "check_id": self.check_id,
            "category": self.category,
            "quantity": self.quantity,
            "production_value": f"{self.production_value:.17g}",
            "reference_value": f"{self.reference_value:.17g}",
            "residual": f"{self.residual:.17g}",
            "tolerance": f"{self.tolerance:.17g}",
            "status": self.status,
            "notes": self.notes,
        }


def _status(residual: float, tolerance: float) -> str:
    return "PASS" if abs(residual) <= tolerance else "FAIL"


def _row(
    check_id: str,
    category: str,
    quantity: str,
    production_value: float,
    reference_value: float,
    tolerance: float,
    notes: str,
) -> ValidationRow:
    residual = production_value - reference_value
    return ValidationRow(
        check_id=check_id,
        category=category,
        quantity=quantity,
        production_value=production_value,
        reference_value=reference_value,
        residual=residual,
        tolerance=tolerance,
        status=_status(residual, tolerance),
        notes=notes,
    )


def independent_isco_radius(a_star: float) -> float:
    """Independent copy of the Bardeen-Press-Teukolsky ISCO expression."""

    if a_star == 0.0:
        return 6.0
    spin_squared = a_star * a_star
    z1 = 1.0 + (1.0 - spin_squared) ** (1.0 / 3.0) * (
        (1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0)
    )
    z2 = sqrt((3.0 * spin_squared) + (z1 * z1))
    sign = 1.0 if a_star > 0.0 else -1.0
    return 3.0 + z2 - sign * sqrt((3.0 - z1) * (3.0 + z1 + (2.0 * z2)))


def independent_specific_energy(a_star: float, radius: float) -> float:
    """Independent circular-orbit specific energy expression."""

    sqrt_radius = sqrt(radius)
    numerator = radius * sqrt_radius - 2.0 * sqrt_radius + a_star
    denominator = radius**0.75 * sqrt(
        radius * sqrt_radius - 3.0 * sqrt_radius + 2.0 * a_star
    )
    return float(numerator / denominator)


def independent_efficiency(a_star: float) -> float:
    """Independent zero-torque efficiency proxy."""

    return 1.0 - independent_specific_energy(a_star, independent_isco_radius(a_star))


def _independent_omega(a_star: float, radius: float) -> float:
    return float(1.0 / (radius**1.5 + a_star))


def _independent_angular_momentum(a_star: float, radius: float) -> float:
    sqrt_radius = sqrt(radius)
    numerator = radius * radius - 2.0 * a_star * sqrt_radius + a_star * a_star
    denominator = radius**0.75 * sqrt(
        radius * sqrt_radius - 3.0 * sqrt_radius + 2.0 * a_star
    )
    return float(numerator / denominator)


def _independent_d_l_dr(a_star: float, radius: float) -> float:
    step = 1.0e-5 * max(1.0, radius)
    return (
        _independent_angular_momentum(a_star, radius + step)
        - _independent_angular_momentum(a_star, radius - step)
    ) / (2.0 * step)


def _independent_d_omega_dr(a_star: float, radius: float) -> float:
    return float(-1.5 * sqrt(radius) / (radius**1.5 + a_star) ** 2)


def independent_page_thorne_flux(a_star: float, radii: FloatArray) -> FloatArray:
    """Independent Page-Thorne numerical integral on a supplied grid."""

    energy = np.array(
        [independent_specific_energy(a_star, radius) for radius in radii],
        dtype=np.float64,
    )
    angular_momentum = np.array(
        [_independent_angular_momentum(a_star, radius) for radius in radii],
        dtype=np.float64,
    )
    omega = np.array(
        [_independent_omega(a_star, radius) for radius in radii],
        dtype=np.float64,
    )
    d_l_dr = np.array(
        [_independent_d_l_dr(a_star, radius) for radius in radii],
        dtype=np.float64,
    )
    energy_angular_momentum = energy - omega * angular_momentum
    integrand = energy_angular_momentum * d_l_dr
    increments = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(radii)
    integral = np.concatenate(
        [np.array([0.0], dtype=np.float64), np.cumsum(increments, dtype=np.float64)]
    )
    flux = (
        1.0
        / (4.0 * pi * radii)
        * (-np.array([_independent_d_omega_dr(a_star, radius) for radius in radii]))
        / (energy_angular_momentum * energy_angular_momentum)
        * integral
    )
    return np.asarray(flux, dtype=np.float64)


def _constant_intensity(frequency_hz: FloatArray, radius: float) -> FloatArray:
    _ = radius
    return np.full_like(frequency_hz, 2.5)


def _manual_transfer_map() -> TransferMap:
    return TransferMap(
        a_star=0.0,
        observer_radius=100.0,
        observer_theta=1.2,
        observer_distance=100.0,
        alpha=np.array([-1.0, 1.0], dtype=np.float64),
        beta=np.array([50.0, 50.0], dtype=np.float64),
        solid_angle=np.array([1.0e-6, 2.0e-6], dtype=np.float64),
        emission_radius=np.array([20.0, 30.0], dtype=np.float64),
        emission_phi=np.array([0.0, 1.0], dtype=np.float64),
        redshift=np.array([0.8, 1.2], dtype=np.float64),
        emission_mu=np.ones(2, dtype=np.float64),
        max_abs_hamiltonian=np.zeros(2, dtype=np.float64),
        steps=np.ones(2, dtype=np.int64),
        outcome_counts={
            "DISK_HIT": 2,
            "HORIZON_CAPTURE": 0,
            "ESCAPED": 0,
            "MAX_STEPS": 0,
            "NUMERICAL_FAILURE": 0,
        },
    )


def run_independent_validation() -> list[ValidationRow]:
    """Run independent Phase 7 comparisons."""

    rows: list[ValidationRow] = []
    spins = [-0.9, -0.5, 0.0, 0.5, 0.9, 0.998]
    for a_star in spins:
        rows.append(
            _row(
                f"isco_{a_star:g}",
                "isco",
                "r_isco",
                isco_radius(a_star),
                independent_isco_radius(a_star),
                1.0e-12,
                "Independent BPT expression coded separately.",
            )
        )
        rows.append(
            _row(
                f"efficiency_{a_star:g}",
                "efficiency",
                "1_minus_e_isco",
                nt_efficiency(a_star),
                independent_efficiency(a_star),
                5.0e-11,
                "Independent circular-orbit energy at independent ISCO.",
            )
        )

    observer_radius = 100.0
    radial_state = initial_photon_covector(0.0, observer_radius, pi / 2.0, 0.0, 0.0)
    schwarzschild_factor = 1.0 - 2.0 / observer_radius
    rows.extend(
        [
            _row(
                "radial_ray_pt",
                "ray_invariant",
                "p_t",
                float(radial_state[4]),
                -sqrt(schwarzschild_factor),
                1.0e-12,
                "Analytic Schwarzschild radial photon from static observer.",
            ),
            _row(
                "radial_ray_pr",
                "ray_invariant",
                "p_r",
                float(radial_state[5]),
                -1.0 / sqrt(schwarzschild_factor),
                1.0e-12,
                "Analytic Schwarzschild radial photon from static observer.",
            ),
            _row(
                "radial_ray_hamiltonian",
                "ray_invariant",
                "H",
                hamiltonian(0.0, radial_state),
                0.0,
                1.0e-12,
                "Independent analytic null expectation.",
            ),
            _row(
                "radial_ray_carter",
                "ray_invariant",
                "Q",
                carter_constant_null(0.0, radial_state),
                0.0,
                1.0e-12,
                "Radial Schwarzschild null ray has Q=0.",
            ),
        ]
    )

    lower = 5.15
    upper = 5.25
    capture = trace_ray(
        0.0,
        initial_photon_covector(0.0, 100.0, pi / 2.0, lower, 0.0),
        step_size=0.02,
        max_steps=20_000,
        escape_radius=120.0,
    )
    escape = trace_ray(
        0.0,
        initial_photon_covector(0.0, 100.0, pi / 2.0, upper, 0.0),
        step_size=0.02,
        max_steps=20_000,
        escape_radius=120.0,
    )
    reference_shadow = sqrt(27.0)
    shadow_residual = 0.0 if lower < reference_shadow < upper else float("inf")
    shadow_status_value = (
        0.0
        if (
            capture.outcome == RayOutcome.HORIZON_CAPTURE
            and escape.outcome == RayOutcome.ESCAPED
        )
        else 1.0
    )
    shadow_passed = shadow_residual == 0.0 and shadow_status_value == 0.0
    rows.append(
        ValidationRow(
            check_id="schwarzschild_shadow_bracket",
            category="shadow",
            quantity="sqrt27_inside_capture_escape_bracket",
            production_value=shadow_status_value,
            reference_value=0.0,
            residual=shadow_residual,
            tolerance=0.0,
            status="PASS" if shadow_passed else "FAIL",
            notes=(
                "Analytic Schwarzschild shadow radius sqrt(27) bracketed "
                "by traced rays."
            ),
        )
    )
    rows.append(
        _row(
            "schwarzschild_shadow_radius_helper",
            "shadow",
            "sqrt27",
            schwarzschild_shadow_radius(),
            reference_shadow,
            1.0e-14,
            "Analytic Schwarzschild shadow radius.",
        )
    )

    a_flux = 0.5
    radii = np.geomspace(independent_isco_radius(a_flux), 200.0, 6_400)
    production_flux = page_thorne_flux_profile(a_flux, radii).flux
    reference_flux = independent_page_thorne_flux(a_flux, radii)
    mask = reference_flux > 1.0e-12
    max_relative_flux_residual = float(
        np.max(
            np.abs(
                (production_flux[mask] - reference_flux[mask]) / reference_flux[mask]
            )
        )
    )
    rows.append(
        ValidationRow(
            check_id="page_thorne_flux_profile_a0p5",
            category="disk_flux",
            quantity="max_relative_flux_residual",
            production_value=max_relative_flux_residual,
            reference_value=0.0,
            residual=max_relative_flux_residual,
            tolerance=2.0e-3,
            status=_status(max_relative_flux_residual, 2.0e-3),
            notes=(
                "Independent integral uses analytic dOmega/dr and separate "
                "finite-difference dL/dr. A coarse 400-point grid initially "
                "failed with 1.09e-2 residual near the ISCO; this resolved grid "
                "tests the converged profile."
            ),
        )
    )

    transfer_map = _manual_transfer_map()
    frequency = np.array([1.0e16, 2.0e16, 4.0e16], dtype=np.float64)
    production_spectrum = observed_flux_density(
        transfer_map,
        frequency,
        _constant_intensity,
    )
    expected_spectrum = np.full_like(
        frequency,
        2.5 * np.sum(transfer_map.redshift**3 * transfer_map.solid_angle),
    )
    spectrum_residual = float(np.max(np.abs(production_spectrum - expected_spectrum)))
    rows.append(
        ValidationRow(
            check_id="constant_intensity_observed_spectrum",
            category="spectrum",
            quantity="max_abs_flux_density_residual",
            production_value=spectrum_residual,
            reference_value=0.0,
            residual=spectrum_residual,
            tolerance=1.0e-18,
            status=_status(spectrum_residual, 1.0e-18),
            notes="Independent analytic sum of g^3 I_nu dOmega for constant intensity.",
        )
    )
    return rows


def write_validation_summary(path: Path, rows: list[ValidationRow]) -> None:
    """Write validation rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_id",
        "category",
        "quantity",
        "production_value",
        "reference_value",
        "residual",
        "tolerance",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
