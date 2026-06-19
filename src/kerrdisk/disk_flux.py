"""Page-Thorne zero-torque thin-disk flux calculations."""

from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray

from kerrdisk.circular_orbits import (
    angular_velocity,
    nt_efficiency,
    specific_angular_momentum,
    specific_energy,
)
from kerrdisk.constants import GRAVITATIONAL_CONSTANT_SI, SPEED_OF_LIGHT_M_PER_S
from kerrdisk.isco import isco_radius
from kerrdisk.metric import _validate_spin

type FloatArray = NDArray[np.float64]

MAX_CONTROLLED_DELTA_ETA = 0.1


@dataclass(frozen=True)
class DiskFluxProfile:
    """Dimensionless Page-Thorne flux profile."""

    a_star: float
    radii: FloatArray
    flux: FloatArray
    cumulative_integral: FloatArray
    mass_accretion_rate: float
    inner_radius: float


@dataclass(frozen=True)
class StressedDiskFluxProfile:
    """Dimensionless Page-Thorne flux profile with Agol-Krolik stress term."""

    a_star: float
    radii: FloatArray
    baseline_flux: FloatArray
    stress_flux: FloatArray
    flux: FloatArray
    delta_eta: float
    mass_accretion_rate: float
    inner_radius: float


def _validate_mass_accretion_rate(mass_accretion_rate: float) -> None:
    if not isfinite(mass_accretion_rate) or mass_accretion_rate <= 0.0:
        msg = "mass_accretion_rate must be finite and positive"
        raise ValueError(msg)


def _validate_delta_eta(delta_eta: float) -> None:
    if not isfinite(delta_eta):
        msg = "delta_eta must be finite"
        raise ValueError(msg)
    if delta_eta < 0.0:
        msg = "delta_eta must be nonnegative"
        raise ValueError(msg)
    if delta_eta > MAX_CONTROLLED_DELTA_ETA:
        msg = (
            f"delta_eta must be <= {MAX_CONTROLLED_DELTA_ETA} for this controlled model"
        )
        raise ValueError(msg)


def _as_strictly_increasing_radii(radii: NDArray[np.float64]) -> FloatArray:
    values = np.asarray(radii, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        msg = "radii must be a one-dimensional array with at least two values"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "radii must contain only finite values"
        raise ValueError(msg)
    if not np.all(np.diff(values) > 0.0):
        msg = "radii must be strictly increasing"
        raise ValueError(msg)
    return values


def _cumulative_trapezoid(y_values: FloatArray, x_values: FloatArray) -> FloatArray:
    increments = 0.5 * (y_values[1:] + y_values[:-1]) * np.diff(x_values)
    return np.concatenate(
        [np.array([0.0], dtype=np.float64), np.cumsum(increments, dtype=np.float64)]
    )


def _prepend_inner_radius(
    radii: FloatArray,
    inner_radius: float,
) -> tuple[FloatArray, int]:
    tolerance = 1.0e-12 * max(1.0, inner_radius)
    if radii[0] < inner_radius - tolerance:
        msg = "radii must lie on or outside the ISCO"
        raise ValueError(msg)
    if abs(radii[0] - inner_radius) <= tolerance:
        return radii, 0
    return np.concatenate([np.array([inner_radius], dtype=np.float64), radii]), 1


def _circular_orbit_arrays(a_star: float, radii: FloatArray) -> tuple[FloatArray, ...]:
    energy = np.array([specific_energy(a_star, radius) for radius in radii])
    angular_momentum = np.array(
        [specific_angular_momentum(a_star, radius) for radius in radii]
    )
    omega = np.array([angular_velocity(a_star, radius) for radius in radii])
    return energy, angular_momentum, omega


def _radial_gradient(values: FloatArray, radii: FloatArray) -> FloatArray:
    gradient = np.empty_like(values)
    if radii.size == 2:
        gradient[:] = (values[1] - values[0]) / (radii[1] - radii[0])
        return gradient

    h0_start = radii[1] - radii[0]
    h1_start = radii[2] - radii[1]
    gradient[0] = (
        -((2.0 * h0_start) + h1_start) / (h0_start * (h0_start + h1_start)) * values[0]
        + (h0_start + h1_start) / (h0_start * h1_start) * values[1]
        - h0_start / (h1_start * (h0_start + h1_start)) * values[2]
    )

    for index in range(1, radii.size - 1):
        h0 = radii[index] - radii[index - 1]
        h1 = radii[index + 1] - radii[index]
        gradient[index] = (
            -h1 / (h0 * (h0 + h1)) * values[index - 1]
            + (h1 - h0) / (h0 * h1) * values[index]
            + h0 / (h1 * (h0 + h1)) * values[index + 1]
        )

    h0_end = radii[-2] - radii[-3]
    h1_end = radii[-1] - radii[-2]
    gradient[-1] = (
        h1_end / (h0_end * (h0_end + h1_end)) * values[-3]
        - (h0_end + h1_end) / (h0_end * h1_end) * values[-2]
        + ((2.0 * h1_end) + h0_end) / (h1_end * (h0_end + h1_end)) * values[-1]
    )
    return gradient


def page_thorne_flux_profile(
    a_star: float,
    radii: NDArray[np.float64],
    *,
    mass_accretion_rate: float = 1.0,
) -> DiskFluxProfile:
    """Return the dimensionless one-face Page-Thorne flux profile."""

    _validate_spin(a_star)
    _validate_mass_accretion_rate(mass_accretion_rate)

    requested_radii = _as_strictly_increasing_radii(radii)
    inner_radius = isco_radius(a_star)
    working_radii, offset = _prepend_inner_radius(requested_radii, inner_radius)
    energy, angular_momentum, omega = _circular_orbit_arrays(a_star, working_radii)

    d_l_dr = _radial_gradient(angular_momentum, working_radii)
    d_omega_dr = _radial_gradient(omega, working_radii)
    energy_angular_momentum = energy - (omega * angular_momentum)
    integrand = energy_angular_momentum * d_l_dr
    cumulative_integral = _cumulative_trapezoid(integrand, working_radii)

    flux = (
        mass_accretion_rate
        / (4.0 * pi * working_radii)
        * (-d_omega_dr)
        / (energy_angular_momentum * energy_angular_momentum)
        * cumulative_integral
    )

    if np.any(flux < -1.0e-14):
        msg = "computed Page-Thorne flux contains negative values"
        raise FloatingPointError(msg)

    return DiskFluxProfile(
        a_star=a_star,
        radii=working_radii[offset:].copy(),
        flux=flux[offset:].copy(),
        cumulative_integral=cumulative_integral[offset:].copy(),
        mass_accretion_rate=mass_accretion_rate,
        inner_radius=inner_radius,
    )


def _agol_krolik_c_factor(a_star: float, radii: FloatArray) -> FloatArray:
    return 1.0 - (3.0 / radii) + (2.0 * a_star / np.power(radii, 1.5))


def agol_krolik_stress_flux(
    a_star: float,
    radii: NDArray[np.float64],
    *,
    delta_eta: float,
    mass_accretion_rate: float = 1.0,
) -> FloatArray:
    """Return the Agol-Krolik added one-face stress flux."""

    _validate_spin(a_star)
    _validate_delta_eta(delta_eta)
    _validate_mass_accretion_rate(mass_accretion_rate)

    requested_radii = _as_strictly_increasing_radii(radii)
    inner_radius = isco_radius(a_star)
    working_radii, offset = _prepend_inner_radius(requested_radii, inner_radius)

    c_values = _agol_krolik_c_factor(a_star, working_radii)
    c_inner = float(_agol_krolik_c_factor(a_star, np.array([inner_radius]))[0])
    if c_inner <= 0.0 or np.any(c_values <= 0.0):
        msg = "Agol-Krolik C(r) must be positive on the stable disk grid"
        raise FloatingPointError(msg)

    numerator = inner_radius**1.5 * np.sqrt(c_inner) * delta_eta
    stress_flux = (
        3.0
        * mass_accretion_rate
        / (8.0 * pi * working_radii**3)
        * numerator
        / (c_values * np.sqrt(working_radii))
    )

    if np.any(stress_flux < -1.0e-14):
        msg = "computed Agol-Krolik stress flux contains negative values"
        raise FloatingPointError(msg)
    stress_flux_array: FloatArray = np.asarray(stress_flux, dtype=np.float64)
    return stress_flux_array[offset:].copy()


def stressed_page_thorne_flux_profile(
    a_star: float,
    radii: NDArray[np.float64],
    *,
    delta_eta: float,
    mass_accretion_rate: float = 1.0,
) -> StressedDiskFluxProfile:
    """Return Page-Thorne flux plus controlled Agol-Krolik stress flux."""

    baseline = page_thorne_flux_profile(
        a_star,
        radii,
        mass_accretion_rate=mass_accretion_rate,
    )
    stress_flux = agol_krolik_stress_flux(
        a_star,
        baseline.radii,
        delta_eta=delta_eta,
        mass_accretion_rate=mass_accretion_rate,
    )
    total_flux = baseline.flux + stress_flux
    return StressedDiskFluxProfile(
        a_star=a_star,
        radii=baseline.radii.copy(),
        baseline_flux=baseline.flux.copy(),
        stress_flux=stress_flux,
        flux=total_flux,
        delta_eta=delta_eta,
        mass_accretion_rate=mass_accretion_rate,
        inner_radius=baseline.inner_radius,
    )


def local_radiated_luminosity(profile: DiskFluxProfile) -> float:
    """Integrate local two-face emitted luminosity in geometric units."""

    integrand = 4.0 * pi * profile.radii * profile.flux
    return float(np.trapezoid(integrand, profile.radii))


def luminosity_at_infinity(profile: DiskFluxProfile) -> float:
    """Integrate two-face disk luminosity at infinity in geometric units."""

    energy = np.array(
        [specific_energy(profile.a_star, radius) for radius in profile.radii],
        dtype=np.float64,
    )
    integrand = 4.0 * pi * profile.radii * profile.flux * energy
    return float(np.trapezoid(integrand, profile.radii))


def stress_luminosity_at_infinity(profile: StressedDiskFluxProfile) -> float:
    """Integrate the Agol-Krolik added luminosity at infinity."""

    energy = np.array(
        [specific_energy(profile.a_star, radius) for radius in profile.radii],
        dtype=np.float64,
    )
    integrand = 4.0 * pi * profile.radii * profile.stress_flux * energy
    return float(np.trapezoid(integrand, profile.radii))


def stressed_luminosity_at_infinity(profile: StressedDiskFluxProfile) -> float:
    """Integrate total two-face stressed-disk luminosity at infinity."""

    energy = np.array(
        [specific_energy(profile.a_star, radius) for radius in profile.radii],
        dtype=np.float64,
    )
    integrand = 4.0 * pi * profile.radii * profile.flux * energy
    return float(np.trapezoid(integrand, profile.radii))


def luminosity_efficiency_error(profile: DiskFluxProfile) -> float:
    """Return luminosity-at-infinity minus the expected NT efficiency."""

    expected = profile.mass_accretion_rate * nt_efficiency(profile.a_star)
    return luminosity_at_infinity(profile) - expected


def flux_si_from_dimensionless(
    dimensionless_flux_per_unit_mdot: NDArray[np.float64],
    *,
    black_hole_mass_kg: float,
    mass_accretion_rate_kg_s: float,
) -> FloatArray:
    """Convert dimensionless one-face flux per unit `mdot` to W/m^2."""

    if not isfinite(black_hole_mass_kg) or black_hole_mass_kg <= 0.0:
        msg = "black_hole_mass_kg must be finite and positive"
        raise ValueError(msg)
    if not isfinite(mass_accretion_rate_kg_s) or mass_accretion_rate_kg_s <= 0.0:
        msg = "mass_accretion_rate_kg_s must be finite and positive"
        raise ValueError(msg)
    flux = np.asarray(dimensionless_flux_per_unit_mdot, dtype=np.float64)
    factor = (
        SPEED_OF_LIGHT_M_PER_S**6
        * mass_accretion_rate_kg_s
        / (GRAVITATIONAL_CONSTANT_SI**2 * black_hole_mass_kg**2)
    )
    return flux * factor


def flux_cgs_from_dimensionless(
    dimensionless_flux_per_unit_mdot: NDArray[np.float64],
    *,
    black_hole_mass_kg: float,
    mass_accretion_rate_kg_s: float,
) -> FloatArray:
    """Convert dimensionless one-face flux per unit `mdot` to erg/s/cm^2."""

    return 1.0e3 * flux_si_from_dimensionless(
        dimensionless_flux_per_unit_mdot,
        black_hole_mass_kg=black_hole_mass_kg,
        mass_accretion_rate_kg_s=mass_accretion_rate_kg_s,
    )
