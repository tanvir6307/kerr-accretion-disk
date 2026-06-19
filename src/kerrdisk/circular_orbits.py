"""Circular equatorial orbit quantities for the Kerr metric."""

from math import isfinite, sqrt

from kerrdisk.isco import isco_radius, validate_stable_orbit_radius
from kerrdisk.metric import _validate_spin


def _orbit_denominator_term(a_star: float, radius: float) -> float:
    sqrt_radius = sqrt(radius)
    return (radius * sqrt_radius) - (3.0 * sqrt_radius) + (2.0 * a_star)


def _validate_orbit(a_star: float, radius: float) -> None:
    _validate_spin(a_star)
    if not isfinite(radius) or radius <= 0.0:
        msg = "radius must be finite and positive"
        raise ValueError(msg)
    validate_stable_orbit_radius(a_star, radius)


def angular_velocity(a_star: float, radius: float) -> float:
    """Return the circular equatorial angular velocity in units of 1/t_g."""

    _validate_orbit(a_star, radius)
    return float(1.0 / ((radius**1.5) + a_star))


def specific_energy(a_star: float, radius: float) -> float:
    """Return circular-orbit specific energy per unit rest mass."""

    _validate_orbit(a_star, radius)
    sqrt_radius = sqrt(radius)
    numerator = (radius * sqrt_radius) - (2.0 * sqrt_radius) + a_star
    denominator = (radius**0.75) * sqrt(_orbit_denominator_term(a_star, radius))
    return float(numerator / denominator)


def specific_angular_momentum(a_star: float, radius: float) -> float:
    """Return circular-orbit specific angular momentum in units of GM/c."""

    _validate_orbit(a_star, radius)
    sqrt_radius = sqrt(radius)
    numerator = (radius * radius) - (2.0 * a_star * sqrt_radius) + (a_star * a_star)
    denominator = (radius**0.75) * sqrt(_orbit_denominator_term(a_star, radius))
    return float(numerator / denominator)


def nt_efficiency(a_star: float) -> float:
    """Return the zero-torque Novikov-Thorne efficiency without photon capture."""

    return 1.0 - specific_energy(a_star, isco_radius(a_star))
