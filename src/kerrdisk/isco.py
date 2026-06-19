"""Equatorial Kerr ISCO utilities."""

from math import isfinite, sqrt

from kerrdisk.metric import _validate_spin


def isco_radius(a_star: float) -> float:
    """Return the equatorial Kerr ISCO radius in units of GM/c^2."""

    _validate_spin(a_star)
    if a_star == 0.0:
        return 6.0

    spin_squared = a_star * a_star
    z1 = 1.0 + (1.0 - spin_squared) ** (1.0 / 3.0) * (
        (1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0)
    )
    z2 = sqrt((3.0 * spin_squared) + (z1 * z1))
    sign = 1.0 if a_star > 0.0 else -1.0
    return 3.0 + z2 - sign * sqrt((3.0 - z1) * (3.0 + z1 + (2.0 * z2)))


def validate_stable_orbit_radius(a_star: float, radius: float) -> None:
    """Require a finite radius on or outside the equatorial ISCO."""

    if not isfinite(radius):
        msg = "radius must be finite"
        raise ValueError(msg)
    r_isco = isco_radius(a_star)
    if radius < r_isco:
        msg = f"radius must be on or outside the ISCO: radius={radius}, r_isco={r_isco}"
        raise ValueError(msg)
