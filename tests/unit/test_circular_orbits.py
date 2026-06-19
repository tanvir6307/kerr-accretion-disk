"""Tests for circular equatorial Kerr orbit quantities."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kerrdisk.circular_orbits import (
    angular_velocity,
    nt_efficiency,
    specific_angular_momentum,
    specific_energy,
)
from kerrdisk.isco import isco_radius


@pytest.mark.parametrize(
    ("a_star", "expected"),
    [
        (-0.9, 0.03899834565),
        (-0.5, 0.04514222695),
        (0.0, 0.05719095842),
        (0.5, 0.08211799334),
        (0.9, 0.15575299199),
        (0.998, 0.32099416562),
    ],
)
def test_nt_efficiency_benchmarks(a_star: float, expected: float) -> None:
    assert nt_efficiency(a_star) == pytest.approx(expected, abs=5.0e-12)


def test_schwarzschild_isco_orbit_values() -> None:
    radius = isco_radius(0.0)

    assert angular_velocity(0.0, radius) == pytest.approx(1.0 / (6.0**1.5))
    assert specific_energy(0.0, radius) == pytest.approx((8.0 / 9.0) ** 0.5)
    assert specific_angular_momentum(0.0, radius) == pytest.approx(12.0**0.5)


@given(
    a_star=st.floats(min_value=-0.95, max_value=0.95, allow_nan=False),
    offset=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
)
def test_stable_orbit_quantities_are_finite(a_star: float, offset: float) -> None:
    radius = isco_radius(a_star) + offset

    assert angular_velocity(a_star, radius) > 0.0
    assert specific_energy(a_star, radius) > 0.0
    assert specific_angular_momentum(a_star, radius) > 0.0


def test_orbit_quantities_reject_inside_isco() -> None:
    with pytest.raises(ValueError, match="ISCO"):
        specific_energy(a_star=0.5, radius=isco_radius(0.5) - 0.01)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan")])
def test_orbit_quantities_reject_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius"):
        specific_angular_momentum(a_star=0.5, radius=radius)


def test_orbit_quantities_reject_invalid_spin() -> None:
    with pytest.raises(ValueError, match="a_star"):
        angular_velocity(a_star=1.0, radius=6.0)
