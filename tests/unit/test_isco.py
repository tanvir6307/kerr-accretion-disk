"""Tests for Kerr ISCO utilities."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kerrdisk.isco import isco_radius, validate_stable_orbit_radius


@pytest.mark.parametrize(
    ("a_star", "expected"),
    [
        (-0.9, 8.717352279606489),
        (-0.5, 7.554584714512358),
        (0.0, 6.000000000000000),
        (0.5, 4.233002529530826),
        (0.9, 2.320883041761887),
        (0.998, 1.236970655175185),
    ],
)
def test_isco_radius_benchmarks(a_star: float, expected: float) -> None:
    assert isco_radius(a_star) == pytest.approx(expected, abs=1.0e-12)


@pytest.mark.parametrize("a_star", [-1.0, 1.0, float("inf"), float("nan")])
def test_isco_radius_rejects_invalid_spin(a_star: float) -> None:
    with pytest.raises(ValueError, match="a_star"):
        isco_radius(a_star)


@given(st.floats(min_value=-0.998, max_value=0.997, allow_nan=False))
def test_isco_radius_decreases_with_spin(a_star: float) -> None:
    upper_spin = min(a_star + 1.0e-3, 0.998)

    assert isco_radius(upper_spin) <= isco_radius(a_star)


def test_validate_stable_orbit_radius_rejects_inside_isco() -> None:
    with pytest.raises(ValueError, match="ISCO"):
        validate_stable_orbit_radius(a_star=0.5, radius=4.0)


def test_validate_stable_orbit_radius_rejects_nonfinite_radius() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_stable_orbit_radius(a_star=0.5, radius=float("nan"))
