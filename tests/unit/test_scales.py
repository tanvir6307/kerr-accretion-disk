"""Tests for physical scale conversions."""

import pytest

from kerrdisk.circular_orbits import nt_efficiency
from kerrdisk.constants import SPEED_OF_LIGHT_M_PER_S
from kerrdisk.scales import (
    accretion_rate_kg_s,
    black_hole_mass_kg,
    distance_m,
    eddington_luminosity_w,
    frequency_hz_from_kev,
    gravitational_radius_m,
    observer_distance_rg,
)


def test_gravitational_radius_of_ten_solar_masses() -> None:
    r_g = gravitational_radius_m(black_hole_mass_kg(10.0))

    # r_g = G M / c^2 is about 14.77 km for ten solar masses.
    assert r_g == pytest.approx(1.477e4, rel=1.0e-2)


def test_eddington_luminosity_matches_standard_value() -> None:
    luminosity_w = eddington_luminosity_w(black_hole_mass_kg(1.0))

    # About 1.26e31 W (1.26e38 erg/s) per solar mass.
    assert luminosity_w == pytest.approx(1.26e31, rel=1.0e-2)


def test_accretion_rate_reproduces_luminosity() -> None:
    mass_kg = black_hole_mass_kg(10.0)
    efficiency = nt_efficiency(0.7)
    mass_accretion_rate = accretion_rate_kg_s(mass_kg, 0.1, efficiency)

    luminosity = efficiency * mass_accretion_rate * SPEED_OF_LIGHT_M_PER_S**2

    assert luminosity == pytest.approx(0.1 * eddington_luminosity_w(mass_kg))


def test_frequency_from_kev_at_one_kev() -> None:
    # 1 keV corresponds to about 2.418e17 Hz.
    assert frequency_hz_from_kev(1.0) == pytest.approx(2.418e17, rel=1.0e-3)


def test_observer_distance_rg_is_large_and_consistent() -> None:
    value = observer_distance_rg(8.0, 10.0)
    expected = distance_m(8.0) / gravitational_radius_m(black_hole_mass_kg(10.0))

    assert value == pytest.approx(expected)
    assert value > 1.0e15


@pytest.mark.parametrize(
    "call",
    [
        lambda: black_hole_mass_kg(0.0),
        lambda: gravitational_radius_m(-1.0),
        lambda: distance_m(0.0),
        lambda: eddington_luminosity_w(0.0),
        lambda: accretion_rate_kg_s(1.0, 0.0, 0.1),
        lambda: accretion_rate_kg_s(1.0, 0.1, 0.0),
        lambda: frequency_hz_from_kev(0.0),
    ],
)
def test_scale_helpers_reject_nonpositive_inputs(call) -> None:  # noqa: ANN001
    with pytest.raises(ValueError):
        call()
