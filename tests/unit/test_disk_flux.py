"""Tests for Page-Thorne disk flux calculations."""

import numpy as np
import pytest
from numpy.typing import NDArray

from kerrdisk.circular_orbits import nt_efficiency
from kerrdisk.constants import GRAVITATIONAL_CONSTANT_SI, SPEED_OF_LIGHT_M_PER_S
from kerrdisk.disk_flux import (
    MAX_CONTROLLED_DELTA_ETA,
    agol_krolik_stress_flux,
    flux_cgs_from_dimensionless,
    flux_si_from_dimensionless,
    local_radiated_luminosity,
    luminosity_at_infinity,
    luminosity_efficiency_error,
    page_thorne_flux_profile,
    stress_luminosity_at_infinity,
    stressed_luminosity_at_infinity,
    stressed_page_thorne_flux_profile,
)
from kerrdisk.isco import isco_radius


def _validation_grid(
    a_star: float,
    outer_radius: float,
    count: int,
) -> NDArray[np.float64]:
    return np.geomspace(isco_radius(a_star), outer_radius, count, dtype=np.float64)


@pytest.mark.parametrize("a_star", [-0.9, 0.0, 0.9])
def test_page_thorne_flux_zero_at_isco_and_positive_outside(a_star: float) -> None:
    radii = _validation_grid(a_star, outer_radius=1_000.0, count=2_000)

    profile = page_thorne_flux_profile(a_star, radii)

    assert profile.flux[0] == pytest.approx(0.0, abs=1.0e-18)
    assert np.all(profile.flux[1:] > 0.0)
    assert profile.inner_radius == pytest.approx(isco_radius(a_star))


def test_page_thorne_flux_accepts_grid_starting_outside_isco() -> None:
    radii = np.geomspace(10.0, 1_000.0, 100, dtype=np.float64)

    profile = page_thorne_flux_profile(0.0, radii)

    assert np.array_equal(profile.radii, radii)
    assert np.all(profile.flux > 0.0)


@pytest.mark.parametrize(
    "radii",
    [
        np.array([5.0, 6.0], dtype=np.float64),
        np.array([6.0], dtype=np.float64),
        np.array([6.0, 6.0, 7.0], dtype=np.float64),
        np.array([6.0, np.nan], dtype=np.float64),
    ],
)
def test_page_thorne_flux_rejects_invalid_grids(radii: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        page_thorne_flux_profile(0.0, radii)


def test_page_thorne_flux_rejects_invalid_mdot() -> None:
    radii = _validation_grid(0.0, outer_radius=100.0, count=100)

    with pytest.raises(ValueError, match="mass_accretion_rate"):
        page_thorne_flux_profile(0.0, radii, mass_accretion_rate=0.0)


def test_large_radius_newtonian_scaling() -> None:
    radii = _validation_grid(0.0, outer_radius=1_000_000.0, count=20_000)

    profile = page_thorne_flux_profile(0.0, radii)
    scaled = profile.flux[-1] * 8.0 * np.pi * profile.radii[-1] ** 3 / 3.0

    assert scaled == pytest.approx(1.0, rel=5.0e-3)


@pytest.mark.parametrize("a_star", [-0.5, 0.0, 0.5, 0.9])
def test_luminosity_at_infinity_matches_nt_efficiency(a_star: float) -> None:
    radii = _validation_grid(a_star, outer_radius=200_000.0, count=30_000)

    profile = page_thorne_flux_profile(a_star, radii)

    assert luminosity_at_infinity(profile) == pytest.approx(
        nt_efficiency(a_star),
        abs=4.0e-5,
    )
    assert abs(luminosity_efficiency_error(profile)) < 4.0e-5


def test_local_luminosity_is_larger_than_luminosity_at_infinity() -> None:
    radii = _validation_grid(0.0, outer_radius=50_000.0, count=10_000)
    profile = page_thorne_flux_profile(0.0, radii)

    assert local_radiated_luminosity(profile) > luminosity_at_infinity(profile)


def test_radial_resolution_convergence() -> None:
    coarse = page_thorne_flux_profile(
        0.5,
        _validation_grid(0.5, outer_radius=50_000.0, count=4_000),
    )
    fine = page_thorne_flux_profile(
        0.5,
        _validation_grid(0.5, outer_radius=50_000.0, count=8_000),
    )

    luminosity_difference = abs(
        luminosity_at_infinity(fine) - luminosity_at_infinity(coarse)
    )

    assert luminosity_difference < 5.0e-7


def test_agol_krolik_zero_stress_recovers_page_thorne_baseline() -> None:
    radii = _validation_grid(0.5, outer_radius=2_000.0, count=2_000)

    baseline = page_thorne_flux_profile(0.5, radii)
    stressed = stressed_page_thorne_flux_profile(0.5, radii, delta_eta=0.0)

    assert np.array_equal(stressed.radii, baseline.radii)
    assert np.array_equal(stressed.stress_flux, np.zeros_like(stressed.stress_flux))
    assert np.array_equal(stressed.flux, baseline.flux)
    assert np.array_equal(stressed.baseline_flux, baseline.flux)


@pytest.mark.parametrize("a_star", [-0.9, 0.0, 0.9])
def test_agol_krolik_stress_flux_is_nonnegative(a_star: float) -> None:
    radii = _validation_grid(a_star, outer_radius=10_000.0, count=4_000)

    stress_flux = agol_krolik_stress_flux(a_star, radii, delta_eta=0.03)

    assert np.all(stress_flux >= 0.0)
    assert stress_flux[0] > 0.0


@pytest.mark.parametrize("a_star", [-0.5, 0.0, 0.5, 0.9])
def test_agol_krolik_added_luminosity_matches_delta_eta(a_star: float) -> None:
    radii = _validation_grid(a_star, outer_radius=1_000_000.0, count=20_000)
    delta_eta = 0.02

    profile = stressed_page_thorne_flux_profile(a_star, radii, delta_eta=delta_eta)

    assert stress_luminosity_at_infinity(profile) == pytest.approx(
        delta_eta,
        abs=2.0e-8,
    )


def test_stressed_luminosity_is_baseline_plus_delta_eta() -> None:
    radii = _validation_grid(0.0, outer_radius=200_000.0, count=30_000)
    delta_eta = 0.01

    baseline = page_thorne_flux_profile(0.0, radii)
    stressed = stressed_page_thorne_flux_profile(0.0, radii, delta_eta=delta_eta)

    assert stressed_luminosity_at_infinity(stressed) == pytest.approx(
        luminosity_at_infinity(baseline) + delta_eta,
        abs=5.0e-8,
    )


def test_agol_krolik_large_radius_slope() -> None:
    radii = _validation_grid(0.5, outer_radius=10_000_000.0, count=10_000)

    stress_flux = agol_krolik_stress_flux(0.5, radii, delta_eta=0.01)
    slope = np.polyfit(np.log(radii[-500:]), np.log(stress_flux[-500:]), deg=1)[0]

    assert slope == pytest.approx(-3.5, abs=2.0e-3)


@pytest.mark.parametrize("delta_eta", [-1.0e-4, float("nan"), 0.11])
def test_agol_krolik_stress_rejects_invalid_delta_eta(delta_eta: float) -> None:
    radii = _validation_grid(0.0, outer_radius=100.0, count=100)

    with pytest.raises(ValueError, match="delta_eta"):
        agol_krolik_stress_flux(0.0, radii, delta_eta=delta_eta)


def test_agol_krolik_stress_accepts_declared_upper_domain() -> None:
    radii = _validation_grid(0.0, outer_radius=100.0, count=100)

    stress_flux = agol_krolik_stress_flux(
        0.0,
        radii,
        delta_eta=MAX_CONTROLLED_DELTA_ETA,
    )

    assert np.all(stress_flux > 0.0)


def test_flux_unit_conversion_matches_newtonian_factor() -> None:
    dimensionless_flux = np.array([3.0 / (8.0 * np.pi * 100.0**3)], dtype=np.float64)
    mass_kg = 10.0 * 1.98847e30
    mdot_kg_s = 1.0e15

    si_flux = flux_si_from_dimensionless(
        dimensionless_flux,
        black_hole_mass_kg=mass_kg,
        mass_accretion_rate_kg_s=mdot_kg_s,
    )
    expected = (
        dimensionless_flux
        * SPEED_OF_LIGHT_M_PER_S**6
        * mdot_kg_s
        / (GRAVITATIONAL_CONSTANT_SI**2 * mass_kg**2)
    )

    assert si_flux == pytest.approx(expected)
    assert flux_cgs_from_dimensionless(
        dimensionless_flux,
        black_hole_mass_kg=mass_kg,
        mass_accretion_rate_kg_s=mdot_kg_s,
    ) == pytest.approx(1.0e3 * expected)


@pytest.mark.parametrize(
    ("mass_kg", "mdot_kg_s"),
    [(0.0, 1.0), (1.0, 0.0), (float("nan"), 1.0), (1.0, float("nan"))],
)
def test_flux_unit_conversion_rejects_invalid_inputs(
    mass_kg: float,
    mdot_kg_s: float,
) -> None:
    with pytest.raises(ValueError):
        flux_si_from_dimensionless(
            np.array([1.0], dtype=np.float64),
            black_hole_mass_kg=mass_kg,
            mass_accretion_rate_kg_s=mdot_kg_s,
        )
