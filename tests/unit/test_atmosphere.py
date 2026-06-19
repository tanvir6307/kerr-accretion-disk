"""Tests for local atmosphere and color-correction models."""

import warnings
from math import pi

import numpy as np
import pytest

from kerrdisk.atmosphere import (
    ConstantFcol,
    EpochwiseFcol,
    LuminosityLawFcol,
    bolometric_intensity,
    diluted_blackbody_nu,
    effective_temperature_from_flux,
    integrate_specific_intensity,
    isotropic_angular_factor,
    planck_nu,
)
from kerrdisk.constants import (
    BOLTZMANN_CONSTANT_J_K,
    PLANCK_CONSTANT_J_S,
    STEFAN_BOLTZMANN_CONSTANT_SI,
)


def _frequency_grid_for_temperature(temperature_k: float) -> np.ndarray:
    x_values = np.geomspace(1.0e-5, 120.0, 60_000, dtype=np.float64)
    return x_values * BOLTZMANN_CONSTANT_J_K * temperature_k / PLANCK_CONSTANT_J_S


def test_effective_temperature_round_trip() -> None:
    temperature = 1.0e7
    flux = STEFAN_BOLTZMANN_CONSTANT_SI * temperature**4

    recovered = effective_temperature_from_flux(np.array([flux], dtype=np.float64))

    assert recovered[0] == pytest.approx(temperature)


def test_effective_temperature_rejects_negative_flux() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        effective_temperature_from_flux(np.array([-1.0], dtype=np.float64))


@pytest.mark.parametrize("temperature", [1.0e5, 1.0e7])
def test_planck_integral_matches_bolometric_intensity(temperature: float) -> None:
    frequencies = _frequency_grid_for_temperature(temperature)
    intensity = planck_nu(frequencies, temperature)

    integrated = integrate_specific_intensity(frequencies, intensity)

    assert integrated == pytest.approx(bolometric_intensity(temperature), rel=1.0e-4)


def test_bolometric_flux_intensity_convention() -> None:
    temperature = 2.0e6

    assert pi * bolometric_intensity(temperature) == pytest.approx(
        STEFAN_BOLTZMANN_CONSTANT_SI * temperature**4
    )


@pytest.mark.parametrize("f_col", [1.0, 1.7, 2.4])
def test_diluted_blackbody_preserves_bolometric_flux(f_col: float) -> None:
    temperature = 1.0e7
    frequencies = _frequency_grid_for_temperature(f_col * temperature)
    intensity = diluted_blackbody_nu(frequencies, temperature, f_col)

    integrated = integrate_specific_intensity(frequencies, intensity)

    assert integrated == pytest.approx(bolometric_intensity(temperature), rel=1.0e-4)


def test_planck_extreme_range_has_no_runtime_warnings() -> None:
    frequencies = np.geomspace(1.0e1, 1.0e25, 1_000, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intensity = planck_nu(frequencies, 1.0e3)

    assert caught == []
    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0.0)
    assert intensity[-1] == pytest.approx(0.0)


def test_planck_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="temperature"):
        planck_nu(np.array([1.0], dtype=np.float64), 0.0)
    with pytest.raises(ValueError, match="frequency"):
        planck_nu(np.array([0.0], dtype=np.float64), 1.0)


def test_integrate_specific_intensity_rejects_invalid_grids() -> None:
    with pytest.raises(ValueError, match="same shape"):
        integrate_specific_intensity(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_specific_intensity(
            np.array([2.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )


def test_isotropic_angular_factor() -> None:
    mu = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    assert np.array_equal(isotropic_angular_factor(mu), np.ones_like(mu))


def test_isotropic_angular_factor_rejects_invalid_mu() -> None:
    with pytest.raises(ValueError, match="mu"):
        isotropic_angular_factor(np.array([-0.1, 0.5], dtype=np.float64))


def test_constant_fcol() -> None:
    model = ConstantFcol(1.7)

    evaluation = model.evaluate()

    assert evaluation.value == pytest.approx(1.7)
    assert evaluation.raw_value == pytest.approx(1.7)
    assert not evaluation.clipped


def test_constant_fcol_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="f_col"):
        ConstantFcol(0.0)


def test_epochwise_fcol() -> None:
    model = EpochwiseFcol((1.5, 1.6, 1.7))

    assert model.evaluate(epoch_index=2).value == pytest.approx(1.7)


def test_epochwise_fcol_requires_epoch_index() -> None:
    model = EpochwiseFcol((1.5,))

    with pytest.raises(ValueError, match="epoch_index"):
        model.evaluate()


def test_epochwise_fcol_rejects_bad_epoch_index() -> None:
    model = EpochwiseFcol((1.5,))

    with pytest.raises(IndexError, match="epoch_index"):
        model.evaluate(epoch_index=1)


def test_luminosity_law_fcol_unclipped_and_clipped() -> None:
    model = LuminosityLawFcol(f0=1.7, f1=0.2, pivot=0.1, lower=1.4, upper=2.0)

    center = model.evaluate(luminosity_eddington=0.1)
    high = model.evaluate(luminosity_eddington=100.0)
    low = model.evaluate(luminosity_eddington=1.0e-5)

    assert center.value == pytest.approx(1.7)
    assert not center.clipped
    assert high.value == pytest.approx(2.0)
    assert high.clipped
    assert low.value == pytest.approx(1.4)
    assert low.clipped


def test_luminosity_law_fcol_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="lower"):
        LuminosityLawFcol(f0=1.7, f1=0.2, pivot=0.1, lower=2.0, upper=1.0)

    model = LuminosityLawFcol(f0=1.7, f1=0.2, pivot=0.1, lower=1.4, upper=2.0)
    with pytest.raises(ValueError, match="luminosity_eddington"):
        model.evaluate()
    with pytest.raises(ValueError, match="luminosity_eddington"):
        model.evaluate(luminosity_eddington=0.0)
