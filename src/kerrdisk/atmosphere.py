"""Local atmosphere and color-correction models."""

from dataclasses import dataclass
from math import isfinite, log10, pi
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrdisk.constants import (
    BOLTZMANN_CONSTANT_J_K,
    PLANCK_CONSTANT_J_S,
    SPEED_OF_LIGHT_M_PER_S,
    STEFAN_BOLTZMANN_CONSTANT_SI,
)

type FloatArray = NDArray[np.float64]

WIEN_EXPONENT_CUTOFF: float = 700.0


@dataclass(frozen=True)
class FcolEvaluation:
    """Evaluated color-correction factor."""

    value: float
    raw_value: float
    clipped: bool = False


class FcolModel(Protocol):
    """Protocol for color-correction models."""

    def evaluate(
        self,
        *,
        luminosity_eddington: float | None = None,
        epoch_index: int | None = None,
    ) -> FcolEvaluation:
        """Return the color-correction factor for an epoch or luminosity."""


def _validate_positive_finite(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)


def _validate_nonnegative_array(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(array < 0.0):
        msg = f"{name} must be nonnegative"
        raise ValueError(msg)
    return array


def _validate_positive_array(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(array <= 0.0):
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return array


def effective_temperature_from_flux(flux_si: ArrayLike) -> FloatArray:
    """Return effective temperature in kelvin from one-face SI flux."""

    flux = _validate_nonnegative_array("flux_si", flux_si)
    return np.power(flux / STEFAN_BOLTZMANN_CONSTANT_SI, 0.25)


def planck_nu(frequency_hz: ArrayLike, temperature_k: float) -> FloatArray:
    """Return Planck spectral radiance per frequency in SI units."""

    _validate_positive_finite("temperature_k", temperature_k)
    frequency = _validate_positive_array("frequency_hz", frequency_hz)
    exponent = (
        PLANCK_CONSTANT_J_S * frequency / (BOLTZMANN_CONSTANT_J_K * temperature_k)
    )
    intensity = np.zeros_like(frequency)
    finite_mask = exponent <= WIEN_EXPONENT_CUTOFF
    prefactor = (
        2.0
        * PLANCK_CONSTANT_J_S
        * np.power(frequency[finite_mask], 3)
        / SPEED_OF_LIGHT_M_PER_S**2
    )
    intensity[finite_mask] = prefactor / np.expm1(exponent[finite_mask])
    return intensity


def diluted_blackbody_nu(
    frequency_hz: ArrayLike,
    effective_temperature_k: float,
    f_col: float,
) -> FloatArray:
    """Return diluted-blackbody local specific intensity."""

    _validate_positive_finite("effective_temperature_k", effective_temperature_k)
    _validate_positive_finite("f_col", f_col)
    return planck_nu(frequency_hz, f_col * effective_temperature_k) / f_col**4


def integrate_specific_intensity(
    frequency_hz: ArrayLike,
    intensity_nu: ArrayLike,
) -> float:
    """Integrate specific intensity over frequency."""

    frequency = _validate_positive_array("frequency_hz", frequency_hz)
    intensity = _validate_nonnegative_array("intensity_nu", intensity_nu)
    if frequency.ndim != 1 or intensity.ndim != 1:
        msg = "frequency_hz and intensity_nu must be one-dimensional"
        raise ValueError(msg)
    if frequency.shape != intensity.shape:
        msg = "frequency_hz and intensity_nu must have the same shape"
        raise ValueError(msg)
    if not np.all(np.diff(frequency) > 0.0):
        msg = "frequency_hz must be strictly increasing"
        raise ValueError(msg)
    return float(np.trapezoid(intensity, frequency))


def bolometric_intensity(temperature_k: float) -> float:
    """Return integrated blackbody intensity `sigma T^4 / pi`."""

    _validate_positive_finite("temperature_k", temperature_k)
    return STEFAN_BOLTZMANN_CONSTANT_SI * temperature_k**4 / pi


def isotropic_angular_factor(mu: ArrayLike) -> FloatArray:
    """Return the baseline isotropic angular-emission factor.

    `mu` is the cosine of the local emission angle and must satisfy
    `0 <= mu <= 1` for the outgoing hemisphere.
    """

    values = np.asarray(mu, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        msg = "mu must contain only finite values"
        raise ValueError(msg)
    if np.any((values < 0.0) | (values > 1.0)):
        msg = "mu must satisfy 0 <= mu <= 1"
        raise ValueError(msg)
    return np.ones_like(values)


@dataclass(frozen=True)
class ConstantFcol:
    """Constant color-correction model."""

    f_col: float

    def __post_init__(self) -> None:
        _validate_positive_finite("f_col", self.f_col)

    def evaluate(
        self,
        *,
        luminosity_eddington: float | None = None,
        epoch_index: int | None = None,
    ) -> FcolEvaluation:
        """Return the constant color-correction factor."""

        _ = (luminosity_eddington, epoch_index)
        return FcolEvaluation(value=self.f_col, raw_value=self.f_col)


@dataclass(frozen=True)
class EpochwiseFcol:
    """Epoch-indexed color-correction model."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            msg = "values must contain at least one f_col"
            raise ValueError(msg)
        for value in self.values:
            _validate_positive_finite("f_col value", value)

    def evaluate(
        self,
        *,
        luminosity_eddington: float | None = None,
        epoch_index: int | None = None,
    ) -> FcolEvaluation:
        """Return the color-correction factor for `epoch_index`."""

        _ = luminosity_eddington
        if epoch_index is None:
            msg = "epoch_index is required for EpochwiseFcol"
            raise ValueError(msg)
        if epoch_index < 0 or epoch_index >= len(self.values):
            msg = "epoch_index is outside the configured f_col values"
            raise IndexError(msg)
        value = self.values[epoch_index]
        return FcolEvaluation(value=value, raw_value=value)


@dataclass(frozen=True)
class LuminosityLawFcol:
    """Controlled luminosity-dependent color-correction model."""

    f0: float
    f1: float
    pivot: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _validate_positive_finite("f0", self.f0)
        if not isfinite(self.f1):
            msg = "f1 must be finite"
            raise ValueError(msg)
        _validate_positive_finite("pivot", self.pivot)
        _validate_positive_finite("lower", self.lower)
        _validate_positive_finite("upper", self.upper)
        if self.lower > self.upper:
            msg = "lower must be less than or equal to upper"
            raise ValueError(msg)

    def evaluate(
        self,
        *,
        luminosity_eddington: float | None = None,
        epoch_index: int | None = None,
    ) -> FcolEvaluation:
        """Return the clipped luminosity-law color-correction factor."""

        _ = epoch_index
        if luminosity_eddington is None:
            msg = "luminosity_eddington is required for LuminosityLawFcol"
            raise ValueError(msg)
        _validate_positive_finite("luminosity_eddington", luminosity_eddington)
        raw = self.f0 + self.f1 * log10(luminosity_eddington / self.pivot)
        value = min(max(raw, self.lower), self.upper)
        return FcolEvaluation(value=value, raw_value=raw, clipped=value != raw)
