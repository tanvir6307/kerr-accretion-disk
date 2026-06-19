"""Likelihood functions for synthetic spectra."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite, lgamma, pi, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrdisk.synthetic import MultiEpochDataset

type FloatArray = NDArray[np.float64]
type EpochExpectations = Sequence[ArrayLike]
type ScalarModelFactory = Callable[[float], EpochExpectations]


@dataclass(frozen=True)
class ScalarFitResult:
    """Result from bounded scalar likelihood maximization."""

    parameter_value: float
    log_likelihood: float
    success: bool
    message: str


def _as_1d(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _same_shape(*arrays: FloatArray) -> bool:
    first_shape = arrays[0].shape
    return all(array.shape == first_shape for array in arrays)


def gaussian_log_likelihood(
    observed: ArrayLike,
    model_expectation: ArrayLike,
    variance: ArrayLike,
) -> float:
    """Return independent Gaussian log-likelihood."""

    y_values = _as_1d("observed", observed)
    model = _as_1d("model_expectation", model_expectation)
    sigma2 = _as_1d("variance", variance)
    if not _same_shape(y_values, model, sigma2):
        msg = "observed, model_expectation, and variance must have the same shape"
        raise ValueError(msg)
    if np.any(model <= 0.0) or np.any(sigma2 <= 0.0):
        return float("-inf")
    residual = y_values - model
    terms = (residual * residual / sigma2) + np.log(2.0 * pi * sigma2)
    return -0.5 * float(np.sum(terms))


def poisson_log_likelihood(
    observed_counts: ArrayLike,
    expected_counts: ArrayLike,
) -> float:
    """Return independent Poisson-count log-likelihood."""

    counts = _as_1d("observed_counts", observed_counts)
    model = _as_1d("expected_counts", expected_counts)
    if counts.shape != model.shape:
        msg = "observed_counts and expected_counts must have the same shape"
        raise ValueError(msg)
    if np.any(counts < 0.0) or np.any(np.floor(counts) != counts):
        msg = "observed_counts must contain nonnegative integer counts"
        raise ValueError(msg)
    if np.any(model <= 0.0):
        return float("-inf")
    log_factorial = np.array([lgamma(float(count) + 1.0) for count in counts])
    terms = counts * np.log(model) - model - log_factorial
    return float(np.sum(terms))


def dataset_log_likelihood(
    dataset: MultiEpochDataset,
    model_expectations: EpochExpectations,
) -> float:
    """Return the matched log-likelihood for a multi-epoch dataset."""

    if len(model_expectations) != len(dataset.epochs):
        msg = "model_expectations length must match dataset epochs"
        raise ValueError(msg)

    total = 0.0
    for epoch, expectation in zip(dataset.epochs, model_expectations, strict=True):
        if epoch.noise_model == "gaussian_relative":
            value = gaussian_log_likelihood(epoch.observed, expectation, epoch.variance)
        else:
            value = poisson_log_likelihood(epoch.observed, expectation)
        if not isfinite(value):
            return float("-inf")
        total += value
    return total


def maximize_scalar_likelihood(
    dataset: MultiEpochDataset,
    *,
    bounds: tuple[float, float],
    model_factory: ScalarModelFactory,
    xatol: float = 1.0e-6,
) -> ScalarFitResult:
    """Maximize a dataset likelihood over one bounded scalar parameter."""

    lower, upper = bounds
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        msg = "bounds must be finite and ordered"
        raise ValueError(msg)
    if not isfinite(xatol) or xatol <= 0.0:
        msg = "xatol must be finite and positive"
        raise ValueError(msg)

    def objective(parameter: float) -> float:
        try:
            log_likelihood = dataset_log_likelihood(
                dataset,
                model_factory(float(parameter)),
            )
        except (FloatingPointError, ValueError):
            return float("inf")
        if not isfinite(log_likelihood):
            return float("inf")
        return -log_likelihood

    inverse_phi = (sqrt(5.0) - 1.0) / 2.0
    inverse_phi_squared = (3.0 - sqrt(5.0)) / 2.0
    left = lower
    right = upper
    width = right - left
    left_probe = left + (inverse_phi_squared * width)
    right_probe = left + (inverse_phi * width)
    left_value = objective(left_probe)
    right_value = objective(right_probe)

    iterations = 0
    while (right - left) > xatol and iterations < 256:
        if left_value < right_value:
            right = right_probe
            right_probe = left_probe
            right_value = left_value
            width = right - left
            left_probe = left + (inverse_phi_squared * width)
            left_value = objective(left_probe)
        else:
            left = left_probe
            left_probe = right_probe
            left_value = right_value
            width = right - left
            right_probe = left + (inverse_phi * width)
            right_value = objective(right_probe)
        iterations += 1

    parameter_value = 0.5 * (left + right)
    max_log_likelihood = -objective(parameter_value)
    success = iterations < 256 and isfinite(max_log_likelihood)
    return ScalarFitResult(
        parameter_value=parameter_value,
        log_likelihood=max_log_likelihood,
        success=success,
        message="bounded golden-section search",
    )


def likelihood_ratio(log_likelihood_a: float, log_likelihood_b: float) -> float:
    """Return `exp(log_likelihood_a - log_likelihood_b)` safely."""

    if not isfinite(log_likelihood_a) or not isfinite(log_likelihood_b):
        msg = "log-likelihood values must be finite"
        raise ValueError(msg)
    return float(np.exp(log_likelihood_a - log_likelihood_b))
