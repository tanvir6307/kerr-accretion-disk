"""Tests for synthetic-data likelihoods."""

from math import log, pi

import numpy as np
import pytest
from scipy.special import gammaln

from kerrdisk.likelihood import (
    dataset_log_likelihood,
    gaussian_log_likelihood,
    likelihood_ratio,
    maximize_scalar_likelihood,
    poisson_log_likelihood,
)
from kerrdisk.synthetic import (
    GaussianRelativeNoise,
    PoissonNoise,
    generate_multi_epoch_dataset,
    make_log_energy_bins,
)


def test_gaussian_log_likelihood_matches_manual_formula() -> None:
    observed = np.array([1.1, 1.9])
    model = np.array([1.0, 2.0])
    variance = np.array([0.01, 0.04])

    value = gaussian_log_likelihood(observed, model, variance)
    manual = -0.5 * np.sum(
        ((observed - model) ** 2 / variance) + np.log(2.0 * pi * variance)
    )

    assert value == pytest.approx(manual)


def test_poisson_log_likelihood_matches_manual_formula() -> None:
    counts = np.array([3.0, 10.0])
    expected = np.array([2.5, 11.0])

    value = poisson_log_likelihood(counts, expected)
    manual = np.sum(counts * np.log(expected) - expected - gammaln(counts + 1.0))

    assert value == pytest.approx(manual)


def test_dataset_log_likelihood_matches_epoch_noise_model() -> None:
    bins = make_log_energy_bins(bin_count=2)
    photon_flux = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    gaussian = generate_multi_epoch_dataset(
        bins,
        photon_flux,
        GaussianRelativeNoise(relative_error=0.01),
        master_seed=10,
        condition_label="gaussian",
    )
    poisson = generate_multi_epoch_dataset(
        bins,
        photon_flux,
        PoissonNoise(effective_area_cm2=100.0, exposure_s=5.0),
        master_seed=10,
        condition_label="poisson",
    )

    gaussian_expectations = [epoch.expectation for epoch in gaussian.epochs]
    poisson_expectations = [epoch.expectation for epoch in poisson.epochs]

    assert np.isfinite(dataset_log_likelihood(gaussian, gaussian_expectations))
    assert np.isfinite(dataset_log_likelihood(poisson, poisson_expectations))


def test_invalid_model_expectation_returns_negative_infinity() -> None:
    assert gaussian_log_likelihood([1.0], [-1.0], [0.1]) == float("-inf")
    assert poisson_log_likelihood([1.0], [0.0]) == float("-inf")


def test_likelihood_rejects_invalid_observed_counts() -> None:
    with pytest.raises(ValueError, match="integer"):
        poisson_log_likelihood([1.5], [2.0])


def test_likelihood_rejects_bad_shapes_and_nonfinite_data() -> None:
    with pytest.raises(ValueError, match="same shape"):
        gaussian_log_likelihood([1.0, 2.0], [1.0], [0.1])
    with pytest.raises(ValueError, match="one-dimensional"):
        gaussian_log_likelihood([[1.0]], [1.0], [0.1])
    with pytest.raises(ValueError, match="finite"):
        poisson_log_likelihood([1.0], [float("nan")])
    with pytest.raises(ValueError, match="same shape"):
        poisson_log_likelihood([1.0, 2.0], [1.0])


def test_likelihood_is_finite_across_valid_scalar_grid() -> None:
    bins = make_log_energy_bins(bin_count=6)
    base = np.exp(-bins.centers_kev / 5.0)
    dataset = generate_multi_epoch_dataset(
        bins,
        [base],
        GaussianRelativeNoise(relative_error=0.01),
        master_seed=55,
        condition_label="finite-grid",
    )

    for spin in np.linspace(-0.9, 0.9, 9):
        expectation = [dataset.epochs[0].expectation * (1.0 + (0.05 * spin))]
        assert np.isfinite(dataset_log_likelihood(dataset, expectation))


def test_high_signal_same_model_scalar_recovery() -> None:
    bins = make_log_energy_bins(bin_count=12)
    template = np.exp(-bins.centers_kev / 4.0)
    true_spin = 0.37
    true_photon_flux = template * (1.0 + (0.25 * true_spin))
    dataset = generate_multi_epoch_dataset(
        bins,
        [true_photon_flux],
        GaussianRelativeNoise(relative_error=1.0e-6),
        master_seed=1234,
        condition_label="spin-recovery",
        truth_metadata={"spin": true_spin},
    )

    def model_factory(spin: float) -> list[np.ndarray]:
        photon_flux = template * (1.0 + (0.25 * spin))
        energy_flux = bins.centers_kev * photon_flux
        return [energy_flux]

    result = maximize_scalar_likelihood(
        dataset,
        bounds=(-0.9, 0.9),
        model_factory=model_factory,
        xatol=1.0e-8,
    )

    assert result.success
    assert result.parameter_value == pytest.approx(true_spin, abs=1.0e-4)


def test_scalar_recovery_handles_invalid_model_region() -> None:
    bins = make_log_energy_bins(bin_count=3)
    photon_flux = np.ones(3)
    dataset = generate_multi_epoch_dataset(
        bins,
        [photon_flux],
        GaussianRelativeNoise(relative_error=1.0e-6),
        master_seed=1,
        condition_label="invalid-region",
    )

    def model_factory(parameter: float) -> list[np.ndarray]:
        return [dataset.epochs[0].expectation * parameter]

    result = maximize_scalar_likelihood(
        dataset,
        bounds=(-1.0, 2.0),
        model_factory=model_factory,
    )

    assert result.success
    assert result.parameter_value == pytest.approx(1.0, abs=1.0e-4)


def test_dataset_likelihood_rejects_wrong_epoch_count() -> None:
    bins = make_log_energy_bins(bin_count=2)
    dataset = generate_multi_epoch_dataset(
        bins,
        [np.ones(2)],
        GaussianRelativeNoise(relative_error=0.01),
        master_seed=1,
        condition_label="bad-count",
    )

    with pytest.raises(ValueError, match="length"):
        dataset_log_likelihood(dataset, [])


def test_scalar_maximizer_rejects_bad_controls() -> None:
    bins = make_log_energy_bins(bin_count=1)
    dataset = generate_multi_epoch_dataset(
        bins,
        [np.ones(1)],
        GaussianRelativeNoise(relative_error=0.01),
        master_seed=1,
        condition_label="bad-controls",
    )

    with pytest.raises(ValueError, match="bounds"):
        maximize_scalar_likelihood(
            dataset,
            bounds=(1.0, 1.0),
            model_factory=lambda parameter: [np.ones(1) * parameter],
        )
    with pytest.raises(ValueError, match="xatol"):
        maximize_scalar_likelihood(
            dataset,
            bounds=(0.5, 1.5),
            model_factory=lambda parameter: [np.ones(1) * parameter],
            xatol=0.0,
        )


def test_scalar_maximizer_reports_all_invalid_models() -> None:
    bins = make_log_energy_bins(bin_count=1)
    dataset = generate_multi_epoch_dataset(
        bins,
        [np.ones(1)],
        GaussianRelativeNoise(relative_error=0.01),
        master_seed=1,
        condition_label="all-invalid",
    )

    result = maximize_scalar_likelihood(
        dataset,
        bounds=(0.0, 1.0),
        model_factory=lambda parameter: [-np.ones(1) * parameter],
    )

    assert not result.success
    assert result.log_likelihood == float("-inf")


def test_likelihood_ratio() -> None:
    assert likelihood_ratio(log(4.0), log(2.0)) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="finite"):
        likelihood_ratio(float("-inf"), 0.0)
