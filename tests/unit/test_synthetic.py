"""Tests for deterministic synthetic spectra."""

import numpy as np
import pytest

from kerrdisk.synthetic import (
    EnergyBins,
    GaussianRelativeNoise,
    PoissonNoise,
    derive_seed,
    generate_gaussian_epoch,
    generate_multi_epoch_dataset,
    generate_poisson_epoch,
    integrated_photon_flux_from_density,
    make_log_energy_bins,
    make_seed_manifest,
    validate_energy_bins,
)


def test_log_energy_bins_have_geometric_centers() -> None:
    bins = make_log_energy_bins(energy_min_kev=1.0, energy_max_kev=4.0, bin_count=2)

    assert bins.edges_kev == pytest.approx(np.array([1.0, 2.0, 4.0]))
    assert bins.centers_kev == pytest.approx(np.array([np.sqrt(2.0), np.sqrt(8.0)]))
    assert bins.widths_kev == pytest.approx(np.array([1.0, 2.0]))


def test_integrated_photon_flux_from_density_uses_midpoint_rule() -> None:
    bins = make_log_energy_bins(energy_min_kev=1.0, energy_max_kev=4.0, bin_count=2)

    flux = integrated_photon_flux_from_density(
        bins,
        lambda energy: 2.0 * energy,
    )

    assert flux == pytest.approx(2.0 * bins.centers_kev * bins.widths_kev)


def test_seed_derivation_is_deterministic_and_hierarchical() -> None:
    first = derive_seed(123, "condition", "a")
    second = derive_seed(123, "condition", "a")
    different = derive_seed(123, "condition", "b")

    assert first == second
    assert first != different


def test_seed_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="master_seed"):
        derive_seed(-1, "bad")
    with pytest.raises(ValueError, match="epoch_index"):
        make_seed_manifest(1, condition_label="bad", epoch_index=-1)


def test_gaussian_epoch_replays_exactly() -> None:
    bins = make_log_energy_bins(bin_count=4)
    photon_flux = np.linspace(1.0, 2.0, bins.centers_kev.size)

    first = generate_gaussian_epoch(
        bins,
        photon_flux,
        GaussianRelativeNoise(relative_error=0.05),
        master_seed=99,
        condition_label="replay",
        epoch_index=0,
        truth_metadata={"spin": 0.7},
    )
    second = generate_gaussian_epoch(
        bins,
        photon_flux,
        GaussianRelativeNoise(relative_error=0.05),
        master_seed=99,
        condition_label="replay",
        epoch_index=0,
        truth_metadata={"spin": 0.7},
    )

    assert np.array_equal(first.observed, second.observed)
    assert first.seed_manifest == second.seed_manifest
    assert first.truth_metadata["spin"] == 0.7


def test_multi_epoch_dataset_stores_distinct_epoch_seeds() -> None:
    bins = make_log_energy_bins(bin_count=3)
    photon_flux_by_epoch = [
        np.ones(bins.centers_kev.size),
        2.0 * np.ones(bins.centers_kev.size),
    ]

    dataset = generate_multi_epoch_dataset(
        bins,
        photon_flux_by_epoch,
        GaussianRelativeNoise(relative_error=0.01),
        master_seed=11,
        condition_label="multi",
        truth_metadata={"spin": 0.5},
        epoch_truth_metadata=[{"eddington_ratio": 0.03}, {"eddington_ratio": 0.1}],
    )

    assert len(dataset.epochs) == 2
    assert dataset.truth_metadata["spin"] == 0.5
    first_seed = dataset.epochs[0].seed_manifest.noise_seed
    second_seed = dataset.epochs[1].seed_manifest.noise_seed
    assert first_seed != second_seed
    assert dataset.epochs[1].truth_metadata["eddington_ratio"] == 0.1


def test_gaussian_noise_moments_match_theory() -> None:
    bins = make_log_energy_bins(energy_min_kev=1.0, energy_max_kev=2.0, bin_count=1)
    target_expectation = 10.0
    photon_flux = np.array([target_expectation / bins.centers_kev[0]])
    observed = []

    for master_seed in range(5_000):
        epoch = generate_gaussian_epoch(
            bins,
            photon_flux,
            GaussianRelativeNoise(relative_error=0.1),
            master_seed=master_seed,
            condition_label="moments",
            epoch_index=0,
        )
        observed.append(epoch.observed[0])

    values = np.array(observed)
    assert values.mean() == pytest.approx(target_expectation, abs=0.04)
    assert values.var(ddof=1) == pytest.approx(1.0, rel=0.05)


def test_poisson_noise_moments_match_theory() -> None:
    bins = make_log_energy_bins(energy_min_kev=1.0, energy_max_kev=2.0, bin_count=1)
    expected_counts = 50.0
    photon_flux = np.array([0.5])
    noise = PoissonNoise(effective_area_cm2=100.0, exposure_s=1.0)
    observed = []

    for master_seed in range(5_000):
        epoch = generate_poisson_epoch(
            bins,
            photon_flux,
            noise,
            master_seed=master_seed,
            condition_label="poisson",
            epoch_index=0,
        )
        observed.append(epoch.observed[0])

    values = np.array(observed)
    assert values.mean() == pytest.approx(expected_counts, rel=0.02)
    assert values.var(ddof=1) == pytest.approx(expected_counts, rel=0.08)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_min_kev": 0.0},
        {"energy_max_kev": 0.01},
        {"bin_count": 0},
    ],
)
def test_log_energy_bins_reject_invalid_inputs(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        make_log_energy_bins(**kwargs)


def test_noise_models_reject_invalid_inputs() -> None:
    bins = make_log_energy_bins(bin_count=1)
    photon_flux = np.ones(1)

    with pytest.raises(ValueError, match="relative_error"):
        generate_gaussian_epoch(
            bins,
            photon_flux,
            GaussianRelativeNoise(relative_error=0.0),
            master_seed=1,
            condition_label="bad",
            epoch_index=0,
        )
    with pytest.raises(ValueError, match="effective_area"):
        generate_poisson_epoch(
            bins,
            photon_flux,
            PoissonNoise(effective_area_cm2=0.0, exposure_s=1.0),
            master_seed=1,
            condition_label="bad",
            epoch_index=0,
        )
    with pytest.raises(ValueError, match="exposure_s"):
        generate_poisson_epoch(
            bins,
            photon_flux,
            PoissonNoise(effective_area_cm2=1.0, exposure_s=0.0),
            master_seed=1,
            condition_label="bad",
            epoch_index=0,
        )


def test_energy_bin_validation_rejects_inconsistent_bins() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_energy_bins(
            EnergyBins(
                edges_kev=np.array([[1.0, 2.0]]),
                centers_kev=np.array([1.5]),
                widths_kev=np.array([1.0]),
            )
        )
    with pytest.raises(ValueError, match="finite"):
        validate_energy_bins(
            EnergyBins(
                edges_kev=np.array([1.0, np.nan]),
                centers_kev=np.array([1.0]),
                widths_kev=np.array([1.0]),
            )
        )
    with pytest.raises(ValueError, match="positive"):
        validate_energy_bins(
            EnergyBins(
                edges_kev=np.array([1.0, 2.0]),
                centers_kev=np.array([1.5]),
                widths_kev=np.array([0.0]),
            )
        )
    with pytest.raises(ValueError, match="inconsistent"):
        validate_energy_bins(
            EnergyBins(
                edges_kev=np.array([1.0, 2.0, 3.0]),
                centers_kev=np.array([1.5]),
                widths_kev=np.array([1.0]),
            )
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_energy_bins(
            EnergyBins(
                edges_kev=np.array([1.0, 2.0, 1.5]),
                centers_kev=np.array([1.5, 1.75]),
                widths_kev=np.array([1.0, 0.5]),
            )
        )


def test_photon_flux_validation_rejects_bad_shapes_and_values() -> None:
    bins = make_log_energy_bins(bin_count=2)

    with pytest.raises(ValueError, match="one value per energy-bin"):
        integrated_photon_flux_from_density(bins, lambda energy: energy[:1])
    with pytest.raises(ValueError, match="nonnegative"):
        integrated_photon_flux_from_density(bins, lambda energy: -energy)
    with pytest.raises(ValueError, match="one value per energy bin"):
        generate_gaussian_epoch(
            bins,
            np.array([1.0]),
            GaussianRelativeNoise(relative_error=0.1),
            master_seed=1,
            condition_label="bad",
            epoch_index=0,
        )
    with pytest.raises(ValueError, match="energy_flux"):
        generate_gaussian_epoch(
            bins,
            np.zeros(2),
            GaussianRelativeNoise(relative_error=0.1),
            master_seed=1,
            condition_label="bad",
            epoch_index=0,
        )
    with pytest.raises(ValueError, match="expected_counts"):
        generate_poisson_epoch(
            bins,
            np.zeros(2),
            PoissonNoise(effective_area_cm2=1.0, exposure_s=1.0),
            master_seed=1,
            condition_label="bad",
            epoch_index=0,
        )


def test_multi_epoch_generation_rejects_bad_epoch_lists() -> None:
    bins = make_log_energy_bins(bin_count=1)

    with pytest.raises(ValueError, match="at least one epoch"):
        generate_multi_epoch_dataset(
            bins,
            [],
            GaussianRelativeNoise(relative_error=0.1),
            master_seed=1,
            condition_label="bad",
        )
    with pytest.raises(ValueError, match="epoch_truth_metadata"):
        generate_multi_epoch_dataset(
            bins,
            [np.ones(1), np.ones(1)],
            GaussianRelativeNoise(relative_error=0.1),
            master_seed=1,
            condition_label="bad",
            epoch_truth_metadata=[{}],
        )
