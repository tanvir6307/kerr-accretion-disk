"""Tests for transfer maps and observed-spectrum integration."""

from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from kerrdisk.metric import covariant_metric
from kerrdisk.raytrace import static_observer_tetrad
from kerrdisk.spectrum import (
    TransferMap,
    build_transfer_map,
    circular_emitter_four_velocity,
    emission_angle_cosine,
    load_transfer_map,
    measured_frequency,
    observed_flux_density,
    redshift_factor,
    save_transfer_map,
)


def _manual_transfer_map(distance: float, samples: int = 4) -> TransferMap:
    solid_angle = np.full(samples, 4.0 / (distance * distance * samples))
    return TransferMap(
        a_star=0.0,
        observer_radius=100.0,
        observer_theta=1.2,
        observer_distance=distance,
        alpha=np.linspace(-1.0, 1.0, samples),
        beta=np.linspace(10.0, 11.0, samples),
        solid_angle=solid_angle,
        emission_radius=np.full(samples, 20.0),
        emission_phi=np.zeros(samples),
        redshift=np.ones(samples),
        emission_mu=np.ones(samples),
        max_abs_hamiltonian=np.zeros(samples),
        steps=np.ones(samples, dtype=np.int64),
        outcome_counts={
            "DISK_HIT": samples,
            "HORIZON_CAPTURE": 0,
            "ESCAPED": 0,
            "MAX_STEPS": 0,
            "NUMERICAL_FAILURE": 0,
        },
    )


def _constant_intensity(frequency_hz: np.ndarray, radius: float) -> np.ndarray:
    _ = radius
    return np.full_like(frequency_hz, 2.0)


def test_circular_emitter_four_velocity_is_normalized() -> None:
    radius = 10.0
    velocity = circular_emitter_four_velocity(0.0, radius)
    metric = covariant_metric(0.0, radius, pi / 2.0)

    assert velocity @ metric @ velocity == pytest.approx(-1.0)


def test_circular_emitter_four_velocity_rejects_inside_isco() -> None:
    with pytest.raises(ValueError, match="ISCO"):
        circular_emitter_four_velocity(0.0, 5.0)


def test_static_schwarzschild_redshift_sanity() -> None:
    photon_covector = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    observer_velocity = static_observer_tetrad(0.0, 100.0, pi / 2.0).time
    emitter_velocity = static_observer_tetrad(0.0, 10.0, pi / 2.0).time

    redshift = redshift_factor(photon_covector, observer_velocity, emitter_velocity)
    expected = sqrt(1.0 - 2.0 / 10.0) / sqrt(1.0 - 2.0 / 100.0)

    assert redshift == pytest.approx(expected)


def test_measured_frequency_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        measured_frequency(np.array([-1.0]), np.ones(4))


def test_measured_frequency_rejects_negative_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        measured_frequency(
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )


def test_measured_frequency_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        measured_frequency(
            np.array([-1.0, np.nan, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )


def test_emission_angle_cosine_for_normal_schwarzschild_photon() -> None:
    radius = 10.0
    emitter_velocity = circular_emitter_four_velocity(0.0, radius)
    metric = covariant_metric(0.0, radius, pi / 2.0)
    contravariant_photon = emitter_velocity + np.array(
        [0.0, 0.0, -1.0 / sqrt(float(metric[2, 2])), 0.0],
        dtype=np.float64,
    )
    photon_covector = metric @ contravariant_photon

    mu = emission_angle_cosine(0.0, photon_covector, emitter_velocity, radius)

    assert mu == pytest.approx(1.0)


def test_build_transfer_map_records_disk_hits_and_symmetry() -> None:
    transfer_map = build_transfer_map(
        0.0,
        np.array([-5.0, 5.0], dtype=np.float64),
        np.array([50.0, 51.0], dtype=np.float64),
        observer_radius=100.0,
        observer_theta=1.2,
        disk_outer_radius=150.0,
        step_size=0.05,
        max_steps=5_000,
        escape_radius=200.0,
    )

    assert transfer_map.outcome_counts["DISK_HIT"] == 4
    assert transfer_map.outcome_counts["MAX_STEPS"] == 0
    assert np.all(transfer_map.redshift > 0.0)
    assert np.all((transfer_map.emission_mu >= 0.0) & (transfer_map.emission_mu <= 1.0))
    assert transfer_map.emission_radius[0] == pytest.approx(
        transfer_map.emission_radius[1]
    )
    assert transfer_map.emission_radius[2] == pytest.approx(
        transfer_map.emission_radius[3]
    )


def test_build_transfer_map_default_escape_radius_path() -> None:
    transfer_map = build_transfer_map(
        0.0,
        np.array([-5.0, 5.0], dtype=np.float64),
        np.array([50.0, 51.0], dtype=np.float64),
        observer_radius=100.0,
        observer_theta=1.2,
        disk_outer_radius=150.0,
        step_size=0.05,
        max_steps=5_000,
    )

    assert transfer_map.outcome_counts["DISK_HIT"] == 4


def test_build_transfer_map_records_non_hit_outcomes() -> None:
    transfer_map = build_transfer_map(
        0.0,
        np.array([-5.0, 5.0], dtype=np.float64),
        np.array([-50.0, -49.0], dtype=np.float64),
        observer_radius=100.0,
        observer_theta=1.2,
        disk_outer_radius=150.0,
        step_size=0.05,
        max_steps=5_000,
        escape_radius=200.0,
    )

    assert transfer_map.alpha.size == 0
    assert transfer_map.outcome_counts["DISK_HIT"] == 0


def test_build_transfer_map_rejects_bad_screen_grid() -> None:
    with pytest.raises(ValueError, match="uniformly spaced"):
        build_transfer_map(
            0.0,
            np.array([-1.0, 0.0, 2.0], dtype=np.float64),
            np.array([50.0, 51.0], dtype=np.float64),
            observer_radius=100.0,
            observer_theta=1.2,
            disk_outer_radius=150.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_values": np.array([1.0], dtype=np.float64)},
        {"alpha_values": np.array([1.0, np.nan], dtype=np.float64)},
        {"alpha_values": np.array([1.0, 0.0], dtype=np.float64)},
        {"observer_radius": 0.0},
        {"observer_distance": 0.0},
        {"disk_outer_radius": 5.0},
    ],
)
def test_build_transfer_map_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "a_star": 0.0,
        "alpha_values": np.array([-1.0, 1.0], dtype=np.float64),
        "beta_values": np.array([50.0, 51.0], dtype=np.float64),
        "observer_radius": 100.0,
        "observer_theta": 1.2,
        "disk_outer_radius": 150.0,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        build_transfer_map(**base)


def test_observed_flux_density_distance_scaling() -> None:
    frequencies = np.array([1.0e16, 2.0e16], dtype=np.float64)
    near = observed_flux_density(
        _manual_transfer_map(100.0),
        frequencies,
        _constant_intensity,
    )
    far = observed_flux_density(
        _manual_transfer_map(200.0),
        frequencies,
        _constant_intensity,
    )

    assert far == pytest.approx(near / 4.0)


def test_observed_flux_density_screen_resolution_convergence() -> None:
    frequencies = np.array([1.0e16, 2.0e16], dtype=np.float64)
    coarse = observed_flux_density(
        _manual_transfer_map(100.0, samples=4),
        frequencies,
        _constant_intensity,
    )
    fine = observed_flux_density(
        _manual_transfer_map(100.0, samples=16),
        frequencies,
        _constant_intensity,
    )

    assert fine == pytest.approx(coarse)


def test_observed_flux_density_reuses_transfer_map_across_frequency_grids() -> None:
    transfer_map = _manual_transfer_map(100.0)
    first = observed_flux_density(
        transfer_map,
        np.array([1.0e16], dtype=np.float64),
        _constant_intensity,
    )
    second = observed_flux_density(
        transfer_map,
        np.array([1.0e16, 2.0e16, 3.0e16], dtype=np.float64),
        _constant_intensity,
    )

    assert first.shape == (1,)
    assert second.shape == (3,)
    assert transfer_map.outcome_counts["DISK_HIT"] == 4


def test_observed_flux_density_rejects_bad_emissivity_shape() -> None:
    def bad_shape(frequency_hz: np.ndarray, radius: float) -> np.ndarray:
        _ = (frequency_hz, radius)
        return np.array([-1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="matching frequency"):
        observed_flux_density(
            _manual_transfer_map(100.0),
            np.array([1.0e16, 2.0e16], dtype=np.float64),
            bad_shape,
        )


def test_observed_flux_density_rejects_bad_emissivity_values() -> None:
    def bad_values(frequency_hz: np.ndarray, radius: float) -> np.ndarray:
        _ = radius
        return np.full_like(frequency_hz, -1.0)

    with pytest.raises(ValueError, match="nonnegative"):
        observed_flux_density(
            _manual_transfer_map(100.0),
            np.array([1.0e16, 2.0e16], dtype=np.float64),
            bad_values,
        )


@pytest.mark.parametrize(
    "frequency",
    [
        np.array([[1.0e16]], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    ],
)
def test_observed_flux_density_rejects_bad_frequency(frequency: np.ndarray) -> None:
    with pytest.raises(ValueError):
        observed_flux_density(
            _manual_transfer_map(100.0),
            frequency,
            _constant_intensity,
        )


def test_transfer_map_cache_round_trip(tmp_path: Path) -> None:
    transfer_map = _manual_transfer_map(100.0)
    path = tmp_path / "transfer_map.npz"

    save_transfer_map(path, transfer_map)
    loaded = load_transfer_map(path)

    assert loaded.a_star == transfer_map.a_star
    assert loaded.outcome_counts == transfer_map.outcome_counts
    assert np.array_equal(loaded.emission_radius, transfer_map.emission_radius)
    assert np.array_equal(loaded.redshift, transfer_map.redshift)
