"""Tests for photon capture and returning-radiation diagnostics."""

from math import pi

import numpy as np
import pytest

from kerrdisk.geodesics import hamiltonian
from kerrdisk.radiation_diagnostics import (
    emitted_photon_state,
    sample_emission_outcomes,
)


def test_emitted_photon_state_is_null() -> None:
    state = emitted_photon_state(
        0.0,
        10.0,
        emission_mu=0.5,
        emission_azimuth=0.25 * pi,
    )

    assert hamiltonian(0.0, state) == pytest.approx(0.0, abs=1.0e-12)


def test_sample_emission_outcomes_accounts_for_all_samples() -> None:
    summary = sample_emission_outcomes(
        0.0,
        10.0,
        mu_values=[0.35, 0.75],
        azimuth_values=[0.0, pi],
        disk_outer_radius=80.0,
        step_size=0.2,
        max_steps=2_000,
        escape_radius=100.0,
    )

    total = (
        summary.escaped
        + summary.captured
        + summary.returning
        + summary.max_steps
        + summary.numerical_failure
    )
    assert total == summary.sample_count == 4
    assert np.isclose(
        summary.escaped_fraction
        + summary.captured_fraction
        + summary.returning_fraction
        + (summary.max_steps / summary.sample_count)
        + (summary.numerical_failure / summary.sample_count),
        1.0,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 5.0},
        {"emission_mu": -0.1},
        {"emission_mu": 1.1},
        {"emission_azimuth": float("nan")},
    ],
)
def test_emitted_photon_state_rejects_invalid_inputs(
    kwargs: dict[str, float],
) -> None:
    params = {
        "a_star": 0.0,
        "radius": 10.0,
        "emission_mu": 0.5,
        "emission_azimuth": 0.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        emitted_photon_state(**params)
