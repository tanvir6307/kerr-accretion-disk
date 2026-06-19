"""Tests for scalar Hamiltonian geodesic utilities."""

from math import pi

import numpy as np
import pytest

from kerrdisk.geodesics import (
    carter_constant_null,
    geodesic_rhs,
    hamiltonian,
    invariants,
    rk4_step,
)
from kerrdisk.raytrace import initial_photon_covector


def test_initial_ray_hamiltonian_is_null() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    assert hamiltonian(0.0, state) == pytest.approx(0.0, abs=1.0e-14)


def test_geodesic_rhs_preserves_stationary_axisymmetric_momenta() -> None:
    state = initial_photon_covector(0.3, 100.0, 1.2, 4.0, 2.0)

    rhs = geodesic_rhs(0.3, state)

    assert rhs[4] == pytest.approx(0.0)
    assert rhs[7] == pytest.approx(0.0)
    assert np.all(np.isfinite(rhs))


def test_rk4_step_keeps_simple_ray_nearly_null() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    next_state = rk4_step(0.0, state, 0.05)

    assert hamiltonian(0.0, next_state) == pytest.approx(0.0, abs=1.0e-12)


def test_invariant_container_values() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    diagnostic = invariants(0.0, state)

    assert diagnostic.energy > 0.0
    assert diagnostic.axial_angular_momentum > 0.0
    assert diagnostic.carter_constant == pytest.approx(carter_constant_null(0.0, state))


def test_rk4_step_rejects_invalid_step_size() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    with pytest.raises(ValueError, match="step_size"):
        rk4_step(0.0, state, 0.0)


def test_hamiltonian_rejects_bad_state_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        hamiltonian(0.0, np.zeros(7, dtype=np.float64))


def test_hamiltonian_rejects_nonfinite_state() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)
    state[1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        hamiltonian(0.0, state)


def test_carter_constant_rejects_axis_state() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)
    state[2] = 0.0

    with pytest.raises(ValueError, match="axis"):
        carter_constant_null(0.0, state)


def test_geodesic_rhs_rejects_radius_inside_horizon() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)
    state[1] = 1.9

    with pytest.raises(ValueError, match="horizon"):
        geodesic_rhs(0.0, state)


def test_geodesic_rhs_rejects_theta_axis() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)
    state[2] = 1.0e-7

    with pytest.raises(ValueError, match="axis"):
        geodesic_rhs(0.0, state)
