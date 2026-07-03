"""Tests for scalar ray tracing."""

from math import pi

import numpy as np
import pytest

from kerrdisk.geodesics import hamiltonian
from kerrdisk.raytrace import (
    RayOutcome,
    initial_photon_covector,
    schwarzschild_shadow_radius,
    static_observer_tetrad,
    tetrad_inner_products,
    trace_ray,
    trace_ray_adaptive,
)


def test_static_observer_tetrad_is_orthonormal() -> None:
    inner_products = tetrad_inner_products(0.5, 100.0, 1.1)

    assert inner_products == pytest.approx(np.diag([-1.0, 1.0, 1.0, 1.0]), abs=1.0e-12)


def test_initial_photon_covector_is_null() -> None:
    state = initial_photon_covector(0.5, 100.0, 1.1, 5.0, 2.0)

    assert hamiltonian(0.5, state) == pytest.approx(0.0, abs=1.0e-13)


def test_static_observer_tetrad_rejects_invalid_observer() -> None:
    with pytest.raises(ValueError, match="observer_radius"):
        static_observer_tetrad(0.0, 1.5, pi / 2.0)


def test_static_observer_tetrad_rejects_invalid_theta() -> None:
    with pytest.raises(ValueError, match="observer_theta"):
        static_observer_tetrad(0.0, 100.0, 0.0)


def test_static_observer_tetrad_rejects_ergosphere_static_observer() -> None:
    with pytest.raises(ValueError, match="g_tt"):
        static_observer_tetrad(0.9, 1.8, pi / 2.0)


def test_initial_photon_covector_rejects_nonfinite_screen_coordinate() -> None:
    with pytest.raises(ValueError, match="screen"):
        initial_photon_covector(0.0, 100.0, pi / 2.0, float("nan"), 0.0)


def test_escaping_ray_conserves_invariants() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    result = trace_ray(
        0.0,
        state,
        step_size=0.03,
        max_steps=10_000,
        escape_radius=120.0,
    )

    assert result.outcome == RayOutcome.ESCAPED
    assert result.diagnostics.max_abs_hamiltonian < 1.0e-8
    assert abs(result.diagnostics.energy_drift) < 1.0e-14
    assert abs(result.diagnostics.angular_momentum_drift) < 1.0e-14
    assert abs(result.diagnostics.carter_drift) < 1.0e-10


def test_schwarzschild_shadow_boundary_is_bracketed() -> None:
    shadow_radius = schwarzschild_shadow_radius()
    capture_state = initial_photon_covector(0.0, 100.0, pi / 2.0, 5.15, 0.0)
    escape_state = initial_photon_covector(0.0, 100.0, pi / 2.0, 5.25, 0.0)

    capture = trace_ray(
        0.0,
        capture_state,
        step_size=0.02,
        max_steps=20_000,
        escape_radius=120.0,
    )
    escape = trace_ray(
        0.0,
        escape_state,
        step_size=0.02,
        max_steps=20_000,
        escape_radius=120.0,
    )

    assert 5.15 < shadow_radius < 5.25
    assert capture.outcome == RayOutcome.HORIZON_CAPTURE
    assert escape.outcome == RayOutcome.ESCAPED


def test_disk_hit_event_records_intersection() -> None:
    state = initial_photon_covector(0.0, 100.0, 1.2, 0.0, 50.0)

    result = trace_ray(
        0.0,
        state,
        step_size=0.05,
        max_steps=5_000,
        escape_radius=200.0,
        disk_inner_radius=6.0,
        disk_outer_radius=150.0,
    )

    assert result.outcome == RayOutcome.DISK_HIT
    assert result.hit_radius is not None
    assert 6.0 <= result.hit_radius <= 150.0
    assert result.hit_theta == pytest.approx(pi / 2.0)
    assert result.diagnostics.disk_event_residual == pytest.approx(0.0, abs=1.0e-14)


def test_max_steps_is_explicit_outcome() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    result = trace_ray(0.0, state, step_size=0.01, max_steps=1, escape_radius=120.0)

    assert result.outcome == RayOutcome.MAX_STEPS
    assert result.diagnostics.steps == 1


def test_numerical_failure_is_explicit_outcome() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 0.0, 0.0)

    result = trace_ray(
        0.0,
        state,
        step_size=0.1,
        max_steps=2_000,
        horizon_buffer=1.0e-4,
        escape_radius=120.0,
    )

    assert result.outcome == RayOutcome.NUMERICAL_FAILURE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step_size": 0.0}, "step_size"),
        ({"max_steps": 0}, "max_steps"),
        ({"horizon_buffer": 0.0}, "horizon_buffer"),
        ({"escape_radius": 0.0}, "escape_radius"),
        ({"disk_inner_radius": 10.0, "disk_outer_radius": 5.0}, "disk radii"),
    ],
)
def test_trace_ray_rejects_invalid_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    with pytest.raises(ValueError, match=message):
        trace_ray(0.0, state, **kwargs)


def test_trace_ray_rejects_invalid_disk_configuration() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    with pytest.raises(ValueError, match="provided together"):
        trace_ray(0.0, state, disk_inner_radius=6.0)


def test_adaptive_escaping_ray_conserves_invariants() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    result = trace_ray_adaptive(0.0, state, rtol=1.0e-10, escape_radius=120.0)

    assert result.outcome == RayOutcome.ESCAPED
    assert result.diagnostics.max_abs_hamiltonian < 1.0e-8
    assert abs(result.diagnostics.energy_drift) < 1.0e-12
    assert abs(result.diagnostics.angular_momentum_drift) < 1.0e-12
    assert abs(result.diagnostics.carter_drift) < 1.0e-9


def test_adaptive_shadow_boundary_is_bracketed() -> None:
    shadow_radius = schwarzschild_shadow_radius()
    capture = trace_ray_adaptive(
        0.0,
        initial_photon_covector(0.0, 100.0, pi / 2.0, 5.15, 0.0),
        escape_radius=120.0,
    )
    escape = trace_ray_adaptive(
        0.0,
        initial_photon_covector(0.0, 100.0, pi / 2.0, 5.25, 0.0),
        escape_radius=120.0,
    )

    assert 5.15 < shadow_radius < 5.25
    assert capture.outcome == RayOutcome.HORIZON_CAPTURE
    assert escape.outcome == RayOutcome.ESCAPED


def test_adaptive_disk_hit_refines_event_to_high_accuracy() -> None:
    state = initial_photon_covector(0.0, 100.0, 1.2, 0.0, 50.0)

    result = trace_ray_adaptive(
        0.0,
        state,
        rtol=1.0e-10,
        escape_radius=200.0,
        disk_inner_radius=6.0,
        disk_outer_radius=150.0,
    )

    assert result.outcome == RayOutcome.DISK_HIT
    assert result.hit_radius is not None
    assert 6.0 <= result.hit_radius <= 150.0
    assert result.hit_theta == pytest.approx(pi / 2.0, abs=1.0e-10)
    assert result.diagnostics.disk_event_residual == pytest.approx(0.0, abs=1.0e-10)


def test_adaptive_agrees_with_fixed_step_hit_radius() -> None:
    state = initial_photon_covector(0.0, 100.0, 1.2, 0.0, 50.0)

    fixed = trace_ray(
        0.0,
        state,
        step_size=0.01,
        max_steps=40_000,
        escape_radius=200.0,
        disk_inner_radius=6.0,
        disk_outer_radius=150.0,
    )
    adaptive = trace_ray_adaptive(
        0.0,
        state,
        rtol=1.0e-11,
        escape_radius=200.0,
        disk_inner_radius=6.0,
        disk_outer_radius=150.0,
    )

    assert fixed.outcome == RayOutcome.DISK_HIT
    assert adaptive.outcome == RayOutcome.DISK_HIT
    assert fixed.hit_radius is not None
    assert adaptive.hit_radius is not None
    assert adaptive.hit_radius == pytest.approx(fixed.hit_radius, abs=1.0e-4)


def test_adaptive_hit_radius_converges_with_tolerance() -> None:
    state = initial_photon_covector(0.3, 100.0, 1.1, 0.0, 45.0)

    loose = trace_ray_adaptive(
        0.3,
        state,
        rtol=1.0e-8,
        escape_radius=200.0,
        disk_inner_radius=6.0,
        disk_outer_radius=150.0,
    )
    tight = trace_ray_adaptive(
        0.3,
        state,
        rtol=1.0e-11,
        escape_radius=200.0,
        disk_inner_radius=6.0,
        disk_outer_radius=150.0,
    )

    assert loose.outcome == RayOutcome.DISK_HIT
    assert tight.outcome == RayOutcome.DISK_HIT
    assert loose.hit_radius is not None
    assert tight.hit_radius is not None
    assert loose.hit_radius == pytest.approx(tight.hit_radius, abs=1.0e-4)


def test_adaptive_max_steps_is_explicit_outcome() -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    result = trace_ray_adaptive(0.0, state, max_steps=1, escape_radius=120.0)

    assert result.outcome == RayOutcome.MAX_STEPS
    assert result.diagnostics.steps == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rtol": 0.0}, "rtol"),
        ({"atol": 0.0}, "atol"),
        ({"min_step": 0.0}, "min_step"),
        ({"max_step": 1.0e-6}, "max_step"),
        ({"max_steps": 0}, "max_steps"),
        ({"escape_radius": 0.0}, "escape_radius"),
        ({"event_tol": 0.0}, "event_tol"),
        ({"disk_inner_radius": 10.0, "disk_outer_radius": 5.0}, "disk radii"),
    ],
)
def test_adaptive_rejects_invalid_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    state = initial_photon_covector(0.0, 100.0, pi / 2.0, 6.0, 0.0)

    with pytest.raises(ValueError, match=message):
        trace_ray_adaptive(0.0, state, **kwargs)
