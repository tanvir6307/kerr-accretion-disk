"""Correctness-first scalar Kerr ray tracer."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, pi, sqrt

import numpy as np
from numpy.typing import NDArray

from kerrdisk.geodesics import (
    Invariants,
    StateVector,
    dormand_prince_step,
    invariants,
    rk4_step,
)
from kerrdisk.metric import _validate_spin, covariant_metric, horizon_radius

type Matrix4 = NDArray[np.float64]


class RayOutcome(StrEnum):
    """Terminal state of a traced ray."""

    DISK_HIT = "DISK_HIT"
    HORIZON_CAPTURE = "HORIZON_CAPTURE"
    ESCAPED = "ESCAPED"
    MAX_STEPS = "MAX_STEPS"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass(frozen=True)
class ObserverTetrad:
    """Contravariant orthonormal tetrad vectors."""

    time: NDArray[np.float64]
    radial: NDArray[np.float64]
    polar: NDArray[np.float64]
    azimuthal: NDArray[np.float64]


@dataclass(frozen=True)
class RayDiagnostics:
    """Diagnostics recorded for a traced scalar ray."""

    initial: Invariants
    final: Invariants | None
    max_abs_hamiltonian: float
    energy_drift: float
    angular_momentum_drift: float
    carter_drift: float
    steps: int
    disk_event_residual: float | None


@dataclass(frozen=True)
class RayTraceResult:
    """Scalar ray-tracing result."""

    outcome: RayOutcome
    final_state: StateVector
    diagnostics: RayDiagnostics
    hit_radius: float | None = None
    hit_theta: float | None = None
    hit_phi: float | None = None


def _validate_observer_coordinates(
    a_star: float,
    observer_radius: float,
    observer_theta: float,
) -> None:
    _validate_spin(a_star)
    if not isfinite(observer_radius) or observer_radius <= horizon_radius(a_star):
        msg = "observer_radius must be finite and outside the horizon"
        raise ValueError(msg)
    if not isfinite(observer_theta) or observer_theta <= 0.0 or observer_theta >= pi:
        msg = "observer_theta must satisfy 0 < theta < pi"
        raise ValueError(msg)


def static_observer_tetrad(
    a_star: float,
    observer_radius: float,
    observer_theta: float,
) -> ObserverTetrad:
    """Return a static-observer orthonormal tetrad at large radius."""

    _validate_observer_coordinates(a_star, observer_radius, observer_theta)
    metric = covariant_metric(a_star, observer_radius, observer_theta)
    if metric[0, 0] >= 0.0:
        msg = "static observer tetrad is invalid where g_tt >= 0"
        raise ValueError(msg)

    time = np.zeros(4, dtype=np.float64)
    radial = np.zeros(4, dtype=np.float64)
    polar = np.zeros(4, dtype=np.float64)
    azimuthal = np.zeros(4, dtype=np.float64)

    time[0] = 1.0 / sqrt(-float(metric[0, 0]))
    radial[1] = 1.0 / sqrt(float(metric[1, 1]))
    polar[2] = 1.0 / sqrt(float(metric[2, 2]))

    normalization = float(metric[3, 3] - (metric[0, 3] * metric[0, 3] / metric[0, 0]))
    if normalization <= 0.0:
        msg = "azimuthal tetrad normalization is not positive"
        raise ValueError(msg)
    azimuthal[3] = 1.0 / sqrt(normalization)
    azimuthal[0] = -float(metric[0, 3]) * azimuthal[3] / float(metric[0, 0])

    return ObserverTetrad(
        time=time,
        radial=radial,
        polar=polar,
        azimuthal=azimuthal,
    )


def tetrad_inner_products(
    a_star: float,
    observer_radius: float,
    observer_theta: float,
) -> Matrix4:
    """Return tetrad inner products with the covariant metric."""

    tetrad = static_observer_tetrad(a_star, observer_radius, observer_theta)
    basis = np.vstack([tetrad.time, tetrad.radial, tetrad.polar, tetrad.azimuthal])
    metric = covariant_metric(a_star, observer_radius, observer_theta)
    return basis @ metric @ basis.T


def initial_photon_covector(
    a_star: float,
    observer_radius: float,
    observer_theta: float,
    alpha: float,
    beta: float,
    *,
    observer_phi: float = 0.0,
) -> StateVector:
    """Return initial ray state from local screen coordinates."""

    _validate_observer_coordinates(a_star, observer_radius, observer_theta)
    if not all(isfinite(value) for value in (alpha, beta, observer_phi)):
        msg = "screen coordinates and observer_phi must be finite"
        raise ValueError(msg)

    tetrad = static_observer_tetrad(a_star, observer_radius, observer_theta)
    direction = np.array(
        [-1.0, beta / observer_radius, alpha / observer_radius],
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    contravariant_momentum = (
        tetrad.time
        + direction[0] * tetrad.radial
        + direction[1] * tetrad.polar
        + direction[2] * tetrad.azimuthal
    )
    metric = covariant_metric(a_star, observer_radius, observer_theta)
    covector = metric @ contravariant_momentum
    return np.array(
        [
            0.0,
            observer_radius,
            observer_theta,
            observer_phi,
            covector[0],
            covector[1],
            covector[2],
            covector[3],
        ],
        dtype=np.float64,
    )


def schwarzschild_shadow_radius() -> float:
    """Return the asymptotic Schwarzschild shadow radius in units of GM/c^2."""

    return sqrt(27.0)


def _interpolate_state(
    previous: StateVector,
    current: StateVector,
    fraction: float,
) -> StateVector:
    return previous + fraction * (current - previous)


def _make_diagnostics(
    a_star: float,
    initial: Invariants,
    final_state: StateVector,
    hamiltonian_history: list[float],
    steps: int,
    disk_event_residual: float | None,
) -> RayDiagnostics:
    try:
        final = invariants(a_star, final_state)
    except ValueError:
        final = None
    if final is None:
        return RayDiagnostics(
            initial=initial,
            final=None,
            max_abs_hamiltonian=max(abs(value) for value in hamiltonian_history),
            energy_drift=float("nan"),
            angular_momentum_drift=float("nan"),
            carter_drift=float("nan"),
            steps=steps,
            disk_event_residual=disk_event_residual,
        )
    return RayDiagnostics(
        initial=initial,
        final=final,
        max_abs_hamiltonian=max(abs(value) for value in hamiltonian_history),
        energy_drift=final.energy - initial.energy,
        angular_momentum_drift=final.axial_angular_momentum
        - initial.axial_angular_momentum,
        carter_drift=final.carter_constant - initial.carter_constant,
        steps=steps,
        disk_event_residual=disk_event_residual,
    )


def trace_ray(
    a_star: float,
    initial_state: StateVector,
    *,
    step_size: float = 0.25,
    max_steps: int = 20_000,
    horizon_buffer: float = 0.1,
    escape_radius: float = 1_000.0,
    disk_inner_radius: float | None = None,
    disk_outer_radius: float | None = None,
) -> RayTraceResult:
    """Trace a single ray until a declared event occurs."""

    _validate_spin(a_star)
    if not isfinite(step_size) or step_size <= 0.0:
        msg = "step_size must be finite and positive"
        raise ValueError(msg)
    if max_steps <= 0:
        msg = "max_steps must be positive"
        raise ValueError(msg)
    if not isfinite(horizon_buffer) or horizon_buffer <= 0.0:
        msg = "horizon_buffer must be finite and positive"
        raise ValueError(msg)
    if not isfinite(escape_radius) or escape_radius <= 0.0:
        msg = "escape_radius must be finite and positive"
        raise ValueError(msg)
    if (disk_inner_radius is None) != (disk_outer_radius is None):
        msg = "disk_inner_radius and disk_outer_radius must be provided together"
        raise ValueError(msg)
    if (
        disk_inner_radius is not None
        and disk_outer_radius is not None
        and (disk_inner_radius <= 0.0 or disk_outer_radius <= disk_inner_radius)
    ):
        msg = "disk radii must satisfy 0 < inner < outer"
        raise ValueError(msg)

    state = initial_state.copy()
    initial = invariants(a_star, state)
    hamiltonian_history = [initial.hamiltonian]
    horizon_stop = horizon_radius(a_star) * (1.0 + horizon_buffer)
    previous_disk_sign = state[2] - (pi / 2.0)

    for step in range(1, max_steps + 1):
        previous = state
        try:
            state = rk4_step(a_star, previous, step_size)
            current_hamiltonian = invariants(a_star, state).hamiltonian
        except (FloatingPointError, ValueError):
            diagnostics = _make_diagnostics(
                a_star,
                initial,
                previous,
                hamiltonian_history,
                step - 1,
                None,
            )
            return RayTraceResult(
                outcome=RayOutcome.NUMERICAL_FAILURE,
                final_state=previous,
                diagnostics=diagnostics,
            )
        hamiltonian_history.append(current_hamiltonian)

        if state[1] <= horizon_stop:
            diagnostics = _make_diagnostics(
                a_star,
                initial,
                state,
                hamiltonian_history,
                step,
                None,
            )
            return RayTraceResult(
                outcome=RayOutcome.HORIZON_CAPTURE,
                final_state=state,
                diagnostics=diagnostics,
            )

        radial_velocity = state[1] - previous[1]
        if state[1] >= escape_radius and radial_velocity > 0.0:
            diagnostics = _make_diagnostics(
                a_star,
                initial,
                state,
                hamiltonian_history,
                step,
                None,
            )
            return RayTraceResult(
                outcome=RayOutcome.ESCAPED,
                final_state=state,
                diagnostics=diagnostics,
            )

        if disk_inner_radius is not None and disk_outer_radius is not None:
            current_disk_sign = state[2] - (pi / 2.0)
            crossed_disk = (
                previous_disk_sign != 0.0
                and previous_disk_sign * current_disk_sign <= 0.0
            )
            if crossed_disk:
                fraction = abs(previous_disk_sign) / (
                    abs(previous_disk_sign) + abs(current_disk_sign)
                )
                hit_state = _interpolate_state(previous, state, fraction)
                hit_radius = float(hit_state[1])
                event_residual = float(hit_state[2] - (pi / 2.0))
                if disk_inner_radius <= hit_radius <= disk_outer_radius:
                    diagnostics = _make_diagnostics(
                        a_star,
                        initial,
                        hit_state,
                        hamiltonian_history,
                        step,
                        event_residual,
                    )
                    return RayTraceResult(
                        outcome=RayOutcome.DISK_HIT,
                        final_state=hit_state,
                        diagnostics=diagnostics,
                        hit_radius=hit_radius,
                        hit_theta=float(hit_state[2]),
                        hit_phi=float(hit_state[3]),
                    )
            previous_disk_sign = current_disk_sign

    diagnostics = _make_diagnostics(
        a_star,
        initial,
        state,
        hamiltonian_history,
        max_steps,
        None,
    )
    return RayTraceResult(
        outcome=RayOutcome.MAX_STEPS,
        final_state=state,
        diagnostics=diagnostics,
    )


# Adaptive step-size controller constants for the Dormand-Prince integrator.
_STEP_SAFETY = 0.9
_STEP_MIN_FACTOR = 0.2
_STEP_MAX_FACTOR = 5.0
_STEP_ERROR_EXPONENT = 0.2  # 1 / (order of embedded error estimate + 1)


def _scaled_error_norm(
    error: StateVector,
    previous_state: StateVector,
    current_state: StateVector,
    rtol: float,
    atol: float,
) -> float:
    scale = atol + rtol * np.maximum(np.abs(previous_state), np.abs(current_state))
    return float(np.sqrt(np.mean(np.square(error / scale))))


def _refine_disk_crossing(
    a_star: float,
    previous_state: StateVector,
    step_size: float,
    event_tol: float,
    *,
    max_iterations: int = 80,
) -> StateVector:
    """Bisect the sub-step to locate the theta = pi/2 disk crossing."""

    lower = 0.0
    upper = step_size
    lower_sign = float(previous_state[2] - (pi / 2.0)) > 0.0
    hit_state = previous_state
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        hit_state, _ = dormand_prince_step(a_star, previous_state, midpoint)
        residual = float(hit_state[2] - (pi / 2.0))
        if abs(residual) <= event_tol:
            return hit_state
        if (residual > 0.0) == lower_sign:
            lower = midpoint
        else:
            upper = midpoint
    return hit_state


def trace_ray_adaptive(
    a_star: float,
    initial_state: StateVector,
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    initial_step: float = 0.5,
    min_step: float = 1.0e-6,
    max_step: float = 16.0,
    max_steps: int = 100_000,
    horizon_buffer: float = 0.1,
    escape_radius: float = 1_000.0,
    disk_inner_radius: float | None = None,
    disk_outer_radius: float | None = None,
    event_tol: float = 1.0e-12,
) -> RayTraceResult:
    """Trace one ray with an adaptive Dormand-Prince RK45 integrator.

    Step size is controlled to a relative/absolute error tolerance, giving a
    single-parameter (`rtol`) convergence knob. The disk-crossing event is
    refined by bisection rather than linear interpolation.
    """

    _validate_spin(a_star)
    _validate_adaptive_controls(
        rtol=rtol,
        atol=atol,
        initial_step=initial_step,
        min_step=min_step,
        max_step=max_step,
        max_steps=max_steps,
        horizon_buffer=horizon_buffer,
        escape_radius=escape_radius,
        event_tol=event_tol,
    )
    if (disk_inner_radius is None) != (disk_outer_radius is None):
        msg = "disk_inner_radius and disk_outer_radius must be provided together"
        raise ValueError(msg)
    if (
        disk_inner_radius is not None
        and disk_outer_radius is not None
        and (disk_inner_radius <= 0.0 or disk_outer_radius <= disk_inner_radius)
    ):
        msg = "disk radii must satisfy 0 < inner < outer"
        raise ValueError(msg)

    state = initial_state.copy()
    initial = invariants(a_star, state)
    hamiltonian_history = [initial.hamiltonian]
    horizon_stop = horizon_radius(a_star) * (1.0 + horizon_buffer)
    previous_disk_sign = float(state[2] - (pi / 2.0))
    step_size = min(initial_step, max_step)
    steps = 0

    while steps < max_steps:
        try:
            candidate, error = dormand_prince_step(a_star, state, step_size)
            candidate_hamiltonian = invariants(a_star, candidate).hamiltonian
        except (FloatingPointError, ValueError):
            step_size *= 0.5
            if step_size < min_step:
                near_horizon = state[1] <= horizon_stop * (1.0 + horizon_buffer)
                outcome = (
                    RayOutcome.HORIZON_CAPTURE
                    if near_horizon
                    else RayOutcome.NUMERICAL_FAILURE
                )
                diagnostics = _make_diagnostics(
                    a_star, initial, state, hamiltonian_history, steps, None
                )
                return RayTraceResult(
                    outcome=outcome, final_state=state, diagnostics=diagnostics
                )
            continue

        error_norm = _scaled_error_norm(error, state, candidate, rtol, atol)
        if error_norm > 1.0 and step_size > min_step:
            factor = max(
                _STEP_MIN_FACTOR,
                _STEP_SAFETY * error_norm**-_STEP_ERROR_EXPONENT,
            )
            step_size = max(min_step, step_size * factor)
            continue

        accepted_step = step_size
        previous = state
        state = candidate
        steps += 1
        hamiltonian_history.append(candidate_hamiltonian)

        if state[1] <= horizon_stop:
            diagnostics = _make_diagnostics(
                a_star, initial, state, hamiltonian_history, steps, None
            )
            return RayTraceResult(
                outcome=RayOutcome.HORIZON_CAPTURE,
                final_state=state,
                diagnostics=diagnostics,
            )

        if state[1] >= escape_radius and state[1] > previous[1]:
            diagnostics = _make_diagnostics(
                a_star, initial, state, hamiltonian_history, steps, None
            )
            return RayTraceResult(
                outcome=RayOutcome.ESCAPED,
                final_state=state,
                diagnostics=diagnostics,
            )

        if disk_inner_radius is not None and disk_outer_radius is not None:
            current_disk_sign = float(state[2] - (pi / 2.0))
            crossed_disk = (
                previous_disk_sign != 0.0
                and previous_disk_sign * current_disk_sign <= 0.0
            )
            if crossed_disk:
                hit_state = _refine_disk_crossing(
                    a_star, previous, accepted_step, event_tol
                )
                hit_radius = float(hit_state[1])
                event_residual = float(hit_state[2] - (pi / 2.0))
                if disk_inner_radius <= hit_radius <= disk_outer_radius:
                    diagnostics = _make_diagnostics(
                        a_star,
                        initial,
                        hit_state,
                        hamiltonian_history,
                        steps,
                        event_residual,
                    )
                    return RayTraceResult(
                        outcome=RayOutcome.DISK_HIT,
                        final_state=hit_state,
                        diagnostics=diagnostics,
                        hit_radius=hit_radius,
                        hit_theta=float(hit_state[2]),
                        hit_phi=float(hit_state[3]),
                    )
            previous_disk_sign = current_disk_sign

        if error_norm == 0.0:
            step_size = min(max_step, accepted_step * _STEP_MAX_FACTOR)
        else:
            growth = min(
                _STEP_MAX_FACTOR,
                _STEP_SAFETY * error_norm**-_STEP_ERROR_EXPONENT,
            )
            step_size = min(max_step, accepted_step * growth)

    diagnostics = _make_diagnostics(
        a_star, initial, state, hamiltonian_history, max_steps, None
    )
    return RayTraceResult(
        outcome=RayOutcome.MAX_STEPS,
        final_state=state,
        diagnostics=diagnostics,
    )


def _validate_adaptive_controls(
    *,
    rtol: float,
    atol: float,
    initial_step: float,
    min_step: float,
    max_step: float,
    max_steps: int,
    horizon_buffer: float,
    escape_radius: float,
    event_tol: float,
) -> None:
    if not isfinite(rtol) or rtol <= 0.0:
        msg = "rtol must be finite and positive"
        raise ValueError(msg)
    if not isfinite(atol) or atol <= 0.0:
        msg = "atol must be finite and positive"
        raise ValueError(msg)
    if not isfinite(min_step) or min_step <= 0.0:
        msg = "min_step must be finite and positive"
        raise ValueError(msg)
    if not isfinite(max_step) or max_step <= min_step:
        msg = "max_step must be finite and greater than min_step"
        raise ValueError(msg)
    if not isfinite(initial_step) or initial_step <= 0.0:
        msg = "initial_step must be finite and positive"
        raise ValueError(msg)
    if max_steps <= 0:
        msg = "max_steps must be positive"
        raise ValueError(msg)
    if not isfinite(horizon_buffer) or horizon_buffer <= 0.0:
        msg = "horizon_buffer must be finite and positive"
        raise ValueError(msg)
    if not isfinite(escape_radius) or escape_radius <= 0.0:
        msg = "escape_radius must be finite and positive"
        raise ValueError(msg)
    if not isfinite(event_tol) or event_tol <= 0.0:
        msg = "event_tol must be finite and positive"
        raise ValueError(msg)
