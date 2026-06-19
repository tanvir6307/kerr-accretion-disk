"""Scalar Hamiltonian Kerr geodesic utilities."""

from dataclasses import dataclass
from math import cos, isfinite, pi, sin

import numpy as np
from numpy.typing import NDArray

from kerrdisk.metric import _validate_spin, contravariant_metric, horizon_radius

type StateVector = NDArray[np.float64]
type Matrix4 = NDArray[np.float64]


@dataclass(frozen=True)
class Invariants:
    """Null-geodesic invariant diagnostics."""

    hamiltonian: float
    energy: float
    axial_angular_momentum: float
    carter_constant: float


def _validate_state(state: StateVector) -> None:
    if state.shape != (8,):
        msg = "state must have shape (8,)"
        raise ValueError(msg)
    if not np.all(np.isfinite(state)):
        msg = "state must contain only finite values"
        raise ValueError(msg)


def _contravariant_metric_derivatives(
    a_star: float,
    radius: float,
    theta: float,
    *,
    step_radius: float = 1.0e-5,
    step_theta: float = 1.0e-5,
) -> tuple[Matrix4, Matrix4]:
    if radius - step_radius <= horizon_radius(a_star):
        msg = "radius derivative stencil must remain outside the horizon"
        raise ValueError(msg)
    if theta - step_theta <= 0.0 or theta + step_theta >= pi:
        msg = "theta derivative stencil must remain away from the coordinate axis"
        raise ValueError(msg)
    d_radius = (
        contravariant_metric(a_star, radius + step_radius, theta)
        - contravariant_metric(a_star, radius - step_radius, theta)
    ) / (2.0 * step_radius)
    d_theta = (
        contravariant_metric(a_star, radius, theta + step_theta)
        - contravariant_metric(a_star, radius, theta - step_theta)
    ) / (2.0 * step_theta)
    return d_radius, d_theta


def hamiltonian(a_star: float, state: StateVector) -> float:
    """Return `1/2 g^{mu nu} p_mu p_nu`."""

    _validate_spin(a_star)
    _validate_state(state)
    radius = float(state[1])
    theta = float(state[2])
    covector = state[4:8]
    inverse_metric = contravariant_metric(a_star, radius, theta)
    return float(0.5 * covector @ inverse_metric @ covector)


def carter_constant_null(a_star: float, state: StateVector) -> float:
    """Return the null-ray Carter constant diagnostic."""

    _validate_spin(a_star)
    _validate_state(state)
    theta = float(state[2])
    if theta <= 0.0 or theta >= pi:
        msg = "theta must remain away from the coordinate axis"
        raise ValueError(msg)
    energy = -float(state[4])
    axial_angular_momentum = float(state[7])
    p_theta = float(state[6])
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    return float(
        p_theta * p_theta
        + cos_theta
        * cos_theta
        * (
            axial_angular_momentum * axial_angular_momentum / (sin_theta * sin_theta)
            - a_star * a_star * energy * energy
        )
    )


def invariants(a_star: float, state: StateVector) -> Invariants:
    """Return invariant diagnostics for a ray state."""

    return Invariants(
        hamiltonian=hamiltonian(a_star, state),
        energy=-float(state[4]),
        axial_angular_momentum=float(state[7]),
        carter_constant=carter_constant_null(a_star, state),
    )


def geodesic_rhs(a_star: float, state: StateVector) -> StateVector:
    """Return Hamiltonian geodesic right-hand side."""

    _validate_spin(a_star)
    _validate_state(state)
    radius = float(state[1])
    theta = float(state[2])
    if not isfinite(radius) or radius <= horizon_radius(a_star):
        msg = "radius must remain outside the outer horizon"
        raise ValueError(msg)
    if not isfinite(theta) or theta <= 0.0 or theta >= pi:
        msg = "theta must remain inside the Boyer-Lindquist coordinate range"
        raise ValueError(msg)

    covector = state[4:8]
    inverse_metric = contravariant_metric(a_star, radius, theta)
    d_radius, d_theta = _contravariant_metric_derivatives(a_star, radius, theta)

    rhs = np.zeros(8, dtype=np.float64)
    rhs[0:4] = inverse_metric @ covector
    rhs[5] = -0.5 * float(covector @ d_radius @ covector)
    rhs[6] = -0.5 * float(covector @ d_theta @ covector)
    return rhs


def rk4_step(a_star: float, state: StateVector, step_size: float) -> StateVector:
    """Advance one explicit RK4 step."""

    if not isfinite(step_size) or step_size <= 0.0:
        msg = "step_size must be finite and positive"
        raise ValueError(msg)
    k1 = geodesic_rhs(a_star, state)
    k2 = geodesic_rhs(a_star, state + 0.5 * step_size * k1)
    k3 = geodesic_rhs(a_star, state + 0.5 * step_size * k2)
    k4 = geodesic_rhs(a_star, state + step_size * k3)
    next_state = state + (step_size / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
    if not np.all(np.isfinite(next_state)):
        msg = "RK4 step produced nonfinite state values"
        raise FloatingPointError(msg)
    return next_state
