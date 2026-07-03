"""Scalar Hamiltonian Kerr geodesic utilities."""

from dataclasses import dataclass
from math import cos, isfinite, pi, sin

import numpy as np
from numpy.typing import NDArray

from kerrdisk.metric import (
    _validate_spin,
    contravariant_metric,
    contravariant_metric_derivatives,
    horizon_radius,
)

type StateVector = NDArray[np.float64]
type Matrix4 = NDArray[np.float64]

# Rays approaching the polar axis enter the Boyer-Lindquist coordinate
# singularity where the contravariant metric diverges as 1/sin^2(theta).
# Reject them explicitly so the integrator reports NUMERICAL_FAILURE rather
# than propagating an unreliable right-hand side.
_AXIS_BUFFER = 1.0e-6


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
) -> tuple[Matrix4, Matrix4]:
    """Return analytic radial and polar contravariant metric derivatives."""

    tensor = contravariant_metric_derivatives(a_star, radius, theta)
    return tensor[1], tensor[2]


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
    if not isfinite(theta) or theta <= _AXIS_BUFFER or theta >= pi - _AXIS_BUFFER:
        msg = "theta must remain away from the Boyer-Lindquist coordinate axis"
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


# Dormand-Prince RK45 Butcher tableau. The geodesic system is autonomous, so
# only the stage combination coefficients are required. The fifth-order weights
# are the FSAL seventh row; the fourth-order weights provide the embedded error
# estimate for adaptive step control.
_DP_A21 = 1.0 / 5.0
_DP_A31, _DP_A32 = 3.0 / 40.0, 9.0 / 40.0
_DP_A41, _DP_A42, _DP_A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
_DP_A51, _DP_A52, _DP_A53, _DP_A54 = (
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
)
_DP_A61, _DP_A62, _DP_A63, _DP_A64, _DP_A65 = (
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
)
_DP_B1, _DP_B3, _DP_B4, _DP_B5, _DP_B6 = (
    35.0 / 384.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
)
_DP_E1 = _DP_B1 - 5179.0 / 57600.0
_DP_E3 = _DP_B3 - 7571.0 / 16695.0
_DP_E4 = _DP_B4 - 393.0 / 640.0
_DP_E5 = _DP_B5 - (-92097.0 / 339200.0)
_DP_E6 = _DP_B6 - 187.0 / 2100.0
_DP_E7 = -1.0 / 40.0


def dormand_prince_step(
    a_star: float,
    state: StateVector,
    step_size: float,
) -> tuple[StateVector, StateVector]:
    """Advance one embedded Dormand-Prince RK45 step.

    Returns the fifth-order solution and the embedded fourth/fifth-order error
    estimate used for adaptive step-size control.
    """

    if not isfinite(step_size) or step_size <= 0.0:
        msg = "step_size must be finite and positive"
        raise ValueError(msg)
    k1 = geodesic_rhs(a_star, state)
    k2 = geodesic_rhs(a_star, state + step_size * (_DP_A21 * k1))
    k3 = geodesic_rhs(a_star, state + step_size * (_DP_A31 * k1 + _DP_A32 * k2))
    k4 = geodesic_rhs(
        a_star, state + step_size * (_DP_A41 * k1 + _DP_A42 * k2 + _DP_A43 * k3)
    )
    k5 = geodesic_rhs(
        a_star,
        state + step_size * (_DP_A51 * k1 + _DP_A52 * k2 + _DP_A53 * k3 + _DP_A54 * k4),
    )
    k6 = geodesic_rhs(
        a_star,
        state
        + step_size
        * (_DP_A61 * k1 + _DP_A62 * k2 + _DP_A63 * k3 + _DP_A64 * k4 + _DP_A65 * k5),
    )
    next_state = state + step_size * (
        _DP_B1 * k1 + _DP_B3 * k3 + _DP_B4 * k4 + _DP_B5 * k5 + _DP_B6 * k6
    )
    k7 = geodesic_rhs(a_star, next_state)
    error = step_size * (
        _DP_E1 * k1
        + _DP_E3 * k3
        + _DP_E4 * k4
        + _DP_E5 * k5
        + _DP_E6 * k6
        + _DP_E7 * k7
    )
    if not np.all(np.isfinite(next_state)):
        msg = "Dormand-Prince step produced nonfinite state values"
        raise FloatingPointError(msg)
    return next_state, error
