"""Kerr metric utilities in Boyer-Lindquist coordinates."""

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

type Matrix4 = NDArray[np.float64]
type DerivativeTensor = NDArray[np.float64]


@dataclass(frozen=True)
class KerrAuxiliary:
    """Common Kerr metric scalar functions."""

    sigma: float
    delta: float
    a_function: float


def _validate_spin(a_star: float) -> None:
    if not isfinite(a_star) or abs(a_star) >= 1.0:
        msg = "a_star must be finite and satisfy -1 < a_star < 1"
        raise ValueError(msg)


def _validate_theta(theta: float, *, require_regular_inverse: bool) -> None:
    if not isfinite(theta) or theta < 0.0 or theta > pi:
        msg = "theta must be finite and satisfy 0 <= theta <= pi"
        raise ValueError(msg)
    if require_regular_inverse and (theta <= 0.0 or theta >= pi):
        msg = "contravariant metric is singular on the Boyer-Lindquist axis"
        raise ValueError(msg)


def horizon_radius(a_star: float) -> float:
    """Return the outer Kerr horizon radius in units of GM/c^2."""

    _validate_spin(a_star)
    return 1.0 + sqrt(1.0 - a_star * a_star)


def kerr_auxiliary(a_star: float, radius: float, theta: float) -> KerrAuxiliary:
    """Return Sigma, Delta, and A for the Kerr metric."""

    _validate_spin(a_star)
    if not isfinite(radius) or radius <= 0.0:
        msg = "radius must be finite and positive"
        raise ValueError(msg)
    _validate_theta(theta, require_regular_inverse=False)

    sin_theta = sin(theta)
    cos_theta = cos(theta)
    radius_squared = radius * radius
    spin_squared = a_star * a_star
    sigma = radius_squared + spin_squared * cos_theta * cos_theta
    delta = radius_squared - 2.0 * radius + spin_squared
    a_function = (radius_squared + spin_squared) ** 2 - (
        spin_squared * delta * sin_theta * sin_theta
    )
    return KerrAuxiliary(sigma=sigma, delta=delta, a_function=a_function)


def covariant_metric(a_star: float, radius: float, theta: float) -> Matrix4:
    """Return covariant Kerr metric components `g_mu_nu`.

    Coordinates are ordered as `(t, r, theta, phi)`.
    """

    aux = kerr_auxiliary(a_star, radius, theta)
    if aux.delta == 0.0:
        msg = "covariant Boyer-Lindquist metric has g_rr singular at Delta = 0"
        raise ValueError(msg)

    sin_theta = sin(theta)
    sin_squared = sin_theta * sin_theta
    metric = np.zeros((4, 4), dtype=np.float64)
    metric[0, 0] = -(1.0 - (2.0 * radius / aux.sigma))
    metric[0, 3] = -(2.0 * a_star * radius * sin_squared / aux.sigma)
    metric[3, 0] = metric[0, 3]
    metric[1, 1] = aux.sigma / aux.delta
    metric[2, 2] = aux.sigma
    metric[3, 3] = aux.a_function * sin_squared / aux.sigma
    return metric


def contravariant_metric(a_star: float, radius: float, theta: float) -> Matrix4:
    """Return contravariant Kerr metric components `g^mu_nu`.

    Coordinates are ordered as `(t, r, theta, phi)`.
    """

    _validate_theta(theta, require_regular_inverse=True)
    if radius <= horizon_radius(a_star):
        msg = "contravariant metric is only evaluated outside the outer horizon"
        raise ValueError(msg)

    aux = kerr_auxiliary(a_star, radius, theta)
    sin_theta = sin(theta)
    sin_squared = sin_theta * sin_theta
    metric = np.zeros((4, 4), dtype=np.float64)
    sigma_delta = aux.sigma * aux.delta
    metric[0, 0] = -aux.a_function / sigma_delta
    metric[0, 3] = -(2.0 * a_star * radius) / sigma_delta
    metric[3, 0] = metric[0, 3]
    metric[1, 1] = aux.delta / aux.sigma
    metric[2, 2] = 1.0 / aux.sigma
    metric[3, 3] = (aux.delta - a_star * a_star * sin_squared) / (
        sigma_delta * sin_squared
    )
    return metric


def metric_inverse_residual(a_star: float, radius: float, theta: float) -> Matrix4:
    """Return `g_mu_alpha g^alpha_nu - delta_mu_nu`."""

    covariant = covariant_metric(a_star, radius, theta)
    contravariant = contravariant_metric(a_star, radius, theta)
    return covariant @ contravariant - np.eye(4, dtype=np.float64)


def metric_derivatives(
    a_star: float,
    radius: float,
    theta: float,
    *,
    step: float = 1.0e-5,
) -> DerivativeTensor:
    """Return finite-difference derivatives of `g_mu_nu`.

    The first axis is the derivative coordinate index `(t, r, theta, phi)`.
    Stationarity and axisymmetry set the `t` and `phi` derivatives to zero.
    """

    if not isfinite(step) or step <= 0.0:
        msg = "step must be finite and positive"
        raise ValueError(msg)
    if radius - step <= horizon_radius(a_star):
        msg = "radius - step must remain outside the outer horizon"
        raise ValueError(msg)
    if theta - step <= 0.0 or theta + step >= pi:
        msg = "theta +/- step must remain away from the coordinate axis"
        raise ValueError(msg)

    derivatives = np.zeros((4, 4, 4), dtype=np.float64)
    derivatives[1] = (
        covariant_metric(a_star, radius + step, theta)
        - covariant_metric(a_star, radius - step, theta)
    ) / (2.0 * step)
    derivatives[2] = (
        covariant_metric(a_star, radius, theta + step)
        - covariant_metric(a_star, radius, theta - step)
    ) / (2.0 * step)
    return derivatives
