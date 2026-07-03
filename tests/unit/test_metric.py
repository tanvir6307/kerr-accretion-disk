"""Tests for Kerr metric utilities."""

from math import pi

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kerrdisk.metric import (
    contravariant_metric,
    contravariant_metric_derivatives,
    covariant_metric,
    covariant_metric_derivatives,
    horizon_radius,
    kerr_auxiliary,
    metric_derivatives,
    metric_inverse_residual,
)


def _finite_difference_contravariant(
    a_star: float,
    radius: float,
    theta: float,
    step: float = 1.0e-6,
) -> np.ndarray:
    tensor = np.zeros((4, 4, 4), dtype=np.float64)
    tensor[1] = (
        contravariant_metric(a_star, radius + step, theta)
        - contravariant_metric(a_star, radius - step, theta)
    ) / (2.0 * step)
    tensor[2] = (
        contravariant_metric(a_star, radius, theta + step)
        - contravariant_metric(a_star, radius, theta - step)
    ) / (2.0 * step)
    return tensor


@pytest.mark.parametrize(
    ("a_star", "expected"),
    [
        (0.0, 2.0),
        (0.5, 1.8660254037844386),
        (-0.5, 1.8660254037844386),
        (0.998, 1.0632139225171164),
    ],
)
def test_horizon_radius(a_star: float, expected: float) -> None:
    assert horizon_radius(a_star) == pytest.approx(expected)


@pytest.mark.parametrize("a_star", [-1.0, 1.0, 1.1, float("nan")])
def test_horizon_radius_rejects_invalid_spin(a_star: float) -> None:
    with pytest.raises(ValueError, match="a_star"):
        horizon_radius(a_star)


def test_kerr_auxiliary_schwarzschild_limit() -> None:
    aux = kerr_auxiliary(a_star=0.0, radius=10.0, theta=pi / 3.0)

    assert aux.sigma == pytest.approx(100.0)
    assert aux.delta == pytest.approx(80.0)
    assert aux.a_function == pytest.approx(10_000.0)


@pytest.mark.parametrize(
    ("radius", "theta"),
    [
        (0.0, pi / 2.0),
        (-1.0, pi / 2.0),
        (5.0, -0.1),
        (5.0, pi + 0.1),
    ],
)
def test_kerr_auxiliary_rejects_invalid_coordinates(
    radius: float,
    theta: float,
) -> None:
    with pytest.raises(ValueError):
        kerr_auxiliary(a_star=0.1, radius=radius, theta=theta)


def test_covariant_metric_schwarzschild_components() -> None:
    radius = 10.0
    theta = pi / 3.0
    metric = covariant_metric(a_star=0.0, radius=radius, theta=theta)

    assert metric[0, 0] == pytest.approx(-(1.0 - (2.0 / radius)))
    assert metric[1, 1] == pytest.approx(1.0 / (1.0 - (2.0 / radius)))
    assert metric[2, 2] == pytest.approx(radius * radius)
    assert metric[3, 3] == pytest.approx(radius * radius * np.sin(theta) ** 2)
    assert metric[0, 3] == pytest.approx(0.0)


def test_covariant_metric_rejects_delta_zero() -> None:
    with pytest.raises(ValueError, match="Delta"):
        covariant_metric(a_star=0.0, radius=2.0, theta=pi / 2.0)


def test_metric_inverse_residual_is_small() -> None:
    residual = metric_inverse_residual(a_star=0.7, radius=8.0, theta=1.1)

    assert np.max(np.abs(residual)) < 1.0e-13


@given(
    a_star=st.floats(min_value=-0.99, max_value=0.99, allow_nan=False),
    theta=st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
    offset=st.floats(min_value=0.25, max_value=100.0, allow_nan=False),
)
def test_metric_inverse_property(a_star: float, theta: float, offset: float) -> None:
    radius = horizon_radius(a_star) + offset

    residual = metric_inverse_residual(a_star=a_star, radius=radius, theta=theta)

    assert np.max(np.abs(residual)) < 1.0e-10


def test_contravariant_metric_rejects_axis() -> None:
    with pytest.raises(ValueError, match="axis"):
        contravariant_metric(a_star=0.2, radius=5.0, theta=0.0)


def test_contravariant_metric_rejects_horizon_or_inside() -> None:
    with pytest.raises(ValueError, match="outside"):
        contravariant_metric(a_star=0.2, radius=horizon_radius(0.2), theta=1.0)


def test_metric_derivatives_shape_and_symmetry() -> None:
    derivatives = metric_derivatives(a_star=0.4, radius=8.0, theta=1.2)

    assert derivatives.shape == (4, 4, 4)
    assert np.allclose(derivatives[0], 0.0)
    assert np.allclose(derivatives[3], 0.0)
    assert np.all(np.isfinite(derivatives))


def test_metric_derivatives_reject_invalid_stencil() -> None:
    with pytest.raises(ValueError, match="outside"):
        metric_derivatives(a_star=0.4, radius=horizon_radius(0.4) + 1.0e-6, theta=1.2)


def test_metric_derivatives_reject_invalid_step() -> None:
    with pytest.raises(ValueError, match="step"):
        metric_derivatives(a_star=0.4, radius=8.0, theta=1.2, step=0.0)


def test_metric_derivatives_reject_axis_crossing_stencil() -> None:
    with pytest.raises(ValueError, match="theta"):
        metric_derivatives(a_star=0.4, radius=8.0, theta=1.0e-6)


def test_analytic_covariant_derivatives_match_finite_difference() -> None:
    analytic = covariant_metric_derivatives(a_star=0.6, radius=7.0, theta=1.1)
    finite = metric_derivatives(a_star=0.6, radius=7.0, theta=1.1)

    assert analytic.shape == (4, 4, 4)
    assert np.allclose(analytic[0], 0.0)
    assert np.allclose(analytic[3], 0.0)
    assert np.allclose(analytic, finite, rtol=1.0e-6, atol=1.0e-8)


def test_analytic_contravariant_derivatives_match_finite_difference() -> None:
    analytic = contravariant_metric_derivatives(a_star=0.6, radius=7.0, theta=1.1)
    finite = _finite_difference_contravariant(a_star=0.6, radius=7.0, theta=1.1)

    assert analytic.shape == (4, 4, 4)
    assert np.allclose(analytic[0], 0.0)
    assert np.allclose(analytic[3], 0.0)
    assert np.allclose(analytic, finite, rtol=1.0e-6, atol=1.0e-8)


@given(
    a_star=st.floats(min_value=-0.95, max_value=0.95, allow_nan=False),
    theta=st.floats(min_value=0.3, max_value=2.8, allow_nan=False),
    offset=st.floats(min_value=0.5, max_value=60.0, allow_nan=False),
)
def test_analytic_derivatives_satisfy_inverse_identity(
    a_star: float,
    theta: float,
    offset: float,
) -> None:
    radius = horizon_radius(a_star) + offset
    inverse = contravariant_metric(a_star, radius, theta)
    covariant_deriv = covariant_metric_derivatives(a_star, radius, theta)
    contravariant_deriv = contravariant_metric_derivatives(a_star, radius, theta)

    for axis in (1, 2):
        expected = -inverse @ covariant_deriv[axis] @ inverse
        assert np.allclose(
            contravariant_deriv[axis], expected, rtol=1.0e-9, atol=1.0e-9
        )


def test_analytic_contravariant_derivatives_reject_horizon() -> None:
    with pytest.raises(ValueError, match="outside"):
        contravariant_metric_derivatives(
            a_star=0.4, radius=horizon_radius(0.4), theta=1.2
        )
