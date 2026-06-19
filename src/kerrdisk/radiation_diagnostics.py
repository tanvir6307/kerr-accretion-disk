"""Photon capture and returning-radiation diagnostics for thin disks."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from kerrdisk.isco import isco_radius
from kerrdisk.metric import _validate_spin, covariant_metric
from kerrdisk.raytrace import RayOutcome, trace_ray
from kerrdisk.spectrum import circular_emitter_four_velocity

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmissionOutcomeSummary:
    """Outcome fractions for sampled photons emitted from one annulus."""

    a_star: float
    emission_radius: float
    sample_count: int
    escaped: int
    captured: int
    returning: int
    max_steps: int
    numerical_failure: int

    @property
    def escaped_fraction(self) -> float:
        return self.escaped / self.sample_count

    @property
    def captured_fraction(self) -> float:
        return self.captured / self.sample_count

    @property
    def returning_fraction(self) -> float:
        return self.returning / self.sample_count

    def as_dict(self) -> dict[str, object]:
        """Return a CSV-ready row."""

        return {
            "a_star": self.a_star,
            "emission_radius": self.emission_radius,
            "sample_count": self.sample_count,
            "escaped": self.escaped,
            "captured": self.captured,
            "returning": self.returning,
            "max_steps": self.max_steps,
            "numerical_failure": self.numerical_failure,
            "escaped_fraction": self.escaped_fraction,
            "captured_fraction": self.captured_fraction,
            "returning_fraction": self.returning_fraction,
        }


def emitted_photon_state(
    a_star: float,
    radius: float,
    emission_mu: float,
    emission_azimuth: float,
) -> FloatArray:
    """Return an initial photon state emitted from the upper disk face."""

    _validate_emission_inputs(a_star, radius, emission_mu, emission_azimuth)
    metric = covariant_metric(a_star, radius, pi / 2.0)
    emitter_velocity = circular_emitter_four_velocity(a_star, radius)
    radial = np.array([0.0, 1.0 / sqrt(float(metric[1, 1])), 0.0, 0.0])
    normal = np.array([0.0, 0.0, -1.0 / sqrt(float(metric[2, 2])), 0.0])
    azimuthal = _azimuthal_emitter_basis(metric, emitter_velocity)
    transverse = sqrt(max(0.0, 1.0 - emission_mu * emission_mu))
    direction = (
        transverse * cos(emission_azimuth) * radial
        + emission_mu * normal
        + transverse * sin(emission_azimuth) * azimuthal
    )
    contravariant = emitter_velocity + direction
    covector = metric @ contravariant
    return np.array(
        [0.0, radius, pi / 2.0, 0.0, *covector],
        dtype=np.float64,
    )


def sample_emission_outcomes(
    a_star: float,
    radius: float,
    *,
    mu_values: Sequence[float],
    azimuth_values: Sequence[float],
    disk_outer_radius: float,
    step_size: float = 0.1,
    max_steps: int = 10_000,
    escape_radius: float = 1_000.0,
) -> EmissionOutcomeSummary:
    """Classify sampled locally emitted photons by terminal outcome."""

    _validate_spin(a_star)
    if not isfinite(disk_outer_radius) or disk_outer_radius <= radius:
        msg = "disk_outer_radius must be finite and larger than radius"
        raise ValueError(msg)
    counts = dict.fromkeys(RayOutcome, 0)
    disk_inner_radius = isco_radius(a_star)
    for emission_mu in mu_values:
        for emission_azimuth in azimuth_values:
            state = emitted_photon_state(
                a_star,
                radius,
                float(emission_mu),
                float(emission_azimuth),
            )
            result = trace_ray(
                a_star,
                state,
                step_size=step_size,
                max_steps=max_steps,
                escape_radius=escape_radius,
                disk_inner_radius=disk_inner_radius,
                disk_outer_radius=disk_outer_radius,
            )
            counts[result.outcome] += 1
    sample_count = len(mu_values) * len(azimuth_values)
    if sample_count == 0:
        msg = "at least one emission sample is required"
        raise ValueError(msg)
    return EmissionOutcomeSummary(
        a_star=a_star,
        emission_radius=radius,
        sample_count=sample_count,
        escaped=counts[RayOutcome.ESCAPED],
        captured=counts[RayOutcome.HORIZON_CAPTURE],
        returning=counts[RayOutcome.DISK_HIT],
        max_steps=counts[RayOutcome.MAX_STEPS],
        numerical_failure=counts[RayOutcome.NUMERICAL_FAILURE],
    )


def _validate_emission_inputs(
    a_star: float,
    radius: float,
    emission_mu: float,
    emission_azimuth: float,
) -> None:
    _validate_spin(a_star)
    if not isfinite(radius) or radius < isco_radius(a_star):
        msg = "radius must be finite and on or outside the ISCO"
        raise ValueError(msg)
    if not isfinite(emission_mu) or not 0.0 <= emission_mu <= 1.0:
        msg = "emission_mu must satisfy 0 <= mu <= 1"
        raise ValueError(msg)
    if not isfinite(emission_azimuth):
        msg = "emission_azimuth must be finite"
        raise ValueError(msg)


def _azimuthal_emitter_basis(
    metric: FloatArray, emitter_velocity: FloatArray
) -> FloatArray:
    denominator = float(metric[0, 0] * emitter_velocity[0])
    denominator += float(metric[0, 3] * emitter_velocity[3])
    numerator = float(metric[0, 3] * emitter_velocity[0])
    numerator += float(metric[3, 3] * emitter_velocity[3])
    time_component = -numerator / denominator
    basis = np.array([time_component, 0.0, 0.0, 1.0], dtype=np.float64)
    norm_squared = float(basis @ metric @ basis)
    if norm_squared <= 0.0:
        msg = "azimuthal emitter basis normalization is not positive"
        raise ValueError(msg)
    return basis / sqrt(norm_squared)
