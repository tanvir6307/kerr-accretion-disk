"""Transfer-map and observed-spectrum assembly."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, pi
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrdisk.circular_orbits import angular_velocity
from kerrdisk.geodesics import StateVector
from kerrdisk.isco import isco_radius
from kerrdisk.metric import _validate_spin, covariant_metric
from kerrdisk.raytrace import (
    RayOutcome,
    initial_photon_covector,
    static_observer_tetrad,
    trace_ray,
    trace_ray_adaptive,
)

type FloatArray = NDArray[np.float64]
type IntensityFunction = Callable[[FloatArray, float], FloatArray]
type IntegratorMode = Literal["fixed", "adaptive"]


@dataclass(frozen=True)
class TransferMap:
    """Successful disk-hit records from an observer screen."""

    a_star: float
    observer_radius: float
    observer_theta: float
    observer_distance: float
    alpha: FloatArray
    beta: FloatArray
    solid_angle: FloatArray
    emission_radius: FloatArray
    emission_phi: FloatArray
    redshift: FloatArray
    emission_mu: FloatArray
    max_abs_hamiltonian: FloatArray
    steps: NDArray[np.int64]
    outcome_counts: dict[str, int]


def _validate_positive_1d(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if np.any(array <= 0.0):
        msg = f"{name} must contain only positive values"
        raise ValueError(msg)
    return array


def _validate_screen_axis(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        msg = f"{name} must be a one-dimensional array with at least two values"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if not np.all(np.diff(array) > 0.0):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)
    return array


def _uniform_spacing(name: str, values: FloatArray) -> float:
    spacing = np.diff(values)
    if not np.allclose(spacing, spacing[0], rtol=1.0e-10, atol=1.0e-12):
        msg = f"{name} must be uniformly spaced for Phase 6 screen integration"
        raise ValueError(msg)
    return float(spacing[0])


def full_disk_screen_axes(
    disk_outer_radius: float,
    screen_size: int,
    *,
    margin: float = 1.4,
) -> tuple[FloatArray, FloatArray]:
    """Return symmetric cell-centered screen axes covering the disk image.

    The image-plane half-width is ``margin * disk_outer_radius`` in
    gravitational radii, large enough to contain the primary disk image (from
    the inner edge near the ISCO out to the outer radius) with room for
    gravitational lensing. Rays are placed at cell centers. The observer must
    be at a radius much larger than this half-width for the flat-screen
    small-angle mapping to remain accurate.
    """

    if not isfinite(disk_outer_radius) or disk_outer_radius <= 0.0:
        msg = "disk_outer_radius must be finite and positive"
        raise ValueError(msg)
    if screen_size < 2:
        msg = "screen_size must be at least two"
        raise ValueError(msg)
    if not isfinite(margin) or margin <= 1.0:
        msg = "margin must be finite and greater than one"
        raise ValueError(msg)
    half_width = margin * disk_outer_radius
    cell_width = (2.0 * half_width) / screen_size
    axis = -half_width + (np.arange(screen_size, dtype=np.float64) + 0.5) * cell_width
    return axis, axis.copy()


def circular_emitter_four_velocity(a_star: float, radius: float) -> FloatArray:
    """Return equatorial circular-emitter four-velocity `u^mu`."""

    _validate_spin(a_star)
    if not isfinite(radius) or radius < isco_radius(a_star):
        msg = "radius must be finite and on or outside the ISCO"
        raise ValueError(msg)
    omega = angular_velocity(a_star, radius)
    metric = covariant_metric(a_star, radius, pi / 2.0)
    normalization = -(
        metric[0, 0] + (2.0 * omega * metric[0, 3]) + (omega * omega * metric[3, 3])
    )
    if normalization <= 0.0:
        msg = "circular-emitter four-velocity normalization is not positive"
        raise ValueError(msg)
    u_t = 1.0 / np.sqrt(normalization)
    return np.array([u_t, 0.0, 0.0, omega * u_t], dtype=np.float64)


def measured_frequency(photon_covector: ArrayLike, four_velocity: ArrayLike) -> float:
    """Return frequency proportional to `-p_mu u^mu`."""

    covector = np.asarray(photon_covector, dtype=np.float64)
    velocity = np.asarray(four_velocity, dtype=np.float64)
    if covector.shape != (4,) or velocity.shape != (4,):
        msg = "photon_covector and four_velocity must both have shape (4,)"
        raise ValueError(msg)
    if not np.all(np.isfinite(covector)) or not np.all(np.isfinite(velocity)):
        msg = "photon_covector and four_velocity must contain finite values"
        raise ValueError(msg)
    frequency = -float(covector @ velocity)
    if frequency <= 0.0:
        msg = "measured frequency must be positive"
        raise ValueError(msg)
    return frequency


def redshift_factor(
    photon_covector: ArrayLike,
    observer_four_velocity: ArrayLike,
    emitter_four_velocity: ArrayLike,
) -> float:
    """Return `nu_obs / nu_em`."""

    nu_obs = measured_frequency(photon_covector, observer_four_velocity)
    nu_em = measured_frequency(photon_covector, emitter_four_velocity)
    return nu_obs / nu_em


def emission_angle_cosine(
    a_star: float,
    photon_covector: ArrayLike,
    emitter_four_velocity: ArrayLike,
    radius: float,
) -> float:
    """Return absolute emission-angle cosine relative to the disk normal."""

    covector = np.asarray(photon_covector, dtype=np.float64)
    velocity = np.asarray(emitter_four_velocity, dtype=np.float64)
    if covector.shape != (4,) or velocity.shape != (4,):
        msg = "photon_covector and emitter_four_velocity must both have shape (4,)"
        raise ValueError(msg)
    if not np.all(np.isfinite(covector)) or not np.all(np.isfinite(velocity)):
        msg = "photon_covector and emitter_four_velocity must contain finite values"
        raise ValueError(msg)
    if not isfinite(radius) or radius <= 0.0:
        msg = "radius must be finite and positive"
        raise ValueError(msg)
    metric = covariant_metric(a_star, radius, pi / 2.0)
    normal_theta = 1.0 / np.sqrt(float(metric[2, 2]))
    normal_projection = abs(float(covector[2]) * normal_theta)
    emitted_frequency = measured_frequency(covector, velocity)
    return float(np.clip(normal_projection / emitted_frequency, 0.0, 1.0))


def _append_record(
    records: dict[str, list[float]],
    *,
    alpha: float,
    beta: float,
    solid_angle: float,
    state: StateVector,
    redshift: float,
    emission_mu: float,
    max_abs_hamiltonian: float,
    steps: int,
) -> None:
    records["alpha"].append(alpha)
    records["beta"].append(beta)
    records["solid_angle"].append(solid_angle)
    records["emission_radius"].append(float(state[1]))
    records["emission_phi"].append(float(state[3]))
    records["redshift"].append(redshift)
    records["emission_mu"].append(emission_mu)
    records["max_abs_hamiltonian"].append(max_abs_hamiltonian)
    records["steps"].append(float(steps))


def build_transfer_map(
    a_star: float,
    alpha_values: ArrayLike,
    beta_values: ArrayLike,
    *,
    observer_radius: float,
    observer_theta: float,
    disk_outer_radius: float,
    observer_distance: float | None = None,
    step_size: float = 0.05,
    max_steps: int = 10_000,
    escape_radius: float | None = None,
    integrator: IntegratorMode = "adaptive",
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
) -> TransferMap:
    """Trace a screen grid and return successful disk-hit transfer records.

    The ``integrator`` selects the adaptive Dormand-Prince RK45 tracer
    (default, error-controlled through ``rtol``/``atol``) or the fixed-step RK4
    reference tracer using ``step_size``.
    """

    _validate_spin(a_star)
    if integrator not in {"fixed", "adaptive"}:
        msg = "integrator must be 'fixed' or 'adaptive'"
        raise ValueError(msg)
    alpha_axis = _validate_screen_axis("alpha_values", alpha_values)
    beta_axis = _validate_screen_axis("beta_values", beta_values)
    if not isfinite(observer_radius) or observer_radius <= 0.0:
        msg = "observer_radius must be finite and positive"
        raise ValueError(msg)
    if observer_distance is None:
        observer_distance = observer_radius
    if not isfinite(observer_distance) or observer_distance <= 0.0:
        msg = "observer_distance must be finite and positive"
        raise ValueError(msg)
    if not isfinite(disk_outer_radius) or disk_outer_radius <= isco_radius(a_star):
        msg = "disk_outer_radius must be finite and larger than the ISCO"
        raise ValueError(msg)

    d_alpha = _uniform_spacing("alpha_values", alpha_axis)
    d_beta = _uniform_spacing("beta_values", beta_axis)
    solid_angle = d_alpha * d_beta / (observer_distance * observer_distance)
    observer_velocity = static_observer_tetrad(
        a_star,
        observer_radius,
        observer_theta,
    ).time
    disk_inner_radius = isco_radius(a_star)
    if escape_radius is None:
        escape_radius = observer_radius * 1.2
    outcome_counts = {outcome.value: 0 for outcome in RayOutcome}
    records: dict[str, list[float]] = {
        "alpha": [],
        "beta": [],
        "solid_angle": [],
        "emission_radius": [],
        "emission_phi": [],
        "redshift": [],
        "emission_mu": [],
        "max_abs_hamiltonian": [],
        "steps": [],
    }

    for beta in beta_axis:
        for alpha in alpha_axis:
            initial_state = initial_photon_covector(
                a_star,
                observer_radius,
                observer_theta,
                float(alpha),
                float(beta),
            )
            if integrator == "adaptive":
                result = trace_ray_adaptive(
                    a_star,
                    initial_state,
                    rtol=rtol,
                    atol=atol,
                    max_steps=max_steps,
                    escape_radius=escape_radius,
                    disk_inner_radius=disk_inner_radius,
                    disk_outer_radius=disk_outer_radius,
                )
            else:
                result = trace_ray(
                    a_star,
                    initial_state,
                    step_size=step_size,
                    max_steps=max_steps,
                    escape_radius=escape_radius,
                    disk_inner_radius=disk_inner_radius,
                    disk_outer_radius=disk_outer_radius,
                )
            outcome_counts[result.outcome.value] += 1
            if result.outcome != RayOutcome.DISK_HIT:
                continue
            photon_covector = result.final_state[4:8]
            emitter_velocity = circular_emitter_four_velocity(
                a_star,
                float(result.final_state[1]),
            )
            emission_mu = emission_angle_cosine(
                a_star,
                photon_covector,
                emitter_velocity,
                float(result.final_state[1]),
            )
            g_factor = redshift_factor(
                photon_covector,
                observer_velocity,
                emitter_velocity,
            )
            _append_record(
                records,
                alpha=float(alpha),
                beta=float(beta),
                solid_angle=solid_angle,
                state=result.final_state,
                redshift=g_factor,
                emission_mu=emission_mu,
                max_abs_hamiltonian=result.diagnostics.max_abs_hamiltonian,
                steps=result.diagnostics.steps,
            )

    return TransferMap(
        a_star=a_star,
        observer_radius=observer_radius,
        observer_theta=observer_theta,
        observer_distance=observer_distance,
        alpha=np.array(records["alpha"], dtype=np.float64),
        beta=np.array(records["beta"], dtype=np.float64),
        solid_angle=np.array(records["solid_angle"], dtype=np.float64),
        emission_radius=np.array(records["emission_radius"], dtype=np.float64),
        emission_phi=np.array(records["emission_phi"], dtype=np.float64),
        redshift=np.array(records["redshift"], dtype=np.float64),
        emission_mu=np.array(records["emission_mu"], dtype=np.float64),
        max_abs_hamiltonian=np.array(
            records["max_abs_hamiltonian"],
            dtype=np.float64,
        ),
        steps=np.array(records["steps"], dtype=np.int64),
        outcome_counts=outcome_counts,
    )


def observed_flux_density(
    transfer_map: TransferMap,
    frequency_hz: ArrayLike,
    emitted_intensity: IntensityFunction,
) -> FloatArray:
    """Integrate observed flux density over a cached transfer map."""

    frequency = _validate_positive_1d("frequency_hz", frequency_hz)
    flux = np.zeros_like(frequency)
    for radius, redshift, solid_angle in zip(
        transfer_map.emission_radius,
        transfer_map.redshift,
        transfer_map.solid_angle,
        strict=True,
    ):
        emitted_frequency = frequency / redshift
        intensity = emitted_intensity(emitted_frequency, float(radius))
        intensity_array = np.asarray(intensity, dtype=np.float64)
        if intensity_array.shape != frequency.shape:
            msg = "emitted_intensity must return an array matching frequency_hz"
            raise ValueError(msg)
        if not np.all(np.isfinite(intensity_array)) or np.any(intensity_array < 0.0):
            msg = "emitted_intensity must return finite nonnegative values"
            raise ValueError(msg)
        flux += redshift**3 * intensity_array * solid_angle
    return flux


def save_transfer_map(path: Path, transfer_map: TransferMap) -> None:
    """Write a transfer map to a compressed NumPy archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "a_star": transfer_map.a_star,
        "observer_radius": transfer_map.observer_radius,
        "observer_theta": transfer_map.observer_theta,
        "observer_distance": transfer_map.observer_distance,
        "outcome_counts": transfer_map.outcome_counts,
    }
    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
        alpha=transfer_map.alpha,
        beta=transfer_map.beta,
        solid_angle=transfer_map.solid_angle,
        emission_radius=transfer_map.emission_radius,
        emission_phi=transfer_map.emission_phi,
        redshift=transfer_map.redshift,
        emission_mu=transfer_map.emission_mu,
        max_abs_hamiltonian=transfer_map.max_abs_hamiltonian,
        steps=transfer_map.steps,
    )


def load_transfer_map(path: Path) -> TransferMap:
    """Load a transfer map written by `save_transfer_map`."""

    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata"]))
        return TransferMap(
            a_star=float(metadata["a_star"]),
            observer_radius=float(metadata["observer_radius"]),
            observer_theta=float(metadata["observer_theta"]),
            observer_distance=float(metadata["observer_distance"]),
            alpha=np.asarray(archive["alpha"], dtype=np.float64),
            beta=np.asarray(archive["beta"], dtype=np.float64),
            solid_angle=np.asarray(archive["solid_angle"], dtype=np.float64),
            emission_radius=np.asarray(archive["emission_radius"], dtype=np.float64),
            emission_phi=np.asarray(archive["emission_phi"], dtype=np.float64),
            redshift=np.asarray(archive["redshift"], dtype=np.float64),
            emission_mu=np.asarray(archive["emission_mu"], dtype=np.float64),
            max_abs_hamiltonian=np.asarray(
                archive["max_abs_hamiltonian"],
                dtype=np.float64,
            ),
            steps=np.asarray(archive["steps"], dtype=np.int64),
            outcome_counts=dict(metadata["outcome_counts"]),
        )
