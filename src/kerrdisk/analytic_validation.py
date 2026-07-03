"""Independent analytic cross-validation of the ray-traced disk spectrum.

The Schwarzschild face-on disk spectrum can be written in closed form: a
circular-orbit emitter observed along the polar axis has no line-of-sight
Doppler shift, so its redshift factor is ``g(r) = 1 / u^t = sqrt(1 - 3 / r)``,
and the observed flux is a redshifted, area-weighted sum of diluted blackbodies
with no light bending. This module implements that calculation from first
principles, independently of the production ray tracer and transfer-map
assembly, and compares it against the ray-traced backend at near face-on
inclination. Agreement validates the ray tracer's redshift factor, solid-angle
normalization, and flux integration.
"""

from dataclasses import dataclass
from math import isfinite, pi, radians

import numpy as np
from numpy.typing import NDArray

from kerrdisk.atmosphere import diluted_blackbody_nu, effective_temperature_from_flux
from kerrdisk.circular_orbits import nt_efficiency
from kerrdisk.constants import (
    KILO_ELECTRON_VOLT_J,
    PLANCK_CONSTANT_J_S,
    WATT_PER_M2_TO_ERG_PER_S_CM2,
)
from kerrdisk.disk_flux import flux_si_from_dimensionless, page_thorne_flux_profile
from kerrdisk.isco import isco_radius
from kerrdisk.scales import (
    accretion_rate_kg_s,
    black_hole_mass_kg,
    distance_m,
    gravitational_radius_m,
    observer_distance_rg,
)
from kerrdisk.spectrum import build_transfer_map, full_disk_screen_axes
from kerrdisk.synthetic import EnergyBins, make_log_energy_bins
from kerrdisk.thermal_spectrum import (
    KerrThinDiskSettings,
    ray_traced_kerr_thin_disk_energy_flux,
)

type FloatArray = NDArray[np.float64]

SCHWARZSCHILD_ISCO = 6.0


@dataclass(frozen=True)
class FaceOnValidationRow:
    """Result of the face-on ray-traced versus analytic comparison."""

    inclination_deg: float
    screen_size: int
    disk_hits: int
    relative_l1_spectrum_delta: float
    total_flux_ratio: float
    tolerance: float
    status: str

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def schwarzschild_face_on_disk_spectrum(
    *,
    eddington_ratio: float,
    f_col: float,
    energy_bins: EnergyBins,
    mass_msun: float = 10.0,
    distance_kpc: float = 8.0,
    radial_grid_count: int = 400,
    disk_outer_radius_rg: float = 80.0,
) -> FloatArray:
    """Return the closed-form Schwarzschild face-on disk energy-flux spectrum.

    The result is the observed binned energy flux in ``erg s^-1 cm^-2`` for a
    zero-torque Page-Thorne disk with the analytic face-on redshift
    ``g(r) = sqrt(1 - 3 / r)`` and no light bending.
    """

    if not isfinite(eddington_ratio) or eddington_ratio <= 0.0:
        msg = "eddington_ratio must be finite and positive"
        raise ValueError(msg)
    if not isfinite(f_col) or f_col <= 0.0:
        msg = "f_col must be finite and positive"
        raise ValueError(msg)
    if radial_grid_count < 8:
        msg = "radial_grid_count must be at least eight"
        raise ValueError(msg)
    if not isfinite(disk_outer_radius_rg) or disk_outer_radius_rg <= SCHWARZSCHILD_ISCO:
        msg = "disk_outer_radius_rg must exceed the Schwarzschild ISCO"
        raise ValueError(msg)

    mass_kg = black_hole_mass_kg(mass_msun)
    r_g = gravitational_radius_m(mass_kg)
    observer_distance = distance_m(distance_kpc)

    radii = np.geomspace(
        isco_radius(0.0), disk_outer_radius_rg, radial_grid_count, dtype=np.float64
    )
    dimensionless_flux = page_thorne_flux_profile(
        0.0, radii, mass_accretion_rate=1.0
    ).flux
    mass_accretion_rate = accretion_rate_kg_s(
        mass_kg, eddington_ratio, nt_efficiency(0.0)
    )
    flux_si = flux_si_from_dimensionless(
        np.maximum(dimensionless_flux, 0.0),
        black_hole_mass_kg=mass_kg,
        mass_accretion_rate_kg_s=mass_accretion_rate,
    )
    temperature_k = effective_temperature_from_flux(flux_si)
    redshift = np.sqrt(np.maximum(1.0 - 3.0 / radii, 0.0))

    frequency = energy_bins.centers_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S
    frequency_width = (
        energy_bins.widths_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S
    )
    radial_width = np.gradient(radii)
    flux_nu = np.zeros_like(frequency)
    for index in range(radii.size):
        temperature = float(temperature_k[index])
        factor = float(redshift[index])
        if temperature <= 0.0 or factor <= 0.0:
            continue
        intensity = diluted_blackbody_nu(frequency / factor, temperature, f_col)
        annulus_area = 2.0 * pi * radii[index] * radial_width[index] * (r_g * r_g)
        flux_nu += factor**3 * intensity * annulus_area / (observer_distance**2)

    return flux_nu * frequency_width * WATT_PER_M2_TO_ERG_PER_S_CM2


def run_face_on_cross_validation(
    *,
    inclination_deg: float = 3.0,
    screen_size: int = 48,
    disk_outer_radius_rg: float = 80.0,
    observer_radius: float = 1_000.0,
    mass_msun: float = 10.0,
    distance_kpc: float = 8.0,
    eddington_ratio: float = 0.1,
    f_col: float = 1.7,
    radial_grid_count: int = 400,
    energy_bin_count: int = 24,
    tolerance: float = 0.10,
) -> FaceOnValidationRow:
    """Compare the ray-traced near-face-on spectrum with the analytic benchmark."""

    if not isfinite(inclination_deg) or not 0.0 < inclination_deg < 15.0:
        msg = "inclination_deg must be a small positive near-face-on angle"
        raise ValueError(msg)
    energy_bins = make_log_energy_bins(
        energy_min_kev=0.1,
        energy_max_kev=20.0,
        bin_count=energy_bin_count,
    )
    analytic = schwarzschild_face_on_disk_spectrum(
        eddington_ratio=eddington_ratio,
        f_col=f_col,
        energy_bins=energy_bins,
        mass_msun=mass_msun,
        distance_kpc=distance_kpc,
        radial_grid_count=radial_grid_count,
        disk_outer_radius_rg=disk_outer_radius_rg,
    )

    alpha, beta = full_disk_screen_axes(disk_outer_radius_rg, screen_size)
    transfer_map = build_transfer_map(
        0.0,
        alpha,
        beta,
        observer_radius=observer_radius,
        observer_theta=radians(inclination_deg),
        disk_outer_radius=disk_outer_radius_rg,
        observer_distance=observer_distance_rg(distance_kpc, mass_msun),
        max_steps=8_000,
        escape_radius=2.0 * observer_radius,
    )
    ray_traced = ray_traced_kerr_thin_disk_energy_flux(
        transfer_map=transfer_map,
        a_star=0.0,
        eddington_ratio=eddington_ratio,
        f_col=f_col,
        delta_eta=0.0,
        energy_bins=energy_bins,
        settings=KerrThinDiskSettings(
            radial_grid_count=radial_grid_count,
            disk_outer_radius_rg=disk_outer_radius_rg,
            mass_msun=mass_msun,
            distance_kpc=distance_kpc,
        ),
        limb_darkening="isotropic",
    )

    analytic_norm = float(np.sum(analytic))
    relative_l1 = float(np.sum(np.abs(ray_traced - analytic)) / analytic_norm)
    total_ratio = float(np.sum(ray_traced) / analytic_norm)
    status = "PASS" if relative_l1 <= tolerance else "FAIL"
    return FaceOnValidationRow(
        inclination_deg=inclination_deg,
        screen_size=screen_size,
        disk_hits=int(transfer_map.emission_radius.size),
        relative_l1_spectrum_delta=relative_l1,
        total_flux_ratio=total_ratio,
        tolerance=tolerance,
        status=status,
    )
