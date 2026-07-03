"""Physically normalized Kerr thin-disk thermal spectra.

The spectrum is anchored to a physical system: a black-hole mass sets the
gravitational radius and the accretion rate through the Eddington ratio, the
Page-Thorne surface flux sets the effective-temperature profile in kelvin, and
the distance sets the observed normalization. The observable is the binned
energy flux in ``erg s^-1 cm^-2`` per bin.
"""

from dataclasses import dataclass
from math import cos, isclose, isfinite, pi, radians
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kerrdisk.atmosphere import (
    diluted_blackbody_nu,
    effective_temperature_from_flux,
)
from kerrdisk.circular_orbits import nt_efficiency
from kerrdisk.constants import (
    KILO_ELECTRON_VOLT_J,
    PLANCK_CONSTANT_J_S,
    WATT_PER_M2_TO_ERG_PER_S_CM2,
)
from kerrdisk.disk_flux import (
    flux_si_from_dimensionless,
    page_thorne_flux_profile,
    stressed_page_thorne_flux_profile,
)
from kerrdisk.isco import isco_radius
from kerrdisk.metric import _validate_spin
from kerrdisk.scales import (
    accretion_rate_kg_s,
    black_hole_mass_kg,
    distance_m,
    gravitational_radius_m,
)
from kerrdisk.spectrum import TransferMap
from kerrdisk.synthetic import EnergyBins

type FloatArray = NDArray[np.float64]
type LimbDarkeningMode = Literal["isotropic", "electron_scattering"]

FIDUCIAL_MASS_MSUN = 10.0
FIDUCIAL_DISTANCE_KPC = 8.0


@dataclass(frozen=True)
class KerrThinDiskSettings:
    """Physical system and numerical settings for the thin-disk spectrum."""

    radial_grid_count: int = 180
    disk_outer_radius_rg: float = 1_000.0
    mass_msun: float = FIDUCIAL_MASS_MSUN
    distance_kpc: float = FIDUCIAL_DISTANCE_KPC


DEFAULT_KERR_THIN_DISK_SETTINGS = KerrThinDiskSettings()


def kerr_thin_disk_energy_flux(
    *,
    a_star: float,
    inclination_deg: float,
    eddington_ratio: float,
    f_col: float,
    delta_eta: float,
    energy_bins: EnergyBins,
    settings: KerrThinDiskSettings = DEFAULT_KERR_THIN_DISK_SETTINGS,
) -> FloatArray:
    """Return binned energy flux for an axisymmetric no-transfer thin disk.

    This backend applies inclination projection and the physical distance
    normalization but no gravitational redshift or lensing. It is a cheap
    comparison model, not the production transfer calculation.
    """

    _validate_system(
        a_star=a_star,
        eddington_ratio=eddington_ratio,
        f_col=f_col,
        delta_eta=delta_eta,
        settings=settings,
    )
    if not isfinite(inclination_deg) or not 0.0 <= inclination_deg < 90.0:
        msg = "inclination_deg must be finite and satisfy 0 <= inclination < 90"
        raise ValueError(msg)

    radii, temperature_k = _effective_temperature_profile(
        a_star=a_star,
        eddington_ratio=eddington_ratio,
        delta_eta=delta_eta,
        settings=settings,
    )
    r_g = gravitational_radius_m(black_hole_mass_kg(settings.mass_msun))
    observer_distance = distance_m(settings.distance_kpc)
    projected = cos(radians(inclination_deg))
    radial_width = np.gradient(radii)
    annulus_area = 2.0 * pi * radii * radial_width * (r_g * r_g) * projected

    frequency = energy_bins.centers_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S
    frequency_width = (
        energy_bins.widths_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S
    )
    flux_nu = np.zeros_like(frequency)
    for temperature, area in zip(temperature_k, annulus_area, strict=True):
        if temperature <= 0.0 or area <= 0.0:
            continue
        intensity = diluted_blackbody_nu(frequency, float(temperature), f_col)
        flux_nu += intensity * area / (observer_distance * observer_distance)

    bin_energy_flux = flux_nu * frequency_width * WATT_PER_M2_TO_ERG_PER_S_CM2
    return _finalize_spectrum(bin_energy_flux)


def ray_traced_kerr_thin_disk_energy_flux(
    *,
    transfer_map: TransferMap,
    a_star: float,
    eddington_ratio: float,
    f_col: float,
    delta_eta: float,
    energy_bins: EnergyBins,
    settings: KerrThinDiskSettings = DEFAULT_KERR_THIN_DISK_SETTINGS,
    limb_darkening: LimbDarkeningMode = "isotropic",
) -> FloatArray:
    """Return binned energy flux by integrating image-plane transfer records.

    The transfer map must be built with ``observer_distance`` equal to the
    astronomical distance in gravitational radii (see
    ``scales.observer_distance_rg``) so its stored solid angle is the physical
    observer solid angle in steradians.
    """

    _validate_system(
        a_star=a_star,
        eddington_ratio=eddington_ratio,
        f_col=f_col,
        delta_eta=delta_eta,
        settings=settings,
    )
    if not isclose(transfer_map.a_star, a_star, rel_tol=0.0, abs_tol=1.0e-12):
        msg = "transfer_map spin must match a_star"
        raise ValueError(msg)
    if transfer_map.emission_radius.size == 0:
        msg = "transfer_map contains no disk-hit records"
        raise ValueError(msg)
    _validate_limb_darkening(limb_darkening)
    _validate_transfer_map_records(transfer_map)

    radii, temperature_k = _effective_temperature_profile(
        a_star=a_star,
        eddington_ratio=eddington_ratio,
        delta_eta=delta_eta,
        settings=settings,
    )
    hit_temperature = np.interp(
        transfer_map.emission_radius,
        radii,
        temperature_k,
        left=0.0,
        right=0.0,
    )

    frequency = energy_bins.centers_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S
    frequency_width = (
        energy_bins.widths_kev * KILO_ELECTRON_VOLT_J / PLANCK_CONSTANT_J_S
    )
    flux_nu = np.zeros_like(frequency)
    for redshift, solid_angle, emission_mu, temperature in zip(
        transfer_map.redshift,
        transfer_map.solid_angle,
        transfer_map.emission_mu,
        hit_temperature,
        strict=True,
    ):
        if temperature <= 0.0:
            continue
        emitted_frequency = frequency / redshift
        intensity = diluted_blackbody_nu(
            emitted_frequency,
            float(temperature),
            f_col,
        )
        angular_factor = _limb_darkening_factor(float(emission_mu), limb_darkening)
        flux_nu += redshift**3 * angular_factor * intensity * solid_angle

    bin_energy_flux = flux_nu * frequency_width * WATT_PER_M2_TO_ERG_PER_S_CM2
    return _finalize_spectrum(bin_energy_flux, require_positive=True)


def _effective_temperature_profile(
    *,
    a_star: float,
    eddington_ratio: float,
    delta_eta: float,
    settings: KerrThinDiskSettings,
) -> tuple[FloatArray, FloatArray]:
    radii, dimensionless_flux = _disk_flux_on_grid(
        a_star=a_star,
        delta_eta=delta_eta,
        settings=settings,
    )
    mass_kg = black_hole_mass_kg(settings.mass_msun)
    efficiency = nt_efficiency(a_star) + delta_eta
    mass_accretion_rate = accretion_rate_kg_s(mass_kg, eddington_ratio, efficiency)
    flux_si = flux_si_from_dimensionless(
        np.maximum(dimensionless_flux, 0.0),
        black_hole_mass_kg=mass_kg,
        mass_accretion_rate_kg_s=mass_accretion_rate,
    )
    temperature_k = effective_temperature_from_flux(flux_si)
    return radii, temperature_k


def _disk_flux_on_grid(
    *,
    a_star: float,
    delta_eta: float,
    settings: KerrThinDiskSettings,
) -> tuple[FloatArray, FloatArray]:
    inner_radius = isco_radius(a_star)
    radii = np.geomspace(
        inner_radius,
        settings.disk_outer_radius_rg,
        settings.radial_grid_count,
        dtype=np.float64,
    )
    if delta_eta == 0.0:
        flux = page_thorne_flux_profile(
            a_star,
            radii,
            mass_accretion_rate=1.0,
        ).flux
    else:
        flux = stressed_page_thorne_flux_profile(
            a_star,
            radii,
            delta_eta=delta_eta,
            mass_accretion_rate=1.0,
        ).flux
    return radii, np.asarray(flux, dtype=np.float64)


def _validate_system(
    *,
    a_star: float,
    eddington_ratio: float,
    f_col: float,
    delta_eta: float,
    settings: KerrThinDiskSettings,
) -> None:
    _validate_spin(a_star)
    if not isfinite(eddington_ratio) or eddington_ratio <= 0.0:
        msg = "eddington_ratio must be finite and positive"
        raise ValueError(msg)
    if not isfinite(f_col) or f_col <= 0.0:
        msg = "f_col must be finite and positive"
        raise ValueError(msg)
    if not isfinite(delta_eta) or delta_eta < 0.0:
        msg = "delta_eta must be finite and nonnegative"
        raise ValueError(msg)
    if settings.radial_grid_count < 8:
        msg = "radial_grid_count must be at least eight"
        raise ValueError(msg)
    if not isfinite(
        settings.disk_outer_radius_rg
    ) or settings.disk_outer_radius_rg <= isco_radius(a_star):
        msg = "disk_outer_radius_rg must be finite and exceed the ISCO"
        raise ValueError(msg)
    if not isfinite(settings.mass_msun) or settings.mass_msun <= 0.0:
        msg = "mass_msun must be finite and positive"
        raise ValueError(msg)
    if not isfinite(settings.distance_kpc) or settings.distance_kpc <= 0.0:
        msg = "distance_kpc must be finite and positive"
        raise ValueError(msg)


def _finalize_spectrum(
    bin_energy_flux: FloatArray,
    *,
    require_positive: bool = False,
) -> FloatArray:
    spectrum = np.asarray(bin_energy_flux, dtype=np.float64)
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        msg = "computed disk spectrum is not finite and nonnegative"
        raise FloatingPointError(msg)
    if require_positive and not np.any(spectrum > 0.0):
        msg = "computed ray-traced disk spectrum is identically zero"
        raise FloatingPointError(msg)
    return spectrum


def _validate_transfer_map_records(transfer_map: TransferMap) -> None:
    fields = (
        transfer_map.emission_radius,
        transfer_map.redshift,
        transfer_map.solid_angle,
        transfer_map.emission_mu,
    )
    size = transfer_map.emission_radius.size
    if any(field.shape != (size,) for field in fields):
        msg = "transfer_map record arrays must have matching one-dimensional shape"
        raise ValueError(msg)
    if not all(np.all(np.isfinite(field)) for field in fields):
        msg = "transfer_map record arrays must contain finite values"
        raise ValueError(msg)
    if np.any(transfer_map.redshift <= 0.0):
        msg = "transfer_map redshift values must be positive"
        raise ValueError(msg)
    if np.any(transfer_map.solid_angle <= 0.0):
        msg = "transfer_map solid_angle values must be positive"
        raise ValueError(msg)
    if np.any((transfer_map.emission_mu < 0.0) | (transfer_map.emission_mu > 1.0)):
        msg = "transfer_map emission_mu values must satisfy 0 <= mu <= 1"
        raise ValueError(msg)


def _validate_limb_darkening(limb_darkening: LimbDarkeningMode) -> None:
    if limb_darkening not in {"isotropic", "electron_scattering"}:
        msg = "limb_darkening must be 'isotropic' or 'electron_scattering'"
        raise ValueError(msg)


def _limb_darkening_factor(
    emission_mu: float,
    limb_darkening: LimbDarkeningMode,
) -> float:
    if limb_darkening == "isotropic":
        return 1.0
    return 0.5 + (0.75 * emission_mu)
