"""Kerr thin-disk thermal spectra for confirmatory campaigns."""

from dataclasses import dataclass
from math import cos, isclose, isfinite, pi, radians
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kerrdisk.disk_flux import (
    page_thorne_flux_profile,
    stressed_page_thorne_flux_profile,
)
from kerrdisk.isco import isco_radius
from kerrdisk.metric import _validate_spin
from kerrdisk.spectrum import TransferMap
from kerrdisk.synthetic import EnergyBins

type FloatArray = NDArray[np.float64]
type LimbDarkeningMode = Literal["isotropic", "electron_scattering"]


@dataclass(frozen=True)
class KerrThinDiskSettings:
    """Numerical settings for the Phase 12 Kerr thin-disk spectrum backend."""

    radial_grid_count: int = 180
    disk_outer_radius_rg: float = 1_000.0
    temperature_scale_kev: float = 20.0


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
    """Return detector-independent bin energy flux for a Kerr thin disk."""

    _validate_inputs(
        a_star=a_star,
        inclination_deg=inclination_deg,
        eddington_ratio=eddington_ratio,
        f_col=f_col,
        delta_eta=delta_eta,
        settings=settings,
    )
    inner_radius = isco_radius(a_star)
    if settings.disk_outer_radius_rg <= inner_radius:
        msg = "disk_outer_radius_rg must exceed the ISCO"
        raise ValueError(msg)
    radii, flux = _disk_flux_on_grid(
        a_star=a_star,
        eddington_ratio=eddington_ratio,
        delta_eta=delta_eta,
        settings=settings,
    )
    positive_flux = np.maximum(flux, 0.0)
    temperature_kev = settings.temperature_scale_kev * np.power(positive_flux, 0.25)
    spectral_flux = _integrate_disk_spectrum(
        energy_bins.centers_kev,
        radii,
        temperature_kev,
        f_col,
        inclination_deg,
    )
    return spectral_flux * energy_bins.widths_kev


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
    """Return bin energy flux by integrating over image-plane transfer records."""

    _validate_inputs(
        a_star=a_star,
        inclination_deg=0.0,
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
    if settings.disk_outer_radius_rg <= isco_radius(a_star):
        msg = "disk_outer_radius_rg must exceed the ISCO"
        raise ValueError(msg)
    _validate_limb_darkening(limb_darkening)
    _validate_transfer_map_records(transfer_map)

    radii, flux = _disk_flux_on_grid(
        a_star=a_star,
        eddington_ratio=eddington_ratio,
        delta_eta=delta_eta,
        settings=settings,
    )
    positive_flux = np.maximum(flux, 0.0)
    temperature_grid = settings.temperature_scale_kev * np.power(positive_flux, 0.25)
    hit_temperature = np.interp(
        transfer_map.emission_radius,
        radii,
        temperature_grid,
        left=0.0,
        right=0.0,
    )

    energy = energy_bins.centers_kev
    spectral_flux = np.zeros_like(energy)
    for redshift, solid_angle, emission_mu, temperature in zip(
        transfer_map.redshift,
        transfer_map.solid_angle,
        transfer_map.emission_mu,
        hit_temperature,
        strict=True,
    ):
        if temperature <= 0.0:
            continue
        emitted_energy = energy / redshift
        intensity = _diluted_blackbody_shape_kev(
            emitted_energy,
            temperature_kev=float(temperature),
            f_col=f_col,
        )
        angular_factor = _limb_darkening_factor(
            float(emission_mu),
            limb_darkening,
        )
        spectral_flux += redshift**3 * angular_factor * intensity * solid_angle

    if not np.all(np.isfinite(spectral_flux)) or np.any(spectral_flux < 0.0):
        msg = "computed ray-traced disk spectrum is not finite and nonnegative"
        raise FloatingPointError(msg)
    if not np.any(spectral_flux > 0.0):
        msg = "computed ray-traced disk spectrum is identically zero"
        raise FloatingPointError(msg)
    return spectral_flux * energy_bins.widths_kev


def _disk_flux_on_grid(
    *,
    a_star: float,
    eddington_ratio: float,
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
            mass_accretion_rate=eddington_ratio,
        ).flux
    else:
        flux = stressed_page_thorne_flux_profile(
            a_star,
            radii,
            delta_eta=delta_eta,
            mass_accretion_rate=eddington_ratio,
        ).flux
    return radii, np.asarray(flux, dtype=np.float64)


def _validate_inputs(
    *,
    a_star: float,
    inclination_deg: float,
    eddington_ratio: float,
    f_col: float,
    delta_eta: float,
    settings: KerrThinDiskSettings,
) -> None:
    _validate_spin(a_star)
    if not isfinite(inclination_deg) or not 0.0 <= inclination_deg < 90.0:
        msg = "inclination_deg must be finite and satisfy 0 <= inclination < 90"
        raise ValueError(msg)
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
    if (
        not isfinite(settings.disk_outer_radius_rg)
        or settings.disk_outer_radius_rg <= 0
    ):
        msg = "disk_outer_radius_rg must be finite and positive"
        raise ValueError(msg)
    if (
        not isfinite(settings.temperature_scale_kev)
        or settings.temperature_scale_kev <= 0
    ):
        msg = "temperature_scale_kev must be finite and positive"
        raise ValueError(msg)


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


def _diluted_blackbody_shape_kev(
    energy_kev: FloatArray,
    *,
    temperature_kev: float,
    f_col: float,
) -> FloatArray:
    exponent = energy_kev / (f_col * temperature_kev)
    local = np.zeros_like(energy_kev)
    finite_mask = exponent <= 700.0
    local[finite_mask] = (
        energy_kev[finite_mask] ** 3 / np.expm1(exponent[finite_mask]) / f_col**4
    )
    return local


def _integrate_disk_spectrum(
    energy_kev: FloatArray,
    radii: FloatArray,
    temperature_kev: FloatArray,
    f_col: float,
    inclination_deg: float,
) -> FloatArray:
    projected_area = cos(radians(inclination_deg))
    radial_width = np.gradient(radii)
    annulus_weight = 2.0 * pi * radii * radial_width * projected_area
    intensity = np.zeros((radii.size, energy_kev.size), dtype=np.float64)
    positive_temperature = temperature_kev > 0.0
    if not np.any(positive_temperature):
        msg = "disk spectrum has no positive-temperature annuli"
        raise FloatingPointError(msg)
    temperature = temperature_kev[positive_temperature, None]
    energy = energy_kev[None, :]
    energy_grid = np.broadcast_to(energy, (temperature.shape[0], energy_kev.size))
    exponent = energy / (f_col * temperature)
    finite_mask = exponent <= 700.0
    local = np.zeros_like(exponent)
    local[finite_mask] = (
        energy_grid[finite_mask] ** 3 / np.expm1(exponent[finite_mask]) / f_col**4
    )
    intensity[positive_temperature, :] = local
    spectrum = annulus_weight @ intensity
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        msg = "computed disk spectrum is not finite and nonnegative"
        raise FloatingPointError(msg)
    return np.asarray(spectrum, dtype=np.float64)
