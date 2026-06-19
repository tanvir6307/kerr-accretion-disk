"""Tests for Kerr thin-disk thermal spectra."""

import numpy as np
import pytest

from kerrdisk.spectrum import TransferMap
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import (
    KerrThinDiskSettings,
    kerr_thin_disk_energy_flux,
    ray_traced_kerr_thin_disk_energy_flux,
)


def _settings() -> KerrThinDiskSettings:
    return KerrThinDiskSettings(
        radial_grid_count=48,
        disk_outer_radius_rg=500.0,
        temperature_scale_kev=20.0,
    )


def _transfer_map(a_star: float = 0.0) -> TransferMap:
    return TransferMap(
        a_star=a_star,
        observer_radius=100.0,
        observer_theta=np.pi / 4.0,
        observer_distance=100.0,
        alpha=np.array([-1.0, 1.0], dtype=np.float64),
        beta=np.array([40.0, 45.0], dtype=np.float64),
        solid_angle=np.array([1.0e-4, 1.0e-4], dtype=np.float64),
        emission_radius=np.array([8.0, 12.0], dtype=np.float64),
        emission_phi=np.array([0.0, 0.1], dtype=np.float64),
        redshift=np.array([0.8, 0.9], dtype=np.float64),
        emission_mu=np.array([1.0, 1.0], dtype=np.float64),
        max_abs_hamiltonian=np.array([1.0e-7, 1.0e-7], dtype=np.float64),
        steps=np.array([100, 110], dtype=np.int64),
        outcome_counts={"DISK_HIT": 2},
    )


def test_kerr_thin_disk_spectrum_is_finite_positive() -> None:
    bins = make_log_energy_bins(bin_count=16)

    spectrum = kerr_thin_disk_energy_flux(
        a_star=0.5,
        inclination_deg=45.0,
        eddington_ratio=0.1,
        f_col=1.7,
        delta_eta=0.0,
        energy_bins=bins,
        settings=_settings(),
    )

    assert spectrum.shape == bins.centers_kev.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)
    assert np.any(spectrum > 0.0)


def test_kerr_thin_disk_spectrum_changes_with_spin_and_stress() -> None:
    bins = make_log_energy_bins(bin_count=16)
    base = kerr_thin_disk_energy_flux(
        a_star=0.0,
        inclination_deg=45.0,
        eddington_ratio=0.1,
        f_col=1.7,
        delta_eta=0.0,
        energy_bins=bins,
        settings=_settings(),
    )
    high_spin = kerr_thin_disk_energy_flux(
        a_star=0.8,
        inclination_deg=45.0,
        eddington_ratio=0.1,
        f_col=1.7,
        delta_eta=0.0,
        energy_bins=bins,
        settings=_settings(),
    )
    stressed = kerr_thin_disk_energy_flux(
        a_star=0.0,
        inclination_deg=45.0,
        eddington_ratio=0.1,
        f_col=1.7,
        delta_eta=0.02,
        energy_bins=bins,
        settings=_settings(),
    )

    assert not np.allclose(base, high_spin)
    assert not np.allclose(base, stressed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inclination_deg": 90.0},
        {"eddington_ratio": 0.0},
        {"f_col": 0.0},
        {"delta_eta": -0.01},
    ],
)
def test_kerr_thin_disk_spectrum_rejects_invalid_inputs(
    kwargs: dict[str, float],
) -> None:
    bins = make_log_energy_bins(bin_count=8)
    params = {
        "a_star": 0.0,
        "inclination_deg": 45.0,
        "eddington_ratio": 0.1,
        "f_col": 1.7,
        "delta_eta": 0.0,
        "energy_bins": bins,
        "settings": _settings(),
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        kerr_thin_disk_energy_flux(**params)


def test_kerr_thin_disk_spectrum_rejects_invalid_settings() -> None:
    bins = make_log_energy_bins(bin_count=8)

    with pytest.raises(ValueError, match="radial_grid_count"):
        kerr_thin_disk_energy_flux(
            a_star=0.0,
            inclination_deg=45.0,
            eddington_ratio=0.1,
            f_col=1.7,
            delta_eta=0.0,
            energy_bins=bins,
            settings=KerrThinDiskSettings(radial_grid_count=4),
        )


def test_ray_traced_kerr_thin_disk_spectrum_is_finite_positive() -> None:
    bins = make_log_energy_bins(bin_count=12)

    spectrum = ray_traced_kerr_thin_disk_energy_flux(
        transfer_map=_transfer_map(),
        a_star=0.0,
        eddington_ratio=0.1,
        f_col=1.7,
        delta_eta=0.0,
        energy_bins=bins,
        settings=KerrThinDiskSettings(
            radial_grid_count=32,
            disk_outer_radius_rg=50.0,
            temperature_scale_kev=20.0,
        ),
    )

    assert spectrum.shape == bins.centers_kev.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)
    assert np.any(spectrum > 0.0)


def test_ray_traced_kerr_thin_disk_spectrum_uses_limb_darkening() -> None:
    bins = make_log_energy_bins(bin_count=12)
    transfer_map = TransferMap(
        **{
            **_transfer_map().__dict__,
            "emission_mu": np.array([0.1, 1.0], dtype=np.float64),
        }
    )
    params = {
        "transfer_map": transfer_map,
        "a_star": 0.0,
        "eddington_ratio": 0.1,
        "f_col": 1.7,
        "delta_eta": 0.0,
        "energy_bins": bins,
        "settings": KerrThinDiskSettings(
            radial_grid_count=32,
            disk_outer_radius_rg=50.0,
            temperature_scale_kev=20.0,
        ),
    }

    isotropic = ray_traced_kerr_thin_disk_energy_flux(**params)
    limb_darkened = ray_traced_kerr_thin_disk_energy_flux(
        **params,
        limb_darkening="electron_scattering",
    )

    assert not np.allclose(isotropic, limb_darkened)


def test_ray_traced_kerr_thin_disk_spectrum_rejects_bad_transfer_map() -> None:
    bins = make_log_energy_bins(bin_count=8)

    with pytest.raises(ValueError, match="spin"):
        ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=_transfer_map(a_star=0.5),
            a_star=0.0,
            eddington_ratio=0.1,
            f_col=1.7,
            delta_eta=0.0,
            energy_bins=bins,
            settings=_settings(),
        )

    empty = _transfer_map()
    empty = TransferMap(
        **{
            **empty.__dict__,
            "alpha": np.array([], dtype=np.float64),
            "beta": np.array([], dtype=np.float64),
            "solid_angle": np.array([], dtype=np.float64),
            "emission_radius": np.array([], dtype=np.float64),
            "emission_phi": np.array([], dtype=np.float64),
            "redshift": np.array([], dtype=np.float64),
            "emission_mu": np.array([], dtype=np.float64),
            "max_abs_hamiltonian": np.array([], dtype=np.float64),
            "steps": np.array([], dtype=np.int64),
        }
    )
    with pytest.raises(ValueError, match="no disk-hit"):
        ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=empty,
            a_star=0.0,
            eddington_ratio=0.1,
            f_col=1.7,
            delta_eta=0.0,
            energy_bins=bins,
            settings=_settings(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "solid_angle",
            np.array([1.0e-4], dtype=np.float64),
            "matching one-dimensional",
        ),
        (
            "emission_radius",
            np.array([8.0, np.nan], dtype=np.float64),
            "finite",
        ),
        ("redshift", np.array([0.8, 0.0], dtype=np.float64), "redshift"),
        ("solid_angle", np.array([1.0e-4, 0.0], dtype=np.float64), "solid_angle"),
        ("emission_mu", np.array([0.5, 1.1], dtype=np.float64), "emission_mu"),
    ],
)
def test_ray_traced_kerr_thin_disk_spectrum_rejects_invalid_records(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    bins = make_log_energy_bins(bin_count=8)
    transfer_map = _transfer_map()
    bad_map = TransferMap(**{**transfer_map.__dict__, field: value})

    with pytest.raises(ValueError, match=message):
        ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=bad_map,
            a_star=0.0,
            eddington_ratio=0.1,
            f_col=1.7,
            delta_eta=0.0,
            energy_bins=bins,
            settings=_settings(),
        )


def test_ray_traced_kerr_thin_disk_spectrum_rejects_bad_limb_darkening() -> None:
    bins = make_log_energy_bins(bin_count=8)

    with pytest.raises(ValueError, match="limb_darkening"):
        ray_traced_kerr_thin_disk_energy_flux(
            transfer_map=_transfer_map(),
            a_star=0.0,
            eddington_ratio=0.1,
            f_col=1.7,
            delta_eta=0.0,
            energy_bins=bins,
            settings=_settings(),
            limb_darkening="bad",  # type: ignore[arg-type]
        )
