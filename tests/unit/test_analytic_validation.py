"""Tests for the independent analytic disk-spectrum cross-validation."""

import numpy as np
import pytest

from kerrdisk.analytic_validation import (
    run_face_on_cross_validation,
    schwarzschild_face_on_disk_spectrum,
)
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import KerrThinDiskSettings, kerr_thin_disk_energy_flux

_BINS = make_log_energy_bins(bin_count=16)


def test_face_on_spectrum_is_finite_positive() -> None:
    spectrum = schwarzschild_face_on_disk_spectrum(
        eddington_ratio=0.1,
        f_col=1.7,
        energy_bins=_BINS,
        radial_grid_count=200,
    )

    assert spectrum.shape == _BINS.centers_kev.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0.0)
    assert np.any(spectrum > 0.0)


def test_face_on_spectrum_scales_as_inverse_distance_squared() -> None:
    def flux(distance_kpc: float) -> np.ndarray:
        return schwarzschild_face_on_disk_spectrum(
            eddington_ratio=0.1,
            f_col=1.7,
            energy_bins=_BINS,
            distance_kpc=distance_kpc,
            radial_grid_count=200,
        )

    assert np.sum(flux(16.0)) == pytest.approx(0.25 * np.sum(flux(8.0)), rel=1.0e-9)


def test_face_on_redshift_reduces_flux_below_no_redshift_model() -> None:
    analytic = schwarzschild_face_on_disk_spectrum(
        eddington_ratio=0.1,
        f_col=1.7,
        energy_bins=_BINS,
        radial_grid_count=200,
        disk_outer_radius_rg=80.0,
    )
    no_redshift = kerr_thin_disk_energy_flux(
        a_star=0.0,
        inclination_deg=0.0,
        eddington_ratio=0.1,
        f_col=1.7,
        delta_eta=0.0,
        energy_bins=_BINS,
        settings=KerrThinDiskSettings(radial_grid_count=200, disk_outer_radius_rg=80.0),
    )

    # The face-on redshift g < 1 dims the observed flux relative to a
    # no-redshift model at the same inclination.
    assert np.sum(analytic) < np.sum(no_redshift)


def test_face_on_spectrum_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="eddington_ratio"):
        schwarzschild_face_on_disk_spectrum(
            eddington_ratio=0.0, f_col=1.7, energy_bins=_BINS
        )
    with pytest.raises(ValueError, match="disk_outer_radius_rg"):
        schwarzschild_face_on_disk_spectrum(
            eddington_ratio=0.1,
            f_col=1.7,
            energy_bins=_BINS,
            disk_outer_radius_rg=5.0,
        )


def test_ray_traced_spectrum_matches_analytic_face_on_benchmark() -> None:
    row = run_face_on_cross_validation(
        inclination_deg=4.0,
        screen_size=18,
        disk_outer_radius_rg=40.0,
        observer_radius=400.0,
        radial_grid_count=200,
        energy_bin_count=16,
        tolerance=0.15,
    )

    assert row.disk_hits > 0
    assert row.status == "PASS"
    assert row.relative_l1_spectrum_delta < 0.15
    # Absolute normalization agrees to within the same band.
    assert row.total_flux_ratio == pytest.approx(1.0, abs=0.15)
