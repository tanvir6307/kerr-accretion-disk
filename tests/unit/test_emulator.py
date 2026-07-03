"""Tests for the precomputed spectral-grid emulator."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from kerrdisk.emulator import build_spectral_grid, load_spectral_grid
from kerrdisk.synthetic import make_log_energy_bins
from kerrdisk.thermal_spectrum import KerrThinDiskSettings, kerr_thin_disk_energy_flux

_BINS = make_log_energy_bins(bin_count=8)
_SETTINGS = KerrThinDiskSettings(radial_grid_count=20, disk_outer_radius_rg=400.0)


def _model(point: Mapping[str, float]) -> np.ndarray:
    return kerr_thin_disk_energy_flux(
        a_star=point["spin"],
        inclination_deg=45.0,
        eddington_ratio=0.1,
        f_col=point["f_col"],
        delta_eta=0.0,
        energy_bins=_BINS,
        settings=_SETTINGS,
    )


def _grid():  # noqa: ANN202
    return build_spectral_grid(
        axes={
            "spin": np.linspace(0.0, 0.9, 5),
            "f_col": np.linspace(1.5, 2.1, 4),
        },
        energy_centers_kev=_BINS.centers_kev,
        model=_model,
    )


def test_grid_reproduces_model_at_nodes() -> None:
    grid = _grid()

    interpolated = grid.evaluate_named(spin=0.45, f_col=1.7)
    direct = _model({"spin": 0.45, "f_col": 1.7})

    assert interpolated.shape == _BINS.centers_kev.shape
    # 0.45 and 1.7 are grid nodes, so interpolation is exact.
    assert np.allclose(interpolated, direct, rtol=1.0e-12, atol=0.0)


def test_grid_interpolates_between_nodes() -> None:
    grid = _grid()

    interpolated = grid.evaluate_named(spin=0.55, f_col=1.85)
    direct = _model({"spin": 0.55, "f_col": 1.85})

    # Off-node multilinear interpolation matches the model to a small error.
    relative = np.max(np.abs(interpolated - direct)) / np.max(direct)
    assert relative < 0.05


def test_grid_save_load_round_trip(tmp_path: Path) -> None:
    grid = _grid()
    path = tmp_path / "grid.npz"

    grid.save(path)
    loaded = load_spectral_grid(path)

    assert loaded.axis_names == grid.axis_names
    point = np.array([0.6, 1.9])
    assert np.allclose(loaded.evaluate(point), grid.evaluate(point))


def test_grid_evaluate_rejects_out_of_bounds() -> None:
    grid = _grid()

    with pytest.raises(ValueError):
        grid.evaluate(np.array([1.5, 1.7]))


def test_grid_evaluate_rejects_wrong_dimension() -> None:
    grid = _grid()

    with pytest.raises(ValueError, match="shape"):
        grid.evaluate(np.array([0.5]))


def test_build_grid_rejects_wrong_model_output() -> None:
    def bad_model(point: Mapping[str, float]) -> np.ndarray:
        _ = point
        return np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="matching energy"):
        build_spectral_grid(
            axes={"spin": np.linspace(0.0, 0.9, 3)},
            energy_centers_kev=_BINS.centers_kev,
            model=bad_model,
        )
