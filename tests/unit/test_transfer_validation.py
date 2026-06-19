"""Tests for Phase 12.5 transfer-validation helpers."""

from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

from kerrdisk import transfer_validation
from kerrdisk.radiation_diagnostics import EmissionOutcomeSummary
from kerrdisk.spectrum import TransferMap


def _transfer_map(screen_size: int = 2) -> TransferMap:
    samples = screen_size * screen_size
    return TransferMap(
        a_star=0.0,
        observer_radius=100.0,
        observer_theta=1.0,
        observer_distance=100.0,
        alpha=np.repeat(np.arange(screen_size, dtype=np.float64), screen_size),
        beta=np.tile(np.arange(screen_size, dtype=np.float64), screen_size),
        solid_angle=np.full(samples, 1.0e-4, dtype=np.float64),
        emission_radius=np.full(samples, 10.0, dtype=np.float64),
        emission_phi=np.zeros(samples, dtype=np.float64),
        redshift=np.full(samples, 0.9, dtype=np.float64),
        emission_mu=np.full(samples, 0.8, dtype=np.float64),
        max_abs_hamiltonian=np.zeros(samples, dtype=np.float64),
        steps=np.ones(samples, dtype=np.int64),
        outcome_counts={"DISK_HIT": samples},
    )


def test_transfer_convergence_runs_with_stubbed_maps(monkeypatch: MonkeyPatch) -> None:
    def fake_map(
        a_star: float, inclination_deg: float, screen_size: int
    ) -> TransferMap:
        _ = (a_star, inclination_deg)
        return _transfer_map(screen_size)

    def fake_spectrum(**kwargs: object) -> np.ndarray:
        transfer_map = kwargs["transfer_map"]
        assert isinstance(transfer_map, TransferMap)
        scale = 1.0 + (1.0 / transfer_map.alpha.size)
        return np.full(4, scale, dtype=np.float64)

    monkeypatch.setattr(transfer_validation, "_validation_transfer_map", fake_map)
    monkeypatch.setattr(
        transfer_validation,
        "ray_traced_kerr_thin_disk_energy_flux",
        fake_spectrum,
    )

    rows = transfer_validation.run_transfer_convergence(screen_sizes=(2, 3))

    assert len(rows) == 4
    assert rows[-1].relative_l1_spectrum_delta == 0.0


def test_external_transfer_comparison_skips_missing() -> None:
    rows = transfer_validation.run_external_transfer_comparison(None)

    assert rows[0].status == "SKIPPED"
    assert rows[0].matched_records == 0


def test_external_transfer_comparison_passes_with_matching_csv(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    path = tmp_path / "external.csv"
    path.write_text(
        "alpha,beta,emission_radius,redshift,emission_mu\n0.0,0.0,10.0,0.9,0.8\n",
        encoding="utf-8",
    )

    def fake_build_transfer_map(*_args: object, **_kwargs: object) -> TransferMap:
        return _transfer_map(1)

    monkeypatch.setattr(
        transfer_validation,
        "build_transfer_map",
        fake_build_transfer_map,
    )

    rows = transfer_validation.run_external_transfer_comparison(path)

    assert rows[0].status == "PASS"
    assert rows[0].matched_records == 1


def test_capture_returning_diagnostics_runs_with_stubbed_sampler(
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_sample(*args: object, **kwargs: object) -> EmissionOutcomeSummary:
        _ = kwargs
        return EmissionOutcomeSummary(
            a_star=float(args[0]),
            emission_radius=float(args[1]),
            sample_count=4,
            escaped=2,
            captured=1,
            returning=1,
            max_steps=0,
            numerical_failure=0,
        )

    monkeypatch.setattr(transfer_validation, "sample_emission_outcomes", fake_sample)

    rows = transfer_validation.run_capture_returning_diagnostics()

    assert len(rows) == 4
    assert rows[0]["captured_fraction"] == 0.25


def test_write_rows_writes_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"

    transfer_validation.write_rows(path, [{"a": 1, "b": 2}])

    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2"]
