"""Tests for Phase 12 confirmatory-campaign utilities."""

import csv
from pathlib import Path

import numpy as np
import pytest

from kerrdisk.confirmatory import (
    ConfirmatoryConfig,
    confirmatory_config_from_mapping,
    load_locked_conditions,
    run_confirmatory_campaign,
    validate_confirmatory_config,
    write_confirmatory_outputs,
)


def _locked_condition_file(tmp_path: Path) -> Path:
    path = tmp_path / "locked.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "condition_id",
                "spin_true",
                "inclination_deg",
                "eddington_ratio",
                "f_col_true",
                "f_col_fit",
                "inner_stress_delta_eta",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition_id": "condition-a",
                "spin_true": "0.0",
                "inclination_deg": "40.0",
                "eddington_ratio": "0.1",
                "f_col_true": "1.7",
                "f_col_fit": "1.7",
                "inner_stress_delta_eta": "0.0",
            }
        )
        writer.writerow(
            {
                "condition_id": "condition-b",
                "spin_true": "0.5",
                "inclination_deg": "70.0",
                "eddington_ratio": "0.18",
                "f_col_true": "1.9",
                "f_col_fit": "1.7",
                "inner_stress_delta_eta": "0.02",
            }
        )
    return path


def _protocol_file(tmp_path: Path) -> Path:
    path = tmp_path / "protocol.md"
    path.write_text("# Frozen protocol\n", encoding="utf-8")
    return path


def _small_config(tmp_path: Path) -> ConfirmatoryConfig:
    return ConfirmatoryConfig(
        config_version="test",
        locked_conditions_path=_locked_condition_file(tmp_path),
        frozen_protocol_path=_protocol_file(tmp_path),
        master_seed=42,
        replicate_count=4,
        energy_min_kev=0.1,
        energy_max_kev=20.0,
        energy_bin_count=12,
        spin_grid_count=101,
        resolution_spin_grid_count=151,
        gaussian_relative_error=0.03,
        bias_stability_abs=0.02,
        model_backend="proxy",
        radial_grid_count=24,
        disk_outer_radius_rg=200.0,
        mass_msun=10.0,
        distance_kpc=8.0,
        ray_screen_alpha_min=-8.0,
        ray_screen_alpha_max=8.0,
        ray_screen_beta_min=35.0,
        ray_screen_beta_max=65.0,
        ray_screen_size=3,
        ray_observer_radius=100.0,
        ray_disk_outer_radius=80.0,
        ray_step_size=0.1,
        ray_max_steps=3000,
        ray_escape_radius=160.0,
        limb_darkening="isotropic",
    )


def test_locked_conditions_get_stable_blind_ids(tmp_path: Path) -> None:
    path = _locked_condition_file(tmp_path)

    first = load_locked_conditions(path, master_seed=1)
    second = load_locked_conditions(path, master_seed=1)
    different = load_locked_conditions(path, master_seed=2)

    assert [condition.blind_id for condition in first] == [
        condition.blind_id for condition in second
    ]
    assert [condition.blind_id for condition in first] != [
        condition.blind_id for condition in different
    ]
    assert first[0].condition_id == "condition-a"


def test_confirmatory_campaign_accounts_for_all_replicates(tmp_path: Path) -> None:
    config = _small_config(tmp_path)

    result = run_confirmatory_campaign(config)

    assert len(result.conditions) == 2
    assert len(result.base_replicates) == 8
    assert len(result.high_resolution_replicates) == 8
    assert len(result.summaries) == 2
    assert all(summary.planned_replicates == 4 for summary in result.summaries)
    assert all(summary.status == "COMPLETED" for summary in result.summaries)
    assert all(np.isfinite(summary.mean_bias) for summary in result.summaries)


def test_resolution_summaries_compare_biases(tmp_path: Path) -> None:
    result = run_confirmatory_campaign(_small_config(tmp_path))

    assert len(result.resolution_summaries) == len(result.conditions)
    assert all(
        summary.abs_bias_difference
        == pytest.approx(
            abs(summary.base_mean_bias - summary.high_resolution_mean_bias)
        )
        for summary in result.resolution_summaries
    )


def test_write_confirmatory_outputs(tmp_path: Path) -> None:
    result = run_confirmatory_campaign(_small_config(tmp_path))
    output_dir = tmp_path / "outputs"

    paths = write_confirmatory_outputs(result, output_dir)

    assert set(paths) == {
        "blinded_conditions",
        "hidden_truths",
        "analysis_freeze",
        "replicates_blinded",
        "results_unblinded",
        "resolution_reruns",
        "failure_summary",
    }
    for path in paths.values():
        assert path.exists()
    hidden_text = paths["hidden_truths"].read_text(encoding="utf-8")
    blinded_text = paths["blinded_conditions"].read_text(encoding="utf-8")
    assert "spin_true" in hidden_text
    assert "spin_true" not in blinded_text
    assert "frozen before hidden truths" in paths["analysis_freeze"].read_text(
        encoding="utf-8"
    )


def test_confirmatory_config_from_mapping(tmp_path: Path) -> None:
    locked = _locked_condition_file(tmp_path)
    protocol = _protocol_file(tmp_path)

    config = confirmatory_config_from_mapping(
        {
            "config_version": "yaml",
            "locked_conditions_path": str(locked),
            "frozen_protocol_path": str(protocol),
            "replicate_count": 2,
            "spin_grid_count": 21,
            "resolution_spin_grid_count": 31,
            "model_backend": "proxy",
        }
    )

    assert config.config_version == "yaml"
    assert config.replicate_count == 2


def test_confirmatory_config_rejects_invalid_values(tmp_path: Path) -> None:
    config = _small_config(tmp_path)

    with pytest.raises(ValueError, match="replicate_count"):
        validate_confirmatory_config(
            config.__class__(**{**config.__dict__, "replicate_count": 0})
        )
    with pytest.raises(ValueError, match="resolution_spin_grid_count"):
        validate_confirmatory_config(
            config.__class__(
                **{
                    **config.__dict__,
                    "spin_grid_count": 101,
                    "resolution_spin_grid_count": 101,
                }
            )
        )
    with pytest.raises(ValueError, match="gaussian_relative_error"):
        confirmatory_config_from_mapping({"gaussian_relative_error": 0.0})
    with pytest.raises(ValueError, match="model_backend"):
        confirmatory_config_from_mapping({"model_backend": "bad"})
    with pytest.raises(ValueError, match="limb_darkening"):
        confirmatory_config_from_mapping({"limb_darkening": "bad"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ray_screen_alpha_max", -9.0, "alpha bounds"),
        ("ray_screen_beta_max", 34.0, "beta bounds"),
        ("ray_screen_size", 1, "ray_screen_size"),
        ("ray_observer_radius", 0.0, "ray_observer_radius"),
        ("ray_disk_outer_radius", 0.0, "ray_disk_outer_radius"),
        ("ray_step_size", 0.0, "ray_step_size"),
        ("ray_max_steps", 0, "ray_max_steps"),
        ("ray_escape_radius", 0.0, "ray_escape_radius"),
    ],
)
def test_confirmatory_config_rejects_invalid_ray_settings(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    config = _small_config(tmp_path)

    with pytest.raises(ValueError, match=message):
        validate_confirmatory_config(
            config.__class__(**{**config.__dict__, field: value})
        )


def test_confirmatory_kerr_thin_disk_backend_runs(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    kerr_config = config.__class__(
        **{
            **config.__dict__,
            "model_backend": "kerr_thin_disk",
            "replicate_count": 1,
            "spin_grid_count": 21,
            "resolution_spin_grid_count": 31,
            "energy_bin_count": 10,
            "radial_grid_count": 16,
        }
    )

    result = run_confirmatory_campaign(kerr_config)

    assert len(result.summaries) == 2
    assert all(summary.status == "COMPLETED" for summary in result.summaries)


def test_confirmatory_ray_traced_transfer_backend_runs(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    ray_config = config.__class__(
        **{
            **config.__dict__,
            "model_backend": "ray_traced_transfer",
            "replicate_count": 1,
            "spin_grid_count": 11,
            "resolution_spin_grid_count": 13,
            "energy_bin_count": 6,
            "radial_grid_count": 16,
            "disk_outer_radius_rg": 80.0,
            "ray_screen_size": 2,
        }
    )

    result = run_confirmatory_campaign(ray_config)

    assert len(result.summaries) == 2
    assert all(summary.status == "COMPLETED" for summary in result.summaries)


def test_confirmatory_requires_frozen_protocol(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    missing = config.__class__(
        **{**config.__dict__, "frozen_protocol_path": tmp_path / "missing.md"}
    )

    with pytest.raises(FileNotFoundError, match="protocol"):
        run_confirmatory_campaign(missing)
