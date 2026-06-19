"""Tests for Phase 11 screening-campaign utilities."""

import csv
from pathlib import Path

import numpy as np
import pytest

from kerrdisk.screening import (
    ReplicateResult,
    ScreeningCondition,
    ScreeningConfig,
    ScreeningThresholds,
    build_screening_conditions,
    load_screening_config,
    run_screening_campaign,
    screening_config_from_mapping,
    summarize_condition,
    write_screening_outputs,
)


def _small_config() -> ScreeningConfig:
    return ScreeningConfig(
        config_version="test",
        master_seed=123,
        spins=(0.0, 0.5),
        inclinations_deg=(40.0,),
        eddington_ratios=(0.1,),
        f_col_true_values=(1.7, 1.9),
        f_col_fit=1.7,
        inner_stress_delta_eta=(0.0,),
        replicate_count=3,
        energy_min_kev=0.1,
        energy_max_kev=20.0,
        energy_bin_count=12,
        spin_grid_count=101,
        gaussian_relative_error=0.03,
        thresholds=ScreeningThresholds(
            high_bias_abs=0.05,
            low_identifiability_width_68=0.25,
            chi2_per_dof_max=2.0,
            failure_rate_max=0.0,
        ),
    )


def test_build_screening_conditions_is_deterministic() -> None:
    config = _small_config()

    first = build_screening_conditions(config)
    second = build_screening_conditions(config)

    assert first == second
    assert len(first) == 4
    assert len({condition.condition_id for condition in first}) == 4


def test_screening_campaign_creates_status_for_every_condition() -> None:
    config = _small_config()

    result = run_screening_campaign(config)

    assert len(result.conditions) == 4
    assert len(result.summaries) == len(result.conditions)
    assert len(result.replicates) == len(result.conditions) * config.replicate_count
    assert all(summary.planned_replicates == 3 for summary in result.summaries)
    assert all(summary.status == "COMPLETED" for summary in result.summaries)


def test_screening_flags_follow_declared_thresholds() -> None:
    config = _small_config()

    result = run_screening_campaign(config)

    mismatched = [
        summary
        for summary in result.summaries
        if summary.f_col_true != summary.f_col_fit
    ]
    matched = [
        summary
        for summary in result.summaries
        if summary.f_col_true == summary.f_col_fit and summary.spin_true == 0.0
    ]

    assert mismatched
    assert any(summary.high_bias_flag for summary in mismatched)
    assert any(summary.needs_refinement for summary in mismatched)
    assert all(not summary.high_bias_flag for summary in matched)


def test_summarize_condition_retains_failed_replicates() -> None:
    condition = ScreeningCondition(
        condition_id="condition",
        spin_true=0.0,
        inclination_deg=40.0,
        eddington_ratio=0.1,
        f_col_true=1.7,
        f_col_fit=1.7,
        inner_stress_delta_eta=0.0,
        replicate_count=2,
    )
    completed = ReplicateResult(
        condition_id="condition",
        replicate_index=0,
        status="COMPLETED",
        failure_cause="",
        spin_true=0.0,
        spin_map=0.01,
        spin_mean=0.02,
        bias=0.02,
        ci68_lower=-0.1,
        ci68_upper=0.1,
        ci95_lower=-0.2,
        ci95_upper=0.2,
        chi2_per_dof=1.0,
        runtime_s=0.01,
        noise_seed=1,
    )
    failed = ReplicateResult(
        condition_id="condition",
        replicate_index=1,
        status="FAILED",
        failure_cause="forced failure",
        spin_true=0.0,
        spin_map=float("nan"),
        spin_mean=float("nan"),
        bias=float("nan"),
        ci68_lower=float("nan"),
        ci68_upper=float("nan"),
        ci95_lower=float("nan"),
        ci95_upper=float("nan"),
        chi2_per_dof=float("nan"),
        runtime_s=0.01,
        noise_seed=2,
    )

    summary = summarize_condition(
        condition,
        [completed, failed],
        ScreeningThresholds(failure_rate_max=0.0),
    )

    assert summary.status == "FAILED"
    assert summary.failed_replicates == 1
    assert summary.failure_rate == pytest.approx(0.5)
    assert summary.needs_refinement
    assert summary.failure_cause == "forced failure"


def test_write_screening_outputs(tmp_path: Path) -> None:
    result = run_screening_campaign(_small_config())

    paths = write_screening_outputs(result, tmp_path)

    assert set(paths) == {
        "conditions",
        "status",
        "replicates",
        "confirmatory_conditions",
        "confirmatory_protocol",
    }
    for path in paths.values():
        assert path.exists()
    with paths["status"].open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(result.conditions)
    assert {"COMPLETED"} == {row["status"] for row in rows}
    assert "frozen before Phase 12" in paths["confirmatory_protocol"].read_text(
        encoding="utf-8"
    )


def test_load_screening_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "screening.yaml"
    path.write_text(
        "\n".join(
            [
                "config_version: yaml-test",
                "master_seed: 7",
                "spins: [0.0]",
                "inclinations_deg: [40.0]",
                "eddington_ratios: [0.1]",
                "f_col_true_values: [1.7]",
                "inner_stress_delta_eta: [0.0]",
                "replicate_count: 2",
            ]
        ),
        encoding="utf-8",
    )

    config = load_screening_config(path)

    assert config.config_version == "yaml-test"
    assert config.master_seed == 7
    assert config.replicate_count == 2


def test_screening_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="replicate_count"):
        screening_config_from_mapping({"replicate_count": 0})
    with pytest.raises(ValueError, match="spin"):
        screening_config_from_mapping({"spins": [1.5]})
    with pytest.raises(TypeError, match="sequence"):
        screening_config_from_mapping({"spins": "bad"})
    with pytest.raises(TypeError, match="thresholds"):
        screening_config_from_mapping({"thresholds": "bad"})


def test_all_default_config_conditions_are_finite() -> None:
    config = screening_config_from_mapping(
        {
            "spins": [0.0],
            "inclinations_deg": [40.0],
            "eddington_ratios": [0.1],
            "f_col_true_values": [1.7],
            "inner_stress_delta_eta": [0.0],
            "replicate_count": 1,
            "energy_bin_count": 12,
            "spin_grid_count": 101,
        }
    )

    result = run_screening_campaign(config)
    summary = result.summaries[0]

    assert np.isfinite(summary.mean_bias)
    assert np.isfinite(summary.mean_width_68)
    assert np.isfinite(summary.mean_chi2_per_dof)
