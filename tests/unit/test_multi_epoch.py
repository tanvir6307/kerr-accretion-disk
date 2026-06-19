"""Tests for joint multi-epoch comparison utilities."""

import csv
from pathlib import Path

import numpy as np
import pytest

from kerrdisk.multi_epoch import (
    MultiEpochConfig,
    build_multi_epoch_groups,
    load_multi_epoch_config,
    run_multi_epoch_campaign,
    validate_multi_epoch_config,
    write_multi_epoch_outputs,
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
        for luminosity in ("0.06", "0.18"):
            writer.writerow(
                {
                    "condition_id": f"condition-l{luminosity}",
                    "spin_true": "0.0",
                    "inclination_deg": "40.0",
                    "eddington_ratio": luminosity,
                    "f_col_true": "1.7",
                    "f_col_fit": "1.7",
                    "inner_stress_delta_eta": "0.0",
                }
            )
    return path


def _confirmatory_config_file(tmp_path: Path) -> Path:
    locked = _locked_condition_file(tmp_path)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# Frozen protocol\n", encoding="utf-8")
    path = tmp_path / "confirmatory.yaml"
    path.write_text(
        "\n".join(
            [
                "config_version: test",
                f"locked_conditions_path: {locked.as_posix()}",
                f"frozen_protocol_path: {protocol.as_posix()}",
                "master_seed: 42",
                "replicate_count: 3",
                "energy_bin_count: 8",
                "spin_grid_count: 31",
                "resolution_spin_grid_count: 41",
                "model_backend: proxy",
                "radial_grid_count: 16",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _multi_epoch_config_file(tmp_path: Path) -> Path:
    confirmatory = _confirmatory_config_file(tmp_path)
    path = tmp_path / "multi.yaml"
    path.write_text(
        "\n".join(
            [
                "config_version: multi-test",
                f"confirmatory_config_path: {confirmatory.as_posix()}",
                "master_seed: 99",
                "replicate_count: 3",
                "min_epoch_count: 2",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_multi_epoch_campaign_runs_and_writes_outputs(tmp_path: Path) -> None:
    config = load_multi_epoch_config(_multi_epoch_config_file(tmp_path))

    result = run_multi_epoch_campaign(config)
    paths = write_multi_epoch_outputs(result, tmp_path / "outputs")

    assert len(result.groups) == 1
    assert len(result.replicates) == 3
    assert result.summaries[0].status == "COMPLETED"
    assert np.isfinite(result.summaries[0].width_68_reduction_fraction)
    assert set(paths) == {
        "groups",
        "replicates",
        "summary",
        "failure_summary",
        "analysis_freeze",
    }
    assert paths["summary"].exists()


def test_multi_epoch_groups_require_multiple_luminosities(tmp_path: Path) -> None:
    config = load_multi_epoch_config(_multi_epoch_config_file(tmp_path))
    result = run_multi_epoch_campaign(config)

    groups = build_multi_epoch_groups(result.groups[0].conditions, min_epoch_count=2)

    assert len(groups) == 1
    assert len(groups[0].conditions) == 2


def test_multi_epoch_config_rejects_invalid_values() -> None:
    config = MultiEpochConfig(
        config_version="bad",
        confirmatory_config_path=Path("missing.yaml"),
        master_seed=1,
        replicate_count=1,
        min_epoch_count=2,
        width_reduction_required_fraction=0.0,
    )

    with pytest.raises(ValueError, match="replicate_count"):
        validate_multi_epoch_config(
            config.__class__(**{**config.__dict__, "replicate_count": 0})
        )
    with pytest.raises(ValueError, match="min_epoch_count"):
        validate_multi_epoch_config(
            config.__class__(**{**config.__dict__, "min_epoch_count": 1})
        )
    with pytest.raises(ValueError, match="width_reduction"):
        validate_multi_epoch_config(
            config.__class__(
                **{**config.__dict__, "width_reduction_required_fraction": 1.0}
            )
        )
