"""Tests for the joint marginalized multi-epoch campaign."""

import csv
from pathlib import Path

import numpy as np
import yaml

from kerrdisk.joint_campaign import JointCampaignConfig
from kerrdisk.joint_multi_epoch import (
    run_joint_multi_epoch_campaign,
    write_joint_multi_epoch_outputs,
)


def _tiny_confirmatory_yaml(tmp_path: Path) -> Path:
    locked = tmp_path / "locked.csv"
    with locked.open("w", encoding="utf-8", newline="") as stream:
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
        # Two epochs of one group: same spin/inclination/f_col, different luminosity.
        for eddington in ("0.06", "0.18"):
            writer.writerow(
                {
                    "condition_id": f"cond-l{eddington}",
                    "spin_true": "0.0",
                    "inclination_deg": "40.0",
                    "eddington_ratio": eddington,
                    "f_col_true": "1.7",
                    "f_col_fit": "1.7",
                    "inner_stress_delta_eta": "0.0",
                }
            )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen\n", encoding="utf-8")
    conf_yaml = tmp_path / "conf.yaml"
    conf_yaml.write_text(
        yaml.safe_dump(
            {
                "config_version": "test_ray",
                "locked_conditions_path": str(locked),
                "frozen_protocol_path": str(protocol),
                "model_backend": "ray_traced_transfer",
                "energy_bin_count": 8,
                "radial_grid_count": 20,
                "disk_outer_radius_rg": 30.0,
                "ray_disk_outer_radius": 30.0,
                "ray_screen_alpha_min": -42.0,
                "ray_screen_alpha_max": 42.0,
                "ray_screen_beta_min": -42.0,
                "ray_screen_beta_max": 42.0,
                "ray_screen_size": 6,
                "ray_observer_radius": 200.0,
                "ray_escape_radius": 400.0,
                "gaussian_relative_error": 0.03,
            }
        ),
        encoding="utf-8",
    )
    return conf_yaml


def _tiny_joint_config(conf_yaml: Path) -> JointCampaignConfig:
    return JointCampaignConfig(
        config_version="test_joint_multi",
        confirmatory_config_path=conf_yaml,
        master_seed=11,
        replicate_count=2,
        spin_node_count=4,
        f_col_node_count=2,
        spin_fit_min=-0.5,
        spin_fit_max=0.9,
        f_col_fit_min=1.5,
        f_col_fit_max=2.0,
        walker_count=8,
        draws=60,
        burn_in=30,
    )


def test_joint_multi_epoch_runs_and_summarizes(tmp_path: Path) -> None:
    conf_yaml = _tiny_confirmatory_yaml(tmp_path)
    result = run_joint_multi_epoch_campaign(_tiny_joint_config(conf_yaml))

    assert len(result.groups) == 1
    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.epoch_count == 2
    assert summary.status == "COMPLETED"
    assert np.isfinite(summary.mean_joint_bias)
    assert summary.mean_joint_width_68 > 0.0


def test_joint_multi_epoch_writes_outputs(tmp_path: Path) -> None:
    conf_yaml = _tiny_confirmatory_yaml(tmp_path)
    result = run_joint_multi_epoch_campaign(_tiny_joint_config(conf_yaml))
    paths = write_joint_multi_epoch_outputs(result, tmp_path / "out")

    assert paths["summary"].exists()
    text = paths["summary"].read_text(encoding="utf-8")
    assert "width_68_reduction_fraction" in text
