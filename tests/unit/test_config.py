"""Tests for bootstrap configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kerrdisk.config import (
    PhysicalDomain,
    ProjectConfig,
    RuntimeConfig,
    load_project_config,
)


def test_default_project_config() -> None:
    config = ProjectConfig()

    assert config.project_name == "KerrDisk-UQ"
    assert config.runtime.random_seed == 20260618
    assert config.physical_domain.luminosity_eddington_min == 0.03


def test_runtime_config_rejects_negative_seed() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(random_seed=-1)


def test_physical_domain_requires_ordered_luminosity_range() -> None:
    with pytest.raises(ValidationError):
        PhysicalDomain(
            luminosity_eddington_min=0.3,
            luminosity_eddington_max=0.03,
        )


def test_physical_domain_requires_ordered_spin_range() -> None:
    with pytest.raises(ValidationError):
        PhysicalDomain(spin_min=0.5, spin_max=-0.5)


def test_load_project_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("project_name: Test\nphase: phase1\n", encoding="utf-8")

    config = load_project_config(path)

    assert config.project_name == "Test"
    assert config.phase == "phase1"


def test_load_project_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TypeError):
        load_project_config(path)
