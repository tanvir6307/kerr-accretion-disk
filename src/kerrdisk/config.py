"""Configuration models for KerrDisk-UQ runs."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PhysicalDomain(BaseModel):
    """Declared validity domain for bootstrap configuration."""

    model_config = ConfigDict(frozen=True)

    luminosity_eddington_min: float = Field(default=0.03, gt=0.0)
    luminosity_eddington_max: float = Field(default=0.30, gt=0.0)
    spin_min: float = Field(default=-0.998, gt=-1.0, lt=1.0)
    spin_max: float = Field(default=0.998, gt=-1.0, lt=1.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "PhysicalDomain":
        """Require ordered validity ranges."""

        if self.luminosity_eddington_min >= self.luminosity_eddington_max:
            msg = "luminosity_eddington_min must be less than luminosity_eddington_max"
            raise ValueError(msg)
        if self.spin_min >= self.spin_max:
            msg = "spin_min must be less than spin_max"
            raise ValueError(msg)
        return self


class RuntimeConfig(BaseModel):
    """Runtime controls shared by validation and production commands."""

    model_config = ConfigDict(frozen=True)

    random_seed: int = Field(default=20260618, ge=0)
    output_root: Path = Path("outputs")
    overwrite: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ProjectConfig(BaseModel):
    """Top-level resolved configuration for a KerrDisk-UQ run."""

    model_config = ConfigDict(frozen=True)

    project_name: str = "KerrDisk-UQ"
    phase: str = "bootstrap"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    physical_domain: PhysicalDomain = Field(default_factory=PhysicalDomain)


def load_project_config(path: Path) -> ProjectConfig:
    """Load a project configuration from YAML."""

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        msg = f"configuration must be a mapping: {path}"
        raise TypeError(msg)
    return ProjectConfig.model_validate(raw)
