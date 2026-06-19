"""Run-manifest helpers."""

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kerrdisk.config import ProjectConfig


class RunManifest(BaseModel):
    """Structured metadata recorded with every reproducible run."""

    model_config = ConfigDict(frozen=True)

    project_name: str
    phase: str
    git_commit: str
    package_versions: dict[str, str]
    random_seed: int
    start_timestamp_utc: str
    end_timestamp_utc: str | None = None
    warnings: list[str] = Field(default_factory=list)
    convergence_status: str = "not_applicable"
    checksum_manifest: dict[str, str] = Field(default_factory=dict)
    resolved_config: dict[str, Any]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def get_git_commit(cwd: Path | None = None) -> str:
    """Return the current git commit, or `UNAVAILABLE` outside a repository."""

    working_directory = cwd if cwd is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "UNAVAILABLE"
    if result.returncode != 0:
        return "UNAVAILABLE"
    commit = result.stdout.strip()
    return commit if commit else "UNAVAILABLE"


def collect_package_versions(packages: Sequence[str]) -> dict[str, str]:
    """Collect installed package versions without failing on missing packages."""

    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "UNAVAILABLE"
    return versions


def create_run_manifest(
    config: ProjectConfig,
    *,
    warnings: Sequence[str] = (),
    cwd: Path | None = None,
) -> RunManifest:
    """Create a manifest for a run that has not executed physics yet."""

    package_versions = collect_package_versions(
        ["kerrdisk-uq", "numpy", "scipy", "pydantic", "typer"]
    )
    return RunManifest(
        project_name=config.project_name,
        phase=config.phase,
        git_commit=get_git_commit(cwd),
        package_versions=package_versions,
        random_seed=config.runtime.random_seed,
        start_timestamp_utc=utc_now_iso(),
        warnings=list(warnings),
        resolved_config=config.model_dump(mode="json"),
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a run manifest as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checksum_manifest(paths: Sequence[Path], *, root: Path) -> dict[str, str]:
    """Return stable relative-path SHA-256 checksums for existing files."""

    manifest: dict[str, str] = {}
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            continue
        key = path.relative_to(root.resolve()).as_posix()
        manifest[key] = sha256_file(path)
    return manifest
