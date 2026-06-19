"""Tests for run-manifest helpers."""

import json
from pathlib import Path

from kerrdisk.config import ProjectConfig
from kerrdisk.io import (
    build_checksum_manifest,
    collect_package_versions,
    create_run_manifest,
    get_git_commit,
    sha256_file,
    write_manifest,
)


def test_create_manifest_records_required_fields() -> None:
    config = ProjectConfig()
    manifest = create_run_manifest(config, warnings=["bootstrap warning"])

    assert manifest.project_name == "KerrDisk-UQ"
    assert manifest.random_seed == config.runtime.random_seed
    assert manifest.warnings == ["bootstrap warning"]
    assert "pydantic" in manifest.package_versions
    assert manifest.resolved_config["project_name"] == "KerrDisk-UQ"


def test_write_manifest_creates_json(tmp_path: Path) -> None:
    manifest = create_run_manifest(ProjectConfig())
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["project_name"] == "KerrDisk-UQ"


def test_collect_package_versions_marks_missing_package() -> None:
    versions = collect_package_versions(["definitely-missing-kerrdisk-package"])

    assert versions["definitely-missing-kerrdisk-package"] == "UNAVAILABLE"


def test_checksum_manifest_uses_relative_paths(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("deterministic\n", encoding="utf-8")

    checksum = sha256_file(path)
    manifest = build_checksum_manifest([path], root=tmp_path)

    assert manifest == {"artifact.txt": checksum}
    assert len(checksum) == 64


def test_get_git_commit_outside_repository(tmp_path: Path) -> None:
    assert get_git_commit(tmp_path) == "UNAVAILABLE"
