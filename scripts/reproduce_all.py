"""Reproduce archived or full paper artifacts and write release manifests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml

from kerrdisk.config import ProjectConfig, RuntimeConfig
from kerrdisk.io import (
    build_checksum_manifest,
    create_run_manifest,
    utc_now_iso,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option("--config", help="Versioned release reproduction config."),
    ] = Path("configs/production/release.yaml"),
    mode: Annotated[
        Literal["archive", "full"],
        typer.Option(
            "--mode",
            help="archive uses existing heavy outputs; full reruns them.",
        ),
    ] = "archive",
) -> None:
    """Regenerate paper figures/tables and write checksums."""

    raw = _read_yaml(config)
    project_config = ProjectConfig(
        project_name=str(raw.get("project_name", "KerrDisk-UQ")),
        phase=str(raw.get("phase", "release_reproduction")),
        runtime=RuntimeConfig.model_validate(raw.get("runtime", {})),
    )
    manifest = create_run_manifest(project_config, cwd=ROOT)
    output_root = ROOT / project_config.runtime.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    try:
        reproduction = raw.get("reproduction", {})
        if not isinstance(reproduction, dict):
            msg = "reproduction config must be a mapping"
            raise TypeError(msg)
        if mode == "full":
            _run_full_reproduction(reproduction)
        else:
            _require_archive_inputs()
        _run_script("scripts/make_tables.py")
        _run_script("scripts/make_figures.py")
    except Exception as exc:
        warnings.append(str(exc))
        final_manifest = manifest.model_copy(
            update={
                "end_timestamp_utc": utc_now_iso(),
                "warnings": warnings,
                "convergence_status": "FAILED",
            }
        )
        write_manifest(output_root / "run_manifest.json", final_manifest)
        raise

    checksum_paths = _checksum_paths(raw, output_root=output_root)
    checksums = build_checksum_manifest(checksum_paths, root=ROOT)
    checksum_csv = output_root / "checksums_sha256.csv"
    _write_checksum_csv(checksum_csv, checksums)
    final_manifest = manifest.model_copy(
        update={
            "end_timestamp_utc": utc_now_iso(),
            "warnings": warnings,
            "convergence_status": "archive_reproduced"
            if mode == "archive"
            else "full_reproduced",
            "checksum_manifest": checksums,
        }
    )
    write_manifest(output_root / "run_manifest.json", final_manifest)
    typer.echo(f"mode: {mode}")
    typer.echo(f"manifest: {output_root / 'run_manifest.json'}")
    typer.echo(f"checksums: {checksum_csv}")


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        msg = f"release config must be a mapping: {path}"
        raise TypeError(msg)
    return raw


def _run_full_reproduction(reproduction: dict[str, object]) -> None:
    phase12_config = str(
        reproduction.get(
            "phase12_config",
            "configs/production/phase12_confirmatory.yaml",
        )
    )
    multi_epoch_config = str(
        reproduction.get(
            "multi_epoch_config",
            "configs/production/phase13p5_multi_epoch.yaml",
        )
    )
    _run_script("scripts/run_validation.py", "page-thorne-flux")
    _run_script("scripts/run_validation.py", "independent")
    _run_script("scripts/run_transfer_validation.py")
    _run_script("scripts/run_confirmatory.py", "--config", phase12_config)
    _run_script("scripts/run_multi_epoch.py", "--config", multi_epoch_config)


def _require_archive_inputs() -> None:
    required = [
        ROOT / "data/processed/validation_summary.csv",
        ROOT / "data/processed/confirmatory/phase12_results_unblinded.csv",
        ROOT / "data/processed/confirmatory/phase12_replicates_blinded.csv",
        ROOT / "data/processed/confirmatory/phase12_resolution_reruns.csv",
        ROOT / "data/processed/transfer_validation/phase12p5_transfer_convergence.csv",
        ROOT
        / "data/processed/transfer_validation"
        / "phase12p5_capture_returning_diagnostics.csv",
        ROOT
        / "data/processed/transfer_validation"
        / "phase12p5_external_transfer_comparison.csv",
        ROOT / "data/processed/multi_epoch/phase13p5_multi_epoch_summary.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        msg = "archive reproduction inputs are missing: " + ", ".join(
            str(path.relative_to(ROOT)) for path in missing
        )
        raise FileNotFoundError(msg)


def _run_script(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def _checksum_paths(raw: dict[str, object], *, output_root: Path) -> list[Path]:
    reproduction = raw.get("reproduction", {})
    configured: object = []
    if isinstance(reproduction, dict):
        configured = reproduction.get("checksum_paths", [])
    if not isinstance(configured, list):
        configured = []
    roots = [ROOT / str(item) for item in configured]
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            if _include_checksum_path(root):
                paths.append(root)
        elif root.is_dir():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and _include_checksum_path(path)
                and not _is_relative_to(path, output_root)
            )
    return paths


def _include_checksum_path(path: Path) -> bool:
    return "__pycache__" not in path.parts and path.suffix != ".pyc"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _write_checksum_csv(path: Path, checksums: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sha256"])
        writer.writeheader()
        for file_path, checksum in sorted(checksums.items()):
            writer.writerow({"path": file_path, "sha256": checksum})


if __name__ == "__main__":
    app()
