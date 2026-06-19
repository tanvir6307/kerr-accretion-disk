"""Audit availability of independent external ray-tracing validation backends."""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

OUTPUT = Path("data/processed/transfer_validation/phase12p5_external_backend_audit.csv")


@dataclass(frozen=True)
class BackendAuditRow:
    backend: str
    backend_type: str
    available: bool
    version_or_reference: str
    validation_role: str
    notes: str


def main() -> None:
    """Write a machine-readable audit of usable external validation backends."""

    rows = [
        _executable_row(
            "GYOTO",
            "gyoto",
            "Candidate independent GR ray-tracing executable.",
            "Expected command-line ray tracer; not bundled with this project.",
        ),
        _executable_row(
            "RAPTOR",
            "raptor",
            "Candidate independent GRRT executable.",
            "Expected compiled executable; not bundled with this project.",
        ),
        _executable_row(
            "grtrans",
            "grtrans",
            "Candidate independent GRRT executable.",
            "Expected compiled executable; not bundled with this project.",
        ),
        _python_module_row(
            "kgeo",
            "kgeo",
            "Candidate independent analytic Kerr ray-tracing Python package.",
            (
                "Not accepted as an agreement backend until screen-coordinate, "
                "redshift, disk-hit, and emission-angle conventions are adapted "
                "and validated."
            ),
        ),
        _python_module_row(
            "EinsteinPy",
            "einsteinpy",
            "Candidate independent geodesic-integration Python package.",
            (
                "Not accepted as an agreement backend until a null-geodesic "
                "screen-to-disk adapter is implemented and benchmarked."
            ),
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)
    print(OUTPUT)


def _executable_row(
    backend: str,
    command: str,
    role: str,
    notes: str,
) -> BackendAuditRow:
    path = shutil.which(command)
    version = ""
    if path is not None:
        version = _version_string(command)
    return BackendAuditRow(
        backend=backend,
        backend_type="executable",
        available=path is not None,
        version_or_reference=version or (path or ""),
        validation_role=role,
        notes=notes if path is None else f"{notes} Detected command: {path}",
    )


def _python_module_row(
    backend: str,
    module: str,
    role: str,
    notes: str,
) -> BackendAuditRow:
    spec = importlib.util.find_spec(module)
    version = ""
    if spec is not None:
        version = _module_version(module)
    return BackendAuditRow(
        backend=backend,
        backend_type="python_module",
        available=spec is not None,
        version_or_reference=version,
        validation_role=role,
        notes=notes,
    )


def _version_string(command: str) -> str:
    for args in ((command, "--version"), (command, "-v")):
        try:
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except OSError:
            continue
        output = (result.stdout or result.stderr).strip().splitlines()
        if output:
            return output[0][:200]
    return ""


def _module_version(module: str) -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version(module)
    except metadata.PackageNotFoundError:
        return ""


if __name__ == "__main__":
    main()
