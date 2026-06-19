"""Tests for independent validation diagnostics."""

from pathlib import Path

from kerrdisk.diagnostics import run_independent_validation, write_validation_summary


def test_independent_validation_rows_pass() -> None:
    rows = run_independent_validation()

    assert rows
    assert {row.category for row in rows} >= {
        "isco",
        "efficiency",
        "ray_invariant",
        "shadow",
        "disk_flux",
        "spectrum",
    }
    assert all(row.status == "PASS" for row in rows)


def test_write_validation_summary(tmp_path: Path) -> None:
    rows = run_independent_validation()
    path = tmp_path / "validation_summary.csv"

    write_validation_summary(path, rows)

    text = path.read_text(encoding="utf-8")
    assert "check_id,category,quantity" in text
    assert "schwarzschild_shadow_bracket" in text
