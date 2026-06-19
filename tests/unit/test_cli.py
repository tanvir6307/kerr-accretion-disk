"""Tests for the bootstrap CLI."""

from typer.testing import CliRunner

from kerrdisk.cli import app


def test_cli_default_message() -> None:
    result = CliRunner().invoke(app)

    assert result.exit_code == 0
    assert "No physics commands" in result.stdout


def test_cli_version_option() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip()
