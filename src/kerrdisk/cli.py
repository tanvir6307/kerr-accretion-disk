"""Command-line entry point for bootstrap checks."""

import typer

from kerrdisk import __version__

app = typer.Typer(help="KerrDisk-UQ development CLI.")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        default=False,
        help="Show the installed package version.",
    ),
) -> None:
    """Run the bootstrap CLI."""

    if version:
        typer.echo(__version__)
        raise typer.Exit
    typer.echo("KerrDisk-UQ bootstrap CLI. No physics commands are implemented yet.")
