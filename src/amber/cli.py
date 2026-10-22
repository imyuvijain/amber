"""Command-line entry point (`amber`). Subcommands are added by later tasks."""

import typer

app = typer.Typer(
    help="Amber: five yous compete to predict your clothing picks.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Amber command-line interface."""
