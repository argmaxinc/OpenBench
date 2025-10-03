# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Dataset command for openbench-cli."""

import typer
from rich.console import Console

from .download import download


console = Console()

app = typer.Typer(
    name="dataset",
    help="Dataset management commands for OpenBench",
    add_completion=False,
)

# Add subcommands
app.command()(download)


def main() -> None:
    """Main entry point for the dataset command."""
    app()


if __name__ == "__main__":
    main()
