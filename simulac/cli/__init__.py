from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Annotated

import typer

from .auth import app as auth_app
APP_HELP = "Simulac CLI"

APP_EPILOG = "\n\n".join(
    [
        "If you are new to Simulac, start with `simulac login` and paste an API "
        f"key from {TOKEN_PORTAL_URL}.",
        "Examples:",
    ]
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=APP_HELP,
    epilog=APP_EPILOG,
)


def _package_version() -> str:
    try:
        return pkg_version("simulac")
    except PackageNotFoundError:
        return "unknown"


def _show_version(value: bool) -> None:
    if not value:
        return

    typer.echo(f"simulac {_package_version()}")
    raise typer.Exit()


def _version_command() -> None:
    typer.echo(f"simulac {_package_version()}")


app.add_typer(auth_app, name="auth", rich_help_panel="Main commands")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
