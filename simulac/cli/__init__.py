from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Annotated

import typer

from simulac.sdk import obtain_runtime

from .auth import app as auth_app
from .benchmark import app as benchmark_app
from .common import TOKEN_PORTAL_URL
from .config import show_config, show_envvars

APP_HELP = "Simulac CLI"

APP_EPILOG = "\n\n".join(
    [
        "If you are new to Simulac, start with `simulac auth login` and paste an API "
        f"key from {TOKEN_PORTAL_URL}.",
        "Examples:",
        "-  $ simulac auth login",
        "-  $ simulac auth whoami",
        "-  $ simulac benchmark list Tektonian/Metaworld",
        "-  $ simulac env",
        "-  $ simulac config",
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


def _version_option_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"{_package_version()}")
    raise typer.Exit()


@app.callback()
def root_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the installed Simulac version and exit.",
            is_eager=True,
            callback=_version_option_callback,
        ),
    ] = False,
) -> None:
    runtime = obtain_runtime()
    ctx.obj = {"runtime": runtime}
    del version


app.add_typer(auth_app, name="auth", rich_help_panel="Main commands")
app.add_typer(benchmark_app, name="benchmark", rich_help_panel="Main commands")
app.command(
    "config",
    short_help="Show the effective Simulac configuration.",
    rich_help_panel="Help commands",
)(show_config)
app.command(
    "env",
    short_help="Show raw Simulac environment variables.",
    rich_help_panel="Help commands",
)(show_envvars)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
