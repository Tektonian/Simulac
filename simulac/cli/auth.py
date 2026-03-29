from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated

import typer

# FIXME: import raw service is anti-pattern
from simulac.base.envvar.envvar_service import EnvvarService

TOKEN_PORTAL_URL = "https://tektonian.com/settings/token"

APP_HELP = "Simulac CLI\n\n"


APP_EPILOG = "\n\n".join(
    [
        "If you are new to Simulac, start with `simulac login` and paste an API "
        f"key from {TOKEN_PORTAL_URL}.\n\n"
        "\n\nLearn more\n",
        "If you want to know detail about Simulac visit https://tektonian.com/docs\n\n",
        "Examples:",
        "-  $ simulac login",
        "-  $ simulac login --apikey tt_sim_your_api_key",
        "-  $ simulac logout",
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


@app.callback()
def cli(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the installed simulac version and exit.",
            callback=_show_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Authenticate this machine with Simulac."""


def _save_token(token_path: Path, token: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token.strip() + "\n", encoding="utf-8")


def _read_token(token_path: Path) -> str | None:
    if not token_path.exists() or not token_path.is_file():
        return None

    lines = token_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None

    token = lines[0].strip()
    return token or None


def _mask_secret(secret: str | None) -> str:
    """Return masked api key

    Args:
        secret (str | None): Api key value. Should look like `tt_sim_***

    Returns:
        str: Masked API key string with only the prefix visible
    """
    if not secret:
        return "preview unavailable"
    return f"{secret[: len('tt_sim_')]}..."


@app.command(
    "login",
    short_help="Save a Simulac API key locally.",
    help=(
        "Save a Simulac API key in the local credential cache.\n\n"
        "If you are new to Simulac, create a key at "
        f"{TOKEN_PORTAL_URL} and paste it when prompted."
    ),
)
def login(
    apikey: Annotated[
        str | None,
        typer.Option(
            "--apikey",
            "-k",
            prompt="Paste your Simulac API key from https://tektonian.com/settings/token\n\nAPI KEY",
            hide_input=True,
            help="Simulac API key to save in the local credential cache.",
        ),
    ] = None,
) -> None:
    env = EnvvarService()

    if apikey is None:
        typer.echo("No API key provided. Nothing was saved.")
        raise typer.Exit(code=1)

    clean_apikey = apikey.strip()
    if not clean_apikey:
        typer.echo(
            "The API key was empty after trimming whitespace. Nothing was saved."
        )
        raise typer.Exit(code=1)

    _save_token(env.token_path, clean_apikey)

    typer.echo(f"Saved: Simulac API key ({_mask_secret(clean_apikey)})")
    typer.echo(f"Location: {env.token_path}")


@app.command(
    short_help="Delete the locally stored API key.",
    help=(
        "Remove the API key stored in the local credential cache.\n\n"
        "This deletes the local file created by `simulac login` when it exists. "
    ),
)
def logout() -> None:
    env = EnvvarService()
    stored_token = _read_token(env.token_path)

    if env.token_path.exists():
        env.token_path.unlink(missing_ok=True)
        typer.echo(f"Removed: API key ({_mask_secret(stored_token)})")
        typer.echo(f"Location: {env.token_path}")
        typer.echo("Deleted file: local plain-text credential cache.")
    else:
        typer.echo("No local credential file was removed.")
        typer.echo(f"Checked: {env.token_path}")
        typer.echo("Data: No Simulac API key file was present.")
