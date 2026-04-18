from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

# FIXME: import raw service is anti-pattern
from simulac.base.envvar.envvar_service import EnvvarService
from simulac.base.error.error import SimulacBaseError

from .common import (
    TOKEN_PORTAL_URL,
    fetch_identity,
    get_token_state,
    mask_secret,
    read_token,
)

AUTH_EPILOG = "\n\n".join(
    [
        "\b",
        "Examples:",
        "  $ simulac auth whoami",
        "  $ simulac auth login",
        "  $ simulac auth logout",
    ]
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Authentication commands.",
    epilog=AUTH_EPILOG,
)


def _save_token(token_path: Path, token: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token.strip() + "\n", encoding="utf-8")


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
        typer.echo("No API key provided. Nothing was saved.", err=True)
        raise typer.Exit(code=1)

    clean_apikey = apikey.strip()
    if not clean_apikey:
        typer.echo(
            "The API key was empty after trimming whitespace. Nothing was saved.",
            err=True,
        )
        raise typer.Exit(code=1)

    _save_token(env.token_path, clean_apikey)

    typer.echo(f"Saved: Simulac API key ({mask_secret(clean_apikey)})")
    typer.echo(f"Location: {env.token_path}")


@app.command(
    "logout",
    short_help="Delete the locally stored API key.",
    help=(
        "Remove the API key stored in the local credential cache.\n\n"
        "This deletes the local file created by `simulac auth login` when it exists."
    ),
)
def logout() -> None:
    env = EnvvarService()
    stored_token = read_token(env.token_path)

    if env.token_path.exists():
        env.token_path.unlink(missing_ok=True)
        typer.echo(f"Removed: API key ({mask_secret(stored_token)})")
        typer.echo(f"Location: {env.token_path}")
        typer.echo("Deleted file: local plain-text credential cache.")
    else:
        typer.echo("No local credential file was removed.")
        typer.echo(f"Checked: {env.token_path}")
        typer.echo("Data: No Simulac API key file was present.")
