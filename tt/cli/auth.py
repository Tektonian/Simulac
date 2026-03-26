from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated, Optional

import typer

from tt.base.envvar.envvar_service import EnvvarService

app = typer.Typer(help="Tektonian CLI — interact with Tektonian services")


def _save_token(token_path: Path, token: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token.strip() + "\n")


API_KEY_OPT = Annotated[
    Optional[str],
    typer.Option(
        help="A private api-key for access Tektonian service. A api-key can generated from http://tektonian.com/settings/token"
    ),
]


@app.command("login", help="Authenticate by storing your API key locally")
def login(
    apikey: Optional[str] = typer.Option(
        None,
        "--apikey",
        "-k",
        prompt=True,
        hide_input=True,
        help="Your Tektonian API key (will be saved locally)",
    ),
) -> None:
    env = EnvvarService()

    if not apikey:
        typer.echo("No API key provided; aborting.")
        raise typer.Exit(code=1)

    _save_token(env.token_path, apikey)
    typer.echo(f"Saved API key to {env.token_path}")


@app.command(help="Remove stored credentials from this machine")
def logout() -> None:
    env = EnvvarService()
    if env.token_path.exists():
        env.token_path.unlink(missing_ok=True)
        typer.echo("Removed stored API key.")
    else:
        typer.echo("No stored API key found.")
