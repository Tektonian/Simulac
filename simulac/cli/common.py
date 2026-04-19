from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import requests
import typer

from simulac.base.envvar.envvar_service import EnvvarKeyValue
from simulac.base.error.error import SimulacBaseError

if TYPE_CHECKING:
    from simulac.sdk.runtime import SimulacRuntime
from simulac.base.envvar.envvar_service import IEnvvarService

TOKEN_PORTAL_URL = "https://tektonian.com/settings/token"
LOG_LEVEL_NAMES = ("off", "trace", "debug", "info", "warning", "error")


@dataclass(slots=True)
class TokenState:
    status: Literal["PRESENT", "INVALID", "MISSING"]
    source: Literal["SIMULAC_API_KEY", "FILE", "NONE"]
    value: str | None
    preview: str | None


@dataclass(slots=True)
class IdentityResult:
    endpoint: str
    user: str | None
    email: str | None


@dataclass(slots=True)
class CliContext:
    runtime: SimulacRuntime


def cast_context(ctx: typer.Context) -> CliContext:
    runtime = ctx.obj["runtime"]
    return CliContext(runtime)


def mask_secret(secret: str | None) -> str | None:
    if not secret:
        return None
    return f"{secret[: len('tt_sim_')]}..."


def read_token(token_path: Path) -> str | None:
    if not token_path.exists() or not token_path.is_file():
        return None

    lines = token_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None

    token = lines[0].strip()
    return token or None


def get_token_state(env: IEnvvarService) -> TokenState:
    env_token = os.environ.get(EnvvarKeyValue.SIMULAC_API_KEY.value)
    file_token = read_token(env.token_path)

    source = "NONE"
    token = None
    if env_token is not None and env_token.strip():
        source = "SIMULAC_API_KEY"
        token = env_token.strip()
    elif file_token:
        source = "FILE"
        token = file_token

    status = "MISSING"
    if token:
        status = "PRESENT" if len(token) > 40 else "INVALID"
    return TokenState(
        status=status,
        source=source,
        value=token,
        preview=mask_secret(token),
    )


def collect_config_snapshot(env: IEnvvarService) -> dict[str, Any]:
    env = env
    token_state = get_token_state(env)
    log_level_index = env.log_level
    log_level = (
        LOG_LEVEL_NAMES[log_level_index]
        if 0 <= log_level_index < len(LOG_LEVEL_NAMES)
        else f"unknown({log_level_index})"
    )

    return {
        "app_root": str(env.app_root),
        "asset_dir": str(env.asset_dir),
        "base_url": env.base_url,
        "cache_dir": str(env.simulac_cache_dir),
        "log_file": str(env.log_file),
        "log_level": log_level,
        "tmp_dir": str(env.tmp_dir),
        "token_path": str(env.token_path),
        "token_source": token_state.source,
        "token_status": token_state.status,
    }


def fetch_identity(base_url: str, token: str) -> IdentityResult:
    headers = {"tt-apikey": token}
    url = f"{base_url}/auth/whoami"

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise SimulacBaseError(
            f"Failed to reach Simulac identity endpoint: {exc}"
        ) from exc

    if response.status_code >= 400:
        raise SimulacBaseError("Authentication failed while validating the API key.")

    try:
        response.raise_for_status()
    except Exception as exc:
        detail = response.text.strip() or str(exc)
        raise SimulacBaseError(
            f"Identity lookup failed: {response.status_code} {detail}"
        ) from exc

    try:
        payload_raw = response.json()
    except ValueError as exc:
        raise SimulacBaseError("Identity lookup returned a non-JSON response.") from exc

    payload: dict[str, str] = (
        payload_raw if isinstance(payload_raw, dict) else {"value": payload_raw}
    )
    return IdentityResult(
        endpoint=url,
        user=payload.get("user", None),
        email=payload.get("email", None),
    )
