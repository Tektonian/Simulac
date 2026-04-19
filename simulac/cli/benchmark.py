from __future__ import annotations

import typer

from simulac.base.error.error import SimulacBaseError
from simulac.lib.gym_style import get_env_list

EPILOG = "\n\n".join(
    [
        "\b",
        "Examples:",
        "  $ simulac benchmark list Tektonian/Libero",
        "  $ simulac benchmark list Tektonian/Metaworld",
    ]
)
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark discovery commands.",
    epilog=EPILOG,
)


@app.command("list", short_help="List environments for a benchmark.")
def list_benchmark(
    benchmark_id: str,
) -> None:
    try:
        env_ids = get_env_list(benchmark_id)
    except SimulacBaseError as exc:
        typer.echo(exc.message, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Failed to fetch benchmark environments: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Benchmark: {benchmark_id}")
    typer.echo(f"Environment Count: {len(env_ids)}")
    for env_id in env_ids:
        typer.echo(env_id)
