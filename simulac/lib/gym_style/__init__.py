from __future__ import annotations

from typing import Any, Optional, overload
from urllib.parse import quote

import requests

from simulac.base.error.error import SimulacBaseError
from simulac.sdk.runtime import obtain_runtime

from .gym_style_environment import BenchmarkEnvironment, BenchmarkVecEnvironment


def init_bench(
    benchmark_id: str,
    env_id: str,
    seed: int = 0,
    /,
    benchmark_specific: dict[str, Any] = {},
) -> BenchmarkEnvironment:
    """Initalize benchmark service

    Args:
        benchmark_id (str): Full benchmark id.\n
            Example: benchmark_id="Tektonian/Metaworld"
        env_id (str): Environment id of the benchmark.\n
            Example: envid="reach-v3"
            If you want to see the full list of the `env_id` visit https://tektonian.com/benchmark
        seed (int, optional): Seed for inital state. Defaults to 0.
        benchmark_specific (dict[str, Any], optional): Benchmark specific option field.\n
            Also, if you want to know the full field and meaning please visit https://tektonian.com/benchmark\n
            Defaults to {}.

    Returns:
        ret (BenchmarkEnvironment):
    """
    runtime = obtain_runtime()
    split_benchmark_id = benchmark_id.split("/")
    normalized_benchmark_id = benchmark_id.strip()

    MESSAGE = "\n".join(
        [
            f"Invalid benchmark_id {benchmark_id!r}. ",
            "Expected '<organization>/<benchmark>', (e.g., 'Tektonian/Metaworld')",
        ]
    )
    if len(split_benchmark_id) == 1:
        raise SimulacBaseError(MESSAGE)
    elif len(split_benchmark_id) != 2:
        runtime.logger.warn(
            "\n".join(
                [
                    MESSAGE,
                    f"Unused fields: '{split_benchmark_id[2:]}' will be removed",
                ]
            )
        )
    elif normalized_benchmark_id != benchmark_id:
        runtime.logger.warn(
            "\n".join(
                [
                    f"benchmark_id has leading or trailing spaces: {benchmark_id!r}. "
                    f"Use {normalized_benchmark_id!r}."
                ]
            )
        )

    # Rename
    (owner_id, world_id) = split_benchmark_id

    env = BenchmarkEnvironment(
        owner_id,
        world_id,
        env_id,
        seed,
        benchmark_specific,
    )

    return env


def get_env_list(benchmark_id: str) -> list[str]:
    # TODO: @gangjeuk
    # Remove group_id later

    # deprecated group_id
    group_id = None

    runtime = obtain_runtime()
    if "/" not in benchmark_id:
        raise SimulacBaseError(
            "Benchmark id format should be owner_id/env_id (e.g., Tektonian/Libero)"
        )
    owner_id, env_id = benchmark_id.split("/", maxsplit=1)
    url = "/".join(
        [
            runtime.environment_variable.base_url,
            "container",
            "scene-list",
            quote(owner_id, safe=""),
            quote(env_id, safe=""),
        ]
    )
    params = {"env_group": group_id} if group_id is not None else None
    res = requests.get(url, params=params, timeout=10)

    res.raise_for_status()

    env_list: list[str] | Any = res.json()
    if not isinstance(env_list, list):
        # Should not be raised. If it happens, it's backend's fault
        raise SimulacBaseError(
            "Scene list response should be a list of environment ids."
        )

    return env_list


def make_vec(envs: list[BenchmarkEnvironment]):
    vec_env = BenchmarkVecEnvironment(envs)
    return vec_env


__all__ = ["init_bench", "make_vec", "get_env_list"]
