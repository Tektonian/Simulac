from __future__ import annotations

from typing import Any, Optional, overload

from .gym_style_environment import BenchmarkEnvironment, BenchmarkVecEnvironment


@overload
def init_bench(
    benchmark_id: str,
    env_id: str,
    seed: Optional[int] = 0,
    /,
    benchmark_specific: dict[str, Any] = {},
) -> BenchmarkEnvironment: ...
@overload
def init_bench(
    benchmark_id: str,
    env_id: None,
    seed: Optional[int] = 0,
    /,
    benchmark_specific: dict[str, Any] = {},
) -> BenchmarkVecEnvironment: ...
def init_bench(
    benchmark_id: str,
    env_id: Optional[str],
    seed: Optional[int] = 0,
    /,
    benchmark_specific: dict[str, Any] = {},
):
    """Initialize environment

    Args:
        env_uri (str): Environment URI could be .json file location, remote prebuilt environment uri
        sim_id (str): _description_
    """

    if env_id is None:
        vec_env = BenchmarkVecEnvironment([])
        return vec_env

    env = BenchmarkEnvironment(
        "id",
        benchmark_id,
        env_id,
        seed,
        benchmark_specific,
    )

    return env


def make_vec(envs: list[BenchmarkEnvironment]):
    vec_env = BenchmarkVecEnvironment(envs)
    return vec_env


__all__ = ["init_bench", "make_vec"]
