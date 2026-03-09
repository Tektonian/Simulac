from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple

from tt.base.error.error import TektonianBaseError
from tt.sdk.environment_service.common.environment_service import (
    IEnvironmentManagementService,
)
from tt.sdk.runner_service.common.runner import IRunner
from tt.sdk.runner_service.common.runner_service import IRunnerManagementService

if TYPE_CHECKING:
    from tt.base.instantiate.instantiate import IInstantiateService


class GymStyleEnvironment:

    def __init__(
        self,
        instantiate_service: IInstantiateService,
        benchmark_id: str,
        remote_env_id: str,
        seed: int,
        benchmark_specific: dict[str, Any],
    ) -> None:
        self.benchmark_id = benchmark_id
        self.remote_env_id = remote_env_id
        self.benchmark_specific = benchmark_specific

        # region [Libero]
        # TODO: mapping task_name to task_id
        self.benchmark_specific["task_name"] = remote_env_id
        # self.benchmark_specific["task_id"] = 0
        self.benchmark_specific["seed"] = seed
        # endregion

        self.env_service: IEnvironmentManagementService = (
            instantiate_service.service_accessor.get(IEnvironmentManagementService)
        )

        self.runner_service: IRunnerManagementService = (
            instantiate_service.service_accessor.get(IRunnerManagementService)
        )
        [owner, env_id] = benchmark_id.split("/")
        # base_url = f"http://0.0.0.0:3000/api/container/{owner}/{env_id}"
        base_url = "https://localhost:3000/ws" # Docker directly
        env_ret = self.env_service.create_environment(
            base_url + "/env.json", base_url + "/act.json", base_url + "/obs.json", seed
        )

        if env_ret[0] is not None:
            self.env = env_ret[0]
            self.runner: None | IRunner = None
        else:
            raise env_ret[1]

    def step(self, action: object) -> object:
        if self.runner is None:
            raise TektonianBaseError("Environment not reset")
        recv = self.runner.step(action)
        import json, json_numpy
        recv_dict = json.loads(recv)
        obs = json_numpy.loads(recv_dict["obs"])
        reward = recv_dict["reward"]
        done = recv_dict["done"]
        info = recv_dict["info"]
        return obs, reward, done, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> None:
        """Reset runner

        Raises:
            runner_ret: _description_
        """
        if self.runner is not None:
            self.runner_service.remove_runner(self.runner.id)

        if seed is None:
            seed = 0

        self.benchmark_specific["seed"] = seed

        runner_ret: Tuple[IRunner, None] | Tuple[None, BaseException] = (
            self.runner_service.create_runner(self.env.id, self.benchmark_specific)
        )
        if runner_ret[0] is not None:
            self.runner = runner_ret[0]
        else:
            raise runner_ret[1]
        
        import json, json_numpy
        recv = self.runner.recv
        recv_dict = json.loads(recv)
        obs = json_numpy.loads(recv_dict['obs'])
        info = recv_dict["info"]
        
        return obs, info

    def render(self) -> None: ...

    def close(self) -> None:
        if self.runner:
            self.runner_service.remove_runner(self.runner.id)
        self.runner = None
