import json
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import urlencode

from websockets.sync.client import ClientConnection, connect

from tt.sdk.runner_service.common.runner import IRunner


@dataclass
class RemoteRunner(IRunner):
    """It is more like BenchmarkRunner for now. Not RemoteRunner
    TODO: Should change logic and add BenchmarkRunner class instead

    Args:
        IRunner (_type_): _description_
    """

    id: str
    env_id: str
    state: object

    def __init__(
        self,
        id: str,
        env_id: str,
        state: object,
        owner: str,
        remote_env_id: str,
        kwargs: dict[str, Any],
    ):

        self.id = id
        self.env_id = env_id
        self.state = state
        self.kwargs = kwargs

        params = urlencode(kwargs)

        self._socket = connect(
            f"ws://localhost:3000/api/container/{owner}/{remote_env_id}?{params}"
        )

        msg = json.dumps({"command": "build_env", "args": self.kwargs})
        print(msg)
        self._socket.send(msg)
        recv = self._socket.recv()
        # print(recv)

    def step(self, action: object) -> object:
        self._socket.send(json.dumps({"command": "step", "args": {"action": action}}))

        recv = self._socket.recv()
        # print(recv)
        return recv

    def set_state(self) -> None:
        pass

    def get_state(self) -> None:
        pass

    def clone_state(self) -> None: ...

    def render(self) -> None:
        pass
