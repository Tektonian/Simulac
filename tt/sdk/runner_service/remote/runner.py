from dataclasses import dataclass
from typing import Any, Dict
import json
from urllib.parse import urlencode

from websockets.sync.client import connect, ClientConnection

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

        # self._socket = connect(
        #     f"ws://localhost:3000/api/container/{owner}/{remote_env_id}?{params}"
        # )
        self._socket = connect(
            f"ws://localhost:3000/ws"
        )

        msg = json.dumps({"command": "build_env", "args": self.kwargs})
        print(msg)
        self._socket.send(msg)
        self.recv = self._socket.recv()

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if hasattr(value, "detach") and callable(value.detach):
            return self._to_jsonable(value.detach().cpu().tolist())
        if hasattr(value, "tolist") and callable(value.tolist):
            return value.tolist()
        if hasattr(value, "item") and callable(value.item):
            return value.item()
        return value

    def step(self, action: object) -> object:
        payload = {"command": "step", "args": {"action": self._to_jsonable(action)}}
        self._socket.send(json.dumps(payload))

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
