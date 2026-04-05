from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import msgpack
import pytest
import zstd
from PIL import Image
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect


def _connect_or_skip() -> ClientConnection:
    LOCAL_SERVER_URL = "ws://localhost:3000/ws"
    return connect(
        LOCAL_SERVER_URL,
        open_timeout=3,
        close_timeout=1,
        ping_interval=None,
    )


def _send_command(
    socket: ClientConnection,
    command: str,
    /,
    **args: Any,
) -> None:
    socket.send(json.dumps({"command": command, "args": args}))


def _receive_packed_response(socket: ClientConnection) -> Any:
    payload = socket.recv(timeout=20, decode=False)
    if not isinstance(payload, (bytes, bytearray)):
        raise AssertionError(f"Expected packed bytes from websocket, got {type(payload)!r}")
    return msgpack.unpackb(zstd.decompress(payload))


def _receive_plain_response(socket: ClientConnection) -> Any:
    payload = socket.recv(timeout=10)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise AssertionError(f"Expected text response from websocket, got {type(payload)!r}")
    return json.loads(payload)


def _assert_images(images: Any) -> None:
    assert isinstance(images, Mapping), "obs.images must be a mapping"
    assert images, "obs.images must not be empty"
    
    rgb_imgs = {camera_name: pixels for camera_name, pixels in images.items() if camera_name.endswith("rgb")}
    for camera_name, pixels in rgb_imgs.items():
        assert isinstance(camera_name, str) and camera_name
        assert isinstance(pixels, Sequence) and pixels, (
            f"{camera_name} image must contain rows"
        )
        height = len(pixels)
        width = len(pixels[0])
        assert height != 3, f"{camera_name} image appears to be in CHW format, expected HWC"
        assert width > 0, f"{camera_name} image must contain columns"

        for row in pixels:
            assert isinstance(row, Sequence)
            assert len(row) == width, f"{camera_name} rows must have a consistent width"
            for pixel in row:
                assert isinstance(pixel, Sequence)
                assert len(pixel) == 3, f"{camera_name} pixels must be RGB triplets"
                for channel in pixel:
                    assert isinstance(channel, int)
                    assert 0 <= channel <= 255

        assert height > 0

    depth_imgs = {camera_name: pixels for camera_name, pixels in images.items() if camera_name.endswith("depth")}
    for camera_name, pixels in depth_imgs.items():
        assert isinstance(camera_name, str) and camera_name
        assert isinstance(pixels, Sequence) and pixels, (
            f"{camera_name} image must contain rows"
        )
        height = len(pixels)
        width = len(pixels[0])
        assert height > 0
        assert width > 0, f"{camera_name} image must contain columns"

        for row in pixels:
            assert isinstance(row, Sequence)
            assert len(row) == width, f"{camera_name} rows must have a consistent width"
            for value in row:
                assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                    f"{camera_name} depth values must be numeric scalars"
                )


def _assert_step_response(response: Any) -> dict[str, Any]:
    assert isinstance(response, dict), "step response must unpack to a dict"
    assert "obs" in response, "step response must include obs"
    assert "reward" in response, "step response must include reward"
    assert "done" in response, "step response must include done"
    assert "info" in response, "step response must include info"

    obs = response["obs"]
    assert isinstance(obs, Mapping), "obs must be a mapping"
    _assert_images(obs.get("images"))
    return response


def _benchmark_result_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "trash" / "benchmark_test-result"


def _image_from_pixels(
    pixels: Sequence[Sequence[Any]],
    *,
    image_name: str,
) -> Image.Image:
    height = len(pixels)
    if height == 0:
        raise ValueError(f"{image_name} has no rows")

    width = len(pixels[0])
    if width == 0:
        raise ValueError(f"{image_name} has no columns")

    first_value = pixels[0][0]
    if isinstance(first_value, Sequence) and not isinstance(first_value, (str, bytes, bytearray)):
        flat_pixels = bytearray()
        for row_index, row in enumerate(pixels):
            if len(row) != width:
                raise ValueError(
                    f"{image_name} row {row_index} has width {len(row)}; expected {width}"
                )
            for col_index, pixel in enumerate(row):
                if len(pixel) != 3:
                    raise ValueError(
                        f"{image_name} pixel ({row_index}, {col_index}) does not have 3 channels"
                    )
                for channel in pixel:
                    value = int(channel)
                    if not 0 <= value <= 255:
                        raise ValueError(
                            f"{image_name} pixel ({row_index}, {col_index}) has channel value {value}"
                        )
                    flat_pixels.append(value)
        return Image.frombytes("RGB", (width, height), bytes(flat_pixels))

    min_value = float(first_value)
    max_value = min_value
    for row_index, row in enumerate(pixels):
        if len(row) != width:
            raise ValueError(
                f"{image_name} row {row_index} has width {len(row)}; expected {width}"
            )
        for value in row:
            numeric_value = float(value)
            if numeric_value < min_value:
                min_value = numeric_value
            elif numeric_value > max_value:
                max_value = numeric_value

    scale = 0.0 if max_value == min_value else 255.0 / (max_value - min_value)
    flat_pixels = bytearray()
    for row in pixels:
        for value in row:
            numeric_value = float(value)
            if scale == 0.0:
                flat_pixels.append(0)
            else:
                flat_pixels.append(int((numeric_value - min_value) * scale))
    return Image.frombytes("L", (width, height), bytes(flat_pixels))


def _save_step_result(
    env_id: str,
    *,
    step_index: int,
    seed: int,
    step_result: Mapping[str, Any],
    prefix: str | None,
) -> None:
    output_dir = _benchmark_result_dir() / env_id if prefix is None else _benchmark_result_dir() / prefix / env_id
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = step_result["obs"]
    assert isinstance(obs, Mapping), "obs must be a mapping"

    images = obs.get("images")
    assert isinstance(images, Mapping), "obs.images must be a mapping"

    for camera_name, pixels in images.items():
        image = _image_from_pixels(
            pixels,
            image_name=f"{env_id}:{camera_name}",
        )
        image.save(output_dir / f"{camera_name}.png")

    json_payload = {
        "obs": {key: value for key, value in obs.items() if key != "images"},
        "reward": step_result["reward"],
        "done": step_result["done"],
        "info": step_result["info"],
    }
    with (output_dir / f"step_{step_index}_seed_{seed}.json").open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=4)


# ENV_ID = "libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate"
# EMPTY_ACTION = [0.0] * 7

CALVIN_TASK_MAP = [
    "calvin_scene_A",
    "calvin_scene_B",
    "calvin_scene_C",
    "calvin_scene_D"
]

# @pytest.mark.integration
# def test_calvin_benchmark_commands_roundtrip() -> None:
#     """Roundtrip against the docker-hosted local FastAPI benchmark server."""
#     EMPTY_ACTION = [0.0] * 7

#     for env_id in CALVIN_TASK_MAP:
#         socket = _connect_or_skip()
#         try:
#             _send_command(socket, "build_env", env_id=env_id)
#             build_response = _receive_packed_response(socket)
#             assert isinstance(build_response, dict), "build_env response must unpack to a dict"

#             _send_command(socket, "reset", seed=0)
#             reset_seed_zero = _receive_packed_response(socket)
#             assert isinstance(reset_seed_zero, dict), "reset response must unpack to a dict"

#             _send_command(socket, "step", action=EMPTY_ACTION)
#             first_step_seed_zero = _assert_step_response(_receive_packed_response(socket))
#             _save_step_result(env_id, step_index=0, seed=0, step_result=first_step_seed_zero)

#             _send_command(socket, "close")
#             try:
#                 close_response = _receive_plain_response(socket)
#             except ConnectionClosed:
#                 close_response = None

#             assert close_response is None or isinstance(close_response, dict)
#         finally:
#             socket.close()


META_WORLD_TASK_MAP = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "reach-v3",
    "push-back-v3",
    "push-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-wall-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3"
]

# @pytest.mark.integration
# def test_meta_world_benchmark_commands_roundtrip() -> None:
#     """Roundtrip against the docker-hosted local FastAPI benchmark server."""
#     EMPTY_ACTION = [0.0] * 4

#     for env_id in META_WORLD_TASK_MAP:
#         socket = _connect_or_skip()
#         try:
#             _send_command(socket, "build_env", env_id=env_id)
#             build_response = _receive_packed_response(socket)
#             assert isinstance(build_response, dict), "build_env response must unpack to a dict"

#             _send_command(socket, "reset", seed=0)
#             reset_seed_zero = _receive_packed_response(socket)
#             assert isinstance(reset_seed_zero, dict), "reset response must unpack to a dict"

#             _send_command(socket, "step", action=EMPTY_ACTION)
#             first_step_seed_zero = _assert_step_response(_receive_packed_response(socket))
#             _save_step_result(env_id, step_index=0, seed=0, step_result=first_step_seed_zero, prefix="meta_world")

#             _send_command(socket, "close")
#             try:
#                 close_response = _receive_plain_response(socket)
#             except ConnectionClosed:
#                 close_response = None

#             assert close_response is None or isinstance(close_response, dict)
#         finally:
#             socket.close()

# BIGYM_TASK_MAP = [
#     "DishwasherClose",
#     "DishwasherCloseTrays",
#     "DishwasherLoadCups",
#     "DishwasherLoadCutlery",
#     "DishwasherLoadPlates",
#     "DishwasherOpen",
#     "DishwasherOpenTrays",
#     "DishwasherUnloadCups",
#     "DishwasherUnloadCupsLong",
#     "DishwasherUnloadCutlery",
#     "DishwasherUnloadCutleryLong",
#     "DishwasherUnloadPlates",
#     "DishwasherUnloadPlatesLong",
#     "DrawerTopClose",
#     "DrawerTopOpen",
#     "DrawersAllClose",
#     "DrawersAllOpen",
#     "FlipCup",
#     "FlipCutlery",
#     "FlipSandwich",
#     "GroceriesStoreLower",
#     "GroceriesStoreUpper",
#     "MovePlate",
#     "MoveTwoPlates",
#     "PickBox",
#     "PutCups",
#     "ReachTarget",
#     "ReachTargetDual",
#     "ReachTargetSingle",
#     "RemoveSandwich",
#     "StackBlocks",
#     "StoreBox",
#     "StoreKitchenware",
#     "TakeCups",
#     "ToastSandwich",
#     "WallCupboardClose",
#     "WallCupboardOpen"
# ]

# @pytest.mark.integration
# def test_bigym_benchmark_commands_roundtrip() -> None:
#     """Roundtrip against the docker-hosted local FastAPI benchmark server."""
#     EMPTY_ACTION = [0.0] * 15

#     for env_id in BIGYM_TASK_MAP:
#         socket = _connect_or_skip()
#         try:
#             _send_command(socket, "build_env", env_id=env_id, floating_base=True, control_frequency=50)
#             build_response = _receive_packed_response(socket)
#             assert isinstance(build_response, dict), "build_env response must unpack to a dict"

#             _send_command(socket, "reset", seed=0)
#             reset_seed_zero = _receive_packed_response(socket)
#             assert isinstance(reset_seed_zero, dict), "reset response must unpack to a dict"

#             _send_command(socket, "step", action=EMPTY_ACTION)
#             first_step_seed_zero = _assert_step_response(_receive_packed_response(socket))
#             _save_step_result(env_id, step_index=0, seed=0, step_result=first_step_seed_zero, prefix="bigym")

#             _send_command(socket, "close")
#             try:
#                 close_response = _receive_plain_response(socket)
#             except ConnectionClosed:
#                 close_response = None

#             assert close_response is None or isinstance(close_response, dict)
#         finally:
#             socket.close()


ROBOMIMIC_TASK_MAP = [
    "Lift",
    "PickPlaceCan",
    "NutAssemblySquare",
    "TwoArmTransport"
]
# BUG: some images are not rendered correctly; cpu rendering would be reason
@pytest.mark.integration
def test_robomimic_benchmark_commands_roundtrip() -> None:
    """Roundtrip against the docker-hosted local FastAPI benchmark server."""
    

    for env_id in ROBOMIMIC_TASK_MAP:
        
        EMPTY_ACTION = [0.0] * 14 if env_id == "TwoArmTransport" else [0.0] * 7
        socket = _connect_or_skip()
        try:
            _send_command(socket, "build_env", env_id=env_id)
            build_response = _receive_packed_response(socket)
            assert isinstance(build_response, dict), "build_env response must unpack to a dict"

            _send_command(socket, "reset", seed=0)
            reset_seed_zero = _receive_packed_response(socket)
            assert isinstance(reset_seed_zero, dict), "reset response must unpack to a dict"

            _send_command(socket, "step", action=EMPTY_ACTION)
            first_step_seed_zero = _assert_step_response(_receive_packed_response(socket))
            _save_step_result(env_id, step_index=0, seed=0, step_result=first_step_seed_zero, prefix="robomimic")

            _send_command(socket, "close")
            try:
                close_response = _receive_plain_response(socket)
            except ConnectionClosed:
                close_response = None

            assert close_response is None or isinstance(close_response, dict)
        finally:
            socket.close()