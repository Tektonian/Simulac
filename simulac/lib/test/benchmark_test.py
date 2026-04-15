from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from websockets.sync.client import connect

from simulac.base.error.error import SimulacBaseError
from simulac.lib.gym_style import get_env_list, init_bench
from simulac.lib.gym_style.gym_style_environment import (
    BenchmarkEnvironment,
    GymEnvResetReturnType,
    GymEnvStepReturnType,
)


BENCHMARK_CONFIGS = {
    "Tektonian/Libero": {
        "benchmark_specific": {
            "control_mode": "OSC_POSE",
        },
        "empty_action": [0.0] * 7,
    },
    "Tektonian/Calvin": {
        "benchmark_specific": {},
        "empty_action": [0.0] * 7,
    },
    "Tektonian/BiGym": {
        "benchmark_specific": {},
        "empty_action": [0.0] * 15,
    },
    "Tektonian/Robomimic": {
        "benchmark_specific": {},
        "empty_action": [0.0] * 7,
    },
    "Tektonian/Metaworld": {
        "benchmark_specific": {},
        "empty_action": [0.0] * 4,
    },
}
IMAGE_RELIABILITY_STEPS = 5
ENV_SAMPLE_SIZE_PER_BENCHMARK = 1
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def _benchmark_result_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "trash" / "benchmark_test-result"


def _test_log_dir() -> Path:
    return Path(__file__).resolve().parent / "logs"


def _test_log_path() -> Path:
    return _test_log_dir() / f"benchmark_test-{RUN_TIMESTAMP}.txt"


def _append_test_log(message: str) -> None:
    log_dir = _test_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    with _test_log_path().open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def _benchmark_slug(benchmark_id: str) -> str:
    return benchmark_id.split("/", maxsplit=1)[1].lower()


def _list_benchmark_envs(benchmark_id: str) -> list[str]:
    envs = get_env_list(benchmark_id)
    if not envs:
        raise AssertionError(f"{benchmark_id} returned no envs.")
    return envs[:ENV_SAMPLE_SIZE_PER_BENCHMARK]


def _log_task_result(
    benchmark_id: str,
    test_name: str,
    env_id: str,
    success: bool,
    *,
    detail: str = "",
) -> None:
    status = "success" if success else "failure"
    message = f"[{status}] benchmark={benchmark_id} test={test_name} env={env_id}"
    if detail:
        message += f" detail={detail}"
    print(message)
    _append_test_log(message)


def _create_benchmark_env(benchmark_id: str, env_id: str) -> BenchmarkEnvironment:
    config = BENCHMARK_CONFIGS[benchmark_id]
    return init_bench(
        benchmark_id,
        env_id,
        0,
        config["benchmark_specific"],
    )


def _empty_action_for(benchmark_id: str, *, env_id: str | None = None) -> list[float]:
    if env_id == "ToolHang":
        return [0.0] * 14

    return list(BENCHMARK_CONFIGS[benchmark_id]["empty_action"])


def _image_from_pixels(
    pixels: Sequence[Sequence[Any]],
    *,
    image_name: str,
) -> Image.Image:
    height = len(pixels)
    if height == 0:
        raise ValueError(f"{image_name} has no rows.")

    width = len(pixels[0])
    if width == 0:
        raise ValueError(f"{image_name} has no columns.")

    first_value = pixels[0][0]
    if isinstance(first_value, Sequence) and not isinstance(
        first_value, (str, bytes, bytearray)
    ):
        flat_pixels = bytearray()
        for row_index, row in enumerate(pixels):
            if len(row) != width:
                raise ValueError(
                    f"{image_name} row {row_index} has width {len(row)}; expected {width}."
                )

            for col_index, pixel in enumerate(row):
                if len(pixel) != 3:
                    raise ValueError(
                        f"{image_name} pixel ({row_index}, {col_index}) does not have 3 channels."
                    )

                for channel in pixel:
                    value = int(channel)
                    if not 0 <= value <= 255:
                        raise ValueError(
                            f"{image_name} pixel ({row_index}, {col_index}) has channel value {value}."
                        )
                    flat_pixels.append(value)

        return Image.frombytes("RGB", (width, height), bytes(flat_pixels))

    min_value = float(first_value)
    max_value = min_value
    for row_index, row in enumerate(pixels):
        if len(row) != width:
            raise ValueError(
                f"{image_name} row {row_index} has width {len(row)}; expected {width}."
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


def _assert_reset_result_schema(
    reset_result: GymEnvResetReturnType,
    *,
    benchmark_id: str,
    test_name: str,
    env_id: str,
) -> None:
    obs, info = reset_result
    assert isinstance(obs, dict), (
        f"{benchmark_id} {test_name} {env_id}: reset obs must be dict, "
        f"got {type(obs).__name__}"
    )
    assert isinstance(info, dict), (
        f"{benchmark_id} {test_name} {env_id}: reset info must be dict, "
        f"got {type(info).__name__}"
    )


def _assert_step_result_schema(
    step_result: GymEnvStepReturnType,
    *,
    benchmark_id: str,
    test_name: str,
    env_id: str,
    step_index: int,
) -> dict[str, Image.Image]:
    obs, reward, done, info = step_result
    assert isinstance(obs, dict), (
        f"{benchmark_id} {test_name} {env_id} step={step_index}: "
        f"obs must be dict, got {type(obs).__name__}"
    )
    assert isinstance(reward, (int, float)), (
        f"{benchmark_id} {test_name} {env_id} step={step_index}: "
        f"reward must be numeric, got {type(reward).__name__}"
    )
    assert isinstance(done, bool), (
        f"{benchmark_id} {test_name} {env_id} step={step_index}: "
        f"done must be bool, got {type(done).__name__}"
    )
    assert isinstance(info, dict), (
        f"{benchmark_id} {test_name} {env_id} step={step_index}: "
        f"info must be dict, got {type(info).__name__}"
    )

    images = obs.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError(
            f"{benchmark_id} {test_name} {env_id} step={step_index}: "
            "obs.images is missing or empty."
        )

    rendered_images: dict[str, Image.Image] = {}
    for camera_name, pixels in images.items():
        rendered_images[camera_name] = _image_from_pixels(
            pixels,
            image_name=f"{benchmark_id}/{env_id}:step{step_index}:{camera_name}",
        )

    return rendered_images


def _save_step_images(
    benchmark_id: str,
    env_id: str,
    step_index: int,
    images: dict[str, Image.Image],
) -> list[Path]:
    output_dir = _benchmark_result_dir()
    task_dir = output_dir / _benchmark_slug(benchmark_id) / env_id / f"step_{step_index:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for camera_name, image in images.items():
        output_path = task_dir / f"{camera_name}.png"
        image.save(output_path)
        output_paths.append(output_path)

    return output_paths


def _raise_for_remote_error_payload(payload: object) -> None:
    if isinstance(payload, dict) and any(
        key in payload for key in ("error", "message", "detail")
    ):
        message = str(payload.get("message") or payload.get("detail") or payload.get("error"))
        raise SimulacBaseError(message, context=payload)

    raise AssertionError(f"Expected remote error payload, got: {payload!r}")


def _assert_invalid_remote_command_raises(
    env: BenchmarkEnvironment,
    command: str,
    **args: Any,
) -> None:
    socket = env._ensure_connected()
    try:
        env._send_command(socket, command, **args)
        payload = env._receive_packed_message(socket)
        _raise_for_remote_error_payload(payload)
    except Exception:
        return

    pytest.fail(
        f"Expected remote command {command!r} to fail for args={args!r}, but it succeeded."
    )


def _assert_invalid_build_env_raises(benchmark_id: str, env_id: str) -> None:
    env = _create_benchmark_env(benchmark_id, env_id)
    try:
        env._create_ticket()
        socket = connect(
            env._build_ws_url(),
            open_timeout=10,
            ping_interval=5,
            ping_timeout=10,
        )
        try:
            env._send_command(
                socket,
                "build_env",
                env_id=env_id,
                seed=0,
                unexpected_arg=True,
                **BENCHMARK_CONFIGS[benchmark_id]["benchmark_specific"],
            )
            payload = env._receive_packed_message(socket)
            _raise_for_remote_error_payload(payload)
        except Exception:
            return
        finally:
            socket.close()

        pytest.fail(
            "Expected remote command 'build_env' to fail for malformed args, "
            "but it succeeded."
        )
    finally:
        env.close()


def _run_error_handling_checks(benchmark_id: str, env_id: str) -> None:
    _assert_invalid_build_env_raises(benchmark_id, env_id)

    env = _create_benchmark_env(benchmark_id, env_id)
    try:
        _assert_invalid_remote_command_raises(
            env,
            "reset",
            seed=0,
            unexpected_arg=True,
        )
        _assert_invalid_remote_command_raises(
            env,
            "step",
            action=_empty_action_for(benchmark_id, env_id=env_id),
            unexpected_arg=True,
        )
        _assert_invalid_remote_command_raises(
            env,
            "close",
            unexpected_arg=True,
        )
        env.close()
        env.close()
    finally:
        env.close()


def _run_seed_reproducibility_checks(benchmark_id: str, env_id: str) -> None:
    env = _create_benchmark_env(benchmark_id, env_id)
    try:
        first_reset = env.reset(0)
        _assert_reset_result_schema(
            first_reset,
            benchmark_id=benchmark_id,
            test_name="test_benchmark_seed_reproducibility",
            env_id=env_id,
        )
        second_reset = env.reset(0)
        _assert_reset_result_schema(
            second_reset,
            benchmark_id=benchmark_id,
            test_name="test_benchmark_seed_reproducibility",
            env_id=env_id,
        )
        assert second_reset == first_reset
    finally:
        env.close()


def _run_image_rendering_checks(benchmark_id: str, env_id: str) -> None:
    env = _create_benchmark_env(benchmark_id, env_id)
    try:
        reset_result = env.reset(0)
        _assert_reset_result_schema(
            reset_result,
            benchmark_id=benchmark_id,
            test_name="test_benchmark_image_rendering_reliability",
            env_id=env_id,
        )

        for step_index in range(IMAGE_RELIABILITY_STEPS):
            step_result = env.step(_empty_action_for(benchmark_id, env_id=env_id))
            rendered_images = _assert_step_result_schema(
                step_result,
                benchmark_id=benchmark_id,
                test_name="test_benchmark_image_rendering_reliability",
                env_id=env_id,
                step_index=step_index,
            )
            _save_step_images(
                benchmark_id,
                env_id,
                step_index,
                rendered_images,
            )
    finally:
        env.close()


def _run_benchmark_env_suite(
    benchmark_id: str,
    test_name: str,
    env_check: Any,
) -> None:
    failures: list[str] = []

    for env_id in _list_benchmark_envs(benchmark_id):
        try:
            env_check(benchmark_id, env_id)
            _log_task_result(benchmark_id, test_name, env_id, True)
        except Exception as exc:
            failures.append(f"{env_id}: {exc!r}")
            _log_task_result(
                benchmark_id,
                test_name,
                env_id,
                False,
                detail=repr(exc),
            )

    assert not failures, (
        f"{benchmark_id} failed {test_name} on {len(failures)} env(s):\n"
        + "\n".join(failures)
    )


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", list(BENCHMARK_CONFIGS))
def test_benchmark_error_handling_feature(benchmark_id: str) -> None:
    _run_benchmark_env_suite(
        benchmark_id,
        "test_benchmark_error_handling_feature",
        _run_error_handling_checks,
    )


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", list(BENCHMARK_CONFIGS))
def test_benchmark_seed_reproducibility(benchmark_id: str) -> None:
    _run_benchmark_env_suite(
        benchmark_id,
        "test_benchmark_seed_reproducibility",
        _run_seed_reproducibility_checks,
    )


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", list(BENCHMARK_CONFIGS))
def test_benchmark_image_rendering_reliability(benchmark_id: str) -> None:
    _run_benchmark_env_suite(
        benchmark_id,
        "test_benchmark_image_rendering_reliability",
        _run_image_rendering_checks,
    )
