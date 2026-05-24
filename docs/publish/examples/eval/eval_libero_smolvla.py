from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any


import collections
import math
import cv2
import numpy as np
import torch
import tqdm
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)

import lerobot.policies.smolvla.processor_smolvla

from simulac.gym_style import get_env_list, init_bench, make_vec


LOG_DIR = Path(__file__).resolve().parent / ".logs"
VIDEO_FPS = 10

BENCHMARK_SLUG = "libero"
MODEL_SLUG = "smolvla"
BENCHMARK_ID = "Tektonian/Libero"
MODEL_ID = "lerobot/smolvla_libero"

VIDEO_CAMERA = "cam_0_rgb"
ACTION_DIM = 7
REPLAN_STEPS = 10


TASK_HORIZONS = {
    "libero_spatial": 280,   # Longest demo: 193 steps
    "libero_object": 280,    # Longest demo: 254 steps
    "libero_goal": 300,      # Longest demo: 270 steps
    "libero_10": 520,        # Longest demo: 505 steps
    "libero_90": 400,        # Longest demo: 373 steps
}


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate {MODEL_ID} on {BENCHMARK_ID} with Simulac."
        )
    )
    parser.add_argument("--episodes-per-env", type=int, default=1)
    parser.add_argument(
        "--max-vec-envs",
        type=int,
        default=2,
        help="Number of seeded env copies to run in each vectorized batch.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Maximum steps per episode. Use 0 to use benchmark task horizon when available.",
    )
    parser.add_argument("--env-limit", type=int, default=None)
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    
    return parser.parse_args()


def configure_logging() -> tuple[Path, Path, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{BENCHMARK_SLUG}_{MODEL_SLUG}_{run_id}"
    log_path = LOG_DIR / f"{prefix}.log"
    result_path = LOG_DIR / f"{prefix}_results.jsonl"
    video_dir = LOG_DIR / f"{prefix}_videos"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("Benchmark: %s", BENCHMARK_ID)
    logging.info("Model: %s", MODEL_ID)
    logging.info("Log file: %s", log_path)
    logging.info("Result file: %s", result_path)
    return log_path, result_path, video_dir


def load_policy_bundle(
    model_id: str,
    device: str,
) -> dict[str, Any]:

    cli_overrides = [f"--device={device}"]
    policy = SmolVLAPolicy.from_pretrained(model_id, cli_overrides=cli_overrides)
    n_params = sum(p.numel() for p in policy.parameters())
    logging.info("Loaded policy from %s with %s parameters", model_id, n_params)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_id,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    state_feature = policy.config.input_features.get("observation.state")
    state_dim = int(state_feature.shape[0]) if state_feature is not None else 0
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        model_id,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "state_dim": state_dim,
    }


def observation_frame(obs: dict, camera_key: str = VIDEO_CAMERA) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(obs["images"][camera_key], dtype=np.uint8))


def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def to_policy_obs(
    obs: dict[str, Any],
    info: dict[str, Any]
) -> dict[str, Any]:
    
    states = obs["states"]
    eef_pos = np.asarray(states["robot_0_eef_pos"], dtype=np.float32)
    eef_euler = _quat_to_euler_xyz(np.asarray(states["robot_0_eef_quat"], dtype=np.float32))
    gripper_qpos = np.asarray(states["robot_0_gripper_qpos"], dtype=np.float32)
    state = torch.from_numpy(np.concatenate([eef_pos, eef_euler, gripper_qpos]))
    
    images = {}
    for key, value in obs["images"].items():
        np_frame = np.asarray(value, dtype=np.uint8)
        images[key] = torch.from_numpy(np_frame).permute(2, 0, 1).float().div(255.0)

    task_description = info.get("task_description")


    policy_obs: dict[str, Any] = {
        "observation.state": state,
        "observation.images.image": images["cam_0_rgb"],
        "observation.images.image2": images["cam_1_rgb"],
        "task": task_description,
    }
    return policy_obs


def infer_action_batch(
    policy_bundle: dict[str, Any],
    policy_obs_list: list[dict[str, Any]],
) -> np.ndarray:
    """Call the SmolVLA policy once for a batch of transformed observations."""

    preprocessor_input = {}

    for key in set(policy_obs_list[0]):
        values = [item.get(key) for item in policy_obs_list]
        
        if all(isinstance(v, torch.Tensor) for v in values):
            preprocessor_input[key] = torch.stack(values, dim=0)
        else:
            preprocessor_input[key] = values

    with torch.no_grad():
        processed = policy_bundle["preprocessor"](preprocessor_input)
        raw_action = policy_bundle["policy"].predict_action_chunk(processed) # B X T X D

    return policy_bundle["postprocessor"](raw_action).cpu().numpy()


def _success_from_info(info: dict[str, Any], done: bool) -> bool:
    for key in ("success", "is_success", "task_success"):
        value = info.get(key)
        if isinstance(value, dict):
            return any(bool(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(bool(item) for item in value)
        if value is not None:
            return bool(value)
    return bool(done and info.get("terminated", False))


def _safe_info(info: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in (
        "task_description",
        "success",
        "is_success",
        "task_success",
        "terminated",
        "truncated",
        "task_progress",
    ):
        if key in info:
            keep[key] = info[key]
    if "obs_meta" in info:
        keep["obs_meta_keys"] = sorted(info["obs_meta"].keys())
    return keep


def run_episode_batch(
    policy_bundle: dict[str, Any],
    env_id: str,
    seeds: list[int],
    horizon: int,
    save_video: bool,
    video_dir: Path,
) -> list[dict[str, Any]]:
    if not seeds:
        return []

    batch_start_time = time.monotonic()
    batch_size = len(seeds)
    vec_env = make_vec([init_bench(BENCHMARK_ID, env_id) for _ in seeds])
    policy_bundle["policy"].reset()
    writer_list: list[cv2.VideoWriter | None] = [None] * batch_size
    success_list = [False for _ in range(batch_size)]
    done_list = [False for _ in range(batch_size)]
    steps_list = [0 for _ in range(batch_size)]
    final_info_list: list[dict[str, Any]] = []

    try:
        reset_results = vec_env.reset(seeds)
        action_plans = [collections.deque() for _ in seeds]
        obs_list = [obs for obs, _ in reset_results]
        info_list = [info for _, info in reset_results]
        prompt_list = [info.get("task_description") for info in info_list]
        final_info_list = list(info_list)

        if save_video:
            for index, (obs, seed) in enumerate(zip(obs_list, seeds)):
                frame = observation_frame(obs)
                video_path = (
                    video_dir
                    / f"{BENCHMARK_SLUG}_{env_id.replace('/', '_')}_seed_{seed:04d}.mp4"
                )

                video_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (width, height))
            
                writer_list[index] = writer
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        for step in tqdm.trange(horizon, desc=f"{env_id} seeds {seeds[0]}-{seeds[-1]}", leave=False):
            active = [i for i, done in enumerate(done_list) if not done]
            if not active:
                break

            replan = [i for i in active if not action_plans[i]]
            if replan:
                batch_obs = [
                    to_policy_obs(obs_list[index], info_list[index])
                    for index in replan
                ]
                batch_action = infer_action_batch(policy_bundle, batch_obs)


                for index, chunk in zip(replan, batch_action):
                    action_plans[index].extend(chunk[:REPLAN_STEPS])
            
            actions = [
                np.zeros(ACTION_DIM, dtype=np.float32).tolist()
                if done
                else action_plans[i].popleft().tolist()
                for i, done in enumerate(done_list)
            ]
            step_results = vec_env.step(actions)

            for index, (obs, _, done, info) in enumerate(step_results):
                if done_list[index]:
                    continue
                obs_list[index] = obs
                success_list[index] = info['success']
                done_list[index] = bool(done or success_list[index])
                final_info_list[index] = info
                steps_list[index] = step + 1

                if writer_list[index] is not None:
                    frame = observation_frame(obs)
                    writer_list[index].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if done_list[index] and writer_list[index] is not None:
                    writer_list[index].release()

            if step % 50 == 0:
                elapsed = time.monotonic() - batch_start_time
                step_rate = (step + 1) / elapsed if elapsed else 0.0
                eta = (horizon - step - 1) / step_rate if step_rate else 0.0
                logging.info(
                    "%s batch step %s/%s active=%s/%s elapsed=%s eta=%s",
                    env_id,
                    step + 1,
                    horizon,
                    active,
                    batch_size,
                    format_seconds(elapsed),
                    format_seconds(eta),
                )

        elapsed = time.monotonic() - batch_start_time
        logging.info(
            "%s batch finished: seeds=%s elapsed=%s",
            env_id,
            seeds,
            format_seconds(elapsed),
        )

        return [
            {
                "benchmark_id": BENCHMARK_ID,
                "benchmark_slug": BENCHMARK_SLUG,
                "model_id": MODEL_ID,
                "model_slug": MODEL_SLUG,
                "env_id": env_id,
                "seed": seed,
                "prompt": prompt_list[index],
                "steps": steps_list[index],
                "horizon": horizon,
                "done": bool(done_list[index]),
                "success": bool(success_list[index]),
                "elapsed_sec": round(elapsed, 3),
                "final_info": _safe_info(final_info_list[index]),
            }
            for index, seed in enumerate(seeds)
        ]
    finally:
        for writer in writer_list:
            if writer is not None and writer.isOpened():
                writer.release()
        vec_env.close()


def evaluate_env(
    policy_bundle: dict[str, Any],
    env_id: str,
    episodes_per_env: int,
    horizon: int,
    max_vec_envs: int,
    result_path: Path,
    save_video: bool,
    video_dir: Path,
) -> list[dict[str, Any]]:
    env_start = time.monotonic()
    results = []
    suite_id = Path(env_id).parent.name
    if horizon == 0:
        horizon = int(TASK_HORIZONS.get(suite_id, 1))
    
    for start in range(0, episodes_per_env, max_vec_envs):
        seeds = list(range(start, min(start + max_vec_envs, episodes_per_env)))
        batch_horizon = max(1, horizon)
        batch_results = run_episode_batch(
            policy_bundle=policy_bundle,
            env_id=env_id,
            seeds=seeds,
            horizon=batch_horizon,
            save_video=save_video,
            video_dir=video_dir,
        )
        results.extend(batch_results)

        for result in batch_results:
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            logging.info(
                "%s seed=%s steps=%s success=%s",
                env_id,
                result["seed"],
                result["steps"],
                result["success"],
            )

        elapsed = time.monotonic() - env_start
        finished = len(results)
        eta = elapsed * (episodes_per_env - finished) / finished if finished else 0.0
        logging.info(
            "%s progress: %s/%s successes=%s elapsed=%s eta=%s",
            env_id,
            finished,
            episodes_per_env,
            sum(1 for item in results if item["success"]),
            format_seconds(elapsed),
            format_seconds(eta),
        )

    return results


def _selected_envs(env_id: str | None, env_limit: int | None) -> list[str]:
    envs = get_env_list(BENCHMARK_ID)
    if env_id:
        if env_id not in envs:
            matches = [item for item in envs if env_id in item]
            if not matches:
                raise ValueError(f"env-id {env_id!r} not found in {BENCHMARK_ID}")
            envs = matches
        else:
            envs = [env_id]
    if env_limit is not None:
        envs = envs[:env_limit]
    return envs


def main() -> None:
    args = parse_args()
    _, result_path, video_dir = configure_logging()

    if args.max_vec_envs < 1:
        raise ValueError("--max-vec-envs must be at least 1")

    policy_bundle = load_policy_bundle(
        MODEL_ID,
        device=args.device
    )

    envs = _selected_envs(args.env_id, args.env_limit)
    logging.info("Selected %s env(s): %s", len(envs), envs[:5])

    all_results: list[dict[str, Any]] = []
    for env_id in envs:
        all_results.extend(
            evaluate_env(
                policy_bundle=policy_bundle,
                env_id=env_id,
                episodes_per_env=args.episodes_per_env,
                horizon=args.horizon,
                max_vec_envs=args.max_vec_envs,
                result_path=result_path,
                save_video=args.save_video,
                video_dir=video_dir,
            )
        )

    successes = sum(1 for item in all_results if item["success"])
    logging.info(
        "Finished %s episodes. success_rate=%.4f result_path=%s",
        len(all_results),
        successes / len(all_results) if all_results else 0.0,
        result_path,
    )


if __name__ == "__main__":
    main()
