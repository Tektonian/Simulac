from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from huggingface_hub import snapshot_download
from openpi import transforms
from openpi.models import model as openpi_model
from openpi.models import pi0_config, tokenizer
from openpi.policies import policy as openpi_policy
from openpi.shared import normalize
from openpi_client import image_tools

from simulac.gym_style import get_env_list, init_bench, make_vec

# This example runs RoboCasa episodes for openPI checkpoints on the Simulac environments.
BENCHMARK_ID = "Tektonian/RobocasaPre"
CHECKPOINT_REPO = "robocasa/robocasa365_checkpoints"
CHECKPOINT_SUBDIRS = {
    "pi0": Path("pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"),
    "pi05": Path("pi05_pretrain_human300/multitask_learning/75000"),
}

ACTION_DIM = 12
MODEL_ACTION_DIM = 32
REPLAN_STEPS = 30
VIDEO_CAMERA = "cam_0_rgb"
VIDEO_FPS = 10

LOG_DIR = Path(__file__).resolve().parent / ".logs"

STATE_KEYS = (
    "robot_0_base_to_eef_pos",
    "robot_0_base_to_eef_quat",
    "robot_0_base_pos",
    "robot_0_base_quat",
    "robot_0_gripper_qpos",
)

BENCHMARK_GROUPS = {
    "atomic_seen": ("Atomic", "Seen"),
    "composite_seen": ("Composite", "Seen"),
    "composite_unseen": ("Composite", "Unseen"),
}

# RoboCasa official task horizons for the 50 benchmark tasks.
TASK_HORIZONS = {
    "ArrangeBreadBasket": 2900,
    "ArrangeTea": 1500,
    "BreadSelection": 1300,
    "CategorizeCondiments": 1100,
    "CloseBlenderLid": 600,
    "CloseFridge": 600,
    "CloseToasterOvenDoor": 300,
    "CoffeeSetupMug": 400,
    "CuttingToolSelection": 800,
    "DeliverStraw": 1700,
    "GarnishPancake": 1800,
    "GatherTableware": 1500,
    "GetToastedBread": 2000,
    "HeatKebabSandwich": 1800,
    "KettleBoiling": 1000,
    "LoadDishwasher": 1200,
    "MakeIceLemonade": 2000,
    "NavigateKitchen": 300,
    "OpenCabinet": 700,
    "OpenDrawer": 500,
    "OpenStandMixerHead": 300,
    "PackIdenticalLunches": 2600,
    "PanTransfer": 1200,
    "PickPlaceCounterToCabinet": 500,
    "PickPlaceCounterToStove": 400,
    "PickPlaceDrawerToCounter": 500,
    "PickPlaceSinkToCounter": 600,
    "PickPlaceToasterToCounter": 400,
    "PortionHotDogs": 1500,
    "PreSoakPan": 1600,
    "PrepareCoffee": 1200,
    "RecycleBottlesByType": 1900,
    "RinseSinkBasin": 900,
    "ScrubCuttingBoard": 800,
    "SearingMeat": 2900,
    "SeparateFreezerRack": 1600,
    "SetUpCuttingStation": 1600,
    "SlideDishwasherRack": 300,
    "StackBowlsCabinet": 1400,
    "SteamInMicrowave": 1400,
    "StirVegetables": 1600,
    "StoreLeftoversInBowl": 1700,
    "TurnOffStove": 500,
    "TurnOnElectricKettle": 300,
    "TurnOnMicrowave": 300,
    "TurnOnSinkFaucet": 400,
    "WaffleReheat": 2700,
    "WashFruitColander": 2100,
    "WashLettuce": 1100,
    "WeighIngredients": 2000,
}


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenPI checkpoint on RoboCasa with Simulac."
    )
    parser.add_argument(
        "--model-name",
        type=str.lower,
        choices=sorted(CHECKPOINT_SUBDIRS),
        default="pi0",
        help="OpenPI RoboCasa checkpoint to run.",
    )
    parser.add_argument(
        "--max-vec-envs",
        type=int,
        default=2,
        help="Number of seeded env copies to run in each vectorized batch.",
    )
    parser.add_argument(
        "--episodes-per-env",
        type=int,
        default=50,
        help="Number of episodes to evaluate per RoboCasa task.",
    )
    return parser.parse_args()


def configure_logging(model_name: str) -> tuple[Path, Path, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_name.lower()
    log_path = LOG_DIR / f"robocasa_{model_slug}_benchmark_{run_id}.log"
    result_path = LOG_DIR / f"robocasa_{model_slug}_benchmark_results_{run_id}.jsonl"
    video_dir = LOG_DIR / f"robocasa_{model_slug}_benchmark_videos_{run_id}"

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
    logging.info("Log file: %s", log_path)
    logging.info("Result file: %s", result_path)

    return log_path, result_path, video_dir


def observation_frame(obs: dict, camera_key: str = VIDEO_CAMERA) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(obs["images"][camera_key], dtype=np.uint8))


def write_video_frame(writer: cv2.VideoWriter, obs: dict) -> None:
    frame = observation_frame(obs, VIDEO_CAMERA)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def robocasa_inputs(data: dict) -> dict:
    base_image = np.ascontiguousarray(
        np.asarray(data["observation/image"], dtype=np.uint8)
    )
    wrist_image = np.ascontiguousarray(
        np.asarray(data["observation/wrist_image"], dtype=np.uint8)
    )
    state = np.asarray(data["observation/state"])

    return {
        "state": np.pad(state, (0, MODEL_ACTION_DIM - state.shape[0])),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
        "prompt": data["prompt"],
    }


def robocasa_outputs(data: dict) -> dict:
    return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


def load_policy(model_name: str) -> openpi_policy.Policy:
    """Load the RoboCasa OpenPI checkpoint without importing RoboCasa training code."""
    checkpoint_subdir = CHECKPOINT_SUBDIRS[model_name.lower()]
    checkpoint_root = Path(
        snapshot_download(
            repo_id=CHECKPOINT_REPO,
            allow_patterns=[
                f"{checkpoint_subdir}/_CHECKPOINT_METADATA",
                f"{checkpoint_subdir}/assets/**",
                f"{checkpoint_subdir}/params/**",
            ],
        )
    )
    checkpoint_dir = checkpoint_root / checkpoint_subdir

    model_config = pi0_config.Pi0Config(max_token_len=96, pi05=model_name == "pi05")
    model = model_config.load(
        openpi_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    )
    norm_stats = normalize.load(checkpoint_dir / "assets")

    return openpi_policy.Policy(
        model,
        transforms=[
            robocasa_inputs,
            transforms.Normalize(norm_stats),
            transforms.InjectDefaultPrompt(None),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(model_config.max_token_len)
            ),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats),
            robocasa_outputs,
        ],
    )


def to_policy_obs(obs_list: list[dict], prompt_list: list[str]) -> list[dict]:
    """Convert Simulac RoboCasa observations into the OpenPI policy format."""
    policy_obs = []
    for obs, prompt in zip(obs_list, prompt_list):
        states = obs["states"]
        state = np.concatenate(
            [
                np.asarray(states[key], dtype=np.float32).reshape(-1)
                for key in STATE_KEYS
            ]
        )
        policy_obs.append(
            {
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        observation_frame(obs, "cam_0_rgb"), 224, 224
                    )
                ),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        observation_frame(obs, "cam_2_rgb"), 224, 224
                    )
                ),
                "observation/state": state,
                "prompt": prompt,
            }
        )
    return policy_obs


def infer_batch(policy: openpi_policy.Policy, policy_obs: list[dict]) -> np.ndarray:
    """Call the OpenPI model once for a batch of transformed observations."""
    transformed = [policy._input_transform(obs) for obs in policy_obs]
    batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *transformed)

    policy._rng, rng = jax.random.split(policy._rng)
    batch_actions = policy._sample_actions(
        rng,
        openpi_model.Observation.from_dict(batch),
        **policy._sample_kwargs,
    )

    return np.stack(
        [
            policy._output_transform(
                {
                    "state": np.asarray(batch["state"][i]),
                    "actions": np.asarray(batch_actions[i]),
                }
            )["actions"]
            for i in range(len(policy_obs))
        ]
    )


def run_episode_batch(
    policy,
    env_id: str,
    seeds: list[int],
    horizon: int,
    video_dir: Path,
) -> int:
    """Run seeded copies of the same env in parallel."""
    batch_start_time = time.monotonic()
    batch_size = len(seeds)
    vec_env = make_vec([init_bench(BENCHMARK_ID, env_id) for _ in seeds])
    video_writers: list[cv2.VideoWriter] = []

    try:
        reset_results = vec_env.reset(seeds)
        obs_list = [obs for obs, _ in reset_results]
        prompt_list = [info["task_description"] for _, info in reset_results]

        action_plans = [collections.deque() for _ in seeds]
        done_list = [False] * batch_size
        success_list = [False] * batch_size

        video_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for seed, obs in zip(seeds, obs_list):
            frame = observation_frame(obs, VIDEO_CAMERA)
            height, width = frame.shape[:2]
            video_path = video_dir / f"{env_id}_seed_{seed:04d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path), fourcc, VIDEO_FPS, (width, height)
            )
            video_writers.append(writer)
            write_video_frame(writer, obs)

        for step in tqdm.trange(
            horizon, desc=f"{env_id} seeds {seeds[0]}-{seeds[-1]}", leave=False
        ):
            active = [i for i, success in enumerate(success_list) if not success]
            if not active:
                break

            replan = [i for i in active if not action_plans[i]]
            if replan:
                chunks = infer_batch(
                    policy,
                    to_policy_obs(
                        [obs_list[i] for i in replan], [prompt_list[i] for i in replan]
                    ),
                )
                for index, chunk in zip(replan, chunks):
                    action_plans[index].extend(chunk[:REPLAN_STEPS])

            actions = [
                np.zeros(ACTION_DIM, dtype=np.float32).tolist()
                if done
                else action_plans[i].popleft().tolist()
                for i, done in enumerate(done_list)
            ]
            step_results = vec_env.step(actions)

            for i, (obs, _, done, info) in enumerate(step_results):
                if done_list[i]:
                    continue
                obs_list[i] = obs
                success_list[i] = success_list[i] or bool(info.get("success", done))
                done_list[i] = bool(done or success_list[i])
                write_video_frame(video_writers[i], obs)
                if done_list[i]:
                    video_writers[i].release()

            if step % 50 == 0:
                elapsed = time.monotonic() - batch_start_time
                step_rate = (step + 1) / elapsed if elapsed else 0.0
                eta = (horizon - step - 1) / step_rate if step_rate else 0.0
                logging.info(
                    "%s step %s/%s done=%s/%s elapsed=%s eta=%s",
                    env_id,
                    step + 1,
                    horizon,
                    sum(done_list),
                    batch_size,
                    format_seconds(elapsed),
                    format_seconds(eta),
                )

        elapsed = time.monotonic() - batch_start_time
        logging.info(
            "%s batch done: seeds=%s elapsed=%s", env_id, seeds, format_seconds(elapsed)
        )
        return sum(success_list)
    finally:
        for writer in video_writers:
            if writer.isOpened():
                writer.release()
        vec_env.close()


def evaluate_env(
    policy,
    env_path: str,
    episodes_per_env: int,
    max_vec_envs: int,
    video_dir: Path,
) -> dict:
    """Evaluate one benchmark env for EPISODES_PER_ENV episodes."""
    env_start_time = time.monotonic()
    env_id = Path(env_path).name
    horizon = int(TASK_HORIZONS[env_id] * 1.5)
    successes = 0

    for start in range(0, episodes_per_env, max_vec_envs):
        seeds = list(range(start, min(start + max_vec_envs, episodes_per_env)))
        successes += run_episode_batch(policy, env_id, seeds, horizon, video_dir)
        elapsed = time.monotonic() - env_start_time
        finished = len(seeds) + start
        eta = elapsed * (episodes_per_env - finished) / finished if finished else 0.0
        logging.info(
            "%s progress: %s/%s successes=%s elapsed=%s eta=%s",
            env_id,
            finished,
            episodes_per_env,
            successes,
            format_seconds(elapsed),
            format_seconds(eta),
        )

    return {
        "env_path": env_path,
        "env_id": env_id,
        "episodes": episodes_per_env,
        "successes": successes,
        "success_rate": successes / episodes_per_env,
    }


def main() -> None:
    args = parse_args()
    _, result_path, video_dir = configure_logging(args.model_name)

    total_start_time = time.monotonic()
    policy = load_policy(args.model_name)
    env_list = get_env_list(BENCHMARK_ID)
    results = []

    for group_name, group_key in BENCHMARK_GROUPS.items():
        env_paths = [
            env_path
            for env_path in env_list
            if tuple(env_path.split("/")[:2]) == group_key
        ]
        logging.info("Starting %s: %s envs", group_name, len(env_paths))

        for env_path in env_paths:
            result = evaluate_env(
                policy,
                env_path,
                args.episodes_per_env,
                args.max_vec_envs,
                video_dir,
            )
            result["group"] = group_name
            results.append(result)
            with result_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
            logging.info(
                "Finished %s: success_rate=%.3f total_elapsed=%s",
                env_path,
                result["success_rate"],
                format_seconds(time.monotonic() - total_start_time),
            )

    for group_name in BENCHMARK_GROUPS:
        group_results = [result for result in results if result["group"] == group_name]
        successes = sum(result["successes"] for result in group_results)
        episodes = sum(result["episodes"] for result in group_results)
        if episodes:
            logging.info(
                "%s success rate: %.4f (%s/%s)",
                group_name,
                successes / episodes,
                successes,
                episodes,
            )

    total_successes = sum(result["successes"] for result in results)
    total_episodes = sum(result["episodes"] for result in results)
    if total_episodes:
        logging.info(
            "total average success rate: %.4f (%s/%s)",
            total_successes / total_episodes,
            total_successes,
            total_episodes,
        )

    logging.info(
        "Benchmark physical time: %s",
        format_seconds(time.monotonic() - total_start_time),
    )


if __name__ == "__main__":
    main()
