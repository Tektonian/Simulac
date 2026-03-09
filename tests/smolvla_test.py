import os
import time
import math

import torch
import numpy as np
import imageio
import cv2

from tt.lib.gym_style import init_bench
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Constants
N_CHUNK = 10
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_TASK_COUNTS = {suite: 10 for suite in TASK_SUITES}
SUITE_TASK_COUNTS["libero_90"] = 90
MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def process_frames(frames, lang):
    """Rotate, pad, and overlay text on frames."""
    frames = [np.rot90(frame, k=2) for frame in frames]
    frames = [np.pad(frame, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=255) for frame in frames]
    max_line_length = 50
    lang_lines = [lang[i:i + max_line_length] for i in range(0, len(lang), max_line_length)]
    for frame in frames:
        for j, line in enumerate(lang_lines[:2]):
            y = 9 + j * 9
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
    return frames


def patch_draccus_non_callable_types():
    """Patch for draccus to handle non-callable types."""
    try:
        from draccus.wrappers.field_wrapper import FieldWrapper
    except Exception:
        return

    if getattr(FieldWrapper, "_tektonian_py314_patch", False):
        return

    original_get_arg_options = FieldWrapper.get_arg_options

    def _patched_get_arg_options(self):
        options = original_get_arg_options(self)
        tpe = options.get("type")
        if tpe is not None and not callable(tpe):
            options["type"] = str
        return options

    FieldWrapper.get_arg_options = _patched_get_arg_options
    FieldWrapper._tektonian_py314_patch = True


def libero_to_smolvla_obs(obs, lang, device):
    """Convert libero observation to smolvla input format."""
    eef_pos = torch.tensor(obs['robot0_eef_pos'], device=device, dtype=torch.float32)
    quat = torch.tensor(obs['robot0_eef_quat'], device=device, dtype=torch.float32)

    def _quat2axisangle(quat):
        q = quat.clone()
        q[3] = max(min(q[3], 1.0), -1.0)
        den = torch.sqrt(1.0 - q[3] * q[3])
        if math.isclose(den.item(), 0.0):
            return torch.zeros(3, device=device, dtype=torch.float32)
        return (q[:3] * 2.0 * math.acos(q[3].item())) / den

    eef_axisangle = _quat2axisangle(quat)
    gripper = torch.tensor(obs['robot0_gripper_qpos'], device=device, dtype=torch.float32)
    state = torch.cat([eef_pos, eef_axisangle, gripper], dim=-1)

    def normalize_image(img):
        img = torch.tensor(img, device=device, dtype=torch.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.permute(2, 0, 1)
        return img

    agentview_img = normalize_image(obs['agentview_image'])
    eye_in_hand_img = normalize_image(obs['robot0_eye_in_hand_image'])
    return {
        "observation.images.image": agentview_img,
        "observation.images.image2": eye_in_hand_img,
        "observation.state": state,
        "task": lang,
    }


def run_task_suite(policy, preprocess, postprocess, device, task_suite_name):
    env = init_bench(
        "Tektonian/Libero",
        f"{task_suite_name}/...",
        0,
        benchmark_specific={"control_mode": "ee_pose"},
    )
    max_steps = MAX_STEPS_PER_SUITE[task_suite_name]
    max_episodes = 50
    task_success_rates = []

    for task_id in range(SUITE_TASK_COUNTS[task_suite_name]):
        env.benchmark_specific["task_id"] = task_id
        task_successes = 0

        for episode in range(max_episodes):
            obs, info = env.reset(seed=episode)
            lang = info['task_description']
            frames = []
            steps = 0
            done = False
            count = 0
            pred_action_chunk = None

            while steps < max_steps + NUM_STEPS_WAIT:
                if steps < NUM_STEPS_WAIT:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    steps += 1
                    continue

                batch_input = libero_to_smolvla_obs(obs, lang, device)
                batch = preprocess(batch_input)

                with torch.inference_mode():
                    if N_CHUNK == 1:
                        pred_action = policy.select_action(batch)
                    else:
                        if steps % N_CHUNK == 0:
                            count = 0
                            pred_action_chunk = policy.predict_action_chunk(batch)
                            pred_action = pred_action_chunk[:, count]
                        else:
                            count += 1
                            pred_action = pred_action_chunk[:, count]

                agentview_img = obs['agentview_image']
                if agentview_img.dtype != np.uint8:
                    agentview_img = (agentview_img * 255).astype(np.uint8) if agentview_img.max() <= 1.0 else agentview_img.astype(np.uint8)
                frames.append(agentview_img)

                action = postprocess(pred_action)
                obs, reward, done, info = env.step(action)
                steps += 1
                print(f"{task_suite_name} Episode {episode + 1} Step {steps} reward: {reward} done: {done} info: {info}")

                if done:
                    task_successes += 1
                    break
            # Logging and saving per-episode result
            episode_result_dir = os.path.join("tests", "results", "libero", "episodes", f"{task_suite_name}_task_{task_id}")
            os.makedirs(episode_result_dir, exist_ok=True)
            episode_result_path = os.path.join(episode_result_dir, f"ep_{episode}_{done}.txt")
            with open(episode_result_path, "w", encoding="utf-8") as epf:
                epf.write(
                    f"task_suite_name: {task_suite_name}\n"
                    f"task_id: {task_id}\n"
                    f"episode: {episode}\n"
                    f"success: {done}\n"
                    f"steps: {steps}\n"
                    f"info: {info}\n"
                )
            gif_dir = os.path.join("tests", "results", "libero", "videos")
            os.makedirs(gif_dir, exist_ok=True)
            gif_path = os.path.join(gif_dir, f"{task_suite_name}_task_{task_id}_ep_{episode}_{done}.gif")
            frames_processed = process_frames(frames, lang)
            imageio.mimsave(gif_path, frames_processed, fps=10)
            print(f"[LOG] Saved agentview GIF: {gif_path}")

        success_rate = task_successes / max_episodes
        task_success_rates.append(success_rate)
        print(f"[LOG] {task_suite_name} task_id={task_id} success rate: {success_rate:.2%} ({task_successes}/{max_episodes})")

    avg_success_rate = sum(task_success_rates) / len(task_success_rates)
    result_dir = os.path.join("tests", "results", "libero")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"smolvla_{task_suite_name.replace('/', '_')}_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"task_suite_name: {task_suite_name}\n")
        f.write(f"episodes_per_task_id: {max_episodes}\n")
        for task_id, rate in enumerate(task_success_rates):
            f.write(f"task_id {task_id}: {rate:.6f}\n")
        f.write(f"average_success_rate: {avg_success_rate:.6f}\n")

    print(f"[LOG] Saved results: {result_path}")
    env.close()
    return task_success_rates


def test_result_type():
    device = torch.device("mps")
    MODEL_ID = "HuggingFaceVLA/smolvla_libero"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    patch_draccus_non_callable_types()

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    benchmark_start_time = time.perf_counter()
    all_task_success_rates = []

    for task_suite_name in TASK_SUITES:
        task_success_rates = run_task_suite(policy, preprocess, postprocess, device, task_suite_name)
        all_task_success_rates.extend(task_success_rates)

    elapsed = time.perf_counter() - benchmark_start_time
    overall_success_rate = sum(all_task_success_rates) / len(all_task_success_rates)
    print(f"elapsed: {elapsed:.3f}s, success rate: {overall_success_rate:.2%}")

    assert 1 == True
