from abc import ABC
import random
import time

from tt.lib.gym_style import init_bench

import os
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import numpy as np
import time
import json
import imageio
import cv2

def test_result_type():
    device = torch.device("mps")  # Set device to Apple Silicon GPU
    # Load SmolVLA policy from HuggingFace ------------------
    MODEL_ID = "HuggingFaceVLA/smolvla_libero"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Avoid tokenizer parallelism warning
    patch_draccus_non_callable_types()

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()  # Load and prepare policy
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    # -------------------------------------------------------
    # NOTE: Benchmark 특성상 task_name을 하나하나 쓰는 것은 귀찮; "libero_10" 정도로 묶어서 쓰는 방법도 고려
    # NOTE: Libero 안에서 libero_10 : task_suite; 하부 작업들이 그밑으로 정의되는 idx
    # NOTE: camera resolution은 고해상도로 해두고 필요에 따라 downsample하는 방식이 좋을듯 (ex: 1280x720 -> 640x360) -> SmolVLA는 224x224로 resize해서 쓰도록 preprocessor에서 처리
    task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] #, "libero_90"]
    suite_task_counts = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    benchmark_start_time = time.perf_counter()
    all_task_success_rates = []
    N_CHUNK = 10
    for task_suite_name in task_suites:
        env = init_bench(
            "Tektonian/Libero",
            task_suite_name+'/...',
            0,
            benchmark_specific={"control_mode": "ee_pose"},
        )

        max_episodes = 50  # Number of episodes to run per task_id
        task_success_rates = []
        for task_id in range(suite_task_counts[task_suite_name]):
            env.benchmark_specific["task_id"] = task_id  # Update task_id for each episode if needed
            task_successes = 0
            for episode in range(max_episodes):
                obs, info = env.reset(seed=episode)  # Reset environment for new episode
                lang = info['task_description']
                frames = []
                start_time = time.perf_counter()
                steps = 0
                done = False
                episode_success = False
                # for _ in range(2):  # Limit to 100 steps per episode to avoid infinite loops
                while not done:
                    # Preprocess observation for the policy
                    # import ipdb; ipdb.set_trace()
                    batch_input = libero_to_smolvla_obs(obs, lang, device)
                    batch = preprocess(batch_input)


                    # Get action from policy
                    with torch.inference_mode():
                        if N_CHUNK == 1:
                            pred_action = policy.select_action(batch)
                        else:
                            if steps % N_CHUNK == 0:
                                count = 0
                                pred_action_chunk = policy.predict_action_chunk(batch)
                                pred_action = pred_action_chunk[:,count]
                            else:
                                count += 1
                                pred_action = pred_action_chunk[:,count]
                        
                    
                    # Save the agentview image for GIF
                    agentview_img = obs['agentview_image']
                    # Convert to uint8 if needed
                    if agentview_img.dtype != np.uint8:
                        agentview_img = (agentview_img * 255).astype(np.uint8) if agentview_img.max() <= 1.0 else agentview_img.astype(np.uint8)
                    frames.append(agentview_img)

                    # Postprocess action and step environment
                    action = postprocess(pred_action)
                    obs, reward, done, info = env.step(action)
                    episode_success = episode_success or info.get("success", False)  # Track success

                    steps += 1
                    print(f"{task_suite_name} Episode {episode + 1} Step {steps} done: {done} info: {info}")

                task_successes += int(episode_success)  # Increment successes if episode was successful

                end_time = time.perf_counter()

                gif_path = f"tests/videos/{task_suite_name}_task_{task_id}_ep_{episode}_{episode_success}.gif"
                # turn 180 degree for better visualization
                frames = [np.rot90(frame, k=2) for frame in frames]
                # add lang info on the top of each frame
                frames = [np.pad(frame, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=255) for frame in frames]
                # Add lang info on the top of each frame with a smaller font, split into two lines if too long
                max_line_length = 50
                lang_lines = [lang[i:i+max_line_length] for i in range(0, len(lang), max_line_length)]
                for i, frame in enumerate(frames):
                    for j, line in enumerate(lang_lines[:2]):  # Only two lines
                        y = 9 + j * 9  # 18 pixels apart
                        cv2.putText(
                            frame,
                            line,
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.25,  # smaller font scale
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA
                        )
                
                imageio.mimsave(gif_path, frames, fps=10)
                print(f"Saved agentview GIF to {gif_path}")

            # Calculate and print per-task_id success rate
            success_rate = task_successes / max_episodes
            task_success_rates.append(success_rate)
            print(f"{task_suite_name} task_id={task_id} success rate: {success_rate:.2%} ({task_successes}/{max_episodes})")

        avg_success_rate = sum(task_success_rates) / len(task_success_rates)
        all_task_success_rates.extend(task_success_rates)
        result_path = f"tests/smolvla_{task_suite_name.replace('/', '_')}_results.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"task_suite_name: {task_suite_name}\n")
            f.write(f"episodes_per_task_id: {max_episodes}\n")
            for task_id, rate in enumerate(task_success_rates):
                f.write(f"task_id {task_id}: {rate:.6f}\n")
            f.write(f"average_success_rate: {avg_success_rate:.6f}\n")

        print(f"Saved results to {result_path}")
        env.close()  # Clean up environment
        
    elapsed = time.perf_counter() - benchmark_start_time
    overall_success_rate = sum(all_task_success_rates) / len(all_task_success_rates)
    print(
        f"elapsed: {elapsed:.3f}s, success rate: {overall_success_rate:.2%}"
    )  # about 3~4 seconds vcpu: 1~1.2 vcpu ram: 3.4GB / 2 instances -> 1.6 vcpu  same ram / 3 instances -> crash
    # 100 steps, in ours: elapsed: 19.06359601020813
    # 100 steps, in docker only: elapsed: 16.273049116134644
    # TODO: step time이 너무 긴데, 원인 분석 필요 (1) Tektonian API 통신 오버헤드? (2) Libero 시뮬레이터 자체의 느린 시뮬레이션 속도? (3) 기타 원인? -> step time breakdown 분석 필요


    assert 1 == True

    return

    init_bench(
        "Tektonian/Bigym",
        "MovePlates",
        0,
        env_specific={"floating_base": True, "control_frequency": 50},
    )

    # "assembly-v3","basketball-v3","bin-picking-v3","box-close-v3","button-press-topdown-v3","button-press-topdown-wall-v3","button-press-v3","button-press-wall-v3","coffee-button-v3","coffee-pull-v3","coffee-push-v3","dial-turn-v3","disassemble-v3","door-close-v3","door-lock-v3","door-open-v3","door-unlock-v3","hand-insert-v3","drawer-close-v3","drawer-open-v3","faucet-open-v3","faucet-close-v3","hammer-v3","handle-press-side-v3","handle-press-v3","handle-pull-side-v3","handle-pull-v3","lever-pull-v3","pick-place-wall-v3","pick-out-of-hole-v3","pick-place-v3","plate-slide-v3","plate-slide-side-v3","plate-slide-back-v3","plate-slide-back-side-v3","peg-insert-side-v3","peg-unplug-side-v3","soccer-v3","stick-push-v3","stick-pull-v3","push-v3","push-wall-v3","push-back-v3","reach-v3","reach-wall-v3","shelf-place-v3","sweep-into-v3","sweep-v3","window-open-v3","window-close-v3",
    # camera_name = '' # one of: ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']
    init_bench("Tektonian/Metaworld", "assembly-v3", 0, env_specific={})

    init_bench("Tektonian/Calvin", "A")

    init_bench(
        "Tektonian/Robosuite",
        "assembly-v3",
        0,
        env_specific={
            """
            Robot, GripperTypes는 default로 설정되도록?
            """
            "robots": [],  # ["Sawyer", "Panda"] https://github.com/ARISE-Initiative/robosuite/blob/6c10ef24a4bb52f59199976125060ce793470e6e/robosuite/robots/__init__.py#L15
            "gripper_types": ["PandaGripper", "RethinkGripper"],
            "env_configuration": "opposed",  # "opposed" | "parallel"
        },
    )


def patch_draccus_non_callable_types() -> None:
    """
    Python 3.14에서 argparse가 non-callable `type=`(예: Dict[...] | None)을
    엄격히 거부해서 draccus config 파싱이 깨지는 문제를 우회한다.
    """
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

def libero_to_smolvla_obs(obs, lang, device) -> dict:
    # ref: https://huggingface.co/docs/lerobot/en/env_processor
    # Extract components similar to _process_observation
    eef_pos = torch.tensor(obs['robot0_eef_pos'], device=device, dtype=torch.float32)  # 3D
    quat = torch.tensor(obs['robot0_eef_quat'], device=device, dtype=torch.float32)    # 4D
    # Convert quaternion to axis-angle
    def quat2euler(q):
        # q: torch tensor of shape (4,)
        w, x, y, z = q

        # roll (x-axis rotation)
        t0 = 2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = 2 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)

        # yaw (z-axis rotation)
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)

        return torch.stack([roll, pitch, yaw])

    eef_axisangle = quat2euler(quat)  # 3D
    gripper = torch.tensor(obs['robot0_gripper_qpos'], device=device, dtype=torch.float32)  # 2D
    state = torch.cat([eef_pos, eef_axisangle, gripper], dim=-1)  # 8D
    

    def normalize_image(img):
        img = torch.tensor(img, device=device, dtype=torch.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:  # HWC
            img = img.permute(2, 0, 1)  # CHW
        return img
    agentview_img = normalize_image(obs['agentview_image'])
    eye_in_hand_img = normalize_image(obs['robot0_eye_in_hand_image'])

    # import ipdb; ipdb.set_trace()
    return {
        "observation.images.image": agentview_img,
        "observation.images.image2": eye_in_hand_img,
        "observation.state": state,
        "task": lang,
    }
