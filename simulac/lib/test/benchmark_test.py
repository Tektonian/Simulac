from pathlib import Path

from pydantic.type_adapter import P
import pytest
from PIL import Image

from simulac.lib.gym_style import init_bench, make_vec


def _benchmark_result_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "trash" / "benchmark_test-result"


def _image_from_rgb_values(
    pixels: list[list[list[int]]],
    *,
    image_name: str,
) -> Image.Image:
    height = len(pixels)
    if height == 0:
        raise ValueError(f"{image_name} has no rows.")

    width = len(pixels[0])
    if width == 0:
        raise ValueError(f"{image_name} has no columns.")

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


def _save_step_images(
    task_dir: Path,
    step_result: dict,
) -> list[Path]:
    images = step_result.get("obs", {}).get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError(f"{task_dir} does not contain obs.images.")

    task_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for camera_name, pixels in images.items():
        image = _image_from_rgb_values(
            pixels,
            image_name=f"{task_dir}:{camera_name}",
        )
        output_path = task_dir / f"{camera_name}.png"
        image.save(output_path)
        output_paths.append(output_path)

    return output_paths

# test start: libero benchmark

# @pytest.mark.integration
# def test_libero_benchmark():
#     env = init_bench(
#         "Tektonian/Libero",
#         "libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
#         0,
#         {
#             "control_mode": "ee_pose",
#         },
#     )
#     EMPTY_ACTION = [0] * 7
#     obs_history = []
#     obs_history.append(env.step(EMPTY_ACTION))
#     for _ in range(4):
#         obs_history.append(env.step(EMPTY_ACTION))

#     reset = env.reset(0)
#     for i in range(len(obs_history)):
#         ret = env.step(EMPTY_ACTION)
#         assert ret == obs_history[i]

#     new_env = init_bench(
#         "Tektonian/Libero",
#         "libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
#         0,
#         {
#             "control_mode": "ee_pose",
#         },
#     )
#     new_obs = new_env.step(EMPTY_ACTION)
#     assert obs_history[0] != new_obs


# LIBERO_TASK_MAP = {
#     "libero_spatial": [
#         "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
#         "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
#     ],
#     "libero_object": [
#         "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
#         "pick_up_the_cream_cheese_and_place_it_in_the_basket",
#         "pick_up_the_salad_dressing_and_place_it_in_the_basket",
#         "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
#         "pick_up_the_ketchup_and_place_it_in_the_basket",
#         "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
#         "pick_up_the_butter_and_place_it_in_the_basket",
#         "pick_up_the_milk_and_place_it_in_the_basket",
#         "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
#         "pick_up_the_orange_juice_and_place_it_in_the_basket",
#     ],
#     "libero_goal": [
#         "open_the_middle_drawer_of_the_cabinet",
#         "put_the_bowl_on_the_stove",
#         "put_the_wine_bottle_on_top_of_the_cabinet",
#         "open_the_top_drawer_and_put_the_bowl_inside",
#         "put_the_bowl_on_top_of_the_cabinet",
#         "push_the_plate_to_the_front_of_the_stove",
#         "put_the_cream_cheese_in_the_bowl",
#         "turn_on_the_stove",
#         "put_the_bowl_on_the_plate",
#         "put_the_wine_bottle_on_the_rack",
#     ],
#     "libero_10": [
#         "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
#         "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
#         "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
#         "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
#         "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
#         "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
#         "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
#         "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
#         "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
#         "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
#     ],
#     "libero_90": [
#         "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
#         "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
#         "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
#         "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
#         "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
#         "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
#         "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
#         "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
#         "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
#         "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
#         "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
#         "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
#         "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
#         "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
#         "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
#         "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
#         "KITCHEN_SCENE3_turn_on_the_stove",
#         "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
#         "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
#         "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
#         "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
#         "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
#         "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
#         "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
#         "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
#         "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
#         "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
#         "KITCHEN_SCENE6_close_the_microwave",
#         "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
#         "KITCHEN_SCENE7_open_the_microwave",
#         "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
#         "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
#         "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
#         "KITCHEN_SCENE8_turn_off_the_stove",
#         "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
#         "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
#         "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
#         "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
#         "KITCHEN_SCENE9_turn_on_the_stove",
#         "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
#         "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
#         "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
#         "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
#         "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
#         "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
#         "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
#         "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
#         "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
#         "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
#         "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
#         "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
#         "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
#         "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
#         "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
#         "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
#         "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
#         "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
#         "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
#         "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
#         "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
#         "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
#         "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
#         "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
#         "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
#         "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
#         "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
#         "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
#         "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
#         "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
#     ],
# }



# @pytest.mark.integration
# def test_all_libero_benchmark():
#     output_dir = _benchmark_result_dir()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     empty_action = [0] * 7
#     failures: list[str] = []

#     for suite_name, task_names in LIBERO_TASK_MAP.items():
#         for task_id, task_name in enumerate(task_names):
#             env_id = f"{suite_name}/{task_name}"
#             env = init_bench(
#                 "Tektonian/Libero",
#                 env_id,
#                 0,
#                 {
#                     "control_mode": "ee_pose",
#                 },
#             )
#             task_dir = output_dir / suite_name / f"{task_id:03d}_{task_name}"
#             if task_dir.exists():
#                 print(f"skip: {task_dir}")
#                 continue
#             try:
#                 print(f"trying: {task_dir}")
#                 step_result = env.step(empty_action)
#                 output_paths = _save_step_images(
#                     task_dir,
#                     step_result,
#                 )
#                 print(
#                     f"[saved] {env_id} -> "
#                     + ", ".join(str(output_path) for output_path in output_paths)
#                 )
#             except Exception as exc:
#                 failures.append(f"{env_id}: {exc!r}")
#             finally:
#                 env.close()

#     assert not failures, "Failed benchmarks:\n" + "\n".join(failures)


# @pytest.mark.integration
# #BUG: seeding does not work in parallel envs
# def test_parallel_libero():
#     EMPTY_ACTION = [0] * 7

#     ast_option = [
#         "Tektonian/Libero",
#         "libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
#         0,
#     ]
#     options = dict(
#         benchmark_specific={
#             "control_mode": "ee_pose",
#         },
#     )

#     env1 = init_bench(*ast_option, **options)
#     env2 = init_bench(*ast_option, **options)
#     env3 = init_bench(*ast_option, **options)

#     vec_env = make_vec([env1, env2, env3])

#     first_step = vec_env.step([EMPTY_ACTION, EMPTY_ACTION, EMPTY_ACTION])
#     assert first_step[0] == first_step[1] == first_step[2]
#     for i in range(4):
#         ret = vec_env.step([EMPTY_ACTION, EMPTY_ACTION, EMPTY_ACTION])
#         assert ret[0] == ret[1] == ret[2]

#     # vec_env.reset_parallel()

#     # second_step = vec_env.step([EMPTY_ACTION, EMPTY_ACTION, EMPTY_ACTION])
#     # assert (
#     #     first_step[0] == second_step[0]
#     #     and first_step[1] == second_step[1]
#     #     and first_step[2] == second_step[2]
#     # )

# test end: libero benchmark

# test start: calvin benchmark
# CALVIN_TASK_MAP = [
#     "calvin_scene_A",
#     "calvin_scene_B",
#     "calvin_scene_C",
#     "calvin_scene_D"
# ]

# @pytest.mark.integration
# def test_calvin_benchmark():
#     env = init_bench(
#         "Tektonian/Calvin",
#         "calvin_scene_A",
#         0,
#     )
#     EMPTY_ACTION = [0] * 7
#     obs_history = []
#     obs_history.append(env.step(EMPTY_ACTION))
#     for _ in range(4):
#         obs_history.append(env.step(EMPTY_ACTION))

#     reset = env.reset(0)
#     for i in range(len(obs_history)):
#         ret = env.step(EMPTY_ACTION)
#         assert ret == obs_history[i]

#     new_env = init_bench(
#         "Tektonian/Calvin",
#         "calvin_scene_A",
#         0,
#     )
#     new_obs = new_env.step(EMPTY_ACTION)
#     assert obs_history[0] != new_obs

# @pytest.mark.integration
# def test_all_calvin_benchmark():
#     output_dir = _benchmark_result_dir()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     empty_action = [0] * 7
#     failures: list[str] = []

#     for suite_name in CALVIN_TASK_MAP:
#         env_id = f"{suite_name}"
#         env = init_bench(
#             "Tektonian/Calvin",
#             env_id,
#             0,
#         )
#         task_dir = output_dir / suite_name

#         try:
#             print(f"trying: {task_dir}")
#             # import ipdb; ipdb.set_trace()
#             step_result = env.step(empty_action)
#             output_paths = _save_step_images(
#                 task_dir,
#                 step_result,
#             )
#             print(
#                 f"[saved] {env_id} -> "
#                 + ", ".join(str(output_path) for output_path in output_paths)
#             )
#         except Exception as exc:
#             failures.append(f"{env_id}: {exc!r}")
#         finally:
#             env.close()

#     assert not failures, "Failed benchmarks:\n" + "\n".join(failures)

#     # test end: calvin benchmark

# test start: metaworld benchmark
# META_WORLD_TASK_MAP = [
#     "assembly-v3",
#     "basketball-v3",
#     "bin-picking-v3",
#     "box-close-v3",
#     "button-press-topdown-v3",
#     "button-press-topdown-wall-v3",
#     "button-press-v3",
#     "button-press-wall-v3",
#     "coffee-button-v3",
#     "coffee-pull-v3",
#     "coffee-push-v3",
#     "dial-turn-v3",
#     "disassemble-v3",
#     "door-close-v3",
#     "door-lock-v3",
#     "door-open-v3",
#     "door-unlock-v3",
#     "hand-insert-v3",
#     "drawer-close-v3",
#     "drawer-open-v3",
#     "faucet-open-v3",
#     "faucet-close-v3",
#     "hammer-v3",
#     "handle-press-side-v3",
#     "handle-press-v3",
#     "handle-pull-side-v3",
#     "handle-pull-v3",
#     "lever-pull-v3",
#     "peg-insert-side-v3",
#     "pick-place-wall-v3",
#     "pick-out-of-hole-v3",
#     "reach-v3",
#     "push-back-v3",
#     "push-v3",
#     "pick-place-v3",
#     "plate-slide-v3",
#     "plate-slide-side-v3",
#     "plate-slide-back-v3",
#     "plate-slide-back-side-v3",
#     "peg-unplug-side-v3",
#     "soccer-v3",
#     "stick-push-v3",
#     "stick-pull-v3",
#     "push-wall-v3",
#     "reach-wall-v3",
#     "shelf-place-v3",
#     "sweep-into-v3",
#     "sweep-v3",
#     "window-open-v3",
#     "window-close-v3"
# ]
# @pytest.mark.integration
# def test_metaworld_benchmark():
#     env = init_bench(
#         "Tektonian/Metaworld",
#         "sweep-v3",
#         0,
#     )
#     EMPTY_ACTION = [0] * 4
#     obs_history = []
#     obs_history.append(env.step(EMPTY_ACTION))
#     for _ in range(4):
#         obs_history.append(env.step(EMPTY_ACTION))

#     reset = env.reset(0)
#     for i in range(len(obs_history)):
#         ret = env.step(EMPTY_ACTION)
#         assert ret == obs_history[i]

#     new_env = init_bench(
#         "Tektonian/Metaworld",
#         "sweep-v3",
#         0,
#     )
#     new_obs = new_env.step(EMPTY_ACTION)
#     assert obs_history[0] != new_obs

# test end: metaworld benchmark

# # test start: robomimic benchmark
# ROBOMIMIC_TASK_MAP = [
#     "Lift",
#     "PickPlaceCan",
#     "NutAssemblySquare",
#     "TwoArmTransport"
# ]

# @pytest.mark.integration
# def test_robomimic_benchmark():
#     env = init_bench(
#         "Tektonian/Robomimic",
#         "Lift",
#         0,
#     )
#     EMPTY_ACTION = [0] * 7
#     obs_history = []
#     obs_history.append(env.step(EMPTY_ACTION))
#     for _ in range(4):
#         obs_history.append(env.step(EMPTY_ACTION))

#     reset = env.reset(0)
#     for i in range(len(obs_history)):
#         ret = env.step(EMPTY_ACTION)
#         assert ret == obs_history[i]

#     new_env = init_bench(
#         "Tektonian/Robomimic",
#         "Lift",
#         0,
#     )
#     new_obs = new_env.step(EMPTY_ACTION)
#     assert obs_history[0] != new_obs
# # test end: robomimic benchmark

# test start: bigym benchmark
BIGYM_TASK_MAP = [
    "DishwasherClose",
    "DishwasherCloseTrays",
    "DishwasherLoadCups",
    "DishwasherLoadCutlery",
    "DishwasherLoadPlates",
    "DishwasherOpen",
    "DishwasherOpenTrays",
    "DishwasherUnloadCups",
    "DishwasherUnloadCupsLong",
    "DishwasherUnloadCutlery",
    "DishwasherUnloadCutleryLong",
    "DishwasherUnloadPlates",
    "DishwasherUnloadPlatesLong",
    "DrawerTopClose",
    "DrawerTopOpen",
    "DrawersAllClose",
    "DrawersAllOpen",
    "FlipCup",
    "FlipCutlery",
    "FlipSandwich",
    "GroceriesStoreLower",
    "GroceriesStoreUpper",
    "MovePlate",
    "MoveTwoPlates",
    "PickBox",
    "PutCups",
    "ReachTarget",
    "ReachTargetDual",
    "ReachTargetSingle",
    "RemoveSandwich",
    "StackBlocks",
    "StoreBox",
    "StoreKitchenware",
    "TakeCups",
    "ToastSandwich",
    "WallCupboardClose",
    "WallCupboardOpen"
]

@pytest.mark.integration
def test_bigym_benchmark():
    env = init_bench(
        "Tektonian/Bigym",
        "MovePlate",
        0,
        benchmark_specific={"floating_base": True, "control_frequency": 50},
    )
    EMPTY_ACTION = [0] * 15
    obs_history = []
    obs_history.append(env.step(EMPTY_ACTION))
    for _ in range(4):
        obs_history.append(env.step(EMPTY_ACTION))

    reset = env.reset(0)
    for i in range(len(obs_history)):
        ret = env.step(EMPTY_ACTION)
        assert ret == obs_history[i]

    new_env = init_bench(
        "Tektonian/Bigym",
        "MovePlate",
        0,
        benchmark_specific={"floating_base": True, "control_frequency": 50},
    )
    new_obs = new_env.step(EMPTY_ACTION)
    assert obs_history[0] != new_obs

# test end: bigym benchmark