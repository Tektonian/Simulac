from abc import ABC
import random
import time

from tt.lib.gym_style import init_bench

# region User Interface example
from tt.lib.world_maker import Runner, World, Environment, Stuff, Robot

from tt.sdk import environment_management_service, runner_management_service
from tt.sdk.environment_service.common.model.component import (
    MJCFPhysicsComponent,
    RenderingComponent,
)
from tt.sdk.environment_service.common.model.entity import EnvironmentMJCFObjectEntity


def test_world_build():
    env_ret = environment_management_service.create_environment("", "", "", 0)
    assert env_ret[0] is not None

    env = env_ret[0]

    ren_entity = RenderingComponent("", "")
    physc_entity = MJCFPhysicsComponent("tt/asset/test/Basket/Basket026/model.xml")
    entity = EnvironmentMJCFObjectEntity(rendering=ren_entity, physics=physc_entity)

    environment_management_service.add_entity(env.id, entity)
    ren_entity = RenderingComponent("", "")
    physc_entity = MJCFPhysicsComponent(
        "tt/asset/test/CheeseGrater/CheeseGrater002/model.xml"
    )
    entity = EnvironmentMJCFObjectEntity(rendering=ren_entity, physics=physc_entity)
    entity.pos = (1, 0, 0)
    environment_management_service.add_entity(env.id, entity)

    runner_env = runner_management_service.create_runner(
        env.id,
    )

    assert runner_env[0] is not None

    runner = runner_env[0]

    runner.step([])

    runner.render()


def test_world_make_example():

    class MockUserPolicy: ...

    user_policy = MockUserPolicy()

    # region When use prebuilt environment

    def mock_env_to_gym_env(env: Environment): ...

    env = Environment("server-side-house-pick-up-sausage-task")

    gym_env = mock_env_to_gym_env(env)

    ## Use env as gym style

    obs, info = gym_env.reset(0)

    for _ in range(100):
        action = user_policy.inference(obs)
        obs, reward, terminate, truncated, info = gym_env.step(action)

    # end-region

    # region When user want to build their own environment

    env = Environment()

    ## remote assets
    franka = Robot("franka")
    basket = Stuff("basket.0")
    block = Stuff("block.0")
    ## local asssets location
    sausage = Stuff("~/assets/sausage.xml")

    franka_obj = env.place_robot_entity(franka)

    TICK = 5  # 5ms
    run = Runner(env, tick=TICK)
    state = env.get_state()

    for _ in range(10):
        action = user_policy.inference(state)
        franka_obj.set_target_position(action)
        state = run.tick()

    # end-region

    # region when user want randomization

    env = Environment()

    franka = Robot("franka")
    basket = Stuff("basket.0")

    franka_obj = env.place_robot_entity(franka)
    basket_obj = env.place_entity(basket)

    ## If user want change stuff's physical attributes
    basket_obj.set_friction(1.0)
    ## or random value
    import random

    random_mass = random.random(0)
    basket_obj.set_mass(random_mass)

    for _ in range(10):
        action = user_policy.inference(state)
        franka_obj.set_target_position(action)
        state = run.tick()

    # end-region

    # region when user want to make parallel world

    ## If user want parallel execution of remote environments
    libero_env0 = Environment("libero_10/KITCHEN_SCENE3")

    libero_env1 = Environment("libero_10/KITCHEN_SCENE2")

    practice_pick_cube_env = Environment("./somewhare/env.json")

    world = World()

    ## For run benchmark parallel
    world.place_env(libero_env0, 1)
    world.place_env(libero_env1, 1)

    ## For learning or testing
    world.place_env(practice_pick_cube_env, 1024)

    states = world.get_state()

    for _ in range(10):
        actions = []
        for state in states:
            action = user_policy.inference(state)
            actions.append(action)

        states = world.step(actions)

    # end-region

    # region when user want async inference

    env = Environment()

    franka = Robot("franka")
    basket = Stuff("basket.0")

    franka_obj = env.place_robot_entity(franka)
    basket_obj = env.place_entity(basket)

    TICK = 5  # 5ms
    run = Runner(env, tick=TICK)
    state = env.get_state()
    import time
    for _ in range(10):
        start_ms = round(time.clock_gettime() * 1000)
        action = user_policy.inference(state)
        end_ms = round(time.time() * 1000)
        
        for _ in range((end_ms - start_ms) // TICK)
            state = run.tick()
            
        franka_obj.set_target_position(action)
    # end-region

    # region when user want async inference with multi agent -> Not implemented yet

    # end-region
# end-region


def test_result_type():
    env = init_bench(
        "Tektonian/Libero",
        "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        0,
    )

    env.reset(0)
    start_time = time.time()
    for _ in range(20):
        env.step([0, 0, 0, 0, 0, 0, 0])
    end_time = time.time()

    print(
        f"elapsed: {end_time - start_time}"
    )  # about 3~4 seconds vcpu: 1~1.2 vcpu ram: 3.4GB / 2 instances -> 1.6 vcpu  same ram / 3 instances -> crash

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
