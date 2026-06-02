<h1 align="center"><b>Simulac</b></h1>

<h3 align="center"><b>⚡️ Full-stack robot framework</b></h3>

---

[Simulac](https://github.com/Tektonian/Simulac) is a full-stack robot framework for robotics engineers who want a **unified API** for robot simulation, scene building, and hosted robot benchmarks.

`Simulac` supports:

1. Integrated physics engines
    - Simulac provides an abstraction layer over industry-standard physics engines
    - Mujoco is default local runtime. Mujoco Warp, Newton and Genesis will be added in the future
2. Digital twin building
    - Simulac provides [fully curated assets](https://tektonian.com/asset) for digital twin
    - Use assets with simple entity constructors `Stuff("Robocasa/mug/mug_01")`
    - Export and load environment definitions as JSON for reuse, review, and experiment setup
3. Robot benchmark
    - Run hosted robot benchmarks without manually reproducing benchmark infrastructure
    - Reduce setup frictions and speed up iteration



## Contents

- [Installation](#installation)
- [Robot Framework](#robot-framework)
- [Local World Building](#local-world-building)
- [Runtime API](#runtime-api)
- [Randomization and Constraints](#randomization-and-constraints)
- [Hosted Benchmark](#hosted-benchmark)
- [CLI](#cli)
- [Authentication](#authentication)
- [Configuration](#configuration)
- [Development](#development)
- [Project Status](#project-status)
- [License](#license)

## Installation

Simulac requires Python 3.12 or later.


Install with `pip`:

```bash
pip install simulac
```

Or install with `uv`:

```bash
uv add simulac
```


## Robot Framework

Simulac's local `Environment` API separates scene definition from runtime
mutation:

1. Create an `Environment`.
2. Add entities such as `Stuff`, `Robot`, `Camera`, and lights.
3. Mutate build-time properties before creating a runner.
4. Create a `Runner`, which freezes the scene definition.
5. Use runtime handles for live simulation changes.


```python
from simulac import Camera, Environment, Robot, Runner, Stuff

env = Environment(default_engine="mujoco")

table = env.add_entity(
    Stuff("path/to/table.xml"),
    entity_id="table",
    pos=(0.0, 0.0, 0.0),
    fixed=True,
)

cube = env.add_entity(
    Stuff("path/to/cube.xml"),
    entity_id="cube",
    pos=(0.4, 0.0, 0.8),
)

robot = env.add_entity(
    Robot("path/to/franka_panda.xml"),
    entity_id="panda",
)

camera = env.add_entity(
    Camera(type="rgb"),
    entity_id="front_rgb",
    pos=(0.8, -0.8, 1.2),
)

cube.set_mass(0.12)
cube.set_friction(1.0)

runner = Runner(env, 0, 5, runtime_engine="mujoco")
state = runner.reset(seed=0)
```

Build-time object handles include:

| Handle | Common operations |
| --- | --- |
| `StuffObject` | `set_pos`, `set_rot`, `set_size`, `set_mass`, `set_fixed`, `set_friction`, `collider`, `joint`, `anchor` |
| `RobotObject` | `set_pos`, `set_rot`, `set_joint_pos`, `collider`, `joint`, `anchor` |
| `CameraObject` | `set_pos`, `set_rot`, `set_fov`, `look_at`, `attach_to`, `follow` |

### Save a scene

```python
env.save_env("scene.json", overwrite=True)
```

### Load a scene

```python
loaded_env = Environment("scene.json", default_engine="mujoco")
```


## Local World Building

Local world building uses entity definitions and object handles.

```python
from simulac import AreaLight, Camera, Environment, Robot, Stuff

env = Environment(default_engine="mujoco")

table = env.add_entity(Stuff("path/to/table.xml"), entity_id="table", fixed=True)
robot = env.add_entity(Robot("path/to/robot.xml"), entity_id="robot")
camera = env.add_entity(Camera(type="rgb"), entity_id="front_rgb")
light = env.add_entity(AreaLight(intensity=0.8), entity_id="main_light")

camera.look_at(table.anchor("workspace_center"))
camera.attach_to(robot.anchor("camera_mount"))
```

Objects can expose named references for placement, constraints, and runtime lookup:

```python
top = table.collider("top")
mount = table.anchor("robot_mount")
joint = robot.joint("joint1")
```

Environment definitions can be exported as JSON:

```python
scene_json = env.dump_env_json(indent=2)
env.save_env("scene.json", overwrite=True)
```

## Runtime API

After a runner is created, use runtime handles for live simulation state.

```python
cube_rt = runner.get_runtime_object(cube)
robot_rt = runner.get_runtime_object(robot)
camera_rt = runner.get_runtime_object(camera)

cube_rt.change_pos((0.42, 0.05, 0.82))
cube_rt.change_mass(0.20)
cube_rt.change_friction(0.9)

robot_rt.change_joint_pos([0.0] * 7)
robot_rt.set_control([0.0] * 7)

livcamera_rte_camera.change_fov(60.0)

state = runner.tick()
image = runner.render(camera, width=640, height=480)
```

Runtime robot handles expose robot state by joint, site, link, and sensor name:

```python
joint = robot_rt.joint("joint1")
site = robot_rt.site("gripper")
link = robot_rt.link("link7")
sensor = robot_rt.sensor("force_sensor")
```

For engine-specific work, access the native context:

```python
ctx = runner.context("mujoco")
model = ctx.model
data = ctx.data
```


## Randomization and Constraints

Randomization specs are sampled during reset. Constraints are used to reject invalid reset candidates.

```python
from simulac import Constraint, Environment, Randomize, Stuff

env = Environment(default_engine="mujoco")

block = env.add_entity(Stuff("path/to/block.xml"), entity_id="block")
goal = env.add_entity(
    Stuff("path/to/goal_marker.xml"),
    entity_id="goal",
    fixed=True,
)

block.set_pos(
    Randomize.uniform(
        (0.25, -0.20, 0.80),
        (0.55, 0.20, 0.80),
        constraints=[
            Constraint.distance("block", "goal", min=0.15, max=0.50),
        ],
    )
)

block.set_mass(Randomize.choice(0.08, 0.12, 0.16))
```

Randomization helpers:

- `Randomize.uniform(min, max)`
- `Randomize.normal(mean, std, clip_min=None, clip_max=None)`
- `Randomize.choice(*values)`

Constraint helpers:

- `Constraint.distance(a, b, min=None, max=None)`
- `Constraint.bbox(target, lower, upper, mode="inside")`
- `Constraint.nonpenetration(*between)`

## Hosted Benchmark

### Run a Benchmark

```python
from simulac.gym_style import init_bench

env = init_bench(
    "Tektonian/Libero",
    "libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    0,
    {"control_mode": "OSC_POSE"},
)

obs, info = env.reset(seed=0)
obs, reward, done, info = env.step([0.0] * 7)
env.close()
```


`init_bench()` accepts:

| Parameter | Description |
| --- | --- |
| `benchmark_id` | Benchmark owner and name, for example `Tektonian/Libero` |
| `env_id` | Benchmark scene or task id |
| `seed` | Initial seed used to build the remote environment |
| `benchmark_specific` | Benchmark-specific options such as controller mode |

Visit [our website](https://tektonian.com/benchmark) to see all available benchmarks

### Authentication

Remote benchmark execution requires a Tektonian API key.

Go to our [website](https://tektonian.com) and get an API key.

Store your API key with CLI command:
```bash
simulac auth login
```

You can also provide the key through an environment variable:
```bash
export SIMULAC_API_KEY=<your_api_key>
```
Note: API keys can only be viewed once when created

Check the active credential:

```bash
simulac auth whoami
```

Remove the locally stored API key:

```bash
simulac auth logout
```




### List Available Environments

```bash
simulac benchmark list Tektonian/Libero
```

```python
from simulac.gym_style import get_env_list

env_ids = get_env_list("Tektonian/Libero")
```
Visit the Tektonian benchmark page to see available benchmarks:

```text
https://tektonian.com/benchmark
```

### Step Multiple Environments

`make_vec()` steps multiple hosted environments concurrently and returns results
in the same order as the input environment list.

```python
from simulac.gym_style import init_bench, make_vec

args = (
    "Tektonian/Libero",
    "libero_90/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    0,
)
options = {"benchmark_specific": {"control_mode": "OSC_POSE"}}

envs = [init_bench(*args, **options) for _ in range(3)]
vec_env = make_vec(envs)

reset_results = vec_env.reset([0, 1, 2])
step_results = vec_env.step([[0.0] * 7 for _ in envs])
vec_env.close()
```

## CLI


## Configuration

Simulac reads configuration from environment variables when present:

- `SIMULAC_API_KEY`: use an API key without running `simulac auth login`
- `SIMULAC_BASE_URL`: override the default Tektonian API endpoint
- `SIMULAC_LOG_LEVEL`: set log verbosity. choose one of `off, trace, debug, info, warning, error`
- `SIMULAC_TELEMETRY=off`: disable telemetry

## Development
For development, start with:

1. See `CONTRIBUTING.md`
2. Read `docs/README.md`
3. Read docs in `docs/internal`

## Project Status

Simulac is currently alpha software.

- [x] - ~~The remote benchmark client is the most complete public surface.~~ (with version 0.1.0 at 01-06-2026 )
- [x] - ~~Local world-building and runner APIs are still evolving.~~ (with version 0.1.0 at 01-06-2026)
- [ ] - Support parallel `Runner`
- [ ] - GUI for build `Environment`
- [ ] - Add more physics engine adapters
     - [ ] - Mujoco Warp
     - [ ] - Newton
     - [ ] - Genesis

## License

Apache-2.0