# Changelog

## 0.1.0 (2026-06-07)

- User side APIs are fully implemented🔥
- Complete mujoco adapter implementation

User side APIs include typed objects, randomzation, constraints, placement, runtime object access, asset/file services, import/export `Environment`, native context access, camera rendering, light APIs.

~~(What a journey... It was so hard...)~~

Additionally, the initial senario concept has been added.

### Features

* **lib:** redefine the world maker interface and add typed randomization helpers, cameras, lights, runtime objects, and parallel runner flows ([#25](https://github.com/Tektonian/Simulac/pull/25)) ([85f426a](https://github.com/Tektonian/Simulac/commit/85f426a))
* **lib:** add articulated object reference APIs and public interfaces for `*Runtime` / `*Object` types ([#26](https://github.com/Tektonian/Simulac/pull/26)) ([7baf61e](https://github.com/Tektonian/Simulac/commit/7baf61e))
* **sdk,lib:** add unified `add_entity()` flow for MuJoCo environments, typed environment dataclasses, and structured camera/light specs ([#27](https://github.com/Tektonian/Simulac/pull/27)) ([57a2f80](https://github.com/Tektonian/Simulac/commit/57a2f80))
* **lib,sdk:** integrate MuJoCo environment assembly, placement, randomization, reset, tick, and step flow ([#28](https://github.com/Tektonian/Simulac/pull/28)) ([d541808](https://github.com/Tektonian/Simulac/commit/d541808))
* **lib,sdk:** implement MuJoCo runtime object access through `Runner.get_runtime_object()` and SDK `IRunner.get_runtime_object(entity_id)` ([#31](https://github.com/Tektonian/Simulac/pull/31)) ([f1bd0bb](https://github.com/Tektonian/Simulac/commit/f1bd0bb))
* **sdk:** integrate the MuJoCo runner runtime, bindings, reset sampling, and `StuffRuntime` operations ([#32](https://github.com/Tektonian/Simulac/pull/32)) ([e5b3159](https://github.com/Tektonian/Simulac/commit/e5b3159))
* **sdk:** implement MuJoCo `RobotRuntime` and `CameraRuntime`, including typed runtime wrappers, camera binding, camera relations, and runtime object dispatch ([#33](https://github.com/Tektonian/Simulac/pull/33)) ([d9efb8e](https://github.com/Tektonian/Simulac/commit/d9efb8e))
* **sdk:** add scene constraints and MuJoCo reset-time constraint checking for distance, bounding box, and non-penetration constraints ([#35](https://github.com/Tektonian/Simulac/pull/35)) ([9da1651](https://github.com/Tektonian/Simulac/commit/9da1651))
* **sdk:** add MuJoCo placement resolver support for surface sampling with source refs, tangent offsets, and local/world offsets ([#36](https://github.com/Tektonian/Simulac/pull/36)) ([042473e](https://github.com/Tektonian/Simulac/commit/042473e))
* **lib:** implement user-side `Environment.constraint(...)` and scene constraint helpers ([#37](https://github.com/Tektonian/Simulac/pull/37)) ([de430c2](https://github.com/Tektonian/Simulac/commit/de430c2))
* **lib,sdk:** support the `fixed` flag for MuJoCo stuff entities and `StuffObject.set_fixed(...)` ([#39](https://github.com/Tektonian/Simulac/pull/39)) ([f04cee5](https://github.com/Tektonian/Simulac/commit/f04cee5))
* **lib,sdk:** finalize the MuJoCo API design with runtime state returns, contact inspection, runtime robot component APIs, and `RobotRuntime.set_control(action)` ([#40](https://github.com/Tektonian/Simulac/pull/40)) ([333e94f](https://github.com/Tektonian/Simulac/commit/333e94f))
* **sdk:** implement `IFileService` with local `file:` and remote `http:` / `https:` providers ([#41](https://github.com/Tektonian/Simulac/pull/41)) ([a5e5b09](https://github.com/Tektonian/Simulac/commit/a5e5b09))
* **sdk:** implement `AssetService` for parsing, resolving, downloading, caching, and verifying asset bundles from native asset references such as `Stuff("Robocasa/mug/mug_01")` ([#42](https://github.com/Tektonian/Simulac/pull/42)) ([de25e40](https://github.com/Tektonian/Simulac/commit/de25e40))
* **lib,sdk:** implement environment import and export through `load_env()`, `dump_env()`, and `dump_env_json()` on the environment management service ([#43](https://github.com/Tektonian/Simulac/pull/43)) ([c220ea2](https://github.com/Tektonian/Simulac/commit/c220ea2))
* **lib,sdk:** add `runner.context("mujoco")` native context access for MuJoCo `mjModel`, `mjData`, and entity bindings ([#44](https://github.com/Tektonian/Simulac/pull/44)) ([269bea5](https://github.com/Tektonian/Simulac/commit/269bea5))
* **lib,sdk:** remove the unpredictable `Stuff.set_density()` API ([#45](https://github.com/Tektonian/Simulac/pull/45)) ([98e5eb6](https://github.com/Tektonian/Simulac/commit/98e5eb6))
* **lib,sdk:** complete build-time `StuffObject` / `RobotObject` APIs, add shared runtime joint state, and add `RobotRuntime.sensor()` / `SensorState` ([#46](https://github.com/Tektonian/Simulac/pull/46)) ([54d545b](https://github.com/Tektonian/Simulac/commit/54d545b))
* **lib,sdk:** add `Environment.remove_object()` and update the environment JSON schema to use `asset_uri` ([#47](https://github.com/Tektonian/Simulac/pull/47)) ([37af6ef](https://github.com/Tektonian/Simulac/commit/37af6ef))
* **lib,sdk:** add camera runtime rendering through `CameraRuntime.render(width, height)` for RGB, depth, segmentation, and typed pointcloud support ([#48](https://github.com/Tektonian/Simulac/pull/48)) ([45f157f](https://github.com/Tektonian/Simulac/commit/45f157f))
* **lib,sdk:** implement user-side, SDK, and MuJoCo light runtime APIs for ambient, point, spot, and area lights ([#49](https://github.com/Tektonian/Simulac/pull/49)) ([cdad679](https://github.com/Tektonian/Simulac/commit/cdad679))
* **lib:** add the initial `Scenario` concept for user-side runtime behavior control ([#50](https://github.com/Tektonian/Simulac/pull/50)) ([7683268](https://github.com/Tektonian/Simulac/commit/7683268))

### Bug Fixes

* **lib,sdk:** allow point refs in `add_entity(..., pos=...)` typing ([#38](https://github.com/Tektonian/Simulac/pull/38)) ([9484499](https://github.com/Tektonian/Simulac/commit/9484499))
* **sdk:** fix `TelemetryService` typing ([8803a42](https://github.com/Tektonian/Simulac/commit/8803a42))

### Documentation

* **project:** add the `eval_libero_smolvla.py` evaluation example ([c481b68](https://github.com/Tektonian/Simulac/commit/c481b68))
* **project:** update README and internal docs, including concept docs, Python rules, and architecture decision records ([2d3b94f](https://github.com/Tektonian/Simulac/commit/2d3b94f))

### Chores

* **lib:** remove `constraints` from `Randomization` ([316a139](https://github.com/Tektonian/Simulac/commit/316a139))
* **lib:** implement user-side docstrings for gym-style and world-maker APIs ([be7f87b](https://github.com/Tektonian/Simulac/commit/be7f87b))
* **lib,sdk:** add telemetry code paths to gym-style environments, asset resolution, world-maker flow, and telemetry service reporting ([bb33864](https://github.com/Tektonian/Simulac/commit/bb33864))

### Breaking Changes

* `Environment` entity pose inputs changed from `quat` to `rot`.
* `Environment` constructor parameter order changed from `[default_engine, prebuilt_env]` to `[prebuilt_env, default_engine]`.
* `Environment.place_entity()` / `Environment.place_object()` were removed in favor of `Environment.add_entity(..., pos=surface.sample(...))` and environment-level constraints.
* `Runner.step(action)` was replaced with `RobotRuntime.set_control(action)`.
* `Runner` now uses `tick_dt_ms` instead of the previous `tick` naming.
* `IEnvironment.load_env()` was removed; import/export moved to the environment management service.
* Some runtime mutation methods for size, fixed state, and density were removed from the common runtime interface.
* `Stuff.set_density()` was removed.
* Camera type literals were narrowed to `rgb`, `depth`, `pointcloud`, and `segmentation`.
* `SetLightTypeOp` was removed, and light type literals were normalized to `ambient`, `pointlight`, `spot`, and `area`.

### Notes

* MuJoCo pointcloud camera rendering is typed but still needs implementation and dependency discussion. (I hate numpy dependency broken)
* MuJoCo area lights are approximated with point lights plus bulb radius, and ambient lights are modeled through MuJoCo ambient light channels.
* `AssetService` is implemented, but backend asset API details and asset curation still need follow-up work.
* `Scenario` is concept-level in this release and is not yet wired into `Runner` execution.
* `ParallelRunner` is exists but not implemented.

## 0.0.4 (2026-05-05)

Add error recovery logic in `BenchmarkEnvironment` to prevent vectorized environments from crashing due to server-side errors.

### Features

* **lib:** add `BenchmarkEnvironment` error-recovery logic and recovery tests ([10758de](https://github.com/Tektonian/Simulac/commit/10758de)) ([#29](https://github.com/Tektonian/Simulac/pull/29))

### Chores

* **project:** prepare 0.0.4 release ([f0e6042](https://github.com/Tektonian/Simulac/commit/f0e6042)) ([#30](https://github.com/Tektonian/Simulac/pull/30))



## 0.0.3 (2026-04-19)

Expand the CLI commands, fixed world-maker entity ID generation, and refactor the gym-style environment code.

### Features

* **cli:** expand CLI commands and add runtime context ([#24](https://github.com/Tektonian/Simulac/pull/24)) ([79cfefb](https://github.com/Tektonian/Simulac/commit/79cfefb))

### Bug Fixes

* **sdk:** generate `entity_id` of world maker from the correct entities ([#22](https://github.com/Tektonian/Simulac/pull/22)) ([26a6594](https://github.com/Tektonian/Simulac/commit/26a6594))

### Refactoring

* **lib:** refactor `step` and `reset` handling in `BenchmarkVecEnvironment` ([#19](https://github.com/Tektonian/Simulac/pull/19)) ([a766200](https://github.com/Tektonian/Simulac/commit/a766200))
* **lib:** clean up `init_bench` and warn when gym-style environments call `step()` before `reset()` ([#20](https://github.com/Tektonian/Simulac/pull/20)) ([2d95893](https://github.com/Tektonian/Simulac/commit/2d95893))

### Chores

* **project:** prepare the `0.0.3` release, update physics engine versions, and add a development guide ([#23](https://github.com/Tektonian/Simulac/pull/23)) ([73e17a8](https://github.com/Tektonian/Simulac/commit/73e17a8))


## 0.0.2 (2026-04-14)

Added missing packages and mostly focused on bug fixing.

### Features

* **lib:** remove the `group_id` parameter from `get_env_list` ([6bd1db2](https://github.com/Tektonian/Simulac/commit/6bd1db2))
* **lib:** change gym env return type from `terminated, truncated` to `done` ([8faf1a2](https://github.com/Tektonian/Simulac/commit/8faf1a2))
* **lib:** fix return type of `gym_env` `step` and `reset` from dict to tuple ([e502cf0](https://github.com/Tektonian/Simulac/commit/e502cf0))
* **base,sdk:** apply the runtime log level from environment variables ([31c05f4](https://github.com/Tektonian/Simulac/commit/31c05f4))

### Bug Fixes

* **project:** add missed `requests` package ([0fd6fc1](https://github.com/Tektonian/Simulac/commit/0fd6fc1))

### Chores

* **project:** change required Python version to 3.12 ([c2081f1](https://github.com/Tektonian/Simulac/commit/c2081f1))
* **project:** prepare `0.0.2` release, clean code, and update README ([6af725c](https://github.com/Tektonian/Simulac/commit/6af725c))



## 0.0.1 (2026-04-14)

Initial 0.0.1 release of `simulac`. Implemented custom DI pattern, SDK and CLI, and adds the first gym-style remote benchmark workflow.

### Features

* **base:** refactor and harden the instantiation/runtime foundation with service extraction, cycle detection, creation tracing, typing cleanup, and env-var/runtime support groundwork ([6039124](https://github.com/Tektonian/Simulac/commit/60391245ea61c92a22cb5e9e59c9377b984624c9)) ([#3](https://github.com/Tektonian/Simulac/pull/3)) ([f1e5858](https://github.com/Tektonian/Simulac/commit/f1e5858c6d206f0c919e6a2b997b00e8d92ea54e)) ([#10](https://github.com/Tektonian/Simulac/pull/10)) ([21ef18f](https://github.com/Tektonian/Simulac/commit/21ef18f3201ccc872ede0ec46fd2d16324503d03))
* **cli:** CLI login/logout flow, improve cache directory handling and user-facing auth messages ([#10](https://github.com/Tektonian/Simulac/pull/10)) ([21ef18f](https://github.com/Tektonian/Simulac/commit/21ef18f3201ccc872ede0ec46fd2d16324503d03)) ([#13](https://github.com/Tektonian/Simulac/pull/13)) ([3074fc1](https://github.com/Tektonian/Simulac/commit/3074fc1222747e0db990b56b658035005b927b15))
* **error:** define the base error type ([86c46c2](https://github.com/Tektonian/Simulac/commit/86c46c2a9eb705057ddfc9d070f2ffc76400526a))
* **lib:** add the `gym_style` module and introduce `GymStyleEnvironment` for gym-like benchmark usage ([7579d38](https://github.com/Tektonian/Simulac/commit/7579d387ce73364dc7d406c93cd0d6af6f7839a0)) ([3354e3e](https://github.com/Tektonian/Simulac/commit/3354e3ea443f66ecbdbf2b185e7c572d644f4d27))
* **lib:** implement user-side world construction and later correct the entity/object responsibility split ([#7](https://github.com/Tektonian/Simulac/pull/7)) ([e5df48a](https://github.com/Tektonian/Simulac/commit/e5df48a1c162831bd6a595fe0544fa1904aecf5d)) ([#15](https://github.com/Tektonian/Simulac/pull/15)) ([767d2c3](https://github.com/Tektonian/Simulac/commit/767d2c314c3da1817baac3cce7d9f2f140e98252))
* **lib:** reimplement the gym-style environment for remote benchmark execution and align it with the updated protocol, including reset support and compressed payload transport ([#9](https://github.com/Tektonian/Simulac/pull/9)) ([dac79e2](https://github.com/Tektonian/Simulac/commit/dac79e2768c2f773d013358ad4170e5fb0b4e3e1)) ([#16](https://github.com/Tektonian/Simulac/pull/16)) ([c875bdd](https://github.com/Tektonian/Simulac/commit/c875bdd6e3f90797ae652e8e8589d02533e53648))
* **lib:** add environment listing and benchmark ticket preflight ([#18](https://github.com/Tektonian/Simulac/pull/18)) ([a31b225](https://github.com/Tektonian/Simulac/commit/a31b22574e1b7f61b46462f0f2a7e09385b8f6b1))
* **project:** reposition the project as a benchmark service ([4ec4f23](https://github.com/Tektonian/Simulac/commit/4ec4f2324a48c514d3621688dd03ca55820ce726))
* **project:** rename the project from `tt` (Tektonian) to `simulac` (Simulac), including import path and prefix updates ([#11](https://github.com/Tektonian/Simulac/pull/11)) ([efdd23b](https://github.com/Tektonian/Simulac/commit/efdd23bfc2f3041296096dab43be3c6002534c87))
* **project:** prepare the repository and packaging baseline for the `0.0.1` pre-alpha release ([#10](https://github.com/Tektonian/Simulac/pull/10)) ([21ef18f](https://github.com/Tektonian/Simulac/commit/21ef18f3201ccc872ede0ec46fd2d16324503d03)) ([#18](https://github.com/Tektonian/Simulac/pull/18)) ([a31b225](https://github.com/Tektonian/Simulac/commit/a31b22574e1b7f61b46462f0f2a7e09385b8f6b1))
* **sdk,lib:** add initial libero-style fields for future benchmark support ([e4929c0](https://github.com/Tektonian/Simulac/commit/e4929c017055115b4f2214fb1ba2d6c473552fb4))
* **sdk:** add telemetry and logging packages, default services, and split environment responsibilities into env, world, simul, and runner layers ([0b3fcf6](https://github.com/Tektonian/Simulac/commit/0b3fcf673b35d506479b007b07d0ea7724516907)) ([65169f2](https://github.com/Tektonian/Simulac/commit/65169f2da6039744091a03d1d03213a79d0da44e)) ([83e90bc](https://github.com/Tektonian/Simulac/commit/83e90bcca4ba2ea870bed04b8198a56e3eb949d8))
* **sdk:** implement the environment builder, adapter pattern, runner factory, and initial Mujoco/Newton adapter support ([#2](https://github.com/Tektonian/Simulac/pull/2)) ([8d74e5c](https://github.com/Tektonian/Simulac/commit/8d74e5c7cd65cbcc732f291d563413c4333fa527))
* **sdk:** add benchmark remote runner support and extend the Newton adapter to handle multiple runners ([a2500b8](https://github.com/Tektonian/Simulac/commit/a2500b847dad22d0c9ea5551387de27c60f828d2)) ([#6](https://github.com/Tektonian/Simulac/pull/6)) ([8fd3436](https://github.com/Tektonian/Simulac/commit/8fd343613162b423af30f386225cebce72e87100))

### Bug Fixes

* **base:** fix duplicate instance creation in the instantiation flow ([4dccb7f](https://github.com/Tektonian/Simulac/commit/4dccb7f88510ba24e134419e9539402a1928cd1a))
* **base/project:** fix tracing and typing issues, including `_NoneTrace` naming and dependency-check adjustments ([#10](https://github.com/Tektonian/Simulac/pull/10)) ([21ef18f](https://github.com/Tektonian/Simulac/commit/21ef18f3201ccc872ede0ec46fd2d16324503d03))
* **lib/sdk:** fix library error codes, object creation defaults, interface naming mismatches, and related test breakages ([#14](https://github.com/Tektonian/Simulac/pull/14)) ([fd9bcea](https://github.com/Tektonian/Simulac/commit/fd9bcea6da49d57699dd40121c81d0a62e78a636))
* **project/lib:** fix URL joining after the project rename and return benchmark error stack traces as strings ([#11](https://github.com/Tektonian/Simulac/pull/11)) ([efdd23b](https://github.com/Tektonian/Simulac/commit/efdd23bfc2f3041296096dab43be3c6002534c87)) ([#18](https://github.com/Tektonian/Simulac/pull/18)) ([a31b225](https://github.com/Tektonian/Simulac/commit/a31b22574e1b7f61b46462f0f2a7e09385b8f6b1))

### Documentation

* **project:** add README, contributing guide, issue templates, pull request template, and GitHub community/configuration files ([#17](https://github.com/Tektonian/Simulac/pull/17)) ([8a34aef](https://github.com/Tektonian/Simulac/commit/8a34aef44bce67f8c6d3f4f16359a4cac714c84b))
* **project:** add a docs README, architecture overview, and architecture decision records ([#18](https://github.com/Tektonian/Simulac/pull/18)) ([a31b225](https://github.com/Tektonian/Simulac/commit/a31b22574e1b7f61b46462f0f2a7e09385b8f6b1))

### Refactoring

* **sdk:** change ID naming conventions and refactor instantiation and environment management ([f5fa316](https://github.com/Tektonian/Simulac/commit/f5fa316f8066eb06696f8b4fa5b155ff149bedf1)) ([#4](https://github.com/Tektonian/Simulac/pull/4)) ([71c0549](https://github.com/Tektonian/Simulac/commit/71c054961fb8e06161346020299d9b2f88e46c3f))
* **lib:** clean up gym environment lifecycle handling and internal benchmark environment structure while updating the remote protocol implementation ([#16](https://github.com/Tektonian/Simulac/pull/16)) ([c875bdd](https://github.com/Tektonian/Simulac/commit/c875bdd6e3f90797ae652e8e8589d02533e53648)) ([#18](https://github.com/Tektonian/Simulac/pull/18)) ([a31b225](https://github.com/Tektonian/Simulac/commit/a31b22574e1b7f61b46462f0f2a7e09385b8f6b1))

### Tests

* **sdk:** add initial simple test coverage ([6b8463c](https://github.com/Tektonian/Simulac/commit/6b8463ce695c1e38337bf89b9ba8595f0a60383b))
* **sdk/lib:** add telemetry, world-maker, benchmark, and integration-oriented benchmark test flows during feature expansion and protocol updates ([#7](https://github.com/Tektonian/Simulac/pull/7)) ([e5df48a](https://github.com/Tektonian/Simulac/commit/e5df48a1c162831bd6a595fe0544fa1904aecf5d)) ([#10](https://github.com/Tektonian/Simulac/pull/10)) ([21ef18f](https://github.com/Tektonian/Simulac/commit/21ef18f3201ccc872ede0ec46fd2d16324503d03)) ([#16](https://github.com/Tektonian/Simulac/pull/16)) ([c875bdd](https://github.com/Tektonian/Simulac/commit/c875bdd6e3f90797ae652e8e8589d02533e53648))

### Chores

* **project:** configure `release-please` for Python releases and add a Husky pre-push branch naming check ([7bb3f9a](https://github.com/Tektonian/Simulac/commit/7bb3f9a585999df56dc29922fa27ebcb1be176aa)) ([406f227](https://github.com/Tektonian/Simulac/commit/406f2270d08664837ded982482f672f9456c2a85))
* **sdk:** separate interface modules and add package `__init__.py` files as the SDK structure stabilized ([c50a69d](https://github.com/Tektonian/Simulac/commit/c50a69d7934c1b2310231e47fa9d03999d6ce2d6)) ([2cff4b0](https://github.com/Tektonian/Simulac/commit/2cff4b04b9d0e12d8fac0e037d52a3a839b35a34))
* **project:** clean dev-only dependencies, rewrite `pyproject.toml`, and refresh editor/tooling settings  ([fa8acb1](https://github.com/Tektonian/Simulac/commit/fa8acb1c82c3b3f598c1508fdad8cec775c7ecc9)) ([#10](https://github.com/Tektonian/Simulac/pull/10)) ([21ef18f](https://github.com/Tektonian/Simulac/commit/21ef18f3201ccc872ede0ec46fd2d16324503d03))
* **project:** finalize the `0.0.1` release preparation and package the initial release notes and supporting docs ([#18](https://github.com/Tektonian/Simulac/pull/18)) ([a31b225](https://github.com/Tektonian/Simulac/commit/a31b22574e1b7f61b46462f0f2a7e09385b8f6b1))
