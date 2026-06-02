# Core Concepts

We strictly seperate boundary between scene definition and runtime execution.

Since Simulac designed with data centric architecture inspried by game engines like Unreal Engine and Unity, the project follows these rules:

1. Treat environment data as the source of truth, and strictly separate data
   definition from the runtime that consumes that data
2. Treat `simulac/lib` as the user-facing API layer
3. Do not expose `simulac/sdk` `*Service` classes through the `simulac/lib`
   API
4. User-side code should obtain process-level capabilities through
   `obtain_runtime()` indirectly, and should access internal services through
   facades such as `simulac/sdk/world_maker.py`

## Define Time

Define time is the phase where user code describes intent in domain terms.
Examples include creating `Environment`, `Stuff`, `Robot`, `Camera`, light
objects, constraints, and randomization specs.

The define-time API should stay small and user-oriented.

At this phase, objects are declarations. They describe what should exist, what
assets should be used, and which references or constraints should be attached to
the scene. 

## Build Time

Build time is the phase where a declared scene becomes an internal environment
definition.

The public `Environment` API forwards user actions to the world-maker facade.
The facade then coordinates the environment and build services. Adding entities
creates internal data models such as:

- `EnvironmentStuffEntity`
- `EnvironmentMachineEntity`
- `EnvironmentCameraEntity`
- `EnvironmentLightEntity`

Build-time handles such as `StuffObject`, `RobotObject`, `CameraObject`, and
`LightObject` should only mutate scene definition data. Examples include setting
position, rotation, mass, friction, camera relations, references, and scene
constraints.

Build-time state can be exported as JSON using the environment schema. This is
used for review, reuse, and later reconstruction of a scene definition.

Build time ends when a `Runner` is created. At that point the public
`Environment` is frozen, and further changes to scene definition data should be
rejected. Runtime mutation must go through runtime handles instead.

## Runtime

Runtime is the phase where an environment definition is consumed by a physics
engine.


Runtime handles are separate from build-time handles:

- `StuffRuntime` changes live object state such as position, rotation, mass, and
  friction.
- `RobotRuntime` applies controls and reads robot state such as joint, link,
  site, and sensor state.
- `CameraRuntime` reads or changes live camera state.

