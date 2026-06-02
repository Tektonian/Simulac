# ADR 0004: Environment Schema

## Status

Pending.

## Context

Simulac needs a durable representation of scene definitions so environments can
be reviewed, saved, loaded, reused, and later consumed by runner adapters.

The public API lets users build scenes with `Environment`, entity constructors,
object handles, references, constraints, and randomization specs. 

## Decision

Use a JSON environment schema. Have no idea which is the perfect data format for now.

The top-level schema contains:

- `id`
- `world_id`
- `physics_engine`
- `stuffs`
- `machines`
- `cameras`
- `lights`
- `relations`
- `constraints`

Especially `id` and `world_id` is for sharing environment setting.

With `<id>/<world_id>` format, such as, `Tektonian/RandomPick`.

Entity groups map to internal model classes:

- `stuffs` uses `EnvironmentStuffEntity`.
- `machines` uses `EnvironmentMachineEntity`.
- `cameras` uses `EnvironmentCameraEntity`.
- `lights` uses `EnvironmentLightEntity`.


### Include randomization

Initially, we did not plan to include randomization, but after confirming its necessity, we added it as an official specification.

## Consequences

The schema keeps scene data reviewable and independent from the user-facing
Python objects that created it.


## Open Work

The schema should eventually define stricter validation for asset formats,
engine compatibility, randomization specs, relation semantics, and complete
runtime-state capture.
