# ADR 0001: Layered Architecture

## Status

Accepted.

## Context

Simulac needs to support several surfaces at the same time:

- scene building and local simulation
- a hosted benchmark client with a Gym-style API
- handle various physics engines

Additional services will be implemented in the future
- remote simulation
- GUI for Environment editing

We should keeping the future requirements plugable and stay user-side API clean.

## Decision
Use a layered architecture:

- `simulac/lib`: user-side Python APIs.
- `simulac/cli`: terminal UX.
- `simulac/sdk`: runtime facade, service interfaces, service implementations,
  and adapters.
- `simulac/base`: cross-cutting infrastructure such as errors, environment
  variables, result types, and dependency injection.

User code should import from top-level `simulac`.

User code should not depend on `sdk/*Service` classes.

`sdk/runtime.py` provides `obtain_runtime()`, which returns one process-level
`SimulacRuntime`. The runtime exposes facades such as `world_maker`, `logger`, etc.

`sdk/main.py` is the composition root. It registers singleton services and
physics adapter factories, then constructs the dependency-injection service
collection.

## Consequences

The public API can stay domain-oriented while internal services remain free to
evolve.

The cost is that some internal calls are indirect. Debugging may require
following the path from `lib` to `sdk/runtime.py`, then to facades and services.


