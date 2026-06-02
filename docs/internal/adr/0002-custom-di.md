# ADR 0002: Custom Dependency Injection

## Status

Accepted.

## Context

Simulac has several long-lived services that should be shared within a Python
process:

- logging;
- environment and world management;
- runner management;
- asset and file services.

## Decision

Instead of using external package, use the local dependency-injection implementation in `simulac/base/instantiate`.

The key pieces are:

- `ServiceIdentifier`: marker base for DI-visible service interfaces.
- `@service_identifier(...)`: decorator that registers interface names.
- `register_singleton(...)`: records singleton service descriptors.
- `ServiceCollection`: stores descriptors and created instances.
- `InstantiateService`: resolves constructor dependencies, creates services,
  caches singleton instances, and checks dependency cycles.

### Pros

Customizable and scable code

### Cons

DI logic depends on `inspect` package, which looks very bad. (blame @gangjeuk)

Type definition is forced to use `*Service`

```python
# ✅
class FooClass:
    def __init__(self, LogService: ILogService): ...

# ⛔️ - Can't use DI
class FooClass:
    def __init__(self, LogService): ...

```

## Consequences

Internal services can depend on interfaces and can be replaced centrally from
the composition root.

The custom DI implementation is small and project-controlled, but it also means
the project owns its edge cases. Constructor annotations, service registration
order, singleton caching, and cycle detection must be maintained carefully.

## Rules

- Service interfaces should inherit from `ServiceIdentifier`.
- Service interfaces should be decorated with `@service_identifier`.
- Must set `I*Service` for using DI pattern

## Open Work

The implementation still contains TODOs around tracing, proxying, and service
dependency handling. Those should be improved without changing the public API
contract that user code stays outside the service container.
