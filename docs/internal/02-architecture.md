# Directory structure and naming rule

If you find any inconsistency please report.
The status of the Simulac is alpha-preview.
Most of the strange code patterns, inconsistency, and stranges codes in Simulac [are just bugs](youtube.com/watch?t=10m52s&v=tTEHyMgBUlY&feature=youtu.be)

1. Layer names
   - base/       = cross-cutting infrastructure
   - sdk/        = runtime services
   - lib/        = user-facing public Python API
   - cli/        = terminal UX
   - server/     = reserved server-side area (currently empty)
   - bddl/       = environment domain data/examples (currently empty)
   - data_types/ = running data saving and reading using sqlite or duckdb (currently empty)

2. Boundary names
    - common = shared contract home; often contains both interface and default impl
    - local  = local-machine implementation; codes including local machine related packages, such as `os`, `sys` etc
    - remote = network/backend implementation; codes including network related packages, such as `websocket`
    - test   = containing unit test and integration test

3. Python `Class` naming rule
    - IName                      = DI-visible interface / service identifier
    - NameService                = singleton, long-lived service
    - NameManagementService      = registry/lifecycle owner
    - NameAdapter                = bridge from domain to engine/backend
    - NameRunner                 = per-execution instance
    - NameEntity                 = class containing data only, not `method`
    - NameObject                 = class created with data in NameEntity class

4. Tests
    - integration test = test case needs external server

# Dependency rule

User code should depend on `lib/*` or top-level exports like `simulac/gym_style.py`

# Singleton rules
1. Process singleton
    - obtain_runtime() -> one SimulacRuntime cache per process

2. Service singleton
    - register_singleton(IThing, ThingService)
        1. registry stores descriptor
        2. InstantiateService resolves constructor dependencies
        3. first access creates instance
        4. later access returns cached instance
        5. (Most DI pattern inspired by Vscode)

3. Use registered singleton

If you want to create some `Class` depends on singleton classes,
Use `create_instance` function to get an instance of it.

```python
class SomeClass:
    def __init__(self, non_leading_argument: int, LogService: ILogService)

some_class_instance = instantiate_service.create_instance(SomeClass, 777)
```

# Overall architecture
```
+--------------------------------------------------------------------------------+
|                                SIMULAC ARCHITECTURE                            |
+--------------------------------------------------------------------------------+

  USER CODE
    |
    |  stable imports only
    |  - e.g.,
    |   - from simulac import Environment, Stuff
    |   - from simulac.gym_style import init_bench
    v
+--------------------------------------------------------------------------------+
| lib/                                                                           |
| user-facing interface layer                                                    |
| e.g.,                                                                          |
|   - gym_style/     : benchmark-oriented UX                                     |
|   - world_maker/   : scene-building oriented UX                                |
|                                                                                |
| philosophy:                                                                    |
| - user code should speak in domain words, hide details.                        |
| - constructors stay simple; wiring stays hidden                                |
+-------------------------------------------+------------------------------------+
                                            |
                                            | obtain_runtime()
                                            v
+--------------------------------------------------------------------------------+
| sdk/runtime.py                                                                 |
| SimulacRuntime singleton contains exposing facade                              |
|                                                                                |
| e.g.: exposes process-wide handlers                                            |
|   - logger                                                                     |
|   - environment_variable                                                       |
|   - world_maker                                                                |
|                                                                                |
| philosophy:                                                                    |
| - one runtime object per process                                               |
| - user API depends on runtime facade, not on individual services directly      |
+-------------------------------------------+------------------------------------+
                                            |
                                            | built once
                                            v
+--------------------------------------------------------------------------------+
| sdk/main.py                                                                    |
| composition root                                                               |
|                                                                                |
| register_singleton(I..., ...)                                                  |
| ServiceCollection(...)                                                         |
| InstantiateService(...)                                                        |
| adapter factory registration                                                   |
|                                                                                |
| philosophy:                                                                    |
| - cross-cutting capabilities are singleton-managed                             |
+-------------------------------------------+------------------------------------+
                                            |
                                            | resolves by interface token
                                            v
+--------------------------------------------------------------------------------+
| base/instantiate/                                                              |
| tiny DI container + dependency graph                                           |
|                                                                                |
| ServiceIdentifier                                                              |
| @service_identifier("I...")                                                    |
| SyncDescriptor                                                                 |
| InstantiateService                                                             |
| Graph + cycle detection                                                        |
|                                                                                |
| philosophy:                                                                    |
| - depend on interface names, not concrete constructors                         |
| - service creation order is framework-owned                                    |
| - lifecycle and dependency rules live below business logic                     |
+----------------------+-------------------------+-------------------------------+
                       |                         |
                       | singleton services      | runtime facade
                       |                         |
                       v                         v
       +------------------------------+   +----------------------------------+
       | sdk/*_service/common/        |   | e.g.,                            |
       |                              |   |   - sdk/world_maker.py           |
       | stateful capability managers |   |   - WorldMakerFacade             |
       | e.g.,                        |   |                                  |
       |   - log_service              |   | simplifies multi-service usage   |
       |   - telemetry_service        |   +----------------------------------+
       |   - world_service            |   
       |   - environment_service      |  
       |   - runner_service           |                    
       |   - runner_service           |                    
       +------------------------------+                    
```
