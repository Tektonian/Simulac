from __future__ import annotations

from typing import Protocol

from .runtime import StuffRuntime


class INativeContext(Protocol):
    engine: str
    """INativeContext class is for exposing native physics engine context.
    We are going to separate API in four layers
    
    User API, Core Runtime API, Capability API, and Native Escape Hatch API.
    
    Since, we can't provide 100% abstract layer to user we intentionally expose physics engine native layer.
    Four layers will expose APIs like below, and abstraction level we become shallow, as the level increasing.
    
        User API
            Environment
            Stuff / Robot / Camera / Light
            Runner
            RuntimeObject

        Core Runtime API
            pose
            body / collider / anchor / joint / link / actuator
            mass / friction / camera output / light control

        Capability API
            fluid
            particle
            deformable

        Native Escape Hatch
            runner.native("mujoco")
            runner.native("newton")
            runner.native("genesis")
    
    MUST have fields for each engine
    
    1. mujoco
        - data: mjData
        - model: mjModel
    2. newton
        - world
        - stages
    3. gs
        - scene
        - entities
        
    Example
    1. mujoco native access
    ```python
    with runner.native("mujoco") as mj:
        geom_id = mj.resolve_geom("cup", "body")
        mj.model.geom_friction[geom_id][0] = 0.8
        mj.forward()
    ```
    
    2. Fluid capability
    ```
    class IFluidCapability(Protocol):
        def add_fluid(
            self,
            *,
            container: VolumeRef,
            material: str,
            amount: float,
            particle_radius: float | None = None,
            entity_id: str | None = None,
        ) -> FluidRuntime: ...

    fluid: IFluidCapability = runner.capability("fluid")
    
    # If runner's physic engine is mujoco, then throw error
    

    ```
    """

    type RuntimeObject = StuffRuntime  # | Robot | etc..

    def binding(self, entity: str | RuntimeObject): ...
    def resolve_body(self, entity: str | RuntimeObject, name: str | None = None): ...
    def resolve_geom(self, entity: str | RuntimeObject, name: str): ...
    def resolve_site(self, entity: str | RuntimeObject, name: str): ...
    def resolve_volume(self, entity: str | RuntimeObject, name: str): ...
    def forward(self) -> None: ...
