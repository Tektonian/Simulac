from tt.base.error.error import TektonianBaseError
from tt.sdk.environment_service.common.model.entity import EnvironmentMJCFObjectEntity
from tt.sdk.runner_service.common.physics_engine_adapter import PhysicsEngineAdapter

MUJOCO_SCENE = """
<mujoco model="scene">
  <statistic center="0.3 0 0.4" extent="1"/>

  <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <custom>
    <numeric data="12" name="max_contact_points"/>
  </custom>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
    <scale contactwidth="0.075" contactheight="0.025" forcewidth="0.05" com="0.05" framewidth="0.01" framelength="0.2"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
  </worldbody>
</mujoco>
"""


import mujoco


class MujocoAdapter(PhysicsEngineAdapter):
  """_summary_
      ![test](https://picsum.photos/200/300)

  Args:
      PhysicsEngineAdapter (_type_): _description_
  """

    def __init__(self) -> None:

        self.root_spec = mujoco.MjSpec.from_string(MUJOCO_SCENE)
        self.root_frame = self.root_spec.worldbody.add_frame()
        
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

    def add_object(self, obj: EnvironmentMJCFObjectEntity) -> str:
        child = mujoco.MjSpec.from_file(
            obj.physics.mjcf_uri,
        )

        child_bodies: list[mujoco.MjsBody] = child.bodies

        # change pos
        for body in child_bodies:
            if body.parent == child.worldbody:
                body.pos = list(obj.pos)
                body.quat = list(obj.quat)

        self.root_spec.attach(child, prefix=obj.uuid)

        return obj.uuid
      
    def remove_object(self, obj_id: int) -> None:...
    
    def start_physics_engine(self) -> None:
        model: mujoco.MjModel = self.root_spec.compile()
        self.data = mujoco.MjData(model)
        self.model = model
    
    def step(self, dt: float) -> None: 
        if self.model is None or self.data is None:
            raise TektonianBaseError('Physics engine not initialized')
        
        mujoco.mj_step(self.model, self.data)
        
    def reset(self) -> None:
        if self.model is None or self.data is None:
            raise TektonianBaseError('Physics engine not initialized')
        
        mujoco.mj_resetData(self.model, self.data)
    