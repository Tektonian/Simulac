from tt.sdk.environment_service.common.model.component import (
    MJCFPhysicsComponent,
    RenderingComponent,
)
from tt.sdk.environment_service.common.model.entity import EnvironmentObjectEntity


class Stuff:
    def __init__(self, obj_uri_or_prebuilt_id: str, name: str | None = None) -> None:
        """_summary_
            TODO:
                1. Handle various name. Expected strings are
                    - Tektonian/cup/cup0 [object with remote owner]
                    - https://tektonian.com/~~ [remote asset]
                    - ./home/mjcf.xml [local asset]
                2. Seperate cases for mjcf, urdf, usd
        Args:
            obj_uri_or_prebuilt_name (str): _description_
        """
        rendering = RenderingComponent("", "")
        physic = MJCFPhysicsComponent(obj_uri_or_prebuilt_id)
        self.entity = EnvironmentObjectEntity(
            rendering=rendering, physics=physic, name=name
        )


class Robot:
    def __init__(self, obj_uri_or_prebuilt_id: str, name: str) -> None:
        rendering = RenderingComponent("", "")
        physic = MJCFPhysicsComponent(obj_uri_or_prebuilt_id)
        self.entity = EnvironmentObjectEntity(
            rendering=rendering, physics=physic, name=name
        )
