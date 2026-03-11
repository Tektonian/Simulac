from tt.lib.world_maker.object import Environment, Stuff, Runner
import mujoco
from PIL import Image


def test_entity_build():
    env = Environment()
    basket = Stuff("tt/asset/test/Bin/Bin001/model.xml")

    basket_obj1 = env.place_stuff_entity(basket, (1, 0, 0))
    basket_obj2 = env.place_stuff_entity(basket, (0, 1, 0))

    runner = Runner(env)

    runner.step([])

    renderer = mujoco.Renderer(runner._runner.mj_model, height=240, width=320)
    try:
        renderer.update_scene(runner._runner._data)
        Image.fromarray(renderer.render()).save("test_entity_build.png")
    finally:
        renderer.close()
