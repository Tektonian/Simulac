from simulac.lib.world_maker.object import Environment, Runner, Stuff


def test_entity_build():
    env = Environment()
    basket = Stuff("trash/assets/asset/test/Basket/Basket026/model.xml")

    basket_obj1 = env.place_entity(basket, (1, 0, 0))
    basket_obj2 = env.place_entity(basket, (0, 1, 0))

    runner = Runner(env)

    runner.step([])

    with runner._debug_render() as viewer:
        while viewer.is_running():
            viewer.sync()
            runner.step([])
