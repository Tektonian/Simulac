from tt.lib.world_maker.object import Environment, Stuff, Runner


def test_entity_build():
    env = Environment()
    basket = Stuff("tt/asset/test/Basket/Basket026/model.xml")

    basket_obj1 = env.place_stuff_entity(basket, (1, 0, 0))
    basket_obj2 = env.place_stuff_entity(basket, (0, 1, 0))

    runner = Runner(env)

    runner.step([])

    with runner.render() as viewer:
        while viewer.is_running():
            viewer.sync()
            runner.step([])
