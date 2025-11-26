from world_models.models.planet import Planet

p = Planet(env="CartPole-v1", bit_depth=5)
p.train(epochs=1)
