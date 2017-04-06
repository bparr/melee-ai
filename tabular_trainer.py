from smash_env import *
import pickle
import sarsa

env = SmashEnv()
history = env.make()

while history[-1].state.players[1].stock != 0:
    history = env.step(3)

env.terminate()