'''
Simple file to test the carla env gymnasium wrapper.
'''

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

register(
    id='CARLAEnv-v0',
    entry_point='env_gym:CARLAEnv',
    max_episode_steps=None,
)


def make_env(gym_id, port):

  def thunk():
    env = gym.make(gym_id, port=port)
    env = gym.wrappers.ClipAction(env)
    return env

  return thunk


envs = gym.vector.SyncVectorEnv([make_env('CARLAEnv-v0', 5555 + i) for i in range(1)])
obs = envs.reset()
i = 0
while True:
  observation, reward, termination, truncation, info = envs.step(np.array([[0.0, 1.0], [0.0, 1.0]]))
  i += 1

print('Finished training.')
