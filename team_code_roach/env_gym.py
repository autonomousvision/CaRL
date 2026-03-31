'''
Gymnasium class for the CARLAEnv. Establishes communication with the env_agent and serves as gymnasium interface.
'''
import os
import math
import pathlib
import time

import gymnasium as gym
from gymnasium import spaces
import zmq
import numpy as np


class CARLAEnv(gym.Env):
  '''
    Gymnasium environment class interface. Handles communication with env_agent.py
  '''

  metadata = {'render_modes': ['rgb_array']}

  def __init__(self, port, config, render_mode='rgb_array'):  # pylint: disable=locally-disabled, unused-argument

    self.num_recv = 0

    self.config = config
    self.camera_shape = (3, self.config.cam_configs[0]['height'],
                         self.config.cam_configs[0]['width'] +
                         self.config.cam_configs[1]['width'] +
                         self.config.cam_configs[2]['width'])
    self.lidar_shape = (int((self.config.lidar_max_x - self.config.lidar_min_x) * self.config.pixels_per_meter),
                        int((self.config.lidar_max_y - self.config.lidar_min_y) * self.config.pixels_per_meter))

    if self.config.use_sensorimotor:
      self.observation_space = spaces.Dict({
          'bev_semantics':
              spaces.Box(0,
                         255,
                         shape=(config.obs_num_channels, config.bev_semantics_height, config.bev_semantics_width),
                         dtype=np.uint8),
          'measurements':
              spaces.Box(-math.inf, math.inf, shape=(config.obs_num_measurements,), dtype=np.float32),
          'value_measurements':
              spaces.Box(-math.inf, math.inf, shape=(config.num_value_measurements,), dtype=np.float32),
          'rgb': spaces.Box(0, 255, shape=self.camera_shape, dtype=np.uint8),
          'lidar': spaces.Box(0, 255, shape=self.lidar_shape, dtype=np.uint8),
          'compass': spaces.Box(-math.inf, math.inf, shape=(1,), dtype=np.float32),
          'speed': spaces.Box(-math.inf, math.inf, shape=(1,), dtype=np.float32),
          'gps': spaces.Box(-math.inf, math.inf, shape=(2,), dtype=np.float32),
          'target_point': spaces.Box(-math.inf, math.inf, shape=(2,), dtype=np.float32),
          'target_point_next': spaces.Box(-math.inf, math.inf, shape=(2,), dtype=np.float32),
      })
    else:
      self.observation_space = spaces.Dict({
        'bev_semantics':
          spaces.Box(0,
                     255,
                     shape=(config.obs_num_channels, config.bev_semantics_height, config.bev_semantics_width),
                     dtype=np.uint8),
        'measurements':
          spaces.Box(-math.inf, math.inf, shape=(config.obs_num_measurements,), dtype=np.float32),
        'value_measurements':
          spaces.Box(-math.inf, math.inf, shape=(config.num_value_measurements,), dtype=np.float32),
      })
    self.action_space = spaces.Box(config.action_space_min,
                                   config.action_space_max,
                                   shape=(config.action_space_dim,),
                                   dtype=np.float32)

    self.metadata['render_fps'] = config.frame_rate
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PAIR)
    self.port = port
    self.initialized = False

  def reset(self, seed=None, options=None):  # pylint: disable=locally-disabled, unused-argument
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    if not self.initialized:
      # Connect to env_agent.
      current_folder = pathlib.Path(__file__).parent.resolve()
      comm_folder = os.path.join(current_folder, 'comm_files')
      pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
      communication_file = os.path.join(comm_folder, str(self.port))
      self.socket.bind(f'ipc://{communication_file}.lock')
      print(f'Connecting to leaderboard gym, port: {self.port}')

      msg = self.socket.recv_string()
      print(msg)
      self.initialized = True

    data = self.socket.recv_multipart(copy=False)
    self.num_recv += 1
    if self.config.use_sensorimotor:
      start = np.frombuffer(data[16], dtype=np.int64).item()
      end = time.monotonic_ns()
      elapsed_ns = end - start  # pure integer
      elapsed_mus = elapsed_ns // 1_000

      # print("Time to send data (µs):", elapsed_mus)
      # if (elapsed_mus > 10000):
      #   print("Large networking delay:")
      observation = {
        'bev_semantics':
          np.frombuffer(data[0],
                        dtype=np.uint8).reshape(self.config.obs_num_channels, self.config.bev_semantics_height,
                                                self.config.bev_semantics_width),
        'measurements':
          np.frombuffer(data[1], dtype=np.float32),
        'value_measurements':
          np.frombuffer(data[2], dtype=np.float32),
        'rgb': np.frombuffer(data[6], dtype=np.uint8).reshape(self.camera_shape),
        'lidar': np.frombuffer(data[7], dtype=np.uint8).reshape(self.lidar_shape),
        'compass': np.frombuffer(data[8], dtype=np.float32),
        'speed': np.frombuffer(data[9], dtype=np.float32),
        'gps': np.frombuffer(data[10], dtype=np.float32).reshape(2),
        'target_point': np.frombuffer(data[11], dtype=np.float32).reshape(2),
        'target_point_next': np.frombuffer(data[12], dtype=np.float32).reshape(2),
      }

      info = {'n_steps': np.frombuffer(data[13], dtype=np.int32), 'suggest': np.frombuffer(data[14], dtype=np.int32)}
      num_sent = np.frombuffer(data[15], dtype=np.uint64).item()
    else:
      observation = {
          'bev_semantics':
              np.frombuffer(data[0],
                            dtype=np.uint8).reshape(self.config.obs_num_channels, self.config.bev_semantics_height,
                                                    self.config.bev_semantics_width),
          'measurements':
              np.frombuffer(data[1], dtype=np.float32),
          'value_measurements':
              np.frombuffer(data[2], dtype=np.float32)
      }

      info = {'n_steps': np.frombuffer(data[6], dtype=np.int32), 'suggest': np.frombuffer(data[7], dtype=np.int32)}
      num_sent = np.frombuffer(data[8], dtype=np.uint64).item()

    if self.num_recv != num_sent:
      raise ValueError(f"Communication breakdown, Leaderboard send more frames than client consumed."
                       f"num_recv: {self.num_recv}, num_sent: {num_sent}")

    return observation, info

  def step(self, action):
    self.socket.send(action.tobytes(), copy=False)
    data = self.socket.recv_multipart(copy=False)
    self.num_recv += 1

    if self.config.use_sensorimotor:
      start = np.frombuffer(data[16], dtype=np.int64).item()
      end = time.monotonic_ns()
      elapsed_ns = end - start  # pure integer
      elapsed_s = elapsed_ns / 1000000000.0

      # print("Time to send data s:", elapsed_s)
      # if (elapsed_mus > 10000):
      #   print("Large networking delay:")

      observation = {
        'bev_semantics':
          np.frombuffer(data[0],
                        dtype=np.uint8).reshape(self.config.obs_num_channels, self.config.bev_semantics_height,
                                                self.config.bev_semantics_width),
        'measurements':
          np.frombuffer(data[1], dtype=np.float32),
        'value_measurements':
          np.frombuffer(data[2], dtype=np.float32),
        'rgb': np.frombuffer(data[6], dtype=np.uint8).reshape(self.camera_shape),
        'lidar': np.frombuffer(data[7], dtype=np.uint8).reshape(self.lidar_shape),
        'compass': np.frombuffer(data[8], dtype=np.float32),
        'speed': np.frombuffer(data[9], dtype=np.float32),
        'gps': np.frombuffer(data[10], dtype=np.float32).reshape(2),
        'target_point': np.frombuffer(data[11], dtype=np.float32).reshape(2),
        'target_point_next': np.frombuffer(data[12], dtype=np.float32).reshape(2),
      }

      reward = np.frombuffer(data[3], dtype=np.float32).item()
      termination = np.frombuffer(data[4], dtype=bool).item()  # True if agent ended in destroy method.
      truncation = np.frombuffer(data[5], dtype=bool).item()  # True if agent timed out.

      info = {
        'n_steps': np.frombuffer(data[13], dtype=np.int32).item(),
        'suggest': np.frombuffer(data[14], dtype=np.int32).item()
      }

      num_sent = np.frombuffer(data[15], dtype=np.uint64).item()
    else:
      observation = {
          'bev_semantics':
              np.frombuffer(data[0],
                            dtype=np.uint8).reshape(self.config.obs_num_channels, self.config.bev_semantics_height,
                                                    self.config.bev_semantics_width),
          'measurements':
              np.frombuffer(data[1], dtype=np.float32),
          'value_measurements':
              np.frombuffer(data[2], dtype=np.float32)
      }

      reward = np.frombuffer(data[3], dtype=np.float32).item()
      termination = np.frombuffer(data[4], dtype=bool).item()  # True if agent ended in destroy method.
      truncation = np.frombuffer(data[5], dtype=bool).item()  # True if agent timed out.

      info = {
          'n_steps': np.frombuffer(data[6], dtype=np.int32).item(),
          'suggest': np.frombuffer(data[7], dtype=np.int32).item()
      }

      num_sent = np.frombuffer(data[8], dtype=np.uint64).item()

    if self.num_recv != num_sent:
      raise ValueError(f"Communication breakdown, Leaderboard send more frames than client consumed."
                       f"num_recv: {self.num_recv}, num_sent: {num_sent}")

    return observation, reward, termination, truncation, info

  def close(self):
    print('Called close!')
