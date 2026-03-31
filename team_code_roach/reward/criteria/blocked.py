'''
Contains a class that checks if a vehicle is stuck for a certain amount of time. Used in reward computation.
'''

import numpy as np


class Blocked():
  '''
  A class that checks if a vehicle is stuck for a certain amount of time.
  '''

  def __init__(self, speed_threshold=0.1, below_threshold_max_time=90.0):
    self._speed_threshold = speed_threshold
    self._below_threshold_max_time = below_threshold_max_time
    self._time_last_valid_state = None
    self.time_till_blocked = 1.0  # For value measurements, percentage of time left, till blocked

  def tick(self, vehicle, timestamp):
    info = None
    linear_speed = self._calculate_speed(vehicle.get_velocity())

    if linear_speed < self._speed_threshold and self._time_last_valid_state:
      time_diff = timestamp - self._time_last_valid_state
      self.time_till_blocked = (self._below_threshold_max_time - time_diff) / self._below_threshold_max_time
      if time_diff > self._below_threshold_max_time:
        # The actor has been "blocked" for too long
        ev_loc = vehicle.get_location()
        info = {'simulation_time': timestamp, 'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]}
    else:
      self._time_last_valid_state = timestamp
      self.time_till_blocked = 1.0
    return info

  @staticmethod
  def _calculate_speed(carla_velocity):
    return np.linalg.norm([carla_velocity.x, carla_velocity.y])
