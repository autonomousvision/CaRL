'''
Heuristic functions that determine whether the ego vehicle is close to colliding.
'''
import numpy as np


def is_within_distance_ahead(target_location, max_distance, up_angle_th=60):
  distance = np.linalg.norm(target_location[0:2])
  if distance < 0.001:
    return True
  if distance > max_distance:
    return False
  x = target_location[0]
  y = target_location[1]
  angle = np.rad2deg(np.arctan2(y, x))
  return abs(angle) < up_angle_th


def lbc_hazard_vehicle(obs_surrounding_vehicles, proximity_threshold=9.5):
  for i, is_valid in enumerate(obs_surrounding_vehicles['binary_mask']):
    if not is_valid:
      continue

    sv_yaw = obs_surrounding_vehicles['rotation'][i][2]
    same_heading = abs(sv_yaw) <= 150

    sv_loc = obs_surrounding_vehicles['location'][i]
    with_distance_ahead = is_within_distance_ahead(sv_loc, proximity_threshold, up_angle_th=45)
    if same_heading and with_distance_ahead:
      return sv_loc
  return None


def lbc_hazard_walker(obs_surrounding_pedestrians, proximity_threshold=9.5):
  for i, is_valid in enumerate(obs_surrounding_pedestrians['binary_mask']):
    if not is_valid:
      continue
    if int(obs_surrounding_pedestrians['on_sidewalk'][i]) == 1:
      continue

    ped_loc = obs_surrounding_pedestrians['location'][i]

    dist = np.linalg.norm(ped_loc)
    degree = 162 / (np.clip(dist, 1.5, 10.5) + 0.3)

    if is_within_distance_ahead(ped_loc, proximity_threshold, up_angle_th=degree):
      return ped_loc
  return None
