'''
Contains a class that computes a reward function.
'''

import numpy as np
import carla

from reward.criteria import run_red_light, run_stop_sign, collision, blocked
from reward.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker
from birds_eye_view.transforms import get_loc_rot_vel_in_ev, loc_global_to_ref
from birds_eye_view.traffic_light import TrafficLightHandler
import rl_utils as rl_u


class RoachReward(object):
  '''
    Reproduced reward function from "End-to-End Urban Driving by Imitating a Reinforcement Learning Coach"
    which is an extension of "End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances"
    '''

  def __init__(self, vehicle, world_map, world, config):
    self.vehicle = vehicle
    self.config = config
    # Note requires TrafficLight handler to be active which is done in the BEV SS observation.
    self.red_light_infraction_detector = run_red_light.RunRedLight(world_map, self.config.penalize_yellow_light)
    self.stop_infraction_detector = run_stop_sign.RunStopSign(world, world_map)
    self.collision_detector = collision.Collision(vehicle, world)
    self.block_detector = blocked.Blocked()
    self.world = world
    self.world_map = world_map
    self.last_lat_dist = 0.0
    self.last_steer = 0.0

  def get(
      self,
      timestamp,
      waypoint_route,
      collision_with_pedestrian,
      vehicles_all=(),  # pylint: disable=locally-disabled, unused-argument
      walkers_all=(),  # pylint: disable=locally-disabled, unused-argument
      static_all=(),  # pylint: disable=locally-disabled, unused-argument
      perc_off_road=None):  # pylint: disable=locally-disabled, unused-argument

    #########################################################################
    # Compute termination conditions and terminal reward.
    #########################################################################
    terminal_reward = 0.0
    closest_route_point = self.get_closest_route_point(waypoint_route)
    # Done condition 1: vehicle blocked
    ego_blocked = self.block_detector.tick(self.vehicle, timestamp) is not None

    # Done condition 2: lateral distance too large
    # ego point is in route coordinate frame. x front y right. so y contains the lateral distance
    lat_dist = abs(closest_route_point[1])

    if lat_dist - self.last_lat_dist > 0.8:
      thresh_lat_dist = lat_dist + 0.5  # Note: Not described in the roach paper, but should almost never trigger?
    else:
      thresh_lat_dist = max(self.config.min_thresh_lat_dist, self.last_lat_dist)

    route_deviation = lat_dist > thresh_lat_dist + 0.01
    self.last_lat_dist = lat_dist

    # Done condition 3: running red light
    ran_red_light = self.red_light_infraction_detector.tick(self.vehicle) is not None

    # # Done condition 4: collision
    collision_detected = self.collision_detector.tick(self.vehicle, timestamp) is not None
    # c_collision = self._ego_vehicle.info_criteria['collision'] is not None
    # Done condition 5: run stop sign
    stop_criteria = self.stop_infraction_detector.tick(self.vehicle)
    ran_stop_sign = (stop_criteria is not None) and (stop_criteria['event'] == 'run')

    # Done condition 6: Collision in Pixel space
    collision_pixel_space = collision_with_pedestrian

    # endless env: timeout means succeed
    timeout = timestamp > self.config.eval_time

    finished_route = False
    if len(waypoint_route) < self.config.num_route_points_rendered:
      finished_route = True

    # terminal guide
    # 0: '', ''
    # 1: 'go', ''
    # 2: 'go', 'turn'
    # 3: 'stop', ''
    exploration_suggest = {'n_steps': 0, 'suggest': 0, 'infraction_type': ''}
    if self.config.use_exploration_suggest:
      if ego_blocked:
        exploration_suggest['n_steps'] = self.config.n_step_exploration
        exploration_suggest['suggest'] = 1
      if route_deviation:
        exploration_suggest['n_steps'] = self.config.n_step_exploration
        exploration_suggest['suggest'] = 2
      if ran_red_light or collision_detected or ran_stop_sign or collision_pixel_space:
        exploration_suggest['n_steps'] = self.config.n_step_exploration
        exploration_suggest['suggest'] = 3

    if route_deviation:
      print('route_deviation')
      exploration_suggest['infraction_type'] = 'route_deviation'
    if ran_red_light:
      print('Run Red Light')
      exploration_suggest['infraction_type'] = 'ran_red_light'
    if ran_stop_sign:
      print('Run Stop Sign')
      exploration_suggest['infraction_type'] = 'ran_stop_sign'
    if collision_detected:
      print('Collision detected')
      exploration_suggest['infraction_type'] = 'collision_detected'
    if collision_pixel_space:
      print('Collision detected in BEV observation')
      exploration_suggest['infraction_type'] = 'collision_pixel_space'
    if ego_blocked:
      print('Vehicle is stuck')
      exploration_suggest['infraction_type'] = 'ego_blocked'
    if timeout:
      print('Success. Agent timed out.')
      exploration_suggest['infraction_type'] = 'timeout'
    if finished_route:
      print('Finished route')
      exploration_suggest['infraction_type'] = 'finished_route'

    termination = (route_deviation or ran_red_light or ran_stop_sign or collision_detected or collision_pixel_space or
                   ego_blocked)

    truncation = timeout or finished_route
    if termination or truncation:
      terminal_reward = -1.0

    ev_vel = self.vehicle.get_velocity()  # in m/s
    ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))
    if ran_red_light or ran_stop_sign or collision_detected or collision_pixel_space:
      terminal_reward -= ev_speed

    #########################################################################
    # Compute shaped per time step rewards
    #########################################################################
    # action reward
    current_control = self.vehicle.get_control()
    if abs(current_control.steer - self.last_steer) > 0.01:
      r_action = -0.1
    else:
      r_action = 0.0
    self.last_steer = current_control.steer

    # desired_speed reward
    obs_vehicle = self.extract_nearby_vehicles()
    obs_pedestrian = self.extract_nearby_pedestrians()

    # all locations in ego_vehicle coordinate
    hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=self.config.rr_vehicle_proximity_threshold)
    hazard_ped_loc = lbc_hazard_walker(obs_pedestrian,
                                       proximity_threshold=self.config.rr_pedestrian_proximity_threshold)
    light_state, light_loc, _ = TrafficLightHandler.get_light_state(self.vehicle,
                                                                    offset=self.config.rr_tl_offset,
                                                                    dist_threshold=self.config.rr_tl_dist_threshold)

    # if hazard_vehicle_loc is not None:
    #   print('Vehicle hazard')
    #
    # if hazard_ped_loc is not None:
    #   print('Walker hazard')

    if self.config.use_speed_limit_as_max_speed:
      speed_limit = self.vehicle.get_speed_limit()
      if isinstance(speed_limit, float):
        # Speed limit is in km/h we compute with m/s, so we convert it by / 3.6
        maximum_speed = speed_limit / 3.6
      else:
        #  Car can have no speed limit right after spawning
        maximum_speed = self.config.rr_maximum_speed
    else:
      maximum_speed = self.config.rr_maximum_speed

    desired_spd_veh = desired_spd_ped = desired_spd_rl = desired_spd_stop = maximum_speed

    if hazard_vehicle_loc is not None:
      dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2]) - 8.0)
      desired_spd_veh = maximum_speed * np.clip(dist_veh, 0.0, 5.0) / 5.0

    if hazard_ped_loc is not None:
      dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2]) - 6.0)
      desired_spd_ped = maximum_speed * np.clip(dist_ped, 0.0, 5.0) / 5.0

    if light_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
      dist_rl = max(0.0, np.linalg.norm(light_loc[0:2]) - 5.0)
      desired_spd_rl = maximum_speed * np.clip(dist_rl, 0.0, 5.0) / 5.0

    # stop sign
    stop_sign = self.stop_infraction_detector.target_stop_sign

    ev_transform = self.vehicle.get_transform()

    if (stop_sign is not None) and (not self.stop_infraction_detector.stop_completed):
      trans = stop_sign.get_transform()
      tv_loc = stop_sign.trigger_volume.location
      loc_in_world = trans.transform(tv_loc)
      loc_in_ev = loc_global_to_ref(loc_in_world, ev_transform)
      stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
      # Stop sign infraction logic changed compared to leaderboard 1.0, so we replace the hardcoded 5 meters with the
      # 4 meters of the PROXIMITY_THRESHOLD
      thresh = run_stop_sign.RunStopSign.PROXIMITY_THRESHOLD
      dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2]) - thresh)
      desired_spd_stop = maximum_speed * np.clip(dist_stop, 0.0, thresh) / thresh

    desired_speed = min(maximum_speed, desired_spd_veh, desired_spd_ped, desired_spd_rl, desired_spd_stop)

    # r_speed
    if ev_speed > maximum_speed:
      # r_speed = 0.0
      r_speed = 1.0 - np.abs(ev_speed - desired_speed) / maximum_speed
    else:
      r_speed = 1.0 - np.abs(ev_speed - desired_speed) / maximum_speed

    # r_position
    r_position = -1.0 * (lat_dist / 2.0)

    # r_rotation
    angle_difference = np.deg2rad(np.abs(closest_route_point[2]))
    r_rotation = -1.0 * angle_difference

    reward = r_speed + r_position + r_rotation + r_action + terminal_reward

    return reward, termination, truncation, exploration_suggest

  def destroy(self):
    self.collision_detector.clean()

  def extract_nearby_vehicles(self):
    ev_transform = self.vehicle.get_transform()
    ev_location = ev_transform.location

    def dist_to_ev(w):
      return w.get_location().distance(ev_location)

    surrounding_vehicles = []
    vehicle_list = self.world.get_actors().filter('*vehicle*')
    for vehicle in vehicle_list:
      has_different_id = self.vehicle.id != vehicle.id
      is_within_distance = dist_to_ev(vehicle) <= self.config.vehicle_distance_threshold
      if has_different_id and is_within_distance:
        surrounding_vehicles.append(vehicle)

    sorted_surrounding_vehicles = sorted(surrounding_vehicles, key=dist_to_ev)

    location, rotation, absolute_velocity = get_loc_rot_vel_in_ev(sorted_surrounding_vehicles, ev_transform)

    binary_mask, extent, road_id, lane_id = [], [], [], []
    for sv in sorted_surrounding_vehicles[:self.config.max_vehicle_detection_number]:
      binary_mask.append(1)

      bbox_extent = sv.bounding_box.extent
      extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

      loc = sv.get_location()
      wp = self.world_map.get_waypoint(loc)
      road_id.append(wp.road_id)
      lane_id.append(wp.lane_id)

    for _ in range(self.config.max_vehicle_detection_number - len(binary_mask)):
      binary_mask.append(0)
      location.append([0, 0, 0])
      rotation.append([0, 0, 0])
      extent.append([0, 0, 0])
      absolute_velocity.append([0, 0, 0])
      road_id.append(0)
      lane_id.append(0)

    obs_dict = {
        'frame': self.world.get_snapshot().frame,
        'binary_mask': np.array(binary_mask, dtype=np.int8),  # Needed
        'location': np.array(location, dtype=np.float32),  # Needed
        'rotation':
            np.array(rotation, dtype=np.float32)  # Needed
    }
    return obs_dict

  def get_closest_route_point(self, waypoint_route):
    '''
    :return: The ego agents position in the coordinate system of the closest route point.
    '''
    ego_vehicle_transform = self.vehicle.get_transform()
    pos = ego_vehicle_transform.location
    pos = np.array([pos.x, pos.y])

    if len(waypoint_route) > 1:
      close_point_global = np.array([waypoint_route[0][0].location.x, waypoint_route[0][0].location.y])
      next_point_global = np.array([waypoint_route[1][0].location.x, waypoint_route[1][0].location.y])
      distance = next_point_global - close_point_global

      # Compute orientation of route.
      if np.linalg.norm(distance) < 0.1:
        # For cases where the points are too close to each other the orientation vector may be too random.
        # We use the orientation of the waypoint itself instead which usually also points in the direction of the route.
        yaw_route = waypoint_route[0][0].rotation.yaw
      else:
        route_vector = distance
        yaw_route = np.rad2deg(np.arctan2(route_vector[1], route_vector[0]))
    else:
      close_point_global = np.array([waypoint_route[0][0].location.x, waypoint_route[0][0].location.y])
      # No next point, so we use the orientation of the route waypoint as direction of the route.
      yaw_route = waypoint_route[0][0].rotation.yaw

    ego_in_route_coordinate = rl_u.inverse_conversion_2d(pos, close_point_global, np.deg2rad(yaw_route))
    ego_in_route_yaw = rl_u.normalize_angle_degree(ego_vehicle_transform.rotation.yaw - yaw_route)
    ego_in_route_coordinate = np.append(ego_in_route_coordinate, ego_in_route_yaw)

    return ego_in_route_coordinate

  def extract_nearby_pedestrians(self):
    ev_transform = self.vehicle.get_transform()
    ev_location = ev_transform.location

    def dist_to_actor(w):
      return w.get_location().distance(ev_location)

    surrounding_pedestrians = []
    pedestrian_list = self.world.get_actors().filter('*walker.pedestrian*')
    for pedestrian in pedestrian_list:
      if dist_to_actor(pedestrian) <= self.config.pedestrian_distance_threshold:
        surrounding_pedestrians.append(pedestrian)

    sorted_surrounding_pedestrians = sorted(surrounding_pedestrians, key=dist_to_actor)

    location, rotation, absolute_velocity = get_loc_rot_vel_in_ev(sorted_surrounding_pedestrians, ev_transform)

    binary_mask, extent, on_sidewalk, road_id, lane_id = [], [], [], [], []
    for ped in sorted_surrounding_pedestrians[:self.config.max_pedestrian_detection_number]:
      binary_mask.append(1)

      bbox_extent = ped.bounding_box.extent
      extent.append([bbox_extent.x, bbox_extent.y, bbox_extent.z])

      loc = ped.get_location()
      wp = self.world_map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
      if wp is None:
        on_sidewalk.append(1)
      else:
        on_sidewalk.append(0)
      wp = self.world_map.get_waypoint(loc)
      road_id.append(wp.road_id)
      lane_id.append(wp.lane_id)

    for _ in range(self.config.max_pedestrian_detection_number - len(binary_mask)):
      binary_mask.append(0)
      location.append([0, 0, 0])
      rotation.append([0, 0, 0])
      absolute_velocity.append([0, 0, 0])
      extent.append([0, 0, 0])
      on_sidewalk.append(0)
      road_id.append(0)
      lane_id.append(0)

    obs_dict = {
        'binary_mask': np.array(binary_mask, dtype=np.int8),
        'location': np.array(location, dtype=np.float32),
        'on_sidewalk': np.array(on_sidewalk, dtype=np.int8)
    }

    return obs_dict
