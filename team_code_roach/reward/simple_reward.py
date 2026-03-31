'''
Contains a class that computes a reward function.
'''
import math
from collections import deque
from dataclasses import dataclass

import carla
import numpy as np
from scipy.signal import savgol_filter

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from reward.criteria import run_red_light, run_stop_sign, run_stop_sign2, collision, blocked, route_completion, in_route_test, outside_route_lanes
import rl_utils as rl_u


@dataclass
class EgoState:
  longitudinal_acceleration = deque(maxlen=15)
  lateral_acceleration = deque(maxlen=15)
  acceleration_magnitude = deque(maxlen=15)
  yaw = deque(maxlen=15)


class SimpleReward(object):
  '''
    A simple reward, that only tells the agent what to do not how to do it. E.g. it does not compute any optimal speed
    like the roach reward. It's designed to mimic the DS (with the same global optimum) but be easier to optimize.
    '''

  def __init__(self, vehicle, world_map, world, config, route):
    self.vehicle = vehicle
    self.config = config
    # Note requires TrafficLight handler to be active which is done in the BEV SS observation.
    self.red_light_infraction_detector = run_red_light.RunRedLight(world_map, self.config.penalize_yellow_light)
    if self.config.use_new_stop_sign_detector:
      self.stop_infraction_detector = run_stop_sign2.RunStopSign2(world, world_map)
    else:
      self.stop_infraction_detector = run_stop_sign.RunStopSign(world, world_map)
    self.collision_detector = collision.Collision(vehicle, world)
    self.block_detector = blocked.Blocked()
    self.route_completion = route_completion.RouteCompletionTest(route, world_map)
    self.outside_route_lanes = outside_route_lanes.OutsideRouteLanesTest(vehicle, world_map)
    self.in_route = in_route_test.InRouteTest(vehicle, route)
    self.world = world
    self.world_map = world_map
    self.last_route_completion = 0.0
    self.last_acceleration = np.array([0.0, 0.0])
    self.last_rotation = None
    self.last_abs_yaw_diff_rad = 0.0
    self.past_mag_jerk = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
    self.past_lon_jerk = deque([0.0, 0.0, 0.0, 0.0, 0.0], maxlen=5)
    self.ego_model = EgoModel(dt=self.config.time_interval)
    self.first_frame = True
    self.last_action = carla.VehicleControl(steer=0.0, throttle=0.0, brake=0.0)

    # For TTC, only need to compute once
    self.ttc_future_time_deltas = np.arange(1.0, 10, self.config.ttc_resolution, dtype=int) * self.config.time_interval
    self.remaining_ttc_penalty_ticks = 0  # How many ticks the agent is still punished for violating TTC.

    # For comfort
    self.ego_state_history = EgoState()
    self.dx = self.config.time_interval * self.config.action_repeat
    self.remaining_comfort_penalty_ticks = np.zeros(6)  # For each of the 6 individual comfort metrics.
    self.last_loc = None
    self.last_velocity = carla.Vector3D()
    self.lane_dist_penalty_counter = 0  # Counts for how many frames the agent has violated the lane dist penalty
    # self.comfort_histogram = {'acc_lon': [],
    #                           'acc_lat': [],
    #                           'jerk': [],
    #                           'jerk_lon': [],
    #                           'yaw_rate': [],
    #                           'yaw_acceleration': []}

  # We keep collision_with_pedestrian to have a common interface but it is not used.
  def get(
      self,
      timestamp,
      waypoint_route,
      collision_with_pedestrian=None,  # pylint: disable=locally-disabled, unused-argument
      vehicles_all=(),
      walkers_all=(),
      static_all=(),
      perc_off_road=None):  # pylint: disable=locally-disabled, unused-argument

    #########################################################################
    # Compute termination conditions and terminal reward.
    #########################################################################
    ego_vehicle_location = self.vehicle.get_location()
    ego_vehicle_transform = self.vehicle.get_transform()
    ev_vel = self.vehicle.get_velocity()  # in m/s
    ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))

    # Done condition 1: vehicle blocked
    ego_blocked = self.block_detector.tick(self.vehicle, timestamp) is not None

    # Done condition 2: lateral distance too large
    # ego point is in route coordinate frame. x front y right. so y contains the lateral distance
    route_deviation = False
    if self.config.use_leave_route_done:
      closest_route_point = self.get_closest_route_point(waypoint_route)
      lat_dist = abs(closest_route_point[1])
      route_deviation = lat_dist > self.config.min_thresh_lat_dist
    # else:
    #   # Ego agent left the road altogether terminate episode.
    #   # NOTE double check this with scenarios where the agent has to drive on the sidewalk.
    #   if self.world_map.get_waypoint(ego_vehicle_location, project_to_road=False) is None:
    #     route_deviation = True

    if self.config.consider_tl:
      # Done condition 3: running red light
      ran_red_light = self.red_light_infraction_detector.tick(self.vehicle) is not None
    else:
      ran_red_light = False

    # # Done condition 4: collision
    collision_detected = self.collision_detector.tick(self.vehicle, timestamp) is not None

    # Done condition 5: run stop sign
    stop_criteria = self.stop_infraction_detector.tick(self.vehicle)
    ran_stop_sign = (stop_criteria is not None) and (stop_criteria['event'] == 'run')

    # Done condition 6: Agent is too close to other actor
    is_vehicle_too_close = False
    if self.config.use_vehicle_close_penalty:
      is_vehicle_too_close = self.vehicle_too_close(ego_vehicle_transform, ev_speed, self.vehicle.get_control(),
                                                    vehicles_all, walkers_all)

    # Done condition 7: Route deviation
    route_deviation_2 = not self.in_route.update()

    # Done condition 8: Driving off the drivable area
    off_road_term = False
    if perc_off_road > self.config.off_road_term_perc and self.config.use_off_road_term:
      off_road_term = True

    # Soft penalty 0: Outside route lanes
    outside_lanes = False
    if self.config.use_outside_route_lanes:
      outside_lanes = self.outside_route_lanes.update()

    # Soft penalty 1: check if agent drives on lane markings. Around 0.2ms
    current_wp = self.world_map.get_waypoint(ego_vehicle_location, project_to_road=True)
    dist_to_centerline_penalty = 1.0

    if not current_wp.is_junction:
      if self.config.use_perc_progress:
        close_point_global = np.array([current_wp.transform.location.x, current_wp.transform.location.y])
        # No next point, so we use the orientation of the route waypoint as direction of the route.
        yaw_route = current_wp.transform.rotation.yaw
        ego_pos = np.array([ego_vehicle_location.x, ego_vehicle_location.y])

        ego_in_lane_coordinate = rl_u.inverse_conversion_2d(ego_pos, close_point_global, np.deg2rad(yaw_route))
        lat_dist_to_closest_lane_center = abs(ego_in_lane_coordinate[1])
        violation_length = np.clip(lat_dist_to_closest_lane_center - self.config.lane_distance_violation_threshold,
                                   a_min=0.0,
                                   a_max=None)

        if violation_length > 0.0:
          self.lane_dist_penalty_counter += 1
        else:
          if self.lane_dist_penalty_counter > 0:
            self.lane_dist_penalty_counter -= 1
          else:
            self.lane_dist_penalty_counter = 0

        if self.lane_dist_penalty_counter > self.config.lane_dist_penalty_time:
          dist_to_centerline_penalty = 0.5

    # Soft penalty 2: Is the agent speed within speed limit?
    agent_too_fast = False
    speed_penalty = 1.0
    if self.config.speeding_infraction:
      speed_limit = self.vehicle.get_speed_limit()
      if isinstance(speed_limit, float):
        # Speed limit is in km/h we compute with m/s, so we convert it by / 3.6
        speed_limit = speed_limit / 3.6
      else:
        #  Car can have no speed limit right after spawning
        speed_limit = self.config.rr_maximum_speed

      exceeding_speed = ev_speed - speed_limit
      if exceeding_speed > 0.0:
        violation_loss = exceeding_speed / self.config.max_overspeed_value_threshold
        speed_penalty = max(0.0, 1.0 - violation_loss)
        agent_too_fast = True

    # Soft penalty 3: Is the agent speed within comfort limit?
    comfort_penalty = 1.0
    if self.config.use_comfort_infraction:
      comfort_penalty = self.compute_comfort_penalty()

    # Soft penalty 4: Is actor too slow?
    fraction_of_speed = 1.0
    if self.config.use_min_speed_infraction:
      fraction_of_speed = self.is_ego_too_slow(ev_speed, vehicles_all)

    # Soft penalty 5: Did action change too much
    action_changed_too_much = False
    if self.config.use_max_change_penalty:
      action = self.vehicle.get_control()
      steer_diff = abs(action.steer - self.last_action.steer)
      throt_diff = abs(action.throttle - self.last_action.throttle)
      brake_diff = abs(action.brake - self.last_action.brake)
      if (steer_diff > self.config.max_change or throt_diff > self.config.max_change or
          brake_diff > self.config.max_change):
        action_changed_too_much = True
      self.last_action = action

    # Soft penalty 6: TTC violation
    ttc_penalty = False
    if self.config.use_ttc:
      if self.remaining_ttc_penalty_ticks > 0:
        self.remaining_ttc_penalty_ticks -= 1

      ttc_penalty = self.does_agent_violate_ttc(ego_vehicle_transform, ev_speed, vehicles_all, walkers_all, static_all)

      if ttc_penalty:
        self.remaining_ttc_penalty_ticks = self.config.ttc_penalty_ticks

      if self.remaining_ttc_penalty_ticks > 0:
        ttc_penalty = True

    timeout = timestamp > self.config.eval_time

    finished_route = False
    if len(waypoint_route) < 2:
      finished_route = True

    if route_deviation:
      print('route_deviation')
    if route_deviation_2:
      print('route_deviation 2')
    if ran_red_light:
      print('Run Red Light')
    if ran_stop_sign:
      print('Run Stop Sign')
    if collision_detected:
      print('Collision detected')
    if ego_blocked:
      print('Vehicle is stuck')
    if timeout:
      print('Agent timed out.')
    if finished_route:
      print('Finished route')
    if is_vehicle_too_close:
      print('Agent is too close to the leading vehicle.')
    if off_road_term:
      print('Agent drove off the drivable area.')
    # if agent_too_fast:
    #   print('Agent is driving too fast.')
    # if agent_drives_uncomfortable:
    #   print('Agent is driving unconformable.')
    # if outside_lanes:
    #   print('outside_lanes')
    # if action_changed_too_much:
    #   print('Action changed too much')

    termination = (route_deviation or ran_red_light or ran_stop_sign or collision_detected or ego_blocked or timeout or
                   is_vehicle_too_close or route_deviation_2 or off_road_term)
    truncation = finished_route

    # No penalty for finishing the route.
    terminal_reward = 0.0
    if termination:
      terminal_reward = self.config.terminal_reward
      if self.config.use_termination_hint:
        if self.config.use_rl_termination_hint:
          condition = (collision_detected or is_vehicle_too_close or ran_red_light)
        else:
          condition = (collision_detected or is_vehicle_too_close)

        if condition:
          terminal_reward -= self.config.terminal_hint

    current_route_completion = self.route_completion.update(self.vehicle)
    progress_reward = current_route_completion - self.last_route_completion
    self.last_route_completion = current_route_completion

    soft_penality = outside_lanes
    if soft_penality:
      progress_reward = 0.0

    if self.config.use_single_reward:
      if agent_too_fast:
        progress_reward = speed_penalty * progress_reward

      if ttc_penalty:
        progress_reward = 0.5 * progress_reward

      if comfort_penalty < 1.0:
        progress_reward = comfort_penalty * progress_reward
    else:  # nuPlan style reward
      length_factor = 2000  # Roughly the length of a completed episode
      scale_factor = 100  # Route completion is in [0,100]
      r_ttc = 0.0 if ttc_penalty else scale_factor
      r_speed = 0.0 if agent_too_fast else scale_factor
      r_comfort = 0.0 if comfort_penalty < 1.0 else scale_factor

      r_ttc /= length_factor
      r_speed /= length_factor
      r_comfort /= length_factor

      # Numbers come from the nuPlan Driving Score metric
      progress_reward = (5 * progress_reward + 5 * r_ttc + 4 * r_speed + 2 * r_comfort) / 16

    if self.config.use_perc_progress:
      progress_reward = dist_to_centerline_penalty * progress_reward

    if self.config.use_min_speed_infraction:
      progress_reward = fraction_of_speed * progress_reward

    if action_changed_too_much:
      progress_reward = 0.5 * progress_reward

    reward = progress_reward + terminal_reward

    if self.config.use_survival_reward:
      reward += self.config.survival_reward_magnitude

    wrong_start = False
    if self.first_frame:
      # Unit test to check if the route is faulty
      if self.world_map.get_waypoint(ego_vehicle_location, project_to_road=False) is None:
        wrong_start = True

    if self.first_frame and (termination or truncation or wrong_start):
      # There is a bug in one of the routes where the agent is spawned at a bad position.
      print(f'Faulty route file:, Map: {str(self.world_map.name)}')

    self.first_frame = False

    info = {'n_steps': 0, 'suggest': 0, 'timeout': timeout, 'infraction_type': ''}
    if termination:
      if route_deviation:
        info['infraction_type'] = 'route_deviation'
      if ran_red_light:
        info['infraction_type'] = 'ran_red_light'
      if ran_stop_sign:
        info['infraction_type'] = 'ran_stop_sign'
      if collision_detected:
        info['infraction_type'] = 'collision'
      if ego_blocked:
        info['infraction_type'] = 'ego_blocked'
      if timeout:
        info['infraction_type'] = 'timeout'
      if is_vehicle_too_close:
        info['infraction_type'] = 'vehicle_too_close'
      if route_deviation_2:
        info['infraction_type'] = 'route_deviation_2'
      if off_road_term:
        info['infraction_type'] = 'off_road_term'

    return reward, termination, truncation, info

  def compute_comfort_penalty(self):
    '''
    Computes the comfort penalty factor
    '''
    self.remaining_comfort_penalty_ticks = np.clip(self.remaining_comfort_penalty_ticks - 1, a_min=0, a_max=None)

    transform = self.vehicle.get_transform()
    forward_vector = transform.get_forward_vector()
    right_vector = transform.get_right_vector()

    inv_dt = self.config.frame_rate  # x / dt = x * (1/dt)
    if self.last_loc is None:
      velocity = carla.Location()
      acceleration = carla.Location()
    else:
      velocity = (transform.location - self.last_loc) * inv_dt
      acceleration = (velocity - self.last_velocity) * inv_dt

    self.last_velocity = velocity
    self.last_loc = transform.location

    # CARLA velocity and acceleration computation seems to be broken (yields different results than finite differences).
    # acceleration = self.vehicle.get_acceleration()  # In world coordinates
    acceleration_magnitude = acceleration.length()

    # Project to local coordinates
    longitudinal_acceleration = (acceleration.x * forward_vector.x + acceleration.y * forward_vector.y +
                                 acceleration.z * forward_vector.z)
    lateral_acceleration = (acceleration.x * right_vector.x + acceleration.y * right_vector.y +
                            acceleration.z * right_vector.z)

    yaw = math.radians(transform.rotation.yaw)

    self.ego_state_history.longitudinal_acceleration.append(longitudinal_acceleration)
    self.ego_state_history.lateral_acceleration.append(lateral_acceleration)
    self.ego_state_history.acceleration_magnitude.append(acceleration_magnitude)
    self.ego_state_history.yaw.append(yaw)

    comfort_penalty = 1.0
    if len(self.ego_state_history.longitudinal_acceleration) >= 8:
      longitudinal_acceleration = savgol_filter(self.ego_state_history.longitudinal_acceleration,
                                                polyorder=2,
                                                window_length=min(
                                                    8, len(self.ego_state_history.longitudinal_acceleration)),
                                                axis=-1)
      lateral_acceleration = savgol_filter(self.ego_state_history.lateral_acceleration,
                                           polyorder=2,
                                           window_length=min(8, len(self.ego_state_history.lateral_acceleration)),
                                           axis=-1)
      acceleration_magnitude = savgol_filter(self.ego_state_history.acceleration_magnitude,
                                             polyorder=2,
                                             window_length=min(8, len(self.ego_state_history.acceleration_magnitude)),
                                             axis=-1)

      jerk = savgol_filter(acceleration_magnitude,
                           polyorder=2,
                           deriv=1,
                           delta=self.dx,
                           window_length=min(15, len(acceleration_magnitude)),
                           axis=-1)
      longitudinal_jerk = savgol_filter(longitudinal_acceleration,
                                        polyorder=2,
                                        deriv=1,
                                        delta=self.dx,
                                        window_length=min(15, len(acceleration_magnitude)),
                                        axis=-1)

      # https://github.com/DanielDauner/tuplan_garage_rl/blob/c6e2a477187c3c124c38bb09d1754e840b8a3222/tuplan_garage/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics_debug.py#L169
      two_pi = 2.0 * np.pi
      yaw_np = np.array(self.ego_state_history.yaw)
      adjustments = np.zeros_like(yaw_np)
      adjustments[1:] = np.cumsum(np.round(np.diff(yaw_np, axis=-1) / two_pi), axis=-1)
      unwrapped_yaw = yaw_np - two_pi * adjustments
      yaw_rate = savgol_filter(unwrapped_yaw,
                               polyorder=2,
                               deriv=1,
                               delta=self.dx,
                               window_length=min(5, len(unwrapped_yaw)),
                               axis=-1)
      yaw_acceleration = savgol_filter(unwrapped_yaw,
                                       polyorder=3,
                                       deriv=2,
                                       delta=self.dx,
                                       window_length=min(5, len(unwrapped_yaw)),
                                       axis=-1)

      # self.comfort_histogram['acc_lon'].append(longitudinal_acceleration[-1])
      # self.comfort_histogram['acc_lat'].append(np.abs(lateral_acceleration)[-1])
      # self.comfort_histogram['jerk'].append(np.abs(jerk)[-1])
      # self.comfort_histogram['jerk_lon'].append(np.abs(longitudinal_jerk)[-1])
      # self.comfort_histogram['yaw_rate'].append(np.abs(yaw_rate)[-1])
      # self.comfort_histogram['yaw_acceleration'].append(np.abs(yaw_acceleration)[-1])
      #
      # from rl_config import GlobalConfig
      #
      # config = GlobalConfig()

      uncomfortable_acc_lon = ((longitudinal_acceleration > self.config.max_lon_accel)[-1] or
                               (longitudinal_acceleration < self.config.min_lon_accel)[-1])
      uncomfortable_acc_lat = (np.abs(lateral_acceleration) > self.config.max_abs_lat_accel)[-1]
      uncomfortable_jerk = (np.abs(jerk) > self.config.max_abs_mag_jerk)[-1]
      uncomfortable_jerk_lon = (np.abs(longitudinal_jerk) > self.config.max_abs_lon_jerk)[-1]
      uncomfortable_yaw_rate = (np.abs(yaw_rate) > self.config.max_abs_yaw_rate)[-1]
      uncomfortable_yaw_acceleration = (np.abs(yaw_acceleration) > self.config.max_abs_yaw_accel)[-1]

      if uncomfortable_acc_lon:
        self.remaining_comfort_penalty_ticks[0] = self.config.comfort_penalty_ticks
      if uncomfortable_acc_lat:
        self.remaining_comfort_penalty_ticks[1] = self.config.comfort_penalty_ticks
      if uncomfortable_jerk:
        self.remaining_comfort_penalty_ticks[2] = self.config.comfort_penalty_ticks
      if uncomfortable_jerk_lon:
        self.remaining_comfort_penalty_ticks[3] = self.config.comfort_penalty_ticks
      if uncomfortable_yaw_rate:
        self.remaining_comfort_penalty_ticks[4] = self.config.comfort_penalty_ticks
      if uncomfortable_yaw_acceleration:
        self.remaining_comfort_penalty_ticks[5] = self.config.comfort_penalty_ticks

      num_infractions = np.sum(self.remaining_comfort_penalty_ticks.astype(bool))
      comfort_penalty = 1.0 - self.config.comfort_penalty_factor * (num_infractions / 6.0)

    return comfort_penalty

  def destroy(self):
    self.collision_detector.clean()

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

  def is_ego_too_slow(self, ego_speed, other_vehicles):
    '''
     Compares the speed of the ego vehicle with the avg speed of surrounding background traffic.
    '''
    background_vehicles = [v for v in other_vehicles if v.attributes['role_name'] == 'background']

    frame_mean_speed = 0.000000000000001
    if len(background_vehicles) > 0:
      for vehicle in background_vehicles:
        frame_mean_speed += CarlaDataProvider.get_velocity(vehicle)
      frame_mean_speed /= len(background_vehicles)

    fraction_of_speed = np.clip(ego_speed / frame_mean_speed, a_min=0.0, a_max=1.0)
    return fraction_of_speed

  # TODO vectorize this function with numpy
  def vehicle_too_close(self, ego_vehicle_transform, speed, ego_control, vehicles_all, walkers_all):
    '''
    Forecasts ego vehicle with WOR kinematic bicycle model. If forcast overlaps with another vehicles current state
    the ego vehicle is considered to be too close to the preceding vehicle. During the forecast brake and throttle
    are considered to be 0 whereas steering is repeated. The speed of the vehicle is clipped to a minimum amount.
    :return: True or False depending on whether the ego vehicle is too close to another traffic participant.
    '''
    next_loc = np.array([ego_vehicle_transform.location.x, ego_vehicle_transform.location.y])

    next_yaw = np.array([np.deg2rad(ego_vehicle_transform.rotation.yaw)])
    ego_action = np.array([ego_control.steer, 0.0, 0.0])

    next_speed = np.array([speed if speed > self.config.ego_forecast_min_speed else self.config.ego_forecast_min_speed])

    ego_bounding_boxes = []

    for _ in range(int(self.config.ego_forecast_time / self.config.time_interval)):
      next_loc, next_yaw, next_speed = self.ego_model.forward(next_loc, next_yaw, next_speed, ego_action)

      delta_yaws = np.rad2deg(next_yaw).item()

      transform = carla.Transform(
          carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=ego_vehicle_transform.location.z),
          carla.Rotation(pitch=ego_vehicle_transform.rotation.pitch,
                         yaw=delta_yaws,
                         roll=ego_vehicle_transform.rotation.roll))

      bounding_box = carla.BoundingBox(transform.location, self.vehicle.bounding_box.extent)
      bounding_box.rotation = transform.rotation

      ego_bounding_boxes.append(bounding_box)

    for ego_bounding_box in ego_bounding_boxes:
      for actor in [*vehicles_all, *walkers_all]:
        if actor.id == self.vehicle.id:
          continue

        traffic_transform = actor.get_transform()
        traffic_bb_center = traffic_transform.transform(actor.bounding_box.location)

        if ego_vehicle_transform.location.distance(traffic_bb_center) < 10.0:
          traffic_bounding_box = carla.BoundingBox(traffic_bb_center, actor.bounding_box.extent)
          traffic_bounding_box.rotation = carla.Rotation(
              pitch=rl_u.normalize_angle_degree(actor.bounding_box.rotation.pitch + traffic_transform.rotation.pitch),
              yaw=rl_u.normalize_angle_degree(actor.bounding_box.rotation.yaw + traffic_transform.rotation.yaw),
              roll=rl_u.normalize_angle_degree(actor.bounding_box.rotation.roll + traffic_transform.rotation.roll))

          # check the first BB of the traffic participant. We don't extrapolate into the future here.
          if rl_u.check_obb_intersection(ego_bounding_box, traffic_bounding_box):
            # self.world.debug.draw_box(box=ego_bounding_box,
            #                           rotation=ego_bounding_box.rotation,
            #                           thickness=0.3,
            #                           color=carla.Color(255, 0, 0, 255),
            #                           life_time=self.config.time_interval + 0.01)
            return True

      # color = carla.Color(0, 255, 0, 255)
      # self.world.debug.draw_box(box=ego_bounding_box,
      #                           rotation=ego_bounding_box.rotation,
      #                           thickness=0.3,
      #                           color=color,
      #                           life_time=self.config.time_interval + 0.01)

    return False

  # TODO vectorize
  def does_agent_violate_ttc(self, ego_vehicle_transform, ev_speed, vehicles_all, walkers_all, static_all):
    stopped_speed_threshold: float = 0.1  # [m/s] (ttc)

    if (len(vehicles_all) <= 0 and len(walkers_all) <= 0) or ev_speed < stopped_speed_threshold:
      return False

    distance = carla.Vector3D(self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y).length()
    ego_transforms = []
    ego_bbs = []

    for delta in self.ttc_future_time_deltas:
      next_loc = ego_vehicle_transform.transform(carla.Location(x=1.0 * delta * ev_speed, y=0.0, z=0.0))

      next_transform = carla.Transform(next_loc, ego_vehicle_transform.rotation)

      ego_bounding_box = carla.BoundingBox(next_transform.location, self.vehicle.bounding_box.extent)
      ego_bounding_box.rotation = next_transform.rotation
      ego_transforms.append(next_transform)
      ego_bbs.append(ego_bounding_box)

      # self.world.debug.draw_box(box=ego_bounding_box, rotation=ego_bounding_box.rotation, thickness=0.3,
      #                           color=carla.Color(255, 0, 0, 255),
      #                           life_time=self.config.time_interval + 0.01)

    for actor in [*vehicles_all, *walkers_all, *static_all]:
      if actor.id == self.vehicle.id:
        continue

      actor_distance = carla.Vector3D(actor.bounding_box.extent.x, actor.bounding_box.extent.y).length()
      actor_transform = actor.get_transform()
      actor_speed = actor.get_velocity().length()
      no_collision_distance = distance + actor_distance

      for idx, delta in enumerate(self.ttc_future_time_deltas):
        actor_next_loc = actor_transform.transform(carla.Location(x=delta * actor_speed, y=0.0, z=0.0))

        # Filter cases that are clearly no collision, using enclosing circles.
        if actor_next_loc.distance(ego_transforms[idx].location) > no_collision_distance:
          continue

        bounding_box = carla.BoundingBox(actor_next_loc, actor.bounding_box.extent)
        bounding_box.rotation = actor_transform.rotation
        # self.world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.3,
        #                           color=carla.Color(255, 255, 0, 255),
        #                           life_time=self.config.time_interval + 0.01)

        if rl_u.check_obb_intersection(ego_bbs[idx], bounding_box):
          return True

    return False


class EgoModel():
  """
    Kinematic bicycle model describing the motion of a car given it's state and
    action. Tuned parameters are taken from World on Rails.
    """

  def __init__(self, dt=1. / 4):
    self.dt = dt

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275

    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    if brake:
      accel = self.brake_accel
    else:
      accel = self.throt_accel * throt

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
    next_spds = spds + accel * self.dt
    next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

    next_locs = np.array([next_locs_0, next_locs_1])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds
