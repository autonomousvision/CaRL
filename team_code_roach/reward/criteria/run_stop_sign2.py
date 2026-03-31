'''
Contains a class that checks if the ego agents has run a stop sign.
'''
import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import shapely
import rl_utils as rl_u


class RunStopSign2:
  """
  Check if an actor is running a stop sign
  Rewritten version, that detects infraction if the agent is driving on a different lane
  Meant to avoid cheating with the simple reward.
  Important parameters:
  """
  PROXIMITY_THRESHOLD = 20.0  # Stops closer than this distance will be detected [m]
  SPEED_THRESHOLD = 0.1  # Minimum speed to consider the actor has stopped [m/s]
  WAYPOINT_STEP = 0.5  # m

  def __init__(self, carla_world, carla_map):
    """
    """
    self.world = carla_world
    self.carla_map = carla_map
    self.list_stop_signs = []
    self.target_stop_sign = None
    self.target_stop_sign_wp = None
    self.stop_completed = False
    self.last_failed_stop = None

    for actor in CarlaDataProvider.get_all_actors():
      if 'traffic.stop' in actor.type_id:
        self.list_stop_signs.append(actor)

  def point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.2):
    """Checks whether a point is inside a bounding box."""

    # pylint: disable=invalid-name
    A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
    B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
    D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
    M = carla.Vector2D(point.x, point.y)

    AB = B - A
    AD = D - A
    AM = M - A
    am_ab = AM.x * AB.x + AM.y * AB.y
    ab_ab = AB.x * AB.x + AB.y * AB.y
    am_ad = AM.x * AD.x + AM.y * AD.y
    ad_ad = AD.x * AD.x + AD.y * AD.y

    return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad  # pylint: disable=chained-comparison

  def is_actor_affected_by_stop(self, wp_list, stop):
    """
    Check if the given actor is affected by the stop.
    Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
    """
    # Quick distance test
    stop_location = stop.get_transform().transform(stop.trigger_volume.location)
    actor_location = wp_list[0].transform.location
    if stop_location.distance(actor_location) > self.PROXIMITY_THRESHOLD:
      return False

    # Check if any of the actor wps is inside the stop's bounding box.
    # Using more than one waypoint removes issues with small trigger volumes and backwards movement
    stop_extent = stop.trigger_volume.extent
    for actor_wp in wp_list:
      if self.point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
        return True

    return False

  def did_actor_run_stop_sign(self, wp_list, stop, vehicle):
    """
    Check if the given actor is affected by the stop.
    Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
    """

    # Quick distance test
    stop_location = stop.get_transform().transform(stop.trigger_volume.location)
    actor_location = wp_list[0].transform.location

    if stop_location.distance(actor_location) > self.PROXIMITY_THRESHOLD:
      return False

    vec_forward = self.target_stop_sign_wp.transform.get_forward_vector()
    vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)
    # Very long line so that the RL agent can not avoid RL infractions by driving on the opposing lane.
    stop_left_loc = self.target_stop_sign_wp.transform.location - 5.0 * self.target_stop_sign_wp.lane_width * vec_right
    stop_right_loc = self.target_stop_sign_wp.transform.location + 5.0 * self.target_stop_sign_wp.lane_width * vec_right

    ev_tra = vehicle.get_transform()
    tail_close_pt = ev_tra.transform(carla.Location(x=0.0))
    tail_far_pt = ev_tra.transform(carla.Location(x=-vehicle.bounding_box.extent.x))

    #Debug
    # stop_left_loc.z = ev_tra.location.z + 0.7
    # stop_right_loc.z = ev_tra.location.z + 0.7
    # self.world.debug.draw_line(stop_left_loc, stop_right_loc, life_time=0.11)
    # loc = stop_transform.transform(stop.trigger_volume.location)
    # loc.z = ev_tra.location.z + 0.7
    # bb = carla.BoundingBox(loc, stop.trigger_volume.extent)
    # bb.rotation = stop_transform.rotation
    # self.world.debug.draw_box(bb, bb.rotation, life_time=0.11)

    crossed = self._is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (stop_left_loc, stop_right_loc))

    return bool(crossed)

  def scan_for_stop_sign(self, actor_transform, wp_list, actor_velocity):
    """
    Check the stop signs to see if any of them affect the actor.
    Ignore all checks when going backwards or through an opposite direction"""
    actor_direction = actor_transform.get_forward_vector()

    # Ignore all when going backwards

    if actor_velocity.dot(actor_direction) < -0.17:  # 100º, just in case
      return None

    for stop in self.list_stop_signs:
      if self.is_actor_affected_by_stop(wp_list, stop):
        return stop

    return None

  def get_waypoints(self, actor):
    """Returns a list of waypoints starting from the ego location and a set amount forward"""
    wp_list = []
    steps = int(self.PROXIMITY_THRESHOLD / self.WAYPOINT_STEP)

    # Add the actor location
    wp = self.carla_map.get_waypoint(actor.get_location())
    wp_list.append(wp)

    # And its forward waypoints
    next_wp = wp
    for _ in range(steps):
      next_wps = next_wp.next(self.WAYPOINT_STEP)
      if not next_wps:
        break
      next_wp = next_wps[0]
      wp_list.append(next_wp)

    return wp_list

  def tick(self, vehicle):
    """
    Check if the actor is running a red light
    """
    info = None
    ev_loc = vehicle.get_location()
    actor_transform = vehicle.get_transform()
    check_wps = self.get_waypoints(vehicle)
    actor_velocity = vehicle.get_velocity()

    if not self.target_stop_sign:
      self.target_stop_sign = self.scan_for_stop_sign(actor_transform, check_wps, actor_velocity)
      if self.target_stop_sign is not None:
        stop_loc = self.target_stop_sign.get_transform().transform(self.target_stop_sign.trigger_volume.location)
        self.target_stop_sign_wp = self.carla_map.get_waypoint(stop_loc)
        stop_loc = self.target_stop_sign.get_location()
        info = {
            'event': 'encounter',
            'id': self.target_stop_sign.id,
            'stop_loc': [stop_loc.x, stop_loc.y, stop_loc.z],
            'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
        }
    else:
      stop_location = self.target_stop_sign.get_transform().transform(self.target_stop_sign.trigger_volume.location)

      if not self.stop_completed:
        current_speed = CarlaDataProvider.get_velocity(vehicle)
        if current_speed < self.SPEED_THRESHOLD:
          vehicle_global_bb = rl_u.local_bounding_box_to_global(vehicle, vehicle.bounding_box)
          stop_sign_global_bb = rl_u.local_bounding_box_to_global(self.target_stop_sign,
                                                                  self.target_stop_sign.trigger_volume)
          stop_sign_global_bb.location.z = 0.0  # Some stop signs have weird z coordinates.
          vehicle_global_bb.location.z = 0.0
          if rl_u.check_obb_intersection(vehicle_global_bb, stop_sign_global_bb):
            self.stop_completed = True

      if (not self.stop_completed and self.last_failed_stop != self.target_stop_sign.id and
          self.did_actor_run_stop_sign(check_wps, self.target_stop_sign, vehicle)):
        self.last_failed_stop = self.target_stop_sign.id
        stop_loc = self.target_stop_sign.get_transform().location
        info = {
            'event': 'run',
            'id': self.target_stop_sign.id,
            'stop_loc': [stop_loc.x, stop_loc.y, stop_loc.z],
            'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
        }
        # Reset state
        self.target_stop_sign = None
        self.target_stop_sign_wp = None
        self.stop_completed = False

      if stop_location.distance(ev_loc) > self.PROXIMITY_THRESHOLD:
        # Reset state
        self.target_stop_sign = None
        self.target_stop_sign_wp = None
        self.stop_completed = False

    return info

  @staticmethod
  def _is_vehicle_crossing_line(seg1, seg2):
    """
        check if vehicle crosses a line segment
        """
    # TODO speedup by not using shapely
    line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
    line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
    inter = line1.intersection(line2)
    return not inter.is_empty
