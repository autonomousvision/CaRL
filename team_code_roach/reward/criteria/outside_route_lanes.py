'''
Class that checks if the vehicle is on a sidewalk or at the wrong lane.
Adapted from scenario runner.
'''
import carla


class OutsideRouteLanesTest:
  """
  Atomic to detect if the vehicle is either on a sidewalk or at a wrong lane. The distance spent outside
  is computed and it is returned as a percentage of the route distance traveled.

  Args:
      actor (carla.ACtor): CARLA actor to be used for this test
      route (list [carla.Location, connection]): series of locations representing the route waypoints
      optional (bool): If True, the result is not considered for an overall pass/fail result
  """

  ALLOWED_OUT_DISTANCE = 0.5  # At least 0.5, due to the mini-shoulder between lanes and sidewalks
  MAX_VEHICLE_ANGLE = 120.0  # Maximum angle between the yaw and waypoint lane
  MAX_WAYPOINT_ANGLE = 150.0  # Maximum change between the yaw-lane angle between frames
  WINDOWS_SIZE = 3  # Amount of additional waypoints checked (in case the first on fails)

  def __init__(self, actor, world_map):
    """
    Constructor
    """
    self.actor = actor

    self.world_map = world_map
    self.last_ego_waypoint = self.world_map.get_waypoint(self.actor.get_location())

    self.outside_lane_active = False
    self.wrong_lane_active = False
    self.last_road_id = None
    self.last_lane_id = None

  def update(self):
    """
    Transforms the actor location and its four corners to waypoints. Depending on its types,
    the actor will be considered to be at driving lanes, sidewalk or offroad.

    returns:
        True if the actor is outside the route lanes.
        False if the actor is inside the route lanes.
    """

    # Some of the vehicle parameters
    location = self.actor.get_location()
    if location is None:
      return False

    self.is_outside_driving_lanes(location)
    self.is_at_wrong_lane(location)

    return self.outside_lane_active or self.wrong_lane_active

  def is_outside_driving_lanes(self, location):
    """
    Detects if the ego_vehicle is outside driving lanes
    """
    driving_wp = self.world_map.get_waypoint(location, lane_type=carla.LaneType.Driving)
    parking_wp = self.world_map.get_waypoint(location, lane_type=carla.LaneType.Parking)

    driving_distance = location.distance(driving_wp.transform.location)
    if parking_wp is not None:  # Some towns have no parking
      parking_distance = location.distance(parking_wp.transform.location)
    else:
      parking_distance = float('inf')

    if driving_distance >= parking_distance:
      distance = parking_distance
      lane_width = parking_wp.lane_width
    else:
      distance = driving_distance
      lane_width = driving_wp.lane_width

    self.outside_lane_active = bool(distance > (lane_width / 2 + self.ALLOWED_OUT_DISTANCE))

  def is_at_wrong_lane(self, location):
    """
    Detects if the ego_vehicle has invaded a wrong lane
    """
    waypoint = self.world_map.get_waypoint(location, lane_type=carla.LaneType.Driving)
    lane_id = waypoint.lane_id
    road_id = waypoint.road_id

    # Lanes and roads are too chaotic at junctions
    if waypoint.is_junction:
      self.wrong_lane_active = False
    elif self.last_road_id != road_id or self.last_lane_id != lane_id:

      if self.last_ego_waypoint.is_junction:
        # Just exited a junction, check the wp direction vs the ego's one
        wp_yaw = waypoint.transform.rotation.yaw % 360
        actor_yaw = self.actor.get_transform().rotation.yaw % 360
        angle = (wp_yaw - actor_yaw) % 360

        if angle < self.MAX_VEHICLE_ANGLE or angle > (360 - self.MAX_VEHICLE_ANGLE):
          self.wrong_lane_active = False
        else:
          self.wrong_lane_active = True

      else:
        # Route direction can be considered continuous, check for a big gap.
        last_wp_yaw = self.last_ego_waypoint.transform.rotation.yaw % 360
        wp_yaw = waypoint.transform.rotation.yaw % 360
        angle = (last_wp_yaw - wp_yaw) % 360

        if self.MAX_WAYPOINT_ANGLE < angle < (360 - self.MAX_WAYPOINT_ANGLE):
          # Is the ego vehicle going back to the lane, or going out? Take the opposite
          self.wrong_lane_active = not bool(self.wrong_lane_active)

    # Remember the last state
    self.last_lane_id = lane_id
    self.last_road_id = road_id
    self.last_ego_waypoint = waypoint
