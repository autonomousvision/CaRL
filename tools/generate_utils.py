import math
from enum import IntEnum

import carla


def generate_road_options(waypoints_trajectory):
  """
  Given a dense trajectory as carla map points, compute the discrete commands associated with the points.
  """

  straight_threshold = 35
  map_route = waypoints_trajectory

  route_trace = []
  in_junction = False
  last_road_option = RoadOption.VOID
  last_map_wp = None

  for idx, map_wp in enumerate(map_route):
    if map_wp.is_junction:
      if in_junction:
        road_option = last_road_option
      else:
        for next_wp_idx in range(idx, len(map_route)):
          tmp_wp = map_route[next_wp_idx]
          if not tmp_wp.is_junction:  # Junction exit
            break
        exit_wp = map_route[next_wp_idx]
        a = map_wp.transform.get_forward_vector()
        b = exit_wp.transform.get_forward_vector()
        # https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
        signed_angle = math.degrees(math.atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y))
        if abs(signed_angle) <= straight_threshold:
          road_option = RoadOption.STRAIGHT
        elif signed_angle < -straight_threshold:
          road_option = RoadOption.LEFT
        elif signed_angle > straight_threshold:
          road_option = RoadOption.RIGHT
        else:
          print('Should not happen')
        in_junction = True
        last_road_option = road_option
    else:
      in_junction = False
      road_option = RoadOption.LANEFOLLOW

      # Overwrite last point if a lane change happened.
      if last_map_wp is not None and not last_map_wp.is_junction:
        # a * b > 0 Checks wether they have the same sign.
        if (map_wp.road_id == last_map_wp.road_id and map_wp.lane_id != last_map_wp.lane_id and
            (map_wp.lane_id * last_map_wp.lane_id) > 0):
          left_lane = min(map_wp.lane_id, last_map_wp.lane_id, key=abs)
          if (last_map_wp.left_lane_marking and
              map_wp.lane_id == left_lane and
              (last_map_wp.left_lane_marking.lane_change == carla.LaneChange.Left
               or last_map_wp.left_lane_marking.lane_change == carla.LaneChange.Both)
          ):
            route_trace[-1][1] = RoadOption.CHANGELANELEFT
          if (last_map_wp.right_lane_marking and
              map_wp.lane_id != left_lane and
              (last_map_wp.right_lane_marking.lane_change == carla.LaneChange.Right or
               last_map_wp.right_lane_marking.lane_change == carla.LaneChange.Both)
          ):
            route_trace[-1][1] = RoadOption.CHANGELANERIGHT


      last_road_option = road_option

    last_map_wp = map_wp

    route_trace.append([map_wp.transform, road_option])

  return route_trace

class RoadOption(IntEnum):
  """
  RoadOption represents the possible topological configurations when moving from a segment of lane to other.

  """
  VOID = -1
  LEFT = 1
  RIGHT = 2
  STRAIGHT = 3
  LANEFOLLOW = 4
  CHANGELANELEFT = 5
  CHANGELANERIGHT = 6