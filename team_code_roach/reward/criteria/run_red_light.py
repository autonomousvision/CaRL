'''
Contains a class that checks if the ego agent run a red light.
'''
import shapely.geometry
import carla
from birds_eye_view.traffic_light import TrafficLightHandler
import numpy as np


class RunRedLight():
  '''
  Class that checks if the ego agent run a red light.
  '''

  def __init__(self, carla_map, penalize_yellow_light, distance_light=30):
    self._map = carla_map
    self._last_red_light_id = None
    self._distance_light = distance_light
    self.penalize_yellow_light = penalize_yellow_light

    # If this assert triggers, then the TrafficLightHandler class was not initialized yet.
    assert TrafficLightHandler.num_tl > 0

  def tick(self, vehicle):
    ev_tra = vehicle.get_transform()
    ev_loc = ev_tra.location
    ev_dir = ev_tra.get_forward_vector()

    # We want the car to stop in front of the red line, so we penalize crossing it with the top part of the car.
    # Which is slightly different from the red light penalty implementation (which considers the backside)
    vehicle_back = ev_tra.transform(carla.Location(x=-vehicle.bounding_box.extent.x))
    vehicle_front = ev_tra.transform(carla.Location(x=vehicle.bounding_box.extent.x))

    info = None
    for idx_tl in range(TrafficLightHandler.num_tl):
      traffic_light = TrafficLightHandler.list_tl_actor[idx_tl]
      tl_tv_loc = TrafficLightHandler.list_tv_loc[idx_tl]
      if tl_tv_loc.distance(ev_loc) > self._distance_light:
        continue

      if self.penalize_yellow_light:
        # We also penalize yellow lights optionally, to prevent the agent from learning to cross the TL last second.
        condition = traffic_light.state not in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
      else:
        condition = traffic_light.state != carla.TrafficLightState.Red
      if condition:
        continue
      if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
        continue

      for idx_wp in range(len(TrafficLightHandler.list_stopline_wps[idx_tl])):
        wp = TrafficLightHandler.list_stopline_wps[idx_tl][idx_wp]
        wp_dir = wp.transform.get_forward_vector()
        dot_ve_wp = ev_dir.x * wp_dir.x + ev_dir.y * wp_dir.y + ev_dir.z * wp_dir.z

        # Based on only vehicle orientation and using longer lines to prevent the agent to drive into other lanes
        if dot_ve_wp > 0:
          # This light is red and is affecting our lane
          stop_left_loc, stop_right_loc = TrafficLightHandler.list_stopline_long_vtx[idx_tl][idx_wp]

          # Debug
          # from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
          # world = CarlaDataProvider.get_world()
          # world.debug.draw_line(stop_left_loc + carla.Location(z=1.0), stop_right_loc + carla.Location(z=1.0),
          # thickness=1.0, color=carla.Color(255, 0, 0), life_time=0.11)

          velocity = vehicle.get_velocity()
          ev_speed = np.linalg.norm(np.array([velocity.x, velocity.y]))
          # Is the vehicle traversing the stop line?
          crossed = self._is_vehicle_crossing_line((vehicle_back, vehicle_front), (stop_left_loc, stop_right_loc))

          if crossed and ev_speed > 0.001:
            tl_loc = traffic_light.get_location()
            self._last_red_light_id = traffic_light.id
            info = {
                'id': traffic_light.id,
                'tl_loc': [tl_loc.x, tl_loc.y, tl_loc.z],
                'ev_loc': [ev_loc.x, ev_loc.y, ev_loc.z]
            }
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
