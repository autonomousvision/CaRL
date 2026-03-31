"""
Utilities to render bird's eye view semantic segmentation maps.
Code adapted from https://github.com/zhejz/carla-roach
"""

import numpy as np
import carla
import cv2 as cv
from collections import deque
from pathlib import Path
import h5py

from birds_eye_view.obs_manager import ObsManagerBase
from birds_eye_view.traffic_light import TrafficLightHandler

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREY = (128, 128, 128)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
  r, g, b = color
  r = int(r + (255 - r) * factor)
  g = int(g + (255 - g) * factor)
  b = int(b + (255 - b) * factor)
  r = min(r, 255)
  g = min(g, 255)
  b = min(b, 255)
  return (r, g, b)


class ObsManager(ObsManagerBase):
  """
  Generates bev semantic segmentation maps.
  """

  def __init__(self, config):
    self._width = int(config.bev_semantics_width)
    self._pixels_ev_to_bottom = config.pixels_ev_to_bottom
    self._pixels_per_meter = config.pixels_per_meter
    self._history_idx = config.history_idx
    self._scale_bbox = config.scale_bbox
    self._scale_mask_col = config.scale_mask_col
    maxlen_queue = max(max(config.history_idx) + 1, -min(config.history_idx))
    self._history_queue = deque(maxlen=maxlen_queue)
    self.config = config

    self._image_channels = 3
    self._masks_channels = 3 + 3 * len(self._history_idx)
    self.vehicle = None
    self._world = None

    # Preallocate memory, so we do not have to recreate the array every time step.
    self.final_masks = np.zeros((15, self._width, self._width), dtype=np.uint8)

    self._map_dir = Path(__file__).resolve().parent / config.map_folder

    super().__init__()

  def attach_ego_vehicle(self, vehicle, criteria_stop, world_map):  # pylint: disable=locally-disabled, unused-argument
    self.vehicle = vehicle
    self.criteria_stop = criteria_stop

    current_world = self.vehicle.get_world()
    if self._world is None or current_world.id != self._world.id:
      self._world = current_world

      maps_h5_path = self._map_dir / (world_map.name.rsplit('/', 1)[1] + '.h5')
      with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
        self.hd_map_array = np.stack((hf['road'], hf['lane_marking_all'], hf['lane_marking_white_broken']), axis=2)
        self.hd_map_array = self.hd_map_array.astype(dtype=np.uint8)
        # self._road = np.array(hf['road'], dtype=np.uint8)
        # self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
        # self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)
        # self._shoulder = np.array(hf['shoulder'], dtype=np.uint8)
        # self._parking = np.array(hf['parking'], dtype=np.uint8)
        # self._sidewalk = np.array(hf['sidewalk'], dtype=np.uint8)
        # self._lane_marking_yellow_broken = np.array(hf['lane_marking_yellow_broken'], dtype=np.uint8)
        # self._lane_marking_yellow_solid = np.array(hf['lane_marking_yellow_solid'], dtype=np.uint8)
        # self._lane_marking_white_solid = np.array(hf['lane_marking_white_solid'], dtype=np.uint8)

        self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
        assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

      self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
    # dilate road mask, lbc draw road polygon with 10px boarder
    # kernel = np.ones((11, 11), np.uint8)
    # self._road = cv.dilate(self._road, kernel, iterations=1)

    TrafficLightHandler.reset(self._world, world_map)

  @staticmethod
  def _get_stops(criteria_stop):
    stop_sign = criteria_stop.target_stop_sign
    stops = []
    if (stop_sign is not None) and (not criteria_stop.stop_completed):
      bb_loc = carla.Location(stop_sign.trigger_volume.location)
      bb_ext = carla.Vector3D(stop_sign.trigger_volume.extent)
      bb_ext.x = max(bb_ext.x, bb_ext.y)
      bb_ext.y = max(bb_ext.x, bb_ext.y)
      trans = stop_sign.get_transform()
      stops = [(carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext)]
    return stops

  # 2-3 ms
  def get_observation(self, waypoint_route, vehicles_all=None, walkers_all=None, static_all=None, debug=False):  # pylint: disable=locally-disabled, unused-argument
    # 0.1 ms
    ev_transform = self.vehicle.get_transform()
    ev_loc = ev_transform.location
    ev_rot = ev_transform.rotation
    ev_bbox = self.vehicle.bounding_box

    def is_within_distance(w):
      c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
          and abs(ev_loc.y - w.location.y) < self._distance_threshold \
          and abs(ev_loc.z - w.location.z) < 8.0
      # Cheap trick to remove the ego bounding box
      c_ev = abs(ev_loc.x - w.location.x) < self.config.ego_extent_y and abs(ev_loc.y -
                                                                             w.location.y) < self.config.ego_extent_y
      return c_distance and (not c_ev)

    actors = self._world.get_actors()
    vehicles = actors.filter('*vehicle*')
    walkers = actors.filter('*walker*')

    # 0.1ms
    # This style of generating bounding boxes is more ugly than just calling world.get_level_bbs in carla but it has
    # the advantage of not causing segfaults in the carla library :)
    vehicle_bbox_list = []
    for vehicle in vehicles:
      if vehicle.id == self.vehicle.id:
        continue
      traffic_transform = vehicle.get_transform()

      # Convert the bounding box to global coordinates
      bounding_box = carla.BoundingBox(traffic_transform.location + vehicle.bounding_box.location,
                                       vehicle.bounding_box.extent)
      # Rotations of the bb are 0.
      bounding_box.rotation = carla.Rotation(pitch=vehicle.bounding_box.rotation.pitch +
                                             traffic_transform.rotation.pitch,
                                             yaw=vehicle.bounding_box.rotation.yaw + traffic_transform.rotation.yaw,
                                             roll=vehicle.bounding_box.rotation.roll + traffic_transform.rotation.roll)

      vehicle_bbox_list.append(bounding_box)

    walker_bbox_list = []
    for walker in walkers:
      walker_transform = walker.get_transform()

      walker_location = walker_transform.location
      transform = carla.Transform(walker_location)
      bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
      bounding_box.rotation = carla.Rotation(pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                                             yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw,
                                             roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)

      walker_bbox_list.append(bounding_box)

    # 0.1 ms
    if self._scale_bbox:
      vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
      walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
    else:
      vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
      walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

    tl_green, tl_yellow, tl_red, _ = TrafficLightHandler.get_stopline_vtx(ev_loc, self._distance_threshold)
    stops = self._get_stops(self.criteria_stop)

    self._history_queue.append((vehicles, walkers, tl_green, tl_yellow, tl_red, stops))

    m_warp = self._get_warp_transform(ev_loc, ev_rot)  # 0.1ms

    # objects with history 1.2 ms
    vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks \
        = self._get_history_masks(m_warp)

    # road_mask, lane_mask 0.3 - 0.5 ms
    # Batched together for higher efficiency.

    warped_hd_map = cv.warpAffine(self.hd_map_array, m_warp, (self._width, self._width))
    lane_mask_broken = warped_hd_map[:, :, 2].astype(bool)

    # 0.1 ms
    route_mask = np.zeros([self._width, self._width], dtype=np.uint8)

    route_idx_len = self.config.num_route_points_rendered
    if len(waypoint_route) < self.config.num_route_points_rendered:
      route_idx_len = len(waypoint_route)

    route_plan = np.array(
        [[waypoint_route[i][0].location.x, waypoint_route[i][0].location.y] for i in range(0, route_idx_len)],
        dtype=np.float32)
    route_in_pixel = self._world_to_pixel_batch(route_plan)[:, np.newaxis, :]

    route_warped = cv.transform(route_in_pixel, m_warp)
    cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)

    ev_mask_col = self._get_mask_from_actor_list(
        [(ev_transform, ev_bbox.location, ev_bbox.extent * self._scale_mask_col)], m_warp)

    # render
    if debug:
      road_mask = warped_hd_map[:, :, 0].astype(bool)
      lane_mask_all = warped_hd_map[:, :, 1].astype(bool)
      route_mask_bool = route_mask.astype(bool)

      # ev_mask
      ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], m_warp)

      image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
      image[road_mask] = COLOR_ALUMINIUM_5
      image[route_mask_bool] = COLOR_ALUMINIUM_3
      #image[sidewalk_mask] = COLOR_GREY
      image[lane_mask_all] = COLOR_MAGENTA
      image[lane_mask_broken] = COLOR_MAGENTA_2

      h_len = len(self._history_idx) - 1
      for i, mask in enumerate(stop_masks):
        image[mask] = tint(COLOR_YELLOW_2, (h_len - i) * 0.2)
      for i, mask in enumerate(tl_green_masks):
        image[mask] = tint(COLOR_GREEN, (h_len - i) * 0.2)
      for i, mask in enumerate(tl_yellow_masks):
        image[mask] = tint(COLOR_YELLOW, (h_len - i) * 0.2)
      for i, mask in enumerate(tl_red_masks):
        image[mask] = tint(COLOR_RED, (h_len - i) * 0.2)

      for i, mask in enumerate(vehicle_masks):
        image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)
      for i, mask in enumerate(walker_masks):
        image[mask] = tint(COLOR_CYAN, (h_len - i) * 0.2)

      image[ev_mask] = COLOR_WHITE

    # 0.6ms
    # masks
    # 0 = Unlabeled

    c_road = warped_hd_map[:, :, 0]
    c_route = route_mask * np.array(255, dtype=np.uint8)
    c_lane = warped_hd_map[:, :, 1]
    c_lane[lane_mask_broken] = np.array(120, dtype=np.uint8)

    # masks with history
    c_tl_history = []
    for i in range(len(self._history_idx)):
      c_tl = np.zeros([self._width, self._width], dtype=np.uint8)
      c_tl[tl_green_masks[i]] = np.array(80, dtype=np.uint8)
      c_tl[tl_yellow_masks[i]] = np.array(170, dtype=np.uint8)
      c_tl[tl_red_masks[i]] = np.array(255, dtype=np.uint8)
      c_tl[stop_masks[i]] = np.array(255, dtype=np.uint8)
      c_tl_history.append(c_tl)

    c_vehicle_history = [m * np.array(255, dtype=np.uint8) for m in vehicle_masks]
    c_walker_history = [m * np.array(255, dtype=np.uint8) for m in walker_masks]

    # Ugly, but 8x faster than np.stack because it avoids array copies.
    self.final_masks[0] = c_road
    self.final_masks[1] = c_route
    self.final_masks[2] = c_lane
    self.final_masks[3] = c_vehicle_history[0]
    self.final_masks[4] = c_vehicle_history[1]
    self.final_masks[5] = c_vehicle_history[2]
    self.final_masks[6] = c_vehicle_history[3]
    self.final_masks[7] = c_walker_history[0]
    self.final_masks[8] = c_walker_history[1]
    self.final_masks[9] = c_walker_history[2]
    self.final_masks[10] = c_walker_history[3]
    self.final_masks[11] = c_tl_history[0]
    self.final_masks[12] = c_tl_history[1]
    self.final_masks[13] = c_tl_history[2]
    self.final_masks[14] = c_tl_history[3]

    obs_dict = {'bev_semantic_classes': self.final_masks}
    if debug:
      # For visualization we don't rotate as
      obs_dict['rendered'] = image

    # NOTE paper says any bounding box overlap is checked, but code seems to only check for pedestrians.
    obs_dict['collision_px'] = np.any(ev_mask_col & walker_masks[-1])
    obs_dict['percentage_off_road'] = np.sum(ev_mask_col & np.logical_not(c_road.astype(bool))) / np.sum(ev_mask_col)

    return obs_dict

  def _get_history_masks(self, m_warp):
    qsize = len(self._history_queue)
    vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
    for idx in self._history_idx:
      idx = max(idx, -1 * qsize)

      vehicles, walkers, tl_green, tl_yellow, tl_red, stops = self._history_queue[idx]

      vehicle_masks.append(self._get_mask_from_actor_list(vehicles, m_warp))  # 0.15 - 0.3ms
      walker_masks.append(self._get_mask_from_actor_list(walkers, m_warp))
      tl_green_masks.append(self._get_mask_from_stopline_vtx(tl_green, m_warp))
      tl_yellow_masks.append(self._get_mask_from_stopline_vtx(tl_yellow, m_warp))
      tl_red_masks.append(self._get_mask_from_stopline_vtx(tl_red, m_warp))
      stop_masks.append(self._get_mask_from_actor_list(stops, m_warp))

    return vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks

  def _get_mask_from_stopline_vtx(self, stopline_vtx, m_warp):
    mask = np.zeros([self._width, self._width], dtype=np.uint8)
    for sp_locs in stopline_vtx:
      stopline_in_pixel = np.array([[x.x, x.y] for x in sp_locs], dtype=np.float32)
      # world_to_pixel Batched and inplace for faster speed
      stopline_in_pixel[:, 0:2] = self._pixels_per_meter * (stopline_in_pixel[:, 0:2] - self._world_offset[0:2])
      stopline_in_pixel = stopline_in_pixel[:, np.newaxis, :]
      stopline_warped = cv.transform(stopline_in_pixel, m_warp)
      # Rounding to pixel coordinates is required by newer opencv versions
      pt1 = (stopline_warped[0, 0] + 0.5).astype(np.int32)
      pt2 = (stopline_warped[1, 0] + 0.5).astype(np.int32)
      cv.line(mask, tuple(pt1), tuple(pt2), color=1, thickness=6)
    return mask.astype(bool)

  def _get_mask_from_actor_list(self, actor_list, m_warp):
    mask = np.zeros([self._width, self._width], dtype=np.uint8)
    for actor_transform, bb_loc, bb_ext in actor_list:
      corners2 = [
          actor_transform.transform(bb_loc + carla.Location(x=-bb_ext.x, y=-bb_ext.y)),
          actor_transform.transform(bb_loc + carla.Location(x=bb_ext.x, y=-bb_ext.y)),
          actor_transform.transform(bb_loc + carla.Location(x=bb_ext.x, y=0)),
          actor_transform.transform(bb_loc + carla.Location(x=bb_ext.x, y=bb_ext.y)),
          actor_transform.transform(bb_loc + carla.Location(x=-bb_ext.x, y=bb_ext.y))
      ]
      corners3 = np.array([[corner.x, corner.y] for corner in corners2], dtype=np.float32)
      corners_in_pixel = self._world_to_pixel_batch(corners3)[:, np.newaxis, :]
      corners_warped = cv.transform(corners_in_pixel, m_warp)

      cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
    return mask.astype(bool)

  @staticmethod
  def _get_surrounding_actors(bbox_list, criterium, scale=None):
    actors = []
    for bbox in bbox_list:
      is_within_distance = criterium(bbox)
      if is_within_distance:
        bb_loc = carla.Location()
        bb_ext = carla.Vector3D(bbox.extent)
        if scale is not None:
          bb_ext = bb_ext * scale
          bb_ext.x = max(bb_ext.x, 0.8)
          bb_ext.y = max(bb_ext.y, 0.8)

        actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
    return actors

  def _get_warp_transform(self, ev_loc, ev_rot):
    ev_loc_in_px = self._world_to_pixel(ev_loc)
    yaw = np.deg2rad(ev_rot.yaw)

    forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
    right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

    bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5 * self._width) * right_vec
    top_left = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec - (0.5 * self._width) * right_vec
    top_right = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec + (0.5 * self._width) * right_vec

    src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
    dst_pts = np.array([[0, self._width - 1], [0, 0], [self._width - 1, 0]], dtype=np.float32)
    return cv.getAffineTransform(src_pts, dst_pts)

  def _world_to_pixel(self, location, projective=False):
    """Converts the world coordinates to pixel coordinates"""
    x = self._pixels_per_meter * (location.x - self._world_offset[0])
    y = self._pixels_per_meter * (location.y - self._world_offset[1])

    if projective:
      p = np.array([x, y, 1], dtype=np.float32)
    else:
      p = np.array([x, y], dtype=np.float32)
    return p

  def _world_to_pixel_batch(self, location):
    """Converts the world coordinates to pixel coordinates in batched form"""
    location[:, 0:2] = self._pixels_per_meter * (location[:, 0:2] - self._world_offset[0:2])
    return location

  def _world_to_pixel_width(self, width):
    """Converts the world units to pixel units"""
    return self._pixels_per_meter * width

  def clean(self):
    self.vehicle = None
    self._world = None
    self._history_queue.clear()
