"""
Some utility functions e.g. for preprocessing traffic lights
Functions for detecting red lights are adapted from scenario runners
atomic_criteria.py
"""
import math

import carla
import numpy as np
# import torch

def normalize_angle(x):
  x = x % (2 * np.pi)  # force in range [0, 2 pi)
  if x > np.pi:  # move to [-pi, pi)
    x -= 2 * np.pi
  return x

def normalize_angle_degree(x):
  x = x % 360.0
  if x > 180.0:
    x -= 360.0
  return x


def rotate_point(point, angle):
  """
  rotate a given point by a given angle
  """
  x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
  y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
  return carla.Vector3D(x_, y_, point.z)


def get_traffic_light_waypoints(traffic_light, carla_map):
  """
  get area of a given traffic light
  """
  base_transform = traffic_light.get_transform()
  base_loc = traffic_light.get_location()
  base_rot = base_transform.rotation.yaw
  area_loc = base_transform.transform(traffic_light.trigger_volume.location)

  # Discretize the trigger box into points
  area_ext = traffic_light.trigger_volume.extent
  x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

  area = []
  for x in x_values:
    point = rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
    point_location = area_loc + carla.Location(x=point.x, y=point.y)
    area.append(point_location)

  # Get the waypoints of these points, removing duplicates
  ini_wps = []
  for pt in area:
    wpx = carla_map.get_waypoint(pt)
    # As x_values are arranged in order, only the last one has to be checked
    if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
      ini_wps.append(wpx)

  # Advance them until the intersection
  wps = []
  eu_wps = []
  for wpx in ini_wps:
    distance_to_light = base_loc.distance(wpx.transform.location)
    eu_wps.append(wpx)
    next_distance_to_light = distance_to_light + 1.0
    while not wpx.is_intersection:
      next_wp = wpx.next(0.5)[0]
      next_distance_to_light = base_loc.distance(next_wp.transform.location)
      if next_wp and not next_wp.is_intersection \
          and next_distance_to_light <= distance_to_light:
        eu_wps.append(next_wp)
        distance_to_light = next_distance_to_light
        wpx = next_wp
      else:
        break

    if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
      wps.append(eu_wps[-4])
    else:
      wps.append(wpx)

  return area_loc, wps


def inverse_conversion_2d(point, translation, yaw):
  """
  Performs a forward coordinate conversion on a 2D point
  :param point: Point to be converted
  :param translation: 2D translation vector of the new coordinate system
  :param yaw: yaw in radian of the new coordinate system
  :return: Converted point
  """
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

  converted_point = rotation_matrix.T @ (point - translation)
  return converted_point


# TODO jit this
def dot_product(vector1, vector2):
  return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z


def cross_product(vector1, vector2):
  return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y,
                        y=vector1.z * vector2.x - vector1.x * vector2.z,
                        z=vector1.x * vector2.y - vector1.y * vector2.x)


def get_separating_plane(r_pos, plane, obb1, obb2):
  ''' Checks if there is a seperating plane
  rPos Vec3
  plane Vec3
  obb1  Bounding Box
  obb2 Bounding Box
  '''
  return (abs(dot_product(r_pos, plane)) >
          (abs(dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
           abs(dot_product((obb1.rotation.get_right_vector() * obb1.extent.y), plane)) +
           abs(dot_product((obb1.rotation.get_up_vector() * obb1.extent.z), plane)) +
           abs(dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
           abs(dot_product((obb2.rotation.get_right_vector() * obb2.extent.y), plane)) +
           abs(dot_product((obb2.rotation.get_up_vector() * obb2.extent.z), plane))))


def check_obb_intersection(obb1, obb2):
  r_pos = obb2.location - obb1.location
  return not (
      get_separating_plane(r_pos, obb1.rotation.get_forward_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb1.rotation.get_right_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb1.rotation.get_up_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb2.rotation.get_forward_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb2.rotation.get_right_vector(), obb1, obb2) or
      get_separating_plane(r_pos, obb2.rotation.get_up_vector(), obb1, obb2) or get_separating_plane(
          r_pos, cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
      get_separating_plane(r_pos, cross_product(
          obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()), obb1, obb2) or get_separating_plane(
              r_pos, cross_product(obb1.rotation.get_up_vector(), obb2.rotation.get_up_vector()), obb1, obb2))


def local_bounding_box_to_global(actor, actor_bb):
  actor_transform = actor.get_transform()
  global_location = actor_transform.transform(actor_bb.location)
  global_bounding_box = carla.BoundingBox(global_location, actor_bb.extent)

  # Assumes that pitch and roll are 0 which they usually are.
  global_bounding_box.rotation = carla.Rotation(pitch=actor_transform.rotation.pitch + actor_bb.rotation.pitch,
                                                yaw=actor_transform.rotation.yaw + actor_bb.rotation.yaw,
                                                roll=actor_transform.rotation.roll + actor_bb.rotation.roll)

  return global_bounding_box

def lidar_to_ego_coordinate(config, lidar):
  """
  Converts the LiDAR points given by the simulator into the ego agents
  coordinate system
  :param config: GlobalConfig, used to read out lidar orientation and location
  :param lidar: the LiDAR point cloud as provided in the input of run_step
  :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
  coordinate system.
  """
  yaw = np.deg2rad(config.lidar_yaw)
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
                             dtype=np.float32)

  translation = np.array([config.lidar_x, config.lidar_y, config.lidar_z], dtype=np.float32)

  # The double transpose is a trick to compute all the points together.
  ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

  return ego_lidar

def lidar_to_histogram_features(lidar, config):
  """
  Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
  :param lidar: (N,3) numpy, LiDAR point cloud
  :param use_ground_plane, whether to use the ground plane
  :return: (2, H, W) numpy, LiDAR as sparse image
  """

  def splat_points(point_cloud):
    # 256 x 256 grid
    xbins = np.linspace(config.lidar_min_x, config.lidar_max_x,
                        (config.lidar_max_x - config.lidar_min_x) * int(config.pixels_per_meter) + 1)
    ybins = np.linspace(config.lidar_min_y, config.lidar_max_y,
                        (config.lidar_max_y - config.lidar_min_y) * int(config.pixels_per_meter) + 1)
    hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
    hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
    overhead_splat = hist * (1.0 / config.hist_max_per_pixel)
    overhead_splat = (overhead_splat * 255).astype(np.uint8)
    # Swaps the image to align with the map elements.
    return overhead_splat[::-1]

  # Remove points above the vehicle
  lidar = lidar[lidar[..., 2] < config.lidar_max_height]
  above = lidar[lidar[..., 2] > config.lidar_split_height]
  return splat_points(above)

def preprocess_compass(compass):
  """
  Checks the compass for Nans and rotates it into the default CARLA coordinate
  system with range [-pi,pi].
  :param compass: compass value provided by the IMU, in radian
  :return: yaw of the car in radian in the CARLA coordinate system.
  """
  if math.isnan(compass):  # simulation bug
    compass = 0.0
  # The minus 90.0 degree is because the compass sensor uses a different
  # coordinate system then CARLA. Check the coordinate_sytems.txt file
  compass = normalize_angle(compass - np.deg2rad(90.0))

  return compass

def lidar_to_ego_coordinate(lidar, pos, yaw_rot):
  """
  Converts the LiDAR points given by the simulator into the ego agents
  coordinate system
  :param lidar: the LiDAR point cloud as provided in the input of run_step
         pos: 3D position of lidar as list
         yaw_rot: Yaw rotation of the lidar in degree
  :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
  coordinate system.
  """
  yaw = np.deg2rad(yaw_rot, dtype=np.float32)
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
                             dtype=np.float32)

  translation = np.array(pos, dtype=np.float32)

  # The double transpose is a trick to compute all the points together.
  ego_lidar = (rotation_matrix @ lidar[:, :3].T).T + translation

  return ego_lidar

class InfoDummy(object):
  """ Info dummy that tries to mimic TIMMs info class"""

  def __init__(self, info):
    super().__init__()
    self.info = info

# @torch.no_grad()
# def hl_gaus_pdf(mean, std, vmin, vmax, bucket_size):
#   std = torch.ones_like(mean) * std
#   bins = torch.arange(vmin - (bucket_size * 0.5), vmax, bucket_size, device=mean.device)
#   bins2 = torch.arange(vmin + (bucket_size * 0.5), vmax + (bucket_size * 0.5) + 0.0001, bucket_size,
#                        device=mean.device)
#   distr = torch.distributions.normal.Normal(mean.unsqueeze(1), std.unsqueeze(1))
#   cdf = distr.cdf(bins.unsqueeze(0))
#   cdf2 = distr.cdf(bins2.unsqueeze(0))
#
#   pdf = cdf2 - cdf
#
#   return pdf
#
#
# @torch.no_grad()
# def hl_gaus_bins(vmin, vmax, bucket_size, device):
#   return torch.arange(vmin, vmax + bucket_size, bucket_size, device=device)
