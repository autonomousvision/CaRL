'''
File that generates routes that randomly drive around town. Need a carla server running to execute.
'''
import random
import pathlib
import gzip
import psutil
import subprocess
import argparse
import time
import socket
import os
import traceback

import carla
from lxml import etree
from tqdm import tqdm
from leaderboard.utils.route_manipulation import interpolate_trajectory
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def spawn_ego_vehicle(ego_transform, world):
  """Spawn the ego vehicle at the first waypoint of the route"""
  elevate_transform = ego_transform
  elevate_transform.location.z += 0.5

  ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz_2020',
                                                    elevate_transform,
                                                    rolename='hero')
  if not ego_vehicle:
    return

  spectator = world.get_spectator()
  spectator.set_transform(carla.Transform(elevate_transform.location + carla.Location(z=50),
                                          carla.Rotation(pitch=-90)))

  world.tick()
  CarlaDataProvider.on_carla_tick()

  return ego_vehicle

def kill(proc_pid):
  if psutil.pid_exists(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
      proc.kill()
    process.kill()
def next_free_port(port=1024, max_port=65535):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  while port <= max_port:
    try:
      sock.bind(('', port))
      sock.close()
      return port
    except OSError:
      port += 1
  raise IOError('no free ports')

def kill_all_carla_servers(ports):
  # Need a failsafe way to find and kill all carla servers. We do so by port.
  for proc in psutil.process_iter():
    # check whether the process name matches
    try:
      proc_connections = proc.connections(kind='all')
    except (PermissionError, psutil.AccessDenied):  # Avoid sudo processes
      proc_connections = None

    if proc_connections is not None:
      for conns in proc_connections:
        if not isinstance(conns.laddr, str):  # Avoid unix paths
          if conns.laddr.port in ports:
            try:
              proc.kill()
            except psutil.NoSuchProcess:  # Catch the error caused by the process no longer existing
              pass  # Ignore it

def compute_route_length(route):
  route_length = 0.0
  previous_location = None

  for transform, _ in route:
    location = transform.location
    if previous_location:
      dist_vec = location - previous_location
      route_length += dist_vec.length()
    previous_location = location

  return route_length

def save_data(filename, routes, town):
  tree = etree.ElementTree(etree.Element('routes'))
  root = tree.getroot()

  for route_id in tqdm(range(len(routes))):
    route = routes.pop()
    new_route = etree.SubElement(root, 'route')
    new_route.set('id', str(route_id))
    new_route.set('town', str(town))
    route_length = round(compute_route_length(route), 1)
    new_route.set('length', str(route_length))
    etree.SubElement(new_route, 'weathers').text = ''
    waypoints = etree.SubElement(new_route, 'waypoints')
    for point in route:
      new_point = etree.SubElement(waypoints, 'position')
      new_point.set('x', str(round(point[0].location.x, 1)))
      new_point.set('y', str(round(point[0].location.y, 1)))
      new_point.set('z', str(round(point[0].location.z, 1)))
      new_point.set('pitch', str(round(point[0].rotation.pitch, 1)))
      new_point.set('yaw', str(round(point[0].rotation.yaw, 1)))
      new_point.set('roll', str(round(point[0].rotation.roll, 1)))
      new_point.set('command', str(point[1].value))

    del route

    etree.SubElement(new_route, 'scenarios').text = ''

  with gzip.open(filename, 'wb') as f:
    test = etree.tostring(tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
    f.write(test)

  del tree
  del routes

def preproces_interpolation(routes, world_map):
  interpolated_routes = []
  original_routes = []
  for route in tqdm(routes):
    try:
      dense_route = interpolate_trajectory(route, world_map=world_map)
    except Exception as e:
      print(f'Skipping route due to error: {e}')
      print(f'Length of route: {len(route)}')
      traceback.print_exc()
      continue
    dense_route_capped = []
    route_length = 0.0
    previous_location = None
    # This route will be at least 1000 meters long, cut off after 1000m to have similar length routes
    for waypoint in dense_route:
      location = waypoint[0].location
      if previous_location:
        dist_vec = location - previous_location
        route_length += dist_vec.length()
      previous_location = location
      dense_route_capped.append(waypoint)
      if route_length > 1000.0:
        break
    interpolated_routes.append(dense_route_capped)
    original_routes.append(route)

  return interpolated_routes, original_routes


def main():
  client_ports = []
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data'
                                r'/roach_preprocessed_routes9/',
                        help='folder where to save the root files')
    parser.add_argument('--carla_root',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/carla_9_15',
                        help='folder containing carla')
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        help='id to run the carla server on')
    parser.add_argument('--start_repetition',
                        default=0,
                        type=int,
                        help='start_repetition to run')
    args, _ = parser.parse_known_args()
    current_port_0 = next_free_port(1024 + (1000*args.start_repetition))
    current_port_1 =  current_port_0 + 3
    current_port_1 = next_free_port(current_port_1)
    carla_servers = []
    client_ports.append(current_port_0)
    carla_servers.append(subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
      f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={current_port_0} -nosound -nullrhi '
      f'-RenderOffScreen -carla-streaming-port={current_port_1} -graphicsadapter={args.gpu_id}',
      shell=True))
    time.sleep(60)
    if carla_servers[0].poll() is not None:
      print('Carla server crashed')
    #  'Town11'
    map_names = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD', 'Town12', 'Town13', 'Town15']
    save_folder = args.save_folder
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    client = carla.Client('localhost', current_port_0)
    client.set_timeout(300.0)
    settings = carla.WorldSettings(
      synchronous_mode=True,
      fixed_delta_seconds=0.1,
      deterministic_ragdolls=True,
      no_rendering_mode=False,
      spectator_as_ego=False,
    )
    client.get_world().apply_settings(settings)
    CarlaDataProvider.set_client(client)
    for repetition in range(args.start_repetition, args.start_repetition+1):
      print(f'Repetition {repetition}')
      for map_name in map_names:
        world = client.load_world(map_name, reset_settings=False)
        settings = world.get_settings()
        settings.tile_stream_distance = 2000  # roughly 2*route length.
        settings.actor_active_distance = 2000  # roughly 2*route length.
        # 0.187191
        world.apply_settings(settings)

        world.reset_all_traffic_lights()
        world.tick()
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.on_carla_tick()
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        spawn_waypoints = [carla_map.get_waypoint(s_p.location, project_to_road=False) for s_p in spawn_points]
        routes = []
          
        for _ in tqdm(range(1000)):
          current_waypoint = random.choice(spawn_waypoints)
          if current_waypoint is None:
            continue
          route_length = 0.0
          route = [current_waypoint.transform.location]
          route_complete = True
          while route_length < 1000.0:
            next_waypoint_list = current_waypoint.next(100.0)
            if len(next_waypoint_list) <= 0:
              route_complete = False
              break
            next_waypoint = random.choice(next_waypoint_list)
            distance = next_waypoint.transform.location.distance(current_waypoint.transform.location)
            route_length += distance
            route.append(next_waypoint.transform.location)
            current_waypoint = next_waypoint

          if route_complete:
            routes.append(route)
        pre_processed_routes, original_routes = preproces_interpolation(routes, carla_map)
        
        # Validate routes:
        # The interpolation algorithm is buggy returning corrupted routes sometimes. Try to filter them.
        validated_routes = []
        for idx, proc_route in enumerate(tqdm(pre_processed_routes)):
          startpoint = carla_map.get_waypoint(proc_route[0][0].location, project_to_road=False)

          valid = True
          if startpoint is None or startpoint.lane_type != carla.LaneType.Driving:
            valid = False
          if proc_route[0][0].location.distance(original_routes[idx][0]) > 5.0:
            valid = False
          ego_vehicle = spawn_ego_vehicle(proc_route[0][0], world)
          if ego_vehicle is None:
            valid = False
          else:
            if CarlaDataProvider.actor_id_exists(ego_vehicle.id):
              CarlaDataProvider.remove_actor_by_id(ego_vehicle.id)

          if valid:
            validated_routes.append(proc_route)
          CarlaDataProvider.cleanup_route()
        
        save_data(os.path.join(save_folder, f'route_{map_name}_{repetition:02}.xml.gz'), validated_routes, map_name)
        # for point in route:
        #   world.debug.draw_point(point.transform.location,  life_time=0)
        print(f'Route generated for {map_name}')

    kill(carla_servers[0].pid)
    print('Done generating routes.')
    kill_all_carla_servers(client_ports)
    del carla_servers

  except (KeyboardInterrupt, RuntimeError) as e:
    print(e)
    kill_all_carla_servers(client_ports)

if __name__ == '__main__':
  main()

