'''
File that generates routes that randomly drive around town. Need a carla server running to execute.
'''
from random import shuffle
import pathlib
import gzip
import psutil
import subprocess
import argparse
import time
import socket
import os

import carla
from lxml import etree
from tqdm import tqdm
from leaderboard.utils.route_manipulation import interpolate_trajectory
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


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
    route = routes[route_id]
    new_route = etree.SubElement(root, 'route')
    new_route.set('id', str(route_id))
    new_route.set('town', str(town))
    route_length = round(compute_route_length(route), 2)
    new_route.set('length', str(route_length))
    etree.SubElement(new_route, 'weathers').text = ''
    waypoints = etree.SubElement(new_route, 'waypoints')
    for point in route:
      new_point = etree.SubElement(waypoints, 'position')
      new_point.set('x', str(round(point[0].location.x, 2)))
      new_point.set('y', str(round(point[0].location.y, 2)))
      new_point.set('z', str(round(point[0].location.z, 2)))
      new_point.set('pitch', str(round(point[0].rotation.pitch, 4)))
      new_point.set('yaw', str(round(point[0].rotation.yaw, 4)))
      new_point.set('roll', str(round(point[0].rotation.roll, 4)))
      new_point.set('command', str(point[1].value))

    etree.SubElement(new_route, 'scenarios').text = ''

  with gzip.open(filename, 'wb') as f:
    test = etree.tostring(tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
    f.write(test)

  del tree

def preproces_interpolation(routes, world_map):
  interpolated_routes = []
  for route in tqdm(routes):
    dense_route = interpolate_trajectory(route, world_map=world_map)
    interpolated_routes.append(dense_route)

  return interpolated_routes


def main():
  client_ports = []
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/'
                                r'data/longest6_no_scenarios_preprocessed/',
                        help='folder where to save the root files')
    parser.add_argument('--start_repetition',
                        default=0,
                        type=int,
                        help='start_repetition to run')
    parser.add_argument('--repetitions',
                        default=16,
                        type=int,
                        help='Number of times the same route file is saved (shuffled)')
    parser.add_argument('--input_route_folder',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/longest6_split',
                        help='folder containing the input routes files to be preprocessed')
    parser.add_argument('--debug',
                        default=0,
                        type=int,
                        help='whether to spawn a carla server.')

    args, _ = parser.parse_known_args()

    if not args.debug:
      current_port_0 = next_free_port(1024 + (1000 * args.start_repetition))
      current_port_1 = current_port_0 + 3
      current_port_1 = next_free_port(current_port_1)
      current_port_2 = current_port_1 + 3
      traffic_manager_port = next_free_port(current_port_2)
      carla_servers = []
      client_ports.append(current_port_0)
      carla_servers.append(subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
        f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={current_port_0} -nosound -nullrhi '
        f'-RenderOffScreen -carla-streaming-port={current_port_1}',
        shell=True))
      time.sleep(60)
      if carla_servers[0].poll() is not None:
        print('Carla server crashed')
      client = carla.Client('localhost', current_port_0)
      num_routes = 500
    else:
      client = carla.Client('localhost', 2000)
      traffic_manager_port = 8000
      num_routes = 10


    #, , 'Town11', 'Town12', 'Town15'
    map_names = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
    save_folder = args.save_folder
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    client.set_timeout(300.0)
    settings = carla.WorldSettings(
      synchronous_mode=True,
      fixed_delta_seconds=0.1,
      deterministic_ragdolls=True,
      no_rendering_mode=False,
      spectator_as_ego=False,
    )
    client.get_world().apply_settings(settings)
    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_hybrid_physics_mode(True)

    CarlaDataProvider.set_client(client)
    for map_name in map_names:
      world = client.load_world(map_name, reset_settings=False)
      world.reset_all_traffic_lights()
      world.tick()
      CarlaDataProvider.set_world(world)
      CarlaDataProvider.set_traffic_manager_port(traffic_manager_port)
      CarlaDataProvider.on_carla_tick()

      carla_map = world.get_map()

      routes = []
      for root, dirs, files in os.walk(args.input_route_folder):
        for file in files:
          if file.endswith(".xml"):
            print(file)
            tree = etree.parse(os.path.join(root, file))
            et_routes = tree.getroot()
            for et_route in et_routes:
              if et_route.attrib['town'] == map_name:
                et_route_points = []
                for position in et_route.find('waypoints').iter('position'):
                  carla_loc = carla.Location(x=float(position.attrib['x']),
                                             y=float(position.attrib['y']),
                                             z=float(position.attrib['z']))
                  et_route_points.append(carla_loc)

                routes.append(et_route_points)


      routes = preproces_interpolation(routes, carla_map)
      for rep in range(args.repetitions):
        shuffle(routes)  # Important so that parallel environments start with different routes
        save_data(os.path.join(save_folder, f'route_{map_name}_{rep:02d}.xml.gz'), routes, map_name)
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

