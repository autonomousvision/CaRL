import random
import pathlib
import gzip
import psutil
import subprocess
import argparse
import time
import socket
import os
import glob
import importlib
import sys
import inspect
import traceback

import carla
from tqdm import tqdm

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_all_scenario_classes(scenario_runner_root):
  """
  Searches through the 'scenarios' folder for all the Python classes
  """
  # Path of all scenario at "srunner/scenarios" folder
  scenarios_list = glob.glob(f'{scenario_runner_root}/srunner/scenarios/*.py')

  all_scenario_classes = {}

  for scenario_file in scenarios_list:

    # Get their module
    module_name = os.path.basename(scenario_file).split('.')[0]
    sys.path.insert(0, os.path.dirname(scenario_file))
    scenario_module = importlib.import_module(module_name)

    # And their members of type class
    for member in inspect.getmembers(scenario_module, inspect.isclass):
      all_scenario_classes[member[0]] = member[1]

  return all_scenario_classes
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
    parser.add_argument('--scenario_runner_root',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/ad_planning/2_carla/original_leaderboard/scenario_runner',
                        help='root folder of scenario runner')
    args, _ = parser.parse_known_args()
    current_port_0 = next_free_port(1024)
    current_port_1 = current_port_0 + 3
    current_port_1 = next_free_port(current_port_1)
    carla_servers = []
    client_ports.append(current_port_0)
    # carla_servers.append(subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
    #   f'bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={current_port_0} -nosound -nullrhi '
    #   f'-RenderOffScreen -carla-streaming-port={current_port_1} -graphicsadapter={args.gpu_id}',
    #   shell=True))
    # time.sleep(60)
    # if carla_servers[0].poll() is not None:
    #   print('Carla server crashed')
    # 'Town07',   'Town10HD' 'Town11', 'Town12', 'Town15','Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD'
    map_names = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
    save_folder = args.save_folder
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    client = carla.Client('localhost', 2000)#current_port_0)
    client.set_timeout(300.0)
    settings = carla.WorldSettings(
      synchronous_mode=True,
      fixed_delta_seconds=0.1,
      deterministic_ragdolls=True,
      no_rendering_mode=False,
      spectator_as_ego=False,
    )
    client.get_world().apply_settings(settings)
    scenario_classes = get_all_scenario_classes(args.scenario_runner_root)
    considered_scenarios = [
      # "ControlLoss",
      # "HardBreakRoute",
      # "DynamicObjectCrossing",  # crossing_angle, direction, distance
      # "VehicleTurningRoute", #  config.route
      # "SignalizedJunctionLeftTurn",
      # "OppositeVehicleRunningRedLight",
      # "SignalizedJunctionRightTurn",
    ]
    CarlaDataProvider.set_client(client)
    selected_scenario_classes = {k: v for k, v in scenario_classes.items() if k in considered_scenarios}
    for map_name in map_names:
      world = client.load_world(map_name, reset_settings=False)

      CarlaDataProvider.set_world(world)
      carla_map = world.get_map()
      list_of_all_waypoints = carla_map.generate_waypoints(10.0)

      spawn_points = carla_map.get_spawn_points()
      blueprint_library = world.get_blueprint_library()
      vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
      dummy_actor = world.spawn_actor(vehicle_bp, spawn_points[0])
      dummy_config = ScenarioConfiguration()
      for scenario in selected_scenario_classes.values():
        for waypoint in tqdm(list_of_all_waypoints):

          try:
            dummy_config.trigger_points = [waypoint.transform]
            scenario_instance = scenario(world, [dummy_actor], dummy_config)

            # if debug:
            #   scenario_loc = scenario_config.trigger_points[0].location
            #   debug_loc = carla_map.get_waypoint(scenario_loc).transform.location + carla.Location(z=0.2)
            #   world.debug.draw_point(
            #     debug_loc, size=0.2, color=carla.Color(128, 0, 0), life_time=self.timeout
            #   )
            #   world.debug.draw_string(
            #     debug_loc, str(scenario_config.name), draw_shadow=False,
            #     color=carla.Color(0, 0, 128), life_time=-1, persistent_lines=True
            #   )

          except Exception as e:
            print(f"\033[93mSkipping scenario '{dummy_config.name}' due to setup error: {e}")
            print(f"\n{traceback.format_exc()}")
            print("\033[0m", end="")
            continue

          print(waypoint)



    kill(carla_servers[0].pid)
    print('Done generating routes.')
    kill_all_carla_servers(client_ports)
    del carla_servers

  except (KeyboardInterrupt, RuntimeError) as e:
    print(e)
    kill_all_carla_servers(client_ports)


if __name__ == '__main__':
  main()
