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
import glob
import importlib
import sys
import inspect
import math
import xml.etree.ElementTree as ET
from random import shuffle
from copy import copy
import ujson

import carla
from lxml import etree
from tqdm import tqdm
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData
from agents.navigation.local_planner import RoadOption

from generate_utils import generate_road_options
from weathers import WEATHER_SETTINGS

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

def same_sign(x, y):
  if (x * y > 0) or (x == 0 and y == 0):
    return True
  else:
    return False

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

def save_data(filename, routes, scenarios_all_routes, town):
  tree = etree.ElementTree(etree.Element('routes'))
  root = tree.getroot()

  for route_id in tqdm(range(len(routes))):
    route = routes.pop()
    scenarios_one_route = scenarios_all_routes.pop()
    new_route = etree.SubElement(root, 'route')
    new_route.set('id', str(route_id))
    new_route.set('town', str(town))
    route_length = round(compute_route_length(route), 2)
    new_route.set('length', str(route_length))

    # randomize weather for visual diversity
    chosen_preset = random.choice(list(WEATHER_SETTINGS.keys()))
    chosen_weather = WEATHER_SETTINGS[chosen_preset]

    weathers = etree.SubElement(new_route, 'weathers')
    weather = etree.SubElement(weathers, 'weather')
    weather.set('route_percentage', "0")  # We only use 1 weather per route

    for weather_parameter in chosen_weather.keys():
      # sun_azimuth_angle is bugged in default weathers. Set it manually to a random value.
      if weather_parameter == 'sun_azimuth_angle':
        sun_azimuth_angle = random.uniform(0, 360)
        weather.set(weather_parameter, str(round(sun_azimuth_angle, 1)))
      else:
        weather.set(weather_parameter, str(round(chosen_weather[weather_parameter], 2)))

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

    scenarios = etree.SubElement(new_route, 'scenarios')
    for scenario in scenarios_one_route:
      new_scenario = etree.SubElement(scenarios, 'scenario')
      new_scenario.set('name', scenario[0])
      new_scenario.set('type', scenario[1])
      for option in scenario[2]:
        new_option = etree.SubElement(new_scenario, option)
        if option == 'trigger_point':
          new_option.set('x', str(round(scenario[2][option].location.x, 1)))
          new_option.set('y', str(round(scenario[2][option].location.y, 1)))
          new_option.set('z', str(round(scenario[2][option].location.z, 1)))
          new_option.set('yaw', str(round(scenario[2][option].rotation.yaw, 1)))
        elif option in ('source_dist_interval', 'frequency'):
          if 'value' in scenario[2][option]:
            new_option.set('value', str(scenario[2][option]['value']))
          else:
            new_option.set('from', str(scenario[2][option]['from']))
            new_option.set('to', str(scenario[2][option]['to']))
        elif option in ('start_actor_flow', 'end_actor_flow', 'other_actor_location'):
          new_option.set('x', str(round(float(scenario[2][option]['x']), 1)))
          new_option.set('y', str(round(float(scenario[2][option]['y']), 1)))
          new_option.set('z', str(round(float(scenario[2][option]['z']), 1)))
        elif option != 'generate_scenarios':  # Debug variable that we don't need to save.  # TODO seems still be saved
          new_option.set('value', str(scenario[2][option]['value']))

    del route
    del scenarios_one_route

  with gzip.open(filename, 'wb') as f:
    test = etree.tostring(tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
    f.write(test)

  del tree
  del routes
  del scenarios_all_routes

def preproces_interpolation(routes, start_scenarios):
  interpolated_routes = []
  original_routes = []
  original_start_scenarios = []
  for idx, route in enumerate(tqdm(routes)):
    try:
      dense_route = generate_road_options(route)
    except Exception as e:
      print(f'Skipping route due to error: {e}')
      print(f'Length of route: {len(route)}')
      traceback.print_exc()
      continue

    interpolated_routes.append(dense_route)
    original_routes.append(route)
    original_start_scenarios.append(start_scenarios[idx])

  return interpolated_routes, original_routes, original_start_scenarios


def main():
  client_ports = []
  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data'
                                r'/debug_routes_with_scenarios/',
                        help='folder where to save the root files')
    parser.add_argument('--carla_root',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/carla_9_15',
                        help='folder containing carla')
    parser.add_argument('--start_repetition',
                        default=0,
                        type=int,
                        help='start_repetition to run')
    parser.add_argument('--scenario_runner_root',
                        type=str,
                        default=r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/scenario_runner',
                        help='root folder of scenario runner')
    parser.add_argument('--scenario_dilation',
                        default=100,
                        type=int,
                        help='every scenario_dilation meters a scenarios will be spawned.')
    parser.add_argument('--debug',
                        default=0,
                        type=int,
                        help='whether to spawn a carla server.')
    parser.add_argument('--generate_scenarios',
                        default=0,
                        type=int,
                        help='whether to generate scenarios. 0: no , 1: yes')
    parser.add_argument('--only_leaderboard_1',
                        default=1,
                        type=int,
                        help='whether only generate the scenario types from leaderboard 1.0.'
                             'Otherwise scenario types from leaderboard 2.0 are also generated.')
    parser.add_argument('--route_length',
                        default=1000.0,
                        type=float,
                        help='Length of the routes to be created.')
    parser.add_argument('--num_routes',
                        default=500,
                        type=int,
                        help='Number of routes to be generated per file')

    args, _ = parser.parse_known_args()
    if not args.debug:
      current_port_0 = next_free_port(1024 + (100*args.start_repetition))
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
      num_routes = args.num_routes
    else:
      client = carla.Client('localhost', 2000)
      traffic_manager_port = 8000
      num_routes = 10

    chance_to_lane_change = 0.10
    min_lane_keep_distance = 200.0
      # 'Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD', 'Town11', 'Town12', 'Town13', 'Town15'
    map_names = [ 'Town07', 'Town10HD'] # 'Town01', 'Town02', 'Town03', 'Town05',  'Town12', 'Town13', 'Town15' 'Town04', 'Town06',
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

    scenario_classes = get_all_scenario_classes(args.scenario_runner_root)
    if args.only_leaderboard_1:
      considered_scenarios = [
        'ControlLoss',  # Done
        'HardBreakRoute',  #Done
        'DynamicObjectCrossing',  # Done
        'VehicleTurningRoute',  # Done
        'SignalizedJunctionLeftTurn', # Done
        'OppositeVehicleRunningRedLight', # Done
        'SignalizedJunctionRightTurn',
      ]
      scenarios_from_file = []

    else:
      considered_scenarios = [
        'Accident',  # Done
        'AccidentTwoWays',  # Done
        'BlockedIntersection',  # Done
        'ConstructionObstacle',  # Done
        'ConstructionObstacleTwoWays',  # Done
        'ControlLoss',  # Done
        'CrossingBicycleFlow',  # Done NOTE very few locations in Town 12 only
        'DynamicObjectCrossing',  # Done
        'EnterActorFlow',  # Done only loaded from dataset
        'EnterActorFlowV2', # Done only loaded from dataset
        'HardBreakRoute',  # Done
        'HazardAtSideLane',  # Done
        'HazardAtSideLaneTwoWays',  # Done
        'HighwayCutIn', # Uses other_actor  # Done only loaded from dataset
        'HighwayExit',  # Done doesn't work well because the route does often not lead off ramp after the scenario started
        'InterurbanActorFlow',  # Done, works less well in town 03 and 05
        'InterurbanAdvancedActorFlow',  # Done
        'InvadingTurn',  # Done
        'MergerIntoSlowTraffic',  # Done only loaded from dataset
        'MergerIntoSlowTrafficV2',  # Done only loaded from dataset
        'NonSignalizedJunctionLeftTurn',  # Done
        'NonSignalizedJunctionRightTurn',  # Done
        'OppositeVehicleRunningRedLight',  # Done
        'OppositeVehicleTakingPriority',  # Done
        'ParkedObstacle',  # Done
        'ParkedObstacleTwoWays',  # Done
        'ParkingCrossingPedestrian',  # Done
        'ParkingCutIn',  # Done
        'ParkingExit',  # Done
        'PedestrianCrossing',  # Done
        'PriorityAtJunction',  # Done
        'SignalizedJunctionLeftTurn',  # Done
        'SignalizedJunctionRightTurn',  # Done
        'StaticCutIn',  # Done but doesn't work very well
        'VehicleOpensDoorTwoWays',  # Done
        'VehicleTurningRoute',  # Done
        'VehicleTurningRoutePedestrian',  # Done
        'YieldToEmergencyVehicle'  # Does not work well in e.g. town 05
      ]

      # Highway entries and exits are not labelled in the CARLA map, so I have no way to automatically generate these
      # scenarios instead we just load the once we have
      scenarios_from_file = ['EnterActorFlow',
                             'EnterActorFlowV2',
                             'HighwayCutIn',
                             'HighwayExit',
                             'MergerIntoSlowTraffic',
                             'MergerIntoSlowTrafficV2',
                             ]
    global_scenario_counts = {}
    for scenario_key in considered_scenarios:
      global_scenario_counts[scenario_key] = 0

    num_scenario_from_file = len(scenarios_from_file)
    num_scenario_types = len(considered_scenarios)

    selected_scenario_classes = {k: v for k, v in scenario_classes.items() if k in considered_scenarios}

    parking_exit_global = selected_scenario_classes.pop('ParkingExit', None)
    for repetition in range(args.start_repetition, args.start_repetition+1):
      print(f'Repetition {repetition}')
      for map_name in map_names:
        world = client.load_world(map_name, reset_settings=False)
        settings = world.get_settings()
        settings.tile_stream_distance = 2000  # roughly 2*route length.
        settings.actor_active_distance = 2000
        # 0.187191
        world.apply_settings(settings)

        world.reset_all_traffic_lights()
        world.tick()
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.set_traffic_manager_port(traffic_manager_port)
        CarlaDataProvider.on_carla_tick()


        carla_map = world.get_map()
        routes = []
        topology = get_unique_topology_wps(carla_map)
        start_scenarios = []

        if map_name in ('Town12', 'Town13'):
          extracted_scenarios = load_scenarios_from_file(scenarios_from_file, map_name)
          frac_from_file = num_scenario_from_file / num_scenario_types
        else:
          frac_from_file = 0.0
          extracted_scenarios = []

        # Debugging
        # settings = carla.WorldSettings(
        #   synchronous_mode=False,
        #   fixed_delta_seconds=0.1,
        #   deterministic_ragdolls=True,
        #   no_rendering_mode=False,
        #   spectator_as_ego=False,
        # )
        # client.get_world().apply_settings(settings)
        # draw_waypoints_no_color(world, topology, 1.0, 0.1)

        for _ in tqdm(range(num_routes)):
          route_length = 0.0
          route_complete = True
          route = []

          # Start route with scenario from file for the proportion of scenarios where this is the case
          if random.uniform(0, 1) < frac_from_file:
            shuffle(extracted_scenarios)
            trigger = extracted_scenarios[0]['trigger_point']
            current_waypoint = carla_map.get_waypoint(trigger.location, project_to_road=False)
            start_scenarios.append(extracted_scenarios[0])
          else:
            current_waypoint = random.choice(topology)
            start_scenarios.append({})

          if current_waypoint is None:
            continue
          route.append(current_waypoint)
          previous_waypoint = current_waypoint
          while route_length < 100.0:
            prev_waypoint_list = current_waypoint.previous(1.0)
            if len(prev_waypoint_list) <= 0:
              break
            current_waypoint = random.choice(prev_waypoint_list)
            distance = previous_waypoint.transform.location.distance(current_waypoint.transform.location)
            route.insert(0, current_waypoint)
            route_length += distance
            previous_waypoint = current_waypoint

          current_waypoint = route[-1]

          last_lane_change = 0.0
          while route_length < args.route_length:
            next_waypoint_list = current_waypoint.next(1.0)
            if len(next_waypoint_list) <= 0:
              route_complete = False
              break

            # Consider doing a lane change.
            next_waypoint = random.choice(next_waypoint_list)
            if not next_waypoint.is_junction:
              left_lane_change_available = next_waypoint.left_lane_marking and (next_waypoint.left_lane_marking.lane_change == carla.LaneChange.Left or next_waypoint.left_lane_marking.lane_change == carla.LaneChange.Both)
              right_lane_change_available = next_waypoint.right_lane_marking and (next_waypoint.right_lane_marking.lane_change == carla.LaneChange.Right or next_waypoint.right_lane_marking.lane_change == carla.LaneChange.Both)
              if left_lane_change_available or right_lane_change_available:
                if random.uniform(0, 1) < chance_to_lane_change:
                  options = []
                  if left_lane_change_available:
                    options.append('left')
                  if right_lane_change_available:
                    options.append('right')
                  direction = random.choice(options)
                  if direction == 'left':
                    changed_waypoint = next_waypoint.get_left_lane()
                  else:
                    changed_waypoint = next_waypoint.get_right_lane()

                  if changed_waypoint is not None \
                      and changed_waypoint.lane_type == carla.LaneType.Driving \
                      and changed_waypoint.road_id == next_waypoint.road_id \
                      and last_lane_change <= 0.0:
                    next_waypoint = changed_waypoint
                    last_lane_change = min_lane_keep_distance

            distance = next_waypoint.transform.location.distance(current_waypoint.transform.location)
            route_length += distance
            last_lane_change -= distance
            route.append(next_waypoint)
            current_waypoint = next_waypoint

          if route_complete:
            routes.append(route)

        pre_processed_routes, original_routes, original_start_scenarios = preproces_interpolation(routes, start_scenarios)

        # Debug
        # import numpy as np
        # from matplotlib import pyplot as plt
        # for idx, checkroute in enumerate(pre_processed_routes):
        #   last_point = None
        #   render = False
        #   for lala in checkroute:
        #     if last_point is not None:
        #       distance = last_point[0].location.distance(lala[0].location)
        #       if distance > 3.0:
        #         render = True
        #         print('Problem')
        #
        #     last_point = lala
        #
        #   if True:#render:
        #     xs = []
        #     ys = []
        #     for lala in checkroute:
        #       xs.append(lala[0].location.x)
        #       ys.append(lala[0].location.y)
        #     plt.scatter(xs, ys, color='r')
        #     old_xs = []
        #     old_ys = []
        #     for old_points in original_routes[idx]:
        #       old_xs.append(old_points.x)
        #       old_ys.append(old_points.y)
        #     plt.scatter(old_xs, old_ys, alpha=0.25, color='b')
        #   plt.show()
        #   plt.close()

        # Validate routes:
        # The interpolation algorithm is buggy returning corrupted routes sometimes. Try to filter them.
        validated_routes = []
        validated_start_scenarios = []
        for idx, proc_route in enumerate(tqdm(pre_processed_routes)):
          startpoint = carla_map.get_waypoint(proc_route[0][0].location, project_to_road=False)
          valid = True
          if startpoint is None or startpoint.lane_type != carla.LaneType.Driving:
            valid = False
          if proc_route[0][0].location.distance(original_routes[idx][0].transform.location) > 5.0:
            valid = False
          ego_vehicle = spawn_ego_vehicle(proc_route[0][0], world)
          if ego_vehicle is None:
            valid = False
          else:
            if CarlaDataProvider.actor_id_exists(ego_vehicle.id):
              CarlaDataProvider.remove_actor_by_id(ego_vehicle.id)
          if valid:
            validated_routes.append(proc_route)
            validated_start_scenarios.append(original_start_scenarios[idx])
          CarlaDataProvider.cleanup_route()

        # Debugging
        # settings = carla.WorldSettings(
        #   synchronous_mode=False,
        #   fixed_delta_seconds=0.1,
        #   deterministic_ragdolls=True,
        #   no_rendering_mode=False,
        #   spectator_as_ego=False,
        # )
        # client.get_world().apply_settings(settings)
        # _draw_waypoints(world, validated_routes, 1.0, 0.1)

        # Generate scenarios
        # Generate scenarios for route
        print('Map Name:', map_name)
        scenarios_all_routes = []

        # Use parking exit scenario?
        parking_exit = True
        if map_name not in ('Town03', 'Town05', 'Town06', 'Town10HD', 'Town12', 'Town13', 'Town15'):
          parking_exit = False

        for idx, proc_route in enumerate(tqdm(validated_routes)):
          scenarios_one_route = []

          scenario_counts = {}

          if not args.generate_scenarios:
            scenarios_all_routes.append(scenarios_one_route)
            continue

          ego_vehicle = spawn_ego_vehicle(proc_route[0][0], world)
          if ego_vehicle is None:
            scenarios_all_routes.append(scenarios_one_route)
            print('Faulty route:')
            continue

          for scenario_key in selected_scenario_classes:
            scenario_counts[scenario_key] = 0

          if parking_exit and parking_exit_global is not None:
            scenario_counts['ParkingExit'] = 0

          route_index = 0
          while route_index < len(proc_route):
            scenario_created = False
            shuffled_scenarios = list(selected_scenario_classes.keys())
            random.shuffle(shuffled_scenarios)

            if route_index == 0 and not len(validated_start_scenarios[idx]) == 0:
              start_s = copy(validated_start_scenarios[idx])
              start_s_type = start_s.pop('type')
              scenario_name = start_s_type + f'_{scenario_counts[start_s_type]:02d}'
              scenario_counts[start_s_type] += 1
              global_scenario_counts[start_s_type] += 1
              scenarios_one_route.append((scenario_name, start_s_type, start_s))
              # The scenario was 100 meters after the start of the route because we backtraced when generating the route.
              # So move scenario index 100 ahead to avoid a place with 2 scenarios
              route_index += 100

            elif route_index == 0 and parking_exit and parking_exit_global is not None:  # Parking Exit
              scenario_type = 'ParkingExit'
              scenario = parking_exit_global

              scenario_created, scenario_options = sample_scenario(args, route_index, proc_route,
                                                                   scenario, scenario_type, carla_map,
                                                                   ego_vehicle, world, scenario_counts,
                                                                   scenarios_one_route, global_scenario_counts)
              if not scenario_created:
                route_index -= args.scenario_dilation - 1  # Try again with a different scenario type than parking exit
            else:
              for scenario_type in shuffled_scenarios:
                scenario = selected_scenario_classes[scenario_type]
                scenario_created, scenario_options = sample_scenario(args, route_index, proc_route,
                                                                     scenario, scenario_type, carla_map,
                                                                     ego_vehicle, world, scenario_counts,
                                                                     scenarios_one_route, global_scenario_counts)

                if scenario_created:
                  break  # Break for loop

            route_index += args.scenario_dilation

          scenarios_all_routes.append(scenarios_one_route)
          CarlaDataProvider.remove_actor_by_id(ego_vehicle.id)
          CarlaDataProvider.cleanup_route()

        save_data(os.path.join(save_folder, f'route_{map_name}_{repetition:03}.xml.gz'), validated_routes, scenarios_all_routes, map_name)
        with open(os.path.join(save_folder, f'scenario_counts_{repetition:03}.json'), 'wt', encoding='utf-8') as fp:
          ujson.dump(global_scenario_counts, fp, sort_keys=True, indent=4)

        print(f'Route generated for {map_name}')

    kill(carla_servers[0].pid)
    print('Done generating routes.')
    kill_all_carla_servers(client_ports)
    del carla_servers

  except (KeyboardInterrupt, RuntimeError) as e:
    print(f"\n{traceback.format_exc()}")
    print(e)
    kill_all_carla_servers(client_ports)


def is_scenario_at_route(trigger_transform, route):
  """
  Check if the scenario is affecting the route.
  This is true if the trigger position is very close to any route point
  """
  dist_threshold = 2.0
  angle_threshold = 10
  def is_trigger_close(trigger_transform, route_transform):
    """Check if the two transforms are similar"""
    dx = trigger_transform.location.x - route_transform.location.x
    dy = trigger_transform.location.y - route_transform.location.y
    dz = trigger_transform.location.z - route_transform.location.z
    dpos = math.sqrt(dx * dx + dy * dy)

    dyaw = (float(trigger_transform.rotation.yaw) - route_transform.rotation.yaw) % 360

    return dz < dist_threshold and dpos < dist_threshold \
      and (dyaw < angle_threshold or dyaw > (360 - angle_threshold))

  for route_transform, _ in route:
    if is_trigger_close(trigger_transform, route_transform):
      return True

  return False

def normalize_angle_degree(x):
  x = x % 360.0
  if x > 180.0:
    x -= 360.0
  return x

def get_min_anlge_wp_from_list(reference_wp, wp_list, carla_map):
  if len(wp_list) == 1:
    return wp_list[0]
  else:
    ref_yaw = reference_wp.transform.rotation.yaw
    min_ref_yaw = 361
    min_wp = None
    for wp in wp_list:
      diff_angle = abs(normalize_angle_degree(wp.transform.rotation.yaw - ref_yaw))
      if diff_angle < min_ref_yaw:
        min_wp = wp
        min_ref_yaw = diff_angle

    if min_wp is not None and (min_ref_yaw < 2.0 or carla_map not in ('Town01', 'Town02')):
      return min_wp
    else:
      raise ValueError('Could not find next waypoint with small angle')

def get_unique_topology_wps(carla_map):
  '''Computes topology removing duplicated from multi-lanes-'''
  topology = carla_map.get_topology()
  topology_points = []
  road_ids = []
  for pair in topology:
    wp_0 = pair[0]
    wp_1 = pair[1]
    if wp_0 is not None and wp_0.road_id not in road_ids:
      road_ids.append(wp_0.road_id)
      topology_points.append(wp_0)
    if wp_1 is not None and wp_1.road_id not in road_ids:
      road_ids.append(wp_1.road_id)
      topology_points.append(wp_1)

  return topology_points

def draw_waypoints_no_color(world, routes, vertical_shift, size):
  for w in routes:
    wp = w.transform.location + carla.Location(z=vertical_shift)
    color = carla.Color(0, 128, 0)  # Green
    world.debug.draw_point(wp, size=size, color=color, life_time=-1)

def _draw_waypoints(world, routes, vertical_shift, size):
  """
  Draw a list of waypoints at a certain height given in vertical_shift.
  """
  for waypoints in routes:
    for i, w in enumerate(waypoints):
      wp = w[0].location + carla.Location(z=vertical_shift)

      if w[1] == RoadOption.LEFT:  # Yellow
        color = carla.Color(128, 128, 0)
      elif w[1] == RoadOption.RIGHT:  # Cyan
        color = carla.Color(0, 128, 128)
      elif w[1] == RoadOption.CHANGELANELEFT:  # Red
        color = carla.Color(255, 0, 0)
      elif w[1] == RoadOption.CHANGELANERIGHT:  # Blue
        color = carla.Color(0, 0, 255)
      elif w[1] == RoadOption.STRAIGHT:  # Gray
        color = carla.Color(64, 64, 64)
      else:  # LANEFOLLOW
        color = carla.Color(0, 128, 0)  # Green

      world.debug.draw_point(wp, size=size, color=color, life_time=-1)

    world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2 * size,
                                color=carla.Color(0, 0, 0), life_time=-1)
    # For the large town you need to spawn the ego to see something.
    # ego_vehicle = spawn_ego_vehicle(waypoints[0][0], world)

    print('Put debug point here to look at individual routes.')
    # if CarlaDataProvider.actor_id_exists(ego_vehicle.id):
    #   CarlaDataProvider.remove_actor_by_id(ego_vehicle.id)

def is_intersection_scenario(scenario_type): # TODO extend for intersection.
  return scenario_type in ('SignalizedJunctionLeftTurn', 'OppositeVehicleRunningRedLight',
                           'SignalizedJunctionRightTurn',)

def sample_scenario(args, route_index, proc_route, scenario, scenario_type,
                    carla_map, ego_vehicle, world, scenario_counts, scenarios_one_route, global_scenario_counts):
  scenario_created = False
  scenario_options = None
  tries = 1 # TODO overthink if and how we need this
  while not scenario_created:
    # During tries randomize start index in range.
    random_start_index = random.randrange(args.scenario_dilation)
    if scenario_type == 'ParkingExit':
      random_start_index = 0
    if random_start_index + route_index < len(proc_route):
      trigger_index = random_start_index + route_index
    else:
      trigger_index = route_index

    map_point = proc_route[trigger_index]

    dummy_config = ScenarioConfiguration()
    tries -= 1
    try:
      dummy_config.trigger_points = [map_point[0]]
      dummy_config.route = proc_route
      dummy_config.name = scenario_type + f'_{scenario_counts[scenario_type]:02d}'
      dummy_config.type = scenario_type
      dummy_config.town = carla_map
      dummy_config.agent = ego_vehicle
      dummy_config.weather = None  # TODO
      dummy_config.friction = None  # TODO
      dummy_config.subtype = None  # TODO might be relevant for  7 to 10
      dummy_config.route_var_name = f"ScenarioRouteNumber{sum(scenario_counts.values())}"
      # dummy_config.other_actors = [] # TODO
      scenario_instance = None

      if is_intersection_scenario(scenario_type):
        trigger_map_wp = carla_map.get_waypoint(map_point[0].location)
        if trigger_map_wp.is_junction:
          raise ValueError("Cannot start intersection scenario in intersection.")

      dummy_config.other_parameters, trigger_points = sample_additional_parameters(args, scenario_type, proc_route,
                                                                                   trigger_index, carla_map)
      if trigger_points is not None:
        dummy_config.trigger_points = trigger_points
      # Flag to use some special code branches for generating scenarios in scenario runner.
      dummy_config.other_parameters['generate_scenarios'] = True
      ego_data = ActorConfigurationData(ego_vehicle.type_id, ego_vehicle.get_transform(), 'hero')

      dummy_config.ego_vehicles = [ego_data]
      scenario_instance = scenario(world, [ego_vehicle], dummy_config, timeout=10000)
      scenario_name = scenario_type + f'_{scenario_counts[scenario_type]:02d}'
      scenario_options = {'trigger_point': dummy_config.trigger_points[0], **dummy_config.other_parameters}

      if not is_scenario_at_route(scenario_options['trigger_point'], proc_route):
        raise ValueError('Scenario is not at the route')

      scenarios_one_route.append((scenario_name, scenario_type, scenario_options))
      scenario_instance.terminate()
      scenario_instance.remove_all_actors()
      scenario_counts[scenario_type] += 1
      global_scenario_counts[scenario_type] += 1
      scenario_created = True

    except Exception as e:
      if scenario_instance is not None:
        scenario_instance.terminate()
        scenario_instance.remove_all_actors()
      print(f"\033[93mSkipping scenario '{dummy_config.name}' due to setup error: {e}")
      print(f"\n{traceback.format_exc()}")
      print("\033[0m", end="")

    # Impossible to create scenario trigger for this point, move on.
    if tries <= 0:
      break  # Break while loop

  return scenario_created, scenario_options

def sample_additional_parameters(args, scenario_type, proc_route, trigger_index, carla_map):
  '''
  Randomly samples a parameterization for scenario types that require that. The possible values are extracted from
   routes_training.xml and routes_validation.xml
  '''
  other_config = {}
  trigger_points = None
  if scenario_type == 'DynamicObjectCrossing':
    # Could add additional constraints to make this more realistic.
    other_config['crossing_angle'] = {
      'value': random.choices((-4, -3, -2, -1, 0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20),
                              weights=(0.01, 0.0, 0.02, 0.01, 0.09, 0.03, 0.03, 0.05, 0.01, 0.06, 0.03, 0.03, 0.02, 0.01, 0.33, 0.05, 0.02, 0.02, 0.12, 0.01, 0.02, 0.01, 0.05))[0]}
    other_config['distance'] = {
      'value': random.choices((15, 30, 35, 38, 40, 44, 45, 50, 55, 58, 60, 64, 65, 66, 68, 70, 75, 78, 90, 98, 105, 113),
                              weights=(0.006, 0.077, 0.039, 0.006, 0.168, 0.004, 0.116, 0.32, 0.034, 0.004, 0.067, 0.004, 0.039, 0.004, 0.004, 0.058, 0.026, 0.004, 0.004, 0.004, 0.004, 0.004))[0]}
    other_config['blocker_model'] = {
      'value': random.choices(('static.prop.advertisement', 'static.prop.busstoplb', 'static.prop.container', 'static.prop.foodcart', 'static.prop.haybalelb', 'static.prop.vendingmachine', 'vehicle.audi.tt', 'vehicle.mini.cooper_s', 'vehicle.nissan.patrol_2021', 'vehicle.tesla.model3'),
                              weights=(0.211, 0.135, 0.239, 0.183, 0.183, 0.004, 0.006, 0.013, 0.013, 0.013))[0]}
    # Left is very rare in Town 12 and 13
    other_config['direction'] = {'value': random.choices(('left', 'right'), weights=(0.026, 0.974))[0]}
  elif scenario_type == 'SignalizedJunctionLeftTurn':
    other_config['flow_speed'] = {'value': random.choices((7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16.6, 17, 18, 19),
                              weights=(0.013, 0.054, 0.04, 0.047, 0.04, 0.201, 0.101, 0.154, 0.148, 0.081, 0.02, 0.06, 0.02, 0.02))[0]}
    interval = random.choices(((10, 45), (10, 50), (10, 55), (10, 60), (10, 65), (10, 70), (10, 75), (10, 80), (12, 55), (14, 60), (15, 50), (15, 55), (15, 60), (15, 65), (15, 70), (15, 75), (17, 60), (18, 55), (20, 40), (20, 45), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (25, 60), (25, 70), (30, 65), (30, 80), (30, 85), (40, 75)),
                              weights=(0.013, 0.054, 0.054, 0.02, 0.02, 0.02, 0.04, 0.04, 0.013, 0.013, 0.02, 0.054, 0.067, 0.04, 0.06, 0.04, 0.027, 0.013, 0.04, 0.027, 0.04, 0.02, 0.04, 0.04, 0.02, 0.02, 0.04, 0.04, 0.02, 0.02, 0.02))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
  elif scenario_type == 'OppositeVehicleRunningRedLight':
    other_config['direction'] = {'value': random.choices(('left', 'right'), weights=(0.488, 0.512))[0]}
  elif scenario_type == 'SignalizedJunctionRightTurn':
    other_config['flow_speed'] = {'value': random.choices((7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18),
                                                        weights=(0.038, 0.013, 0.013, 0.231, 0.064, 0.167, 0.147, 0.16, 0.071, 0.077, 0.019))[0]}
    interval = random.choices(((10, 35), (10, 45), (10, 50), (10, 55), (10, 65), (10, 70), (10, 75), (15, 35), (15, 40), (15, 45), (15, 50), (15, 55), (15, 58), (15, 60), (15, 65), (15, 75), (15, 80), (16, 56), (17, 55), (17, 66), (18, 50), (19, 50), (19, 65), (20, 50), (20, 55), (20, 60), (20, 66), (20, 70), (25, 55), (25, 60), (25, 65), (25, 70), (30, 60), (30, 80), (40, 70), (40, 75)),
                              weights=(0.019, 0.032, 0.019, 0.026, 0.019, 0.019, 0.019, 0.019, 0.019, 0.038, 0.103, 0.032, 0.013, 0.122, 0.071, 0.019, 0.019, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.051, 0.038, 0.019, 0.013, 0.019, 0.038, 0.019, 0.019, 0.019, 0.019, 0.019, 0.019, 0.019))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
  elif scenario_type == 'Accident':
    other_config['distance'] = {'value': random.choices((50, 55, 60, 65, 70, 72, 75, 80, 90, 100, 110, 120, 130, 150, 160, 170, 180, 200),
                                                        weights=(0.05, 0.02, 0.01, 0.03, 0.08, 0.01, 0.03, 0.15, 0.07, 0.11, 0.04, 0.2, 0.04, 0.04, 0.04, 0.02, 0.02, 0.02))[0]}
    other_config['direction'] = {'value': random.choices(('left', 'right'), weights=(0.18, 0.82))[0]}
    other_config['speed'] = {'value': random.choices((40, 45, 47, 50, 55, 60, 70),
                                                     weights=(0.08, 0.04, 0.01, 0.09, 0.07, 0.69, 0.02))[0]}
  elif scenario_type == 'AccidentTwoWays':
    other_config['distance'] = {
      'value': random.choices((60, 65, 70, 74, 75, 76, 77, 80, 85, 88, 90, 95, 100, 110, 120, 130, 140, 150, 170),
                              weights=(0.01, 0.01, 0.2, 0.01, 0.11, 0.01, 0.02, 0.1, 0.01, 0.01, 0.13, 0.01, 0.06, 0.01, 0.21, 0.03, 0.01, 0.03, 0.01))[0]}
    other_config['direction'] = {'value': 'right'}
    other_config['speed'] = {'value': 60}
    interval = random.choices(((10, 99), (15, 85), (18, 89), (20, 88), (20, 95), (20, 98), (25, 88), (25, 89), (25, 90), (25, 98), (25, 100), (25, 110), (28, 100), (30, 135), (35, 88), (35, 90), (35, 120), (40, 83), (40, 90), (40, 95), (40, 100), (40, 110), (40, 115), (40, 120), (40, 125), (40, 180), (40, 190), (45, 90), (45, 99), (45, 105), (45, 110), (45, 115), (45, 125), (50, 105), (50, 115), (50, 130), (55, 115), (55, 120), (55, 165), (55, 190), (55, 220), (60, 130), (60, 135), (60, 140), (60, 145), (60, 180), (60, 200), (60, 220), (65, 120), (65, 145), (65, 180), (65, 200), (70, 105), (70, 150), (70, 205), (70, 220), (75, 220)),
                              weights=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.07, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.03, 0.08, 0.08, 0.01, 0.03, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01))[0]
    other_config['frequency'] = {'from': interval[0],
                                 'to': interval[1]}

  elif scenario_type == 'ConstructionObstacle':
    other_config['distance'] = {
      'value': random.choices((50, 70, 75, 78, 80, 88, 90, 100, 120, 130, 150, 200),
                              weights=(0.01, 0.18, 0.03, 0.01, 0.15, 0.01, 0.05, 0.43, 0.06, 0.02, 0.02, 0.02))[0]}
    other_config['direction'] = {'value': random.choices(('left', 'right'),
                             weights=(0.1, 0.9))[0]}
    other_config['speed'] = {'value': random.choices((40, 45, 49, 50, 55, 57, 60),
                             weights=(0.04, 0.03, 0.01, 0.25, 0.09, 0.01, 0.56))[0]}

  elif scenario_type == 'ConstructionObstacleTwoWays':
    other_config['distance'] = {
      'value': random.choices((65, 70, 75, 76, 77, 78, 80, 85, 88, 90, 100, 110, 115, 120, 140, 300),
                              weights=(0.01, 0.09, 0.07, 0.01, 0.01, 0.01, 0.25, 0.02, 0.01, 0.03, 0.3, 0.03, 0.02, 0.03, 0.05, 0.05))[0]}
    other_config['direction'] = {'value': 'right'}
    other_config['speed'] = {'value': 60}
    interval = random.choices(((10, 55), (15, 50), (15, 80), (19, 90), (20, 70), (20, 105), (25, 99), (35, 95), (35, 99), (35, 100), (35, 110), (35, 115), (35, 120), (40, 80), (40, 90), (40, 100), (40, 105), (40, 110), (40, 115), (40, 120), (40, 145), (45, 95), (45, 100), (45, 105), (45, 110), (45, 115), (45, 120), (45, 125), (45, 170), (50, 100), (50, 105), (50, 110), (50, 115), (50, 125), (50, 140), (55, 115), (55, 130), (55, 140), (55, 170), (55, 200), (60, 130), (60, 140), (60, 180), (60, 200), (65, 130), (65, 135), (65, 150), (65, 170), (65, 180), (70, 125), (70, 180), (75, 200), (75, 220)),
                              weights=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01, 0.02, 0.06, 0.05, 0.02, 0.02, 0.01, 0.01, 0.03, 0.04, 0.05, 0.01, 0.02, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02))[0]
    other_config['frequency'] = {'from': interval[0],
                                 'to': interval[1]}

  elif scenario_type == 'CrossingBicycleFlow':
    current_point = proc_route[trigger_index][0].location

    current_map_point = carla_map.get_waypoint(current_point, lane_type=carla.LaneType.Biking)
    # LaneType.Biking is rare, this can happen to project onto another road.
    if current_map_point is None or current_point.distance(current_map_point.transform.location) > 100:
      return {}  # Scenario config not possible
    other_config['start_actor_flow'] = {'x': current_map_point.transform.location.x,
                                        'y': current_map_point.transform.location.y,
                                        'z': current_map_point.transform.location.z}
    other_config['flow_speed'] = {
      'value': random.choices((5, 6, 7, 8, 9, 10),
                              weights=(0.07, 0.29, 0.14, 0.36, 0.07, 0.07))[0]}
    interval = random.choices(([5, 18], [5, 20], [5, 25], [5, 30], [10, 25], [10, 30], [10, 35], [10, 55], [10, 3255], [15, 25], [30, 60]),
                              weights=(0.07, 0.14, 0.14, 0.07, 0.07, 0.14, 0.07, 0.07, 0.07, 0.07, 0.07))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
  elif scenario_type == 'HazardAtSideLane':
    other_config['distance'] = {
      'value': random.choices((50, 55, 60, 70, 75, 80, 90),
                              weights=(0.13, 0.04, 0.13, 0.31, 0.04, 0.27, 0.07))[0]}
    other_config['speed'] = {'value': random.choices((30, 40, 50, 60),
                              weights=(0.07, 0.13, 0.13, 0.67))[0]}
    other_config['bicycle_speed'] = {'value': random.choices((8, 9, 10, 11, 12, 13, 15),
                              weights=(0.33, 0.04, 0.04, 0.04, 0.27, 0.13, 0.13))[0]}
    other_config['bicycle_drive_distance'] = {'value': random.choices((60, 70, 80, 90, 100, 150, 200),
                              weights=(0.13, 0.27, 0.16, 0.04, 0.27, 0.07, 0.07))[0]}
  elif scenario_type == 'HazardAtSideLaneTwoWays':
    other_config['distance'] = {
      'value': random.choices((40, 50, 55, 60, 65, 70, 75, 80, 88, 90, 95, 100, 110, 120, 135),
                              weights=(0.03, 0.06, 0.01, 0.12, 0.01, 0.26, 0.09, 0.14, 0.01, 0.08, 0.01, 0.13, 0.01, 0.01, 0.01))[0]}
    other_config['speed'] = {'value': 60}
    other_config['bicycle_speed'] = {'value': random.choices((5, 6, 7, 8, 9, 10, 11, 12, 13, 15),
                              weights=(0.01, 0.04, 0.12, 0.16, 0.16, 0.28, 0.11, 0.07, 0.03, 0.01))[0]}
    other_config['bicycle_drive_distance'] = {'value': random.choices((40, 50, 60, 65, 70, 80, 89, 90, 95, 100, 120, 150, 200, 300),
                              weights=(0.03, 0.1, 0.18, 0.01, 0.08, 0.06, 0.01, 0.11, 0.02, 0.31, 0.02, 0.03, 0.04, 0.01))[0]}
    other_config['frequency'] = {'value': random.choices((40, 75, 77, 78, 80, 82, 85, 88, 90, 95, 100, 110, 120, 125, 130, 140),
                              weights=(0.01, 0.15, 0.01, 0.01, 0.22, 0.01, 0.12, 0.01, 0.21, 0.07, 0.08, 0.01, 0.03, 0.04, 0.01, 0.01))[0]}
  elif scenario_type == 'InvadingTurn':
    other_config['distance'] = {
      'value': random.choices((60, 70, 75, 80, 90, 95, 100, 110, 115, 120, 125, 130, 140, 150, 160, 170, 180, 200, 230),
                              weights=(0.01, 0.04, 0.01, 0.01, 0.08, 0.02, 0.31, 0.14, 0.02, 0.09, 0.01, 0.07, 0.01, 0.06, 0.03, 0.02, 0.03, 0.01, 0.01))[0]}
    other_config['offset'] = {'value': random.choices((0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6),
                              weights=(0.13, 0.2, 0.12, 0.18, 0.21, 0.09, 0.08))[0]}
  elif scenario_type == 'NonSignalizedJunctionLeftTurn':
    other_config['flow_speed'] = {
      'value': random.choices((6, 7, 8, 8.3, 9, 10, 11, 12, 13, 15, 16.6, 17, 20),
                              weights=(0.01, 0.03, 0.13, 0.04, 0.15, 0.29, 0.03, 0.07, 0.14, 0.05, 0.02, 0.02, 0.02))[0]}
    interval = random.choices(((10, 40), (10, 50), (10, 55), (10, 60), (10, 70), (10, 78), (10, 85), (10, 88), (12, 50), (12, 62), (12, 73), (12, 79), (13, 45), (13, 48), (13, 50), (15, 45), (15, 50), (15, 55), (15, 60), (15, 66), (15, 69), (15, 78), (16, 54), (16, 80), (17, 55), (17, 62), (18, 45), (18, 53), (18, 55), (18, 65), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (25, 40), (25, 55), (25, 60), (30, 50), (30, 65), (30, 80), (40, 70), (80, 110)),
                              weights=(0.03, 0.08, 0.01, 0.02, 0.02, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.08, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01, 0.01, 0.06, 0.01, 0.02, 0.02, 0.04, 0.02, 0.01, 0.04, 0.02, 0.02, 0.02, 0.04, 0.02))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
  elif scenario_type == 'NonSignalizedJunctionRightTurn':
    other_config['flow_speed'] = {
      'value': random.choices((5, 6, 7, 8, 8.3, 9, 10, 10.6, 11, 12, 13, 14, 15, 16, 21),
                              weights=(0.01, 0.02, 0.04, 0.13, 0.02, 0.11, 0.13, 0.01, 0.05, 0.09, 0.09, 0.09, 0.13, 0.04, 0.01))[
        0]}
    interval = random.choices(((10, 30), (10, 45), (10, 47), (10, 48), (10, 50), (10, 55), (10, 58), (10, 60), (10, 65), (10, 70), (12, 48), (13, 50), (13, 55), (13, 60), (13, 80), (14, 48), (14, 59), (15, 30), (15, 40), (15, 50), (15, 53), (15, 55), (15, 60), (15, 62), (15, 65), (15, 70), (17, 55), (17, 60), (18, 50), (18, 55), (18, 56), (18, 58), (18, 65), (18, 66), (20, 45), (20, 50), (20, 55), (20, 57), (20, 60), (20, 65), (20, 66), (20, 70), (20, 80), (20, 110), (25, 40), (25, 55), (25, 65), (30, 60), (30, 70), (35, 60), (40, 70), (50, 65)),
                              weights=(0.02, 0.04, 0.01, 0.02, 0.05, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.04, 0.01, 0.07, 0.03, 0.01, 0.04, 0.02, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.02, 0.06, 0.04, 0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
  elif scenario_type == 'OppositeVehicleTakingPriority':
    other_config['direction'] = {
      'value': random.choices(('left', 'right'),
                              weights=(0.42, 0.58))[0]}
  elif scenario_type == 'ParkedObstacle':
    other_config['distance'] = {
      'value': random.choices((50, 55, 60, 66, 70, 80, 90, 100, 110, 120, 160),
                              weights=(0.07, 0.02, 0.02, 0.02, 0.21, 0.13, 0.03, 0.03, 0.13, 0.3, 0.03))[0]}
    other_config['direction'] = {
      'value': random.choices(('left', 'right'),
                              weights=(0.16, 0.84))[0]}
    other_config['speed'] = {
      'value': random.choices((40, 45, 49, 50, 55, 60, 70),
                              weights=(0.07, 0.02, 0.02, 0.07, 0.03, 0.76, 0.03))[0]}
  elif scenario_type == 'ParkedObstacleTwoWays':
    other_config['distance'] = {
      'value': random.choices((50, 55, 60, 65, 66, 70, 74, 75, 78, 80, 85, 90, 100, 110, 120, 130),
                              weights=(0.01, 0.01, 0.02, 0.01, 0.01, 0.14, 0.01, 0.07, 0.01, 0.17, 0.04, 0.07, 0.1, 0.04, 0.29, 0.01))[0]}
    other_config['direction'] = {'value': 'right'}
    other_config['speed'] = {'value': 60}
    interval = random.choices(((10, 110), (15, 85), (15, 95), (15, 100), (18, 90), (18, 99), (19, 60), (25, 75), (25, 90), (25, 95), (25, 110), (30, 85), (30, 90), (30, 105), (35, 85), (35, 90), (35, 95), (35, 100), (35, 105), (35, 110), (35, 115), (38, 58), (40, 80), (40, 85), (40, 90), (40, 95), (40, 100), (40, 105), (40, 110), (45, 85), (45, 90), (45, 95), (45, 105), (45, 110), (45, 115), (45, 120), (50, 85), (50, 95), (50, 100), (50, 105), (50, 110), (50, 115), (50, 125), (50, 150), (55, 90), (55, 130), (55, 140), (55, 185), (60, 140), (60, 160), (60, 190), (65, 120), (65, 145), (65, 150), (65, 195), (70, 110), (70, 140), (70, 180), (70, 200), (75, 145)),
                              weights=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.05, 0.05, 0.02, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01))[0]
    other_config['frequency'] = {'from': interval[0],
                                 'to': interval[1]}

  elif scenario_type == 'ParkingCrossingPedestrian':
    other_config['distance'] = {
      'value': random.choices((12, 30, 35, 40, 45, 50, 52, 55, 60, 65, 70, 73, 75, 78, 80),
                              weights=(0.21, 0.11, 0.06, 0.16, 0.06, 0.18, 0.01, 0.02, 0.05, 0.01, 0.05, 0.01, 0.04, 0.02, 0.01))[0]}
    other_config['crossing_angle'] = {
      'value': random.choices((-2, 0, 1, 2, 3, 4, 5, 8, 10, 14, 15),
                              weights=(0.01, 0.65, 0.02, 0.03, 0.07, 0.01, 0.04, 0.01, 0.1, 0.01, 0.03))[0]}
    other_config['direction'] = {
      'value': random.choices(('left', 'right'),
                              weights=(0.18, 0.82))[0]}

  elif scenario_type == 'ParkingCutIn':
    other_config['direction'] = {
      'value': random.choices(('left', 'right'),
                              weights=(0.16, 0.84))[0]}

  elif scenario_type == 'ParkingExit':
    other_config['front_vehicle_distance'] = {'value': 9}
    other_config['behind_vehicle_distance'] = {'value': 9}
    other_config['direction'] = {'value': random.choices(('left', 'right'), weights=(0.06, 0.94))[0]}
    other_config['flow_distance'] = {'value': 25}
    other_config['speed'] = {'value': 60}

  elif scenario_type == 'StaticCutIn':
    other_config['distance'] = {
      'value': random.choices((80, 90, 100, 102, 105, 109, 110, 112, 113, 120, 130, 150, 170),
                              weights=(0.04, 0.2, 0.39, 0.01, 0.06, 0.01, 0.06, 0.01, 0.01, 0.14, 0.02, 0.02, 0.02))[0]}
    other_config['direction'] = {'value': random.choices(('left', 'right'), weights=(0.11, 0.89))[0]}

  elif scenario_type == 'VehicleOpensDoorTwoWays':
    other_config['distance'] = {
      'value': random.choices((35, 40, 45, 50, 55, 60, 65, 67, 70, 75, 77, 80),
                              weights=(0.04, 0.09, 0.02, 0.15, 0.01, 0.24, 0.07, 0.02, 0.15, 0.07, 0.01, 0.1))[0]}
    other_config['direction'] = {'value': 'right'}
    other_config['speed'] = {'value': 60}
    interval = random.choices(((15, 85), (20, 75), (20, 100), (25, 65), (25, 85), (25, 90), (25, 95), (30, 85), (30, 90), (30, 95), (30, 100), (35, 65), (35, 80), (35, 90), (35, 95), (35, 100), (35, 110), (40, 70), (40, 80), (40, 90), (40, 95), (40, 100), (40, 105), (40, 110), (40, 115), (45, 65), (45, 80), (45, 90), (45, 100)),
                              weights=(0.01, 0.01, 0.04, 0.01, 0.07, 0.08, 0.04, 0.02, 0.04, 0.02, 0.02, 0.01, 0.02, 0.15, 0.07, 0.07, 0.01, 0.01, 0.01, 0.02, 0.07, 0.04, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01))[0]
    other_config['frequency'] = {'from': interval[0],
                                 'to': interval[1]}
  elif scenario_type == 'YieldToEmergencyVehicle':
    other_config['distance'] = {
      'value': random.choices((110, 115, 120, 125, 130, 135, 140, 150),
                              weights=(0.03, 0.03, 0.13, 0.16, 0.3, 0.16, 0.15, 0.04))[0]}
  elif scenario_type == 'InterurbanActorFlow':
    other_config['flow_speed'] = {
      'value': random.choices((15, 16.6, 17, 18, 20),
                              weights=(0.14, 0.14, 0.09, 0.23, 0.41))[0]}
    interval = random.choices(((25, 80), (25, 95), (30, 100), (35, 75), (35, 85), (40, 120), (50, 100), (65, 90)),
                              weights=(0.14, 0.09, 0.09, 0.14, 0.14, 0.14, 0.14, 0.14))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
    left_turn_location = None
    for index in range(trigger_index, min(trigger_index+int(args.scenario_dilation), len(proc_route))):
      if proc_route[index][1] == RoadOption.LEFT:
        local_index = index
        while local_index > 0:
          local_index -= 1
          # Find the point before the intersection
          if proc_route[local_index][1] != RoadOption.LEFT:
            left_turn_location = proc_route[local_index][0]
            break
        break

    if left_turn_location is None:
      raise ValueError('Did not find a left turn for scenario InterurbanActorFlow')

    current_map_point = carla_map.get_waypoint(left_turn_location.location)

    left_lane = current_map_point
    last_id = left_lane.lane_id
    tries = 0
    # Finds the lane to the left of the ego for the opposing traffic.
    while True:
      left_lane = left_lane.get_left_lane()
      if left_lane is None:
        break
      if not same_sign(left_lane.lane_id, last_id):
        break

      tries += 1
      last_id = left_lane.lane_id

      if tries > 10:
        left_lane = None
        break

    if left_lane is None:
      raise ValueError('Did not find a left lane for scenario InterurbanActorFlow')

    end_actor_flow = left_lane
    last_wp = end_actor_flow

    for i in range(3):
      end_actor_flow = end_actor_flow.next(10)
      if len(end_actor_flow) > 0:
        end_actor_flow = get_min_anlge_wp_from_list(last_wp, end_actor_flow, carla_map) # TODO we would ideally find the middle lane
      else:
        raise ValueError('Did not find an end position for scenario InterurbanActorFlow')
      last_wp = end_actor_flow

    start_actor_flow = left_lane
    last_wp = start_actor_flow
    for i in range(5):
      start_actor_flow = start_actor_flow.previous(10)
      if len(start_actor_flow) > 0:
        start_actor_flow = get_min_anlge_wp_from_list(last_wp, start_actor_flow, carla_map)
      else:
        raise ValueError('Did not find a start position for scenario InterurbanActorFlow')
      last_wp = start_actor_flow

    other_config['start_actor_flow'] = {'x': start_actor_flow.transform.location.x,
                                        'y': start_actor_flow.transform.location.y,
                                        'z': start_actor_flow.transform.location.z}
    other_config['end_actor_flow'] = {'x': end_actor_flow.transform.location.x,
                                      'y': end_actor_flow.transform.location.y,
                                      'z': end_actor_flow.transform.location.z}
    print(current_map_point)
    # Overwrite trigger point with one that is closer to the intersection
    trigger_points = [proc_route[max(0, local_index-10)][0]]
  elif scenario_type == 'InterurbanAdvancedActorFlow':
    other_config['flow_speed'] = {
      'value': random.choices((15, 16, 17, 18),
                              weights=(0.38, 0.08, 0.12, 0.42))[0]}
    interval = random.choices(((17, 65), (20, 75), (23, 88), (25, 75), (25, 100), (30, 85), (30, 105), (40, 80), (50, 80)),
                              weights=(0.08, 0.12, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12))[0]
    other_config['source_dist_interval'] = {'from': interval[0],
                                            'to': interval[1]}
    left_turn_location = None
    for index in range(trigger_index, min(trigger_index+int(args.scenario_dilation), len(proc_route))):
      if proc_route[index][1] == RoadOption.LEFT:
        local_index = index
        while local_index > 0:
          local_index -= 1
          # Find the point before the intersection
          if proc_route[local_index][1] != RoadOption.LEFT:
            left_turn_location = proc_route[local_index][0]
            break

        index_after_intersection = index
        while index_after_intersection < len(proc_route):
          index_after_intersection += 1
          if proc_route[index_after_intersection][1] != RoadOption.LEFT:
            sink_transform = proc_route[min(index_after_intersection+25, len(proc_route)-1)][0]
            break
        break

    if left_turn_location is None:
      raise ValueError('Did not find a left turn for scenario InterurbanAdvancedActorFlow')

    current_map_point = carla_map.get_waypoint(sink_transform.location)

    start_actor_flow = current_map_point
    last_wp = start_actor_flow
    for i in range(8):
      start_actor_flow = start_actor_flow.previous(10)
      if len(start_actor_flow) > 0:
        start_actor_flow = get_min_anlge_wp_from_list(last_wp, start_actor_flow, carla_map)
      else:
        raise ValueError('Did not find a start position for scenario InterurbanActorFlow')
      last_wp = start_actor_flow

    other_config['start_actor_flow'] = {'x': start_actor_flow.transform.location.x,
                                        'y': start_actor_flow.transform.location.y,
                                        'z': start_actor_flow.transform.location.z}
    other_config['end_actor_flow'] = {'x': sink_transform.location.x,
                                      'y': sink_transform.location.y,
                                      'z': sink_transform.location.z}
    print(current_map_point)
    # Overwrite trigger point with one that is closer to the intersection
    trigger_points = [proc_route[max(0, local_index-10)][0]]

  return other_config, trigger_points

def load_scenarios_from_file(scenario_types, map_name):
  extracted_scenarios = []
  for scenario_type in scenario_types:
    tree = ET.parse(rf'scenarios/{scenario_type}_{map_name}.xml')
    scenarios = tree.getroot()

    for scenario in scenarios:
      if scenario.attrib['type'] == scenario_type:
        extracted_scenario = {'type': scenario_type}
        for parameter in scenario:
          if parameter.tag == 'trigger_point':
            trigger_loc = carla.Location(x=float(parameter.attrib['x']), y=float(parameter.attrib['y']), z=float(parameter.attrib['z']))
            trigger_rot = carla.Rotation(yaw=float(parameter.attrib['yaw']))
            trigger_transf = carla.Transform(trigger_loc, trigger_rot)
            extracted_scenario[parameter.tag] = trigger_transf
          else:
            extracted_scenario[parameter.tag] = parameter.attrib
        extracted_scenarios.append(extracted_scenario)

  return extracted_scenarios

if __name__ == '__main__':
  main()

