#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import time
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import os
import sys

import signal
import socket

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid, SensorInterface
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, NextRoute
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer

import pathlib


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer',
    'sensor.camera.semantic_segmentation': 'carla_camera', # for datagen
    'sensor.camera.depth':      'carla_camera', # for datagen
}

def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')

class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    # Tunable parameters
    client_timeout = 10.0  # in seconds

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None
        self.ego_vehicle = None

        self.statistics_manager = statistics_manager

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None

        print('Wait for CARLA server')
        # Wait until CARLA server has started.
        waited_for = 0.0  # How many seconds we waited for.
        while True:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = s.connect_ex((args.host, args.port))

            if result == 0:
                break
            s.close()

            waited_for += 1.0
            if waited_for > args.timeout:
                raise ValueError(f'CARLA server port {args.port} did not start in time.')

            time.sleep(1.0)

        # Setup the simulation
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(args)
        print("Client version:", self.client.get_client_version())
        # NOTE: If I work with a debug compilation of CARLA and load the world too fast the client causes
        #  a segfault here. Hacked in a sleep 3 which works around the issue. Remeber to comment when running release
        # time.sleep(3)
        # Load Town 01 to decrease RAM usage (uses less than default Town10HD) before we load the routes into the RAM.
        world = self.client.load_world('Town01', reset_settings=False)  # 1.96 seconds
        world.tick()

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, self.statistics_manager, args.runtime_timeout, args.debug)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        self._agent_watchdog = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self._client_timed_out = False

        self.init_world = False
        self.init_agent = False
        self.current_town = ''

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        print('Signal exit')
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            # Sometimes these Runtime errors can get stuck in catch blocks.
            # Shutdown completely so that the restart script can restart the client.
            os._exit(6)
            raise RuntimeError("Timeout: Agent took longer than {}s to setup".format(self.client_timeout))
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'manager') and self.manager:
            del self.manager

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self, results=None):
        """
        Remove and destroy all actors
        """
        CarlaDataProvider.cleanup_route()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            if self.agent_instance:
                self.agent_instance.destroy(results) # Only indicated to the agent that the route has finished now.

        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        if self.route_scenario:
            self.route_scenario.remove_all_actors()
            del self.route_scenario
            self.route_scenario = None
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()
        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup()

        # from pytictoc import TicToc
        # t = TicToc()
        # t.tic()
        self.world.tick()
        # t.toc(msg='3 CARLA step time: ')

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        client = carla.Client(args.host, args.port, worker_threads=1)

        if args.timeout:
            client_timeout = args.timeout
        client.set_timeout(client_timeout)

        settings = carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / args.frame_rate,
            deterministic_ragdolls = True,
            no_rendering_mode = bool(args.no_rendering_mode),
            spectator_as_ego = False,
        )
        client.get_world().apply_settings(settings)

        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        return client, client_timeout, traffic_manager

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """
        CarlaDataProvider.cleanup()
        self.world = self.client.load_world(town, reset_settings=False) # 1.96 seconds

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        # 0.187191
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        self.world.tick()  # 5ms

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

    def _reload_world(self, args):
        self.world.reset_all_traffic_lights()
        self.world.tick()  # 5ms

    def _register_statistics(self, route_date_string, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        #print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_entry_status(entry_status)
        current_stats_record = self.statistics_manager.compute_route_statistics(
            route_date_string, route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
        )

        return current_stats_record


    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"
        from pytictoc import TicToc


        # print("[========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        self._agent_watchdog = Watchdog(args.timeout - 20)
        self._agent_watchdog.start()
        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        self.statistics_manager.create_route_data(route_name, config.index)
        # print("00005", flush=True)
        # Load the world and the scenario
        try:
            # print(f"Current route file: {args.routes}")
            # Only reload the world when the town changed.
            if not self.init_world or self.current_town != config.town:
                self._load_and_wait_for_world(args, config.town)

                # During the first 1800 ticks after loading a town CARLA will free GPU memory and RAM.
                # If many servers are needed on one GPU this can be used
                # Only needed if CARLA is started with sensor rendering,
                if args.debug > 0 or args.no_rendering_mode:
                    num_ticks = 1
                else:
                    num_ticks = 1800
                for _ in range(num_ticks):
                    self.world.tick()  # 5ms

                self.init_world = True
                self.current_town = config.town
            else:
                self._reload_world(args)

            # TODO slow 150 ms
            self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug, criteria_enable=True, ego_vehicle=self.ego_vehicle)
            self.ego_vehicle = self.route_scenario.ego_vehicle
            self.statistics_manager.set_scenario(self.route_scenario, config.route_length)

            if not self.init_world:
                self._agent_wrapper.setup_sensors(self.ego_vehicle)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print(f"\n{traceback.format_exc()}\033[0m")
            print(f"Faulty route file: {args.routes}")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics('', config.index, entry_status, crash_message)
            self._cleanup()
            # Go to the next route if the problem was just that the ego could not be spawned.
            return e.args[0] != "Shutting down, couldn't spawn the ego vehicle"

        self._agent_watchdog.stop()
        self._agent_watchdog = None
        # print("00006", flush=True)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            now = datetime.now()
            # route_string = pathlib.Path(os.environ.get('ROUTES', '')).stem + '_'
            route_string = pathlib.Path(args.routes).stem + '_'
            route_string += f'route{config.index}'
            route_date_string = route_string + '_' + '_'.join(
                map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second))
            )
            if not self.init_agent:
                self._agent_watchdog = Watchdog(args.timeout)
                self._agent_watchdog.start()
                agent_class_name = getattr(self.module_agent, 'get_entry_point')()
                agent_class_obj = getattr(self.module_agent, agent_class_name)

                # Start the ROS1 bridge server only for ROS1 based agents.
                if getattr(agent_class_obj, 'get_ros_version')() == 1 and self._ros1_server is None:
                    from leaderboard.autoagents.ros1_agent import ROS1Server
                    self._ros1_server = ROS1Server()
                    self._ros1_server.start()

                self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
                self.agent_instance.setup(args.agent_config, args.gym_port, config)

                # Check and store the sensors
                if not self.sensors:
                    self.sensors = self.agent_instance.sensors()
                    track = self.agent_instance.track

                    validate_sensor_configuration(self.sensors, track, args.track)

                    self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                    self.statistics_manager.save_sensors(self.sensor_icons)
                    self.statistics_manager.write_statistics() # 3ms

                    self.sensors_initialized = True

                # Set after agent_instance.sensors() such that the agent config is already initialized
                self.agent_instance.set_global_plan(self.route_scenario.route)
                self.manager.register_agent(self.agent_instance, self.ego_vehicle)

                self._agent_watchdog.stop()
                self._agent_watchdog = None
                self.init_agent = True
            else:
                # Note I delete the old sensor interface when cleaning up the sensors. Need to make a new one.
                self._agent_watchdog = Watchdog(args.timeout)
                self._agent_watchdog.start()

                self.agent_instance.set_global_plan(self.route_scenario.route)
                self.agent_instance.setup(args.agent_config, args.gym_port, config)

                self._agent_watchdog.stop()
                self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            self._cleanup(result)
            return True

        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            self._cleanup(result)
            return False

        # print("00007", flush=True)

        # Run the scenario
        try:
            self._agent_watchdog = Watchdog(args.timeout - 20)
            self._agent_watchdog.start()
            # Load scenario and run it
            if args.record:
                # self.client.start_recorder("{}.log".format(args.record), True)
                self.client.start_recorder("{}.log".format(args.record), False) # changed to False, otherwise the log file become too large
            # 30-120 ms if sensors are used
            self.manager.load_scenario(self.route_scenario, config.index, config.repetition_index)

            self._agent_watchdog.stop()
            self._agent_watchdog = None
            # 5-10 ms
            self.manager.run_scenario()

        except NextRoute:
            # The agent requested the next route
            print("\n\033[91m Agent requested next route")
            entry_status, crash_message = "Started", "Next Route"
        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
        except KeyError:
            print("\n\033[91mError in Scenario Runner:")
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)
            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
        except RuntimeError:
            print("\n\033[91mError in Scenario Runner:")
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)
            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]
        except Exception:
            print("\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)
            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]

        # print("00008", flush=True)

        # Stop the scenario
        self._agent_watchdog = Watchdog(args.timeout - 20)
        self._agent_watchdog.start()
        try:
            t = TicToc()
            #print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            result = self._register_statistics(route_date_string, config.index, entry_status, crash_message)
            if args.record:
                self.client.stop_recorder()
            self._cleanup(result)

            if crash_message == "Simulation crashed":
                if self.agent_instance:
                    del self.agent_instance
                self.init_agent = False
                CarlaDataProvider.cleanup()

        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print(f"\n{traceback.format_exc()}\033[0m")

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # print("00009", flush=True)

        self._agent_watchdog.stop()
        self._agent_watchdog = None
        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
        """
        resume_from_file = args.resume
        while True:
            route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

            if resume_from_file:
                resume = route_indexer.validate_and_resume(args.checkpoint)
            else:
                resume = False


            if resume:
                self.statistics_manager.add_file_records(args.checkpoint)
            else:
                self.statistics_manager.clear_records()


            # The last run was a forced exit. Log the route as crashed and skip it.
            if args.skip_next_route and route_indexer.peek():
                print('Skip next three routes')
                for _ in range(3):
                    config = route_indexer.get_next_config()
                    entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
                    route_name = f"{config.name}_rep{config.repetition_index}"
                    self.statistics_manager.create_route_data(route_name, config.index)
                    result = self._register_statistics('', config.index, entry_status, crash_message)


            self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            self.statistics_manager.write_statistics()

            crashed = False
            while route_indexer.peek() and not crashed:

                # Run the scenario
                config = route_indexer.get_next_config()
                crashed = self._load_and_run_scenario(args, config)

                # Save the progress and write the route statistics
                self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
                self.statistics_manager.write_statistics()

            # Shutdown ROS1 bridge server if necessary
            if self._ros1_server is not None:
                self._ros1_server.shutdown()

            print(f'NOTE: Leaderboard with port: {args.port} ran out of routes to train with. Reloading routes.', file=sys.stderr, flush=True)

            del route_indexer
            resume_from_file = False

            if crashed:
                break

        return crashed

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=0, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int,
                        help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=300.0, type=float,
                        help='Timeout for setting everything up')
    parser.add_argument('--runtime_timeout', default=60.0, type=float,
                        help='Timeout per step. Lower than startup timeout.')

    # simulation setup
    parser.add_argument('--routes', required=True,
                        help='Name of the routes file to be executed.')
    parser.add_argument('--routes-subset', default='', type=str,
                        help='Execute a specific set of routes')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str,
                        help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    parser.add_argument('--gym_port', type=int, required=True, help='Port used for gym communication.')
    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")
    parser.add_argument("--frame_rate", type=float, default=20.0,
                        help="Frame_rate of the simulator")
    parser.add_argument('--no_rendering_mode',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='if toggled, carla will not render anything including sensors.'
                             'Unfortunately seems to not improve render speed in synchronous mode.')
    parser.add_argument('--skip_next_route',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='Skip the next route while continuing. Useful if that route crashed.')

    arguments = parser.parse_args()
    CarlaDataProvider.set_random_seed(arguments.traffic_manager_seed)

    pathlib.Path(arguments.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    except ValueError as e:
        print(e)
        sys.exit(-1)

    crashed = leaderboard_evaluator.run(arguments)
    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
