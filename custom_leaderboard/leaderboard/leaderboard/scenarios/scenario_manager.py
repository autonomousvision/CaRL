#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time
import os

import py_trees
import carla
from greenlet import greenlet

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapperFactory, AgentError, NextRoute
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

from pytictoc import TicToc


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, statistics_manager, runtime_timeout, debug_mode=0):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.route_index = None
        self.scenario = None
        self.scenario_tree = None
        self.ego_vehicles = None
        self.other_actors = None
        self._watchdog = None
        self._scenario_greenlet = None
        self.tick_count = 0

        self._debug_mode = debug_mode
        self._agent_wrapper = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)
        self._runtime_timeout = float(runtime_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        # Detects if the simulation or agent is down
        self.initialized = False

        self._statistics_manager = statistics_manager

        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        # Sometimes these Runtime errors can get stuck in catch blocks.
        # Shutdown completely so that the restart script can restart the client.
        print('Signal exit')
        os._exit(5)
        if self._watchdog and not self._watchdog.get_status():
            raise RuntimeError("The simulation or agent took longer than {}s to update".format(self._timeout))
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self._spectator = None


    def register_agent(self, agent, ego_vehicle):
        self._agent_wrapper = AgentWrapperFactory.get_wrapper(agent)
        self._agent_wrapper.setup_sensors(ego_vehicle)

    def load_scenario(self, scenario, route_index, rep_number):
        """
        Load a new scenario
        """
        GameTime.restart()
        self.route_index = route_index
        self.scenario = scenario
        self.scenario_tree = scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        self._spectator = CarlaDataProvider.get_world().get_spectator()

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)
        # Move sensor setup outside of scenarios to not redo it per route.
        # self._agent_wrapper.setup_sensors(self.ego_vehicles[0])

    def build_scenarios_loop(self, debug):
        """
        Keep periodically trying to start the scenarios that are close to the ego vehicle
        Additionally, do the same for the spawned vehicles
        """
        while self._running:
            self.scenario.build_scenarios(self.ego_vehicles[0], debug=debug)
            self.scenario.spawn_parked_vehicles(self.ego_vehicles[0])
            self.gr_main.switch()

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._running = True
        # stackful coroutine for build_scenarios
        self.gr_main = greenlet.getcurrent()
        self._scenario_greenlet = greenlet(self.build_scenarios_loop)
        while self._running:
            self._tick_scenario()

    def _tick_scenario(self):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        self._watchdog = Watchdog(self._timeout)
        self._watchdog.start()

        self._scenario_greenlet.switch(self._debug_mode > 0)
        # t = TicToc()
        # t.tic()
        # print("Before CarlaDataProvider.get_world().tick():", flush=True)
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick()  # 3.5ms
        self.tick_count += 1
        # t.toc(msg=f'0 CARLA step time {self.tick_count:06d}: ')
        # NOTE We draw debug per timestep because we don't reload the world in between episodes
        if self._debug_mode > 0:
            self._draw_waypoints(self.scenario.route, vertical_shift=1.0, size=0.1, downsample=10)
            self._draw_scenario_triggers()

        # print("70001", flush=True)
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            # print("70002", flush=True)

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            # print("70003", flush=True)

            CarlaDataProvider.on_carla_tick()
            # print("70004", flush=True)

            try:
                ego_action = self._agent_wrapper() # 7ms
                # print("70005", flush=True)

            except NextRoute as e:
                self._watchdog.stop()
                self._watchdog = None
                raise NextRoute(e)

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                self._watchdog.stop()
                self._watchdog = None
                raise RuntimeError(e)

            except Exception as e:
                self._watchdog.stop()
                self._watchdog = None
                raise AgentError(e)

            # print("10000", flush=True)
            # After being initialized (agent ticked for one step, set timeout to runtime.
            # Second condition is to check that the training has actually started.
            if not self.initialized and not (ego_action.steer==0.0 and ego_action.throttle==0.0 and ego_action.brake==1.0):
                self._timeout = self._runtime_timeout
                self.initialized = True

            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario. Add the ego control to the blackboard in case some behaviors want to change it
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            # print("10001", flush=True)

            self.scenario_tree.tick_once() # 1ms
            # print("10002", flush=True)

            if self._debug_mode > 1:
                self.compute_duration_time()

                # Update live statistics
                self._statistics_manager.compute_route_statistics(
                    "",
                    self.route_index,
                    self.scenario_duration_system,
                    self.scenario_duration_game,
                    failure_message=""
                )
                self._statistics_manager.write_live_results(
                    self.route_index,
                    self.ego_vehicles[0].get_velocity().length(),
                    ego_action,
                    self.ego_vehicles[0].get_location()
                )

            if self._debug_mode > 2:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            ego_trans = self.ego_vehicles[0].get_transform()

            # self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=70),
                                                        #   carla.Rotation(pitch=-90)))
            # NOTE Spectator use -90 for BEV spectator
            # For third-person view
            location = ego_trans.transform(carla.Location(x=-4.5, z=2.3))
            self._spectator.set_transform(carla.Transform(location, carla.Rotation(pitch=-15.0, yaw=ego_trans.rotation.yaw)))
            # self._spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
            #                                               carla.Rotation(pitch=-90)))

        self._watchdog.stop()
        self._watchdog = None
        # print("10003", flush=True)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._watchdog:
            return self._watchdog.get_status()
        return True

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

        self.compute_duration_time()

        # Set running to false. The coroutine will now exit at the next iteration.
        self._running = False
        # The coroutine might be busy setting up a scenario. To ensure a clean exit we will let it finish setting up
        # by providing the needed CARLA ticks.
        if self._scenario_greenlet is not None:
            self._scenario_greenlet.switch(self._debug_mode > 0)

            while not self._scenario_greenlet.dead:
                CarlaDataProvider.get_world().tick()
                self._scenario_greenlet.switch(self._debug_mode > 0)

            self._scenario_greenlet = None

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent_wrapper is not None:
                self._agent_wrapper.cleanup() # 10 ms # NOTE does nothing anymore, since we keep the sensors

            self.analyze_scenario()

        del self.scenario
        del self.scenario_tree

        self.scenario = None
        self.scenario_tree = None

    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def _draw_scenario_triggers(self):
        render_lifetime = self.scenario.world.get_settings().fixed_delta_seconds + 0.01
        for scenario_config in self.scenario.scenario_configurations:
            scenario_loc = scenario_config.trigger_points[0].location
            debug_loc = self.scenario.map.get_waypoint(scenario_loc).transform.location + carla.Location(z=0.2)
            self.scenario.world.debug.draw_point(
                debug_loc, size=0.2, color=carla.Color(128, 0, 0), life_time=render_lifetime
            )
            self.scenario.world.debug.draw_string(
                debug_loc, str(scenario_config.name), draw_shadow=False,
                color=carla.Color(0, 0, 128), life_time=render_lifetime
            )

    # pylint: disable=no-self-use
    def _draw_waypoints(self, waypoints, vertical_shift, size, downsample=1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        render_lifetime = self.scenario.world.get_settings().fixed_delta_seconds + 0.01

        for i, w in enumerate(waypoints):
            if i % downsample != 0:
                continue

            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            self.scenario.world.debug.draw_point(wp, size=size, color=color, life_time=render_lifetime)

        self.scenario.world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(0, 0, 128), life_time=render_lifetime)
        self.scenario.world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(128, 128, 128), life_time=render_lifetime)
    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        ResultOutputProvider(self)
