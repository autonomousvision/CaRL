import xml.etree.ElementTree as ET
from collections import Counter
import ast
import numpy as np

IN_PATH = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_training.xml'
IN_PATH2 = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation.xml'


# scenario_type = 'DynamicObjectCrossing'
# options = {'crossing_angle': [],
#            'distance': [],
#            'blocker_model': [],
#            'direction': []}
#
# default_option = {'crossing_angle': 0.0,
#                   'distance': 12,
#                   'blocker_model': 'static.prop.vendingmachine',
#                   'direction': 'right'
#                   }

# scenario_type = 'SignalizedJunctionLeftTurn'
# options = {'flow_speed': [],
#            'source_dist_interval': [],
# }
# default_option = {'flow_speed': 20,
#                   'source_dist_interval': (25, 50)
#                   }

# scenario_type = 'OppositeVehicleRunningRedLight'
# options = {'direction': [] }
# default_option = {'direction': 'right'}

scenario_type = 'SignalizedJunctionRightTurn'
options = {'flow_speed': [],
           'source_dist_interval': [],
}

default_option = {'flow_speed': 20, 'source_dist_interval': (25, 50)}

# scenario_type = 'Accident'
# options = {'distance': [],
#            'direction': [],
#            'speed': []
# }
# default_option = {'distance': 120,
#            'direction': 'right',
#            'speed': 60
# }

# scenario_type = 'AccidentTwoWays'
# options = {'distance': [],
#            'direction': [],
#            'speed': [],
#            'frequency': [],
# }
# default_option = {'distance': 120,
#            'direction': 'right',
#            'speed': 60,
#            'frequency': (20, 100),
# }


# scenario_type = 'ConstructionObstacle'
# options = {'distance': [],
#            'direction': [],
#            'speed': [],
# }
# default_option = {'distance': 100,
#            'direction': 'right',
#            'speed': 60,
# }

# scenario_type = 'ConstructionObstacleTwoWays'
# options = {'distance': [],
#            'direction': [],
#            'speed': [],
#            'frequency': [],
# }
# default_option = {'distance': 100,
#            'direction': 'right',
#            'speed': 60,
#            'frequency': (20, 100),
# }

# scenario_type = 'CrossingBicycleFlow'
# options = {'start_actor_flow': [],
#            'flow_speed': [],
#            'source_dist_interval': [],
# }
# default_option = {'flow_speed': 10,
#                   'source_dist_interval': (20, 50),
# }

# scenario_type = 'HazardAtSideLane'
# options = {'distance': [],
#            'speed': [],
#            'bicycle_speed': [],
#            'bicycle_drive_distance' : []
# }
# default_option = {'distance': 100,
#                   'speed': 60,
#                   'bicycle_speed': 10,
#                   'bicycle_drive_distance': 50
# }

# scenario_type = 'HazardAtSideLaneTwoWays'
# options = {'distance': [],
#            'speed': [],
#            'bicycle_speed': [],
#            'bicycle_drive_distance' : [],
#            'frequency' : []
# }
# default_option = {'distance': 100,
#                   'speed': 60,
#                   'bicycle_speed': 10,
#                   'bicycle_drive_distance': 50,
#                   'frequency' : 100
# }

# scenario_type = 'InvadingTurn'
# options = {'distance': [],
#            'offset': []
# }
# default_option = {'distance': 100,
#                   'offset': 0.25,
# }

# scenario_type = 'NonSignalizedJunctionLeftTurn'
# options = {'flow_speed': [],
#            'source_dist_interval': []
# }
# default_option = {'flow_speed': 20,
#                   'source_dist_interval': (25, 50),
# }

# scenario_type = 'NonSignalizedJunctionRightTurn'
# options = {'flow_speed': [],
#            'source_dist_interval': []
# }
# default_option = {'flow_speed': 20,
#                   'source_dist_interval': (25, 50),
# }

# scenario_type = 'OppositeVehicleTakingPriority'
# options = {'direction': []
# }
# default_option = {'direction': 'right'
# }

# scenario_type = 'ParkedObstacle'
# options = {'distance': [],
#            'direction': [],
#            'speed': [],
# }
# default_option = {'distance': 120,
#                   'direction': 'right',
#                   'speed': 60,
# }

# scenario_type = 'ParkedObstacleTwoWays'
# options = {'distance': [],
#            'direction': [],
#            'speed': [],
#            'frequency': []
#           }
# default_option = {'distance': 120,
#                   'direction': 'right',
#                   'speed': 60,
#                   'frequency': (20, 100)
#                   }

# scenario_type = 'ParkingCrossingPedestrian'
# options = {'distance': [],
#            'crossing_angle': [],
#            'direction': []
#           }
# default_option = {'distance': 12,
#                   'crossing_angle': 0,
#                   'direction': 'right'
#                   }

# scenario_type = 'ParkingCutIn' # Some buggy value in file need to turn off sorting to display
# options = {'direction': []
#           }
# default_option = {'direction': 'right'
#                   }

# scenario_type = 'ParkingExit'
# options = {'front_vehicle_distance': [],
#            'behind_vehicle_distance': [],
#            'direction': [],
#            'flow_distance': [],
#            'speed': [],
#           }
# default_option = {'front_vehicle_distance': 20,
#                   'behind_vehicle_distance': 10,
#                   'direction': 'right',
#                   'flow_distance': 25,
#                   'speed': 60,
#                   }

# scenario_type = 'StaticCutIn'
# options = {'distance': [],
#            'direction': [],
#           }
# default_option = {'distance': 100,
#                   'direction': 'right',
#                   }

# scenario_type = 'VehicleOpensDoorTwoWays'
# options = {'distance': [],
#            'direction': [],
#            'speed': [],
#            'frequency': []
#           }
# default_option = {'distance': 50,
#                   'direction': 'right',
#                   'speed': 60,
#                   'frequency': (20, 100),
#                   }

# scenario_type = 'YieldToEmergencyVehicle'
# options = {'distance': [],
#           }
# default_option = {'distance': 140,
#                   }


# scenario_type = 'InterurbanActorFlow'
# options = {'start_actor_flow': [],
#            'end_actor_flow': [],
#            'flow_speed': [],
#            'source_dist_interval': [],
#           }
# default_option = {'flow_speed': 10,
#                   'source_dist_interval': (20, 50)
#                   }

# scenario_type = 'InterurbanAdvancedActorFlow'
# options = {'start_actor_flow': [],
#            'end_actor_flow': [],
#            'flow_speed': [],
#            'source_dist_interval': [],
#           }
# default_option = {'flow_speed': 10,
#                   'source_dist_interval': (20, 50)
#                   }

# scenario_type = 'MergerIntoSlowTraffic'
# options = {'start_actor_flow': [],
#            'end_actor_flow': [],
#            'flow_speed': [],
#            'source_dist_interval': [],
#           }
# default_option = {'flow_speed': 10,
#                   'source_dist_interval': (20, 50)
#                   }

tree = ET.parse(IN_PATH)
routes = tree.getroot()
for route in routes:
    if route.find('scenarios') is not None:  # waypoint conversion
        scenarios = route.find('scenarios')
        for scenario in scenarios:
            if scenario.attrib['type'] == scenario_type:
                for parameter_name in options:
                    parameter = scenario.find(parameter_name)
                    if parameter is not None:
                        # These are coordinates, we extract the difference to the trigger point instead.
                        if parameter_name in ('start_actor_flow', 'end_actor_flow'):
                            trigger_parameter = scenario.find('trigger_point')
                            before = np.array((ast.literal_eval(trigger_parameter.attrib['x']),
                                               ast.literal_eval(trigger_parameter.attrib['y']),
                                               ast.literal_eval(trigger_parameter.attrib['z'])))
                            after = np.array((ast.literal_eval(parameter.attrib['x']),
                                              ast.literal_eval(parameter.attrib['y']),
                                              ast.literal_eval(parameter.attrib['z'])))
                            options[parameter_name].append(np.linalg.norm(before - after).item())
                        else:
                            try:
                                options[parameter_name].append(ast.literal_eval(parameter.attrib['value']))
                            except KeyError:
                                options[parameter_name].append((ast.literal_eval(parameter.attrib['from']), ast.literal_eval(parameter.attrib['to'])))
                            except ValueError:
                                options[parameter_name].append(str(parameter.attrib['value']))
                    else:
                        options[parameter_name].append(default_option[parameter_name])


tree = ET.parse(IN_PATH2)
routes = tree.getroot()
for route in routes:
    if route.find('scenarios') is not None:  # waypoint conversion
        scenarios = route.find('scenarios')
        for scenario in scenarios:
            if scenario.attrib['type'] == scenario_type:
                for parameter_name in options:
                    parameter = scenario.find(parameter_name)
                    if parameter is not None:
                        # These are coordinates, we extract the difference to the trigger point instead.
                        if parameter_name in ('start_actor_flow', 'end_actor_flow'):
                            trigger_parameter = scenario.find('trigger_point')
                            before = np.array((ast.literal_eval(trigger_parameter.attrib['x']),
                                               ast.literal_eval(trigger_parameter.attrib['y']),
                                               ast.literal_eval(trigger_parameter.attrib['z'])))
                            after = np.array((ast.literal_eval(parameter.attrib['x']),
                                              ast.literal_eval(parameter.attrib['y']),
                                              ast.literal_eval(parameter.attrib['z'])))
                            options[parameter_name].append(np.linalg.norm(before - after))
                        else:
                            try:
                                options[parameter_name].append(ast.literal_eval(parameter.attrib['value']))
                            except KeyError:
                                options[parameter_name].append((ast.literal_eval(parameter.attrib['from']), ast.literal_eval(parameter.attrib['to'])))
                            except ValueError:
                                options[parameter_name].append(str(parameter.attrib['value']))
                    else:
                        options[parameter_name].append(default_option[parameter_name])


for parameter_name in options:
    if scenario_type == 'ParkingCutIn':
      options[parameter_name] = [i for i in options[parameter_name] if i != 40]  # remove buggy values
    c = Counter(options[parameter_name])
    c = sorted(c.items())
    #print(parameter_name,  np.around(sorted(set(options[parameter_name])),2).tolist())
    print(parameter_name,  sorted(set(options[parameter_name])))
    print(parameter_name, c)
    print(parameter_name, [round(i[1] / len(options[parameter_name]), 3) for i in c])


# print(IN_PATH)
# tree = ET.parse(IN_PATH)
# routes = tree.getroot()
# for route in routes:
#     if route.find('waypoints') is None:  # waypoint conversion
#         new = ET.SubElement(route, 'waypoints')
#         for element in route:
#             if element.tag != 'waypoint':
#                 continue
#             element.tag = 'position'
#             del element.attrib['pitch']
#             del element.attrib['roll']
#             del element.attrib['yaw']
#             new.append(element)
#         for wp in route.findall('position'):
#             route.remove(wp)
#
#     if route.find('scenarios') is None:  # add scenario child
#         new = ET.SubElement(route, 'scenarios')
#
# tree.write(OUT_PATH)
