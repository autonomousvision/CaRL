import xml.etree.ElementTree as ET

IN_PATH = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_training.xml'
IN_PATH2 = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation.xml'

all_types = []

tree = ET.parse(IN_PATH)
routes = tree.getroot()
for route in routes:
    if route.find('scenarios') is not None:  # waypoint conversion
        scenarios = route.find('scenarios')
        for scenario in scenarios:
            if not (scenario.attrib['type'] in all_types):
                all_types.append(scenario.attrib['type'])

tree = ET.parse(IN_PATH2)
routes = tree.getroot()
for route in routes:
    if route.find('scenarios') is not None:  # waypoint conversion
        scenarios = route.find('scenarios')
        for scenario in scenarios:
            if not (scenario.attrib['type'] in all_types):
                all_types.append(scenario.attrib['type'])

all_types.sort()
print(all_types)
