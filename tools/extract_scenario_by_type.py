import xml.etree.ElementTree as ET

IN_PATH = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_training.xml'
IN_PATH2 = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation.xml'
#scenario_type = 'EnterActorFlow'
#scenario_type = 'EnterActorFlowV2'
#scenario_type = 'HighwayCutIn'
#scenario_type = 'HighwayExit'
#scenario_type = 'MergerIntoSlowTraffic'
scenario_type = 'MergerIntoSlowTrafficV2'
save_path = rf'scenarios/{scenario_type}_Town12.xml'
save_path2 = rf'scenarios/{scenario_type}_Town13.xml'





new_routes = ET.Element('scenarios')
new_tree = ET.ElementTree(new_routes)

tree = ET.parse(IN_PATH)
routes = tree.getroot()
for route in routes:
    if route.find('scenarios') is not None:  # waypoint conversion
        scenarios = route.find('scenarios')
        for scenario in scenarios:
            if scenario.attrib['type'] == scenario_type:
                new_routes.append(scenario)

new_routes2 = ET.Element('scenarios')
new_tree2 = ET.ElementTree(new_routes2)
tree = ET.parse(IN_PATH2)
routes = tree.getroot()
for route in routes:
    if route.find('scenarios') is not None:  # waypoint conversion
        scenarios = route.find('scenarios')
        for scenario in scenarios:
            if scenario.attrib['type'] == scenario_type:
              new_routes2.append(scenario)

new_tree.write(save_path)
new_tree2.write(save_path2)