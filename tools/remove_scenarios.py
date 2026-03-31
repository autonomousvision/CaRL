import xml.etree.ElementTree as ET

IN_PATH = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation.xml'
OUT_PATH = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation_no_scenarios.xml'

tree = ET.parse(IN_PATH)
routes = tree.getroot()
for route in routes:
    route.remove(route.find('scenarios'))

    if route.find('scenarios') is None:  # add empty scenario child
        new = ET.SubElement(route,'scenarios')

tree.write(OUT_PATH)
