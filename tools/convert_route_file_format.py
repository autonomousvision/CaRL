import xml.etree.ElementTree as ET
import os

IN_PATH = '/home/jaeger/ordnung/internal/carla_garage/leaderboard/data/longest6_split'
# OUT_PATH = '/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/longest6.xml'
OUT_PATH = '/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/longest6_no_scenarios_split'

for root, dirs, files in os.walk(IN_PATH):
    for file in files:
        if file.endswith(".xml"):
            print(file)
            tree = ET.parse(os.path.join(root, file))
            routes = tree.getroot()
            for route in routes:
                if route.find('waypoints') is None:  # waypoint conversion
                    new = ET.SubElement(route,'waypoints')
                    for element in route:
                        if element.tag != 'waypoint':
                            continue
                        element.tag = 'position'
                        del element.attrib['pitch']
                        del element.attrib['roll']
                        del element.attrib['yaw']
                        new.append(element)
                    for wp in route.findall('position'):
                        route.remove(wp)

                if route.find('scenarios') is None:  # add scenario child
                    new = ET.SubElement(route,'scenarios')

            tree.write(os.path.join(OUT_PATH, file))

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
