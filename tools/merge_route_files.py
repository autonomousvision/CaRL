import os
from lxml import etree as ET
from random import shuffle

town = 'Town02'
IN_PATH = '/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/transfuser_routes'
OUT_PATH = '/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/transfuser_routes_merged'

# New xml data
new_routes = ET.Element('routes')
new_tree = ET.ElementTree(new_routes)
route_list = []
for root, dirs, files in os.walk(IN_PATH):
    for file in files:
        if file.endswith(".xml") and file.startswith(town):
            print(file)
            tree = ET.parse(os.path.join(root, file))
            routes = tree.getroot()
            for route in routes:
                route_list.append(route)



outname = 'route_' + town + '.xml'

shuffle(route_list)
new_route_id = 0
for new_route in route_list:
    new_route.set('id', str(new_route_id))
    new_routes.insert(new_route_id, new_route)
    new_route_id += 1

with open(os.path.join(OUT_PATH, outname), 'wb') as f:
    test = ET.tostring(new_tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
    f.write(test)
