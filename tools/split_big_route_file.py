from lxml import etree
import os

file = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation.xml'
out_folder = r'/home/jaeger/ordnung/internal/ad_planning/2_carla/custom_leaderboard/leaderboard/data/routes_validation_split'



tree = etree.parse(file)
routes = tree.getroot()
id = 0
for route in routes:
    new_tree = etree.ElementTree(etree.Element('routes'))
    root = new_tree.getroot()
    root.append(route)
    filename = f'routes_validation_{id:02d}.xml'
    with open(os.path.join(out_folder, filename), 'wb') as f:
        test = etree.tostring(new_tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
        f.write(test)
    new_tree.write(os.path.join(out_folder, filename))
    id += 1
