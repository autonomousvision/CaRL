import carla

# Start CARLA server normally before running this script.
# Doesn't seem to reproduce the bug.

# Setup a standard synchronous client
client = carla.Client('localhost', 2000)
client.set_timeout(300.0)
settings = carla.WorldSettings(
  synchronous_mode=True,
  fixed_delta_seconds=0.1,
  deterministic_ragdolls=True,
  no_rendering_mode=False,
  spectator_as_ego=True,
)
client.get_world().apply_settings(settings)
world = client.load_world('Town04', reset_settings=False)
blueprint_library = world.get_blueprint_library()
world.tick()

carla_map = world.get_map()
with open("/home/jaeger/ordnung/internal/custom_carla_0_9_15/carla/custom_maps/Town04.xodr", "r", encoding="utf-8") as f:
  content = f.read()
test_map = carla.Map("Town04", content)

bad_wp_00 = test_map.get_waypoint(carla.Location(x=-34.72969436645508, y=-88.61881256103516, z=0.0010084686800837517))
bad_wp_01 = test_map.get_waypoint(carla.Location(x=-69.64703369140625, y=37.3536376953125, z=9.807632446289062))

while bad_wp_01.is_junction:
  bad_wp_01s = bad_wp_01.previous(0.2)

  if len(bad_wp_01s) == 0:
    break  # Stop when there's no prev
  bad_wp_01 = bad_wp_01s[0]
# Happens even without further world.tick()
print('Finished')