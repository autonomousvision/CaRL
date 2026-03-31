import carla

# Start CARLA server normally before running this script.

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
world = client.load_world('Town01', reset_settings=False)
world.tick()
print('The CARLA server will now not simulate anything (since it is waiting for the tick()), but still consume '
      '100% utilization of 1 CPU core')
