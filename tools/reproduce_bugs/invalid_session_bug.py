import carla

# Start CARLA server normally before running this script.
show_bug = True

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
blueprint_library = world.get_blueprint_library()
world.tick()

collision_bp = blueprint_library.find('sensor.other.collision')
# One can also attach the sensor to a vehicle. The bug happens in both cases
collision_sensor = world.spawn_actor(collision_bp, carla.Transform())
world.tick()
# Callback function for collisions
def on_collision(event):
    print(f"Collision detected with {event.other_actor.type_id} at {event.normal_impulse}")

collision_sensor.listen(on_collision)
# Now we destroy the vehicle before destroying the collision sensor. This will trigger the bug.
world.tick()
if not show_bug:
  # No bug occurs if the original reference to the collision sensor is used to destroy it.
  collision_sensor.stop()
  collision_sensor.destroy()
else:
  # The bug seems to only happen if the sensor reference was aquired with world.get_actors()
  actors = world.get_actors()
  sensors = actors.filter('*sensor*')
  for s in sensors:
    if s.is_alive:
      # The following warning is printed after trying to stop the sensor:
      # WARNING: attempting to unsubscribe from stream but sensor wasn't listening: Actor 371 (sensor.other.collision)
      s.stop()
      # After calling s.destroy() the server terminal will spam continuously:
      # ERROR: Invalid session: no stream available with id
      s.destroy()

# Happens even without further world.tick()
print('Finished')