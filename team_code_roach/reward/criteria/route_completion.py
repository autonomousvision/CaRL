'''
Contains a class that computes the route completion of the ego vehicle.
'''


class RouteCompletionTest:
  """
  Check at which stage of the route is the actor at each tick

  Important parameters:
  - actor: CARLA actor to be used for this test
  - route: Route to be checked
  - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
  """
  WINDOWS_SIZE = 2

  # Thresholds to return that a route has been completed
  DISTANCE_THRESHOLD = 10.0  # meters

  def __init__(self, route, carla_map):
    """
    """
    self.route = route
    self.carla_map = carla_map

    self.index = 0
    self.route_length = len(self.route)
    self.route_transforms, _ = zip(*self.route)
    self.route_accum_perc = self.get_acummulated_percentages()
    self.actual_value = 0

  def get_acummulated_percentages(self):
    """Gets the accumulated percentage of each of the route transforms"""
    accum_meters = []
    prev_loc = self.route_transforms[0].location
    for i, tran in enumerate(self.route_transforms):
      d = tran.location.distance(prev_loc)
      new_d = 0 if i == 0 else accum_meters[i - 1]

      accum_meters.append(d + new_d)
      prev_loc = tran.location

    max_dist = accum_meters[-1]
    return [x / max_dist * 100 for x in accum_meters]

  def update(self, vehicle):
    """
    Check if the actor location is within trigger region
    """
    location = vehicle.get_transform().location

    for index in range(self.index, min(self.index + self.WINDOWS_SIZE + 1, self.route_length)):
      # Get the dot product to know if it has passed this location
      route_transform = self.route_transforms[index]
      route_location = route_transform.location
      wp_dir = route_transform.get_forward_vector()  # Waypoint's forward vector
      wp_veh = location - route_location  # vector route - vehicle

      if wp_veh.dot(wp_dir) > 0:
        self.index = index
        self.actual_value = self.route_accum_perc[self.index]

    self.actual_value = round(self.actual_value, 2)

    return self.actual_value
