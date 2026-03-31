'''
Contains a class that checks if the agent left the route (like taking a wrong turn).
'''
import math


class InRouteTest:
  """
  The test is a success if the actor is never outside route. The actor can go outside of the route
  but only for a certain amount of distance

  Important parameters:
  - actor: CARLA actor to be used for this test
  - route: Route to be checked
  - offroad_max: Maximum distance (in meters) the actor can deviate from the route
  - offroad_min: Maximum safe distance (in meters). Might eventually cause failure
  - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
  """
  MAX_ROUTE_PERCENTAGE = 30  # %
  WINDOWS_SIZE = 5  # Amount of additional waypoints checked

  def __init__(self, actor, route, offroad_min=None, offroad_max=30):
    """
    """
    self.actor = actor
    self.units = None  # We care about whether or not it fails, no units attached

    self.route = route
    self.offroad_max = offroad_max
    # Unless specified, halve of the max value
    if offroad_min is None:
      self.offroad_min = self.offroad_max / 2
    else:
      self.offroad_min = self.offroad_min

    self.route_transforms, _ = zip(*self.route)
    self.route_length = len(self.route)
    self.current_index = 0
    self.out_route_distance = 0
    self.in_safe_route = True

    self.accum_meters = []
    prev_loc = self.route_transforms[0].location
    for i, tran in enumerate(self.route_transforms):
      loc = tran.location
      d = loc.distance(prev_loc)
      accum = 0 if i == 0 else self.accum_meters[i - 1]

      self.accum_meters.append(d + accum)
      prev_loc = loc

  def update(self):
    """
    Check if the actor location is within trigger region
    """
    location = self.actor.get_location()

    if location is None:
      return True

    off_route = True

    shortest_distance = float('inf')
    closest_index = -1

    # Get the closest distance
    for index in range(self.current_index, min(self.current_index + self.WINDOWS_SIZE + 1, self.route_length)):
      ref_location = self.route_transforms[index].location
      distance = math.sqrt(((location.x - ref_location.x)**2) + ((location.y - ref_location.y)**2))
      if distance <= shortest_distance:
        closest_index = index
        shortest_distance = distance

    if closest_index == -1 or shortest_distance == float('inf'):
      return True

    # Check if the actor is out of route
    if shortest_distance < self.offroad_max:
      off_route = False
      self.in_safe_route = bool(shortest_distance < self.offroad_min)

    # If actor advanced a step, record the distance
    if self.current_index != closest_index:

      new_dist = self.accum_meters[closest_index] - self.accum_meters[self.current_index]

      # If too far from the route, add it and check if its value
      if not self.in_safe_route:
        self.out_route_distance += new_dist
        out_route_percentage = 100 * self.out_route_distance / self.accum_meters[-1]
        if out_route_percentage > self.MAX_ROUTE_PERCENTAGE:
          off_route = True

      self.current_index = closest_index

    if off_route:
      return False

    return True
