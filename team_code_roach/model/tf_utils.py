"""
Some utility functions
"""
import numpy as np

# Taken from https://stackoverflow.com/a/47381058/9173068
def trapez(y, y0, w):
  return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)
def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
  # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
  # If either of these cases are violated, do some switches.
  if abs(c1 - c0) < abs(r1 - r0):
    # Switch x and y, and switch again when returning.
    xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)  # pylint: disable=locally-disabled, arguments-out-of-order
    return (yy, xx, val)

  # At this point we know that the distance in columns (x) is greater
  # than that in rows (y). Possibly one more switch if c0 > c1.
  if c0 > c1:
    return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)  # pylint: disable=locally-disabled, arguments-out-of-order

  # The following is now always < 1 in abs
  if (c1 - c0) != 0.0:
    slope = (r1 - r0) / (c1 - c0)
  else:
    slope = 0.0

  # Adjust weight by the slope
  w *= np.sqrt(1 + np.abs(slope)) / 2

  # We write y as a function of x, because the slope is always <= 1
  # (in absolute value)
  x = np.arange(c0, c1 + 1, dtype=float)
  if (c1 - c0) != 0.0:
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)
  else:
    y = np.zeros_like(x)

  # Now instead of 2 values for y, we have 2*np.ceil(w/2).
  # All values are 1 except the upmost and bottommost.
  thickness = np.ceil(w / 2)
  yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
  xx = np.repeat(x, yy.shape[1])
  vals = trapez(yy, y.reshape(-1, 1), w).flatten()

  yy = yy.flatten()

  # Exclude useless parts and those outside of the interval
  # to avoid parts outside of the picture
  mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

  return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])
def draw_line(img, start_row, start_column, end_row, end_column, color=(255, 255, 255), thickness=1, rmax=256):

  if start_row == end_row and start_column == end_column:
    rr, cc, val = start_row, start_column, 1.0
  else:
    rr, cc, val = weighted_line(r0=start_row, c0=start_column, r1=end_row, c1=end_column, w=thickness, rmax=rmax)

  img[rr, cc, 0] = val * color[0] + (1.0 - val) * img[rr, cc, 0]
  img[rr, cc, 1] = val * color[1] + (1.0 - val) * img[rr, cc, 1]
  img[rr, cc, 2] = val * color[2] + (1.0 - val) * img[rr, cc, 2]
  return img
def draw_box(img, box, color=(255, 255, 255), pixel_per_meter=4, thickness=1):
  translation = np.array([[box[0], box[1]]])
  width = box[2]
  height = box[3]
  yaw = box[4]
  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
  speed = box[5] * pixel_per_meter
  speed_coords = np.array([[0.0, 0.0], [0.0, speed]])
  corners = np.array([[-width, -height], [width, -height], [width, height], [-width, height]])
  corner_global = (rotation_matrix @ corners.T).T + translation
  speed_coords_global = (rotation_matrix @ speed_coords.T).T + translation
  corner_global = corner_global.astype(np.int64)
  speed_coords_global = speed_coords_global.astype(np.int64)

  # Only the center is guaranteed to be within the image. Need to clip the corner points.
  max_row = img.shape[0]
  max_column = img.shape[1]
  corner_global[:, 0] = np.clip(corner_global[:, 0], a_min=0, a_max=max_row - 1)
  corner_global[:, 1] = np.clip(corner_global[:, 1], a_min=0, a_max=max_column - 1)
  speed_coords_global[:, 0] = np.clip(speed_coords_global[:, 0], a_min=0, a_max=max_row - 1)
  speed_coords_global[:, 1] = np.clip(speed_coords_global[:, 1], a_min=0, a_max=max_column - 1)

  img = draw_line(img,
                  start_row=corner_global[0, 0],
                  start_column=corner_global[0, 1],
                  end_row=corner_global[1, 0],
                  end_column=corner_global[1, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=corner_global[1, 0],
                  start_column=corner_global[1, 1],
                  end_row=corner_global[2, 0],
                  end_column=corner_global[2, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=corner_global[2, 0],
                  start_column=corner_global[2, 1],
                  end_row=corner_global[3, 0],
                  end_column=corner_global[3, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=corner_global[3, 0],
                  start_column=corner_global[3, 1],
                  end_row=corner_global[0, 0],
                  end_column=corner_global[0, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)
  img = draw_line(img,
                  start_row=speed_coords_global[0, 0],
                  start_column=speed_coords_global[0, 1],
                  end_row=speed_coords_global[1, 0],
                  end_column=speed_coords_global[1, 1],
                  color=color,
                  thickness=thickness,
                  rmax=max_row)

  return img

def normalize_imagenet_func(x):
  """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
  x = x.clone()
  x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
  x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
  x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
  return x

def normalize_image_bf16(x):
  """
    Losslessly normalizes images in bfloat16 to approximate ImageNet stats.
    Maps uint8 inputs [0, 255] to exactly [-2.0, 1.984375].
  Args:
      x (tensor): Input images (already bfloat16)
  """
  x = x.clone()
  # Equivalent to: (x / 256.0 - 0.5) / 0.25
  x = (x - 128.0) / 64.0
  return x
