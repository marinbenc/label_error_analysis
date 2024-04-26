import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def is_inside_contour(point, contours):
  '''
  Checks if a point is inside a contour.
  Args:
    point: point to check (x, y)
    contours: list of contours obtained from cv.findContours
  Returns:
    True if the point is inside a contour, False otherwise
  '''
  point = point.astype(float)
  for contour in contours:
    if cv.pointPolygonTest(contour, (point[0], point[1]), False) > 0:
      return True
  return False

def get_simplified_label(label: np.ndarray, dilation_size: int) -> np.ndarray:
  '''
  Gets a simplified version of the label and erodes/dilates it.
  Args:
    label: label image (binary image)
    dilation_size: size of the structuring element for erosion (if <0) or dilation (if >0) (in pixels), if 0, no erosion or dilation is performed
  Returns:
    simplified label image
  '''
  label_area = cv.countNonZero(label)
  label_area = label_area / label.shape[0] / label.shape[1]

  blur_size = 256 * label_area
  blur_size = max(1, blur_size)
  blur_size = round(blur_size)
  blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

  # blur
  label_blur = cv.blur(label, (blur_size, blur_size), borderType=cv.BORDER_CONSTANT)

  # threshold
  thresh = cv.threshold(label_blur, 128, 255, cv.THRESH_BINARY)[1]

  if dilation_size == 0:
    return thresh

  kernel_size = abs(int(round(dilation_size)))
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

  if dilation_size < 0:
    thresh = cv.erode(thresh, kernel, iterations=1)
  elif dilation_size > 0:
    thresh = cv.dilate(thresh, kernel, iterations=1)

  return thresh

def get_max_distance(label1: np.ndarray, label2: np.ndarray) -> int:
  '''
  Gets the maximum distance between the contours of two labels.
  Args:
    label1: label image (binary image)
    label2: label image (binary image)
  Returns:
    maximum distance between the contours of the two labels (in pixels)
  '''
  # get contours
  contours1, _ = cv.findContours(label1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contours2, _ = cv.findContours(label2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  if len(contours1) == 0 or len(contours2) == 0:
    return 0

  # get largest contour
  contour1 = max(contours1, key=lambda x: cv.contourArea(x)).squeeze()
  contour2 = max(contours2, key=lambda x: cv.contourArea(x)).squeeze()

  # get max distance
  max_dist = 0
  for point in contour1:
    point = point.astype(float)
    dist = cv.pointPolygonTest(contour2, tuple(point), True)
    dist = abs(dist)
    if dist > max_dist:
      max_dist = dist
  return max_dist

def get_initial_label_for_dilation(label, max_dist):
  kernel = np.ones((int(max_dist) * 2, int(max_dist) * 2), np.uint8)
  return cv.erode(label, kernel, iterations=1)

def get_contour_polygon_points(label):
  '''
  Simplifies a contour into a polygon.
  Args:
    label: label image (binary image)
  Returns:
    array of points of the polygon
  '''
  label = label.copy()

  # morphological closing
  kernel = np.ones((5, 5), np.uint8)
  label = cv.morphologyEx(label, cv.MORPH_OPEN, kernel)

  contours = cv.findContours(label, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
  contour = max(contours, key=lambda x: cv.contourArea(x)).squeeze()
  epsilon = 0.002 * cv.arcLength(contour, True)
  approx = cv.approxPolyDP(contour, epsilon, True)

  points = []
  for point in approx:
    points.append(point[0])
  return np.array(points)

def make_error_label(gt_label, simplified_label, percent_error):
  '''
  Makes a label with error by pulling some points of the label towards the simplified label.
  Args:
    gt_label: ground truth label image (binary image)
    simplified_label: simplified label image (binary image)
    percent_error: percentage of points to pull towards the simplified label
  Returns:
    label with error (binary image)
  '''
  points = get_contour_polygon_points(gt_label)
  #print(len(points))

  # label points as groups of 1s or 0s
  point_groups = []
  group = []

  simplified_contours, _ = cv.findContours(simplified_label, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  point_above_contour = [is_inside_contour(p, simplified_contours) for p in points]

  for i, point in enumerate(points):
    if i == 0:
      group.append(point)
    else:
      if point_above_contour[i] == point_above_contour[i - 1]:
        group.append(point)
      else:
        point_groups.append(group)
        group = [point]

  # choose random groups to pull towards contour
  n_groups_to_pull = round(percent_error * len(point_groups))
  random_group_idxs = np.random.randint(len(point_groups), size=n_groups_to_pull)

  new_points = np.array(points)
  all_contour_points = simplified_contours[0].squeeze()

  # pull chosen groups towards contour
  for i, point in enumerate(points):
    # check if point is in random group
    in_group = False
    for group_idx in random_group_idxs:
      if np.sum(point == point_groups[group_idx]) > 0:
        in_group = True
        break
    
    if not in_group:
      continue

    # get closest point on error contour
    closest_point = None
    min_dist = float('inf')
    for contour_point in all_contour_points:
      dist = np.linalg.norm(contour_point - point)
      if dist < min_dist:
        min_dist = dist
        closest_point = contour_point

    new_points[i] = closest_point
  
  # draw contour from new_points
  contour = np.array([new_points], dtype=np.int32)
  error_label = np.zeros(gt_label.shape, np.uint8)
  error_label = cv.drawContours(error_label, contour, -1, 255, 2)

  return error_label

if __name__ == '__main__':
  fn_im = 'data/isic/train/input/ISIC_0000018.jpg'
  fn_anno = 'data/isic/train/label/ISIC_0000018.jpg'

  label = cv.imread(fn_anno, cv.IMREAD_GRAYSCALE)
  label = cv.resize(label, (512, 512), interpolation=cv.INTER_NEAREST)
  simplified_label = get_simplified_label(label, 0)
  max_dist = get_max_distance(label, simplified_label)

  ratios = np.arange(-3, 1.5, 0.4)

  labels_for_ratios = []
  for ratio in ratios:
    dilation = round(ratio * max_dist * 2)
    current_label = get_simplified_label(label, dilation)
    error_label = make_error_label(label, current_label, 0.5)
    labels_for_ratios.append(error_label)

  # ratio colors for plotting
  colors = plt.cm.Blues(np.linspace(0, 1, len(ratios)))

  # plot
  fig, ax = plt.subplots(figsize=(7, 7))
  label_rgb = cv.cvtColor(label, cv.COLOR_GRAY2RGB)
  label_rgb[label > 128] = (255, 0, 0)
  label_rgb[label <= 128] = (0, 0, 0)
  ax.imshow(label_rgb)

  for i, ratio in enumerate(ratios):
    contours, _ = cv.findContours(labels_for_ratios[i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = contours[0].squeeze()
    ax.plot(contour[:, 0], contour[:, 1], color=colors[i], linewidth=1.5, label=round(ratio, 2))
  ax.legend()
  ax.axis('off')
  plt.show()