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
  epsilon = 0.005 * cv.arcLength(contour, True)
  approx = cv.approxPolyDP(contour, epsilon, True)

  points = []
  for point in approx:
    points.append(point[0])

  return np.array(points)

def make_error_label(gt_label, percent_error, bias):
  '''
  Makes a label with error by pulling some points of the label towards the simplified label.
  Args:
    gt_label: ground truth label image (binary image)
    percent_error: percentage of points to pull towards the simplified label
    bias: no bias (0), bias towards false positives (1), bias towards false negatives (-1)
  Returns:
    label with error (binary image)
  '''
  points = get_contour_polygon_points(gt_label)

  # draw points
  viz_img = np.zeros((gt_label.shape[0], gt_label.shape[1], 3), np.uint8)
  viz_img[gt_label > 128] = (255, 0, 0)
  # for point in points:
  #   gt_label = cv.circle(viz_img, tuple(point), 3, (0, 255, 0), -1)

  # draw convex hull
  hull = cv.convexHull(points, returnPoints=False)
  #hull_points = points[hull.squeeze()]
  #viz_img = cv.drawContours(viz_img, [hull_points], -1, (0, 0, 255), 2)

  hull_points = []
  for i in hull.squeeze():
    hull_points.append(points[i])
  hull_points = np.array(hull_points)

  # find defects
  defects = cv.convexityDefects(points, hull)
  if defects is not None:
    defect_points = points[defects[:, 0, 2]]
  else:
    defect_points = np.empty((0, 2))

  key_points = np.concatenate((hull_points, defect_points))
  is_defect = np.zeros(len(key_points)).astype(bool)
  is_defect[len(hull_points):] = 1

  # sort key_points based on points
  points = np.array(points)
  sorting = np.zeros(len(key_points), np.uint8)
  for idx in range(len(key_points)):
    # find closest point in points
    idx_in_points = np.argmin(np.linalg.norm(points - key_points[idx], axis=1))
    sorting[idx] = idx_in_points

  sorting = np.argsort(sorting)
  key_points = key_points[sorting]
  is_defect = is_defect[sorting]

  # shift so that key_points[0] is a defect point
  if not is_defect[0]:
    key_points = np.roll(key_points, -1, axis=0)
    is_defect = np.roll(is_defect, -1, axis=0)

  # make two arrays of groups of point indices
  # one for hull points and one for defect points

  hull_groups = []
  defect_groups = []

  for i, point in enumerate(key_points):
    # check previous point
    if i > 0:
      # if point is in the same group as the previous point
      # otherwise, start a new group
      if is_defect[i] == is_defect[i - 1]:
        if is_defect[i]:
          defect_groups[-1].append(i)
        else:
          if len(hull_groups[-1]) >= 3:
            hull_groups.append([i]) # limit the maximum number of points in a hull group to 3
          else:
            hull_groups[-1].append(i)
      else:
        if is_defect[i]:
          defect_groups.append([i])
        else:
          hull_groups.append([i])
    else:
      if is_defect[i]:
        defect_groups.append([i])
      else:
        hull_groups.append([i])

  n_groups_to_remove = round(len(defect_groups) * percent_error)
  n_groups_to_remove = min(n_groups_to_remove, len(defect_groups))
  n_groups_to_remove = max(n_groups_to_remove, 0)

  hull_group_idxs_to_remove = []
  defect_group_idxs_to_remove = []

  if bias == 0:
    hull_group_idxs_to_remove = np.random.choice(len(hull_groups), n_groups_to_remove // 2, replace=False)
    defect_group_idxs_to_remove = np.random.choice(len(defect_groups), n_groups_to_remove // 2, replace=False)
  elif bias == 1:
    hull_group_idxs_to_remove = np.random.choice(len(hull_groups), n_groups_to_remove, replace=False)
  elif bias == -1:
    defect_group_idxs_to_remove = np.random.choice(len(defect_groups), n_groups_to_remove, replace=False)
  else:
    raise ValueError('bias must be -1, 0, or 1')

  hull_points_to_remove = [p for group in hull_group_idxs_to_remove for p in hull_groups[group]]
  defect_points_to_remove = [p for group in defect_group_idxs_to_remove for p in defect_groups[group]]

  # remove groups
  new_key_points = []
  for i, point in enumerate(key_points):
    if i not in hull_points_to_remove and i not in defect_points_to_remove:
      new_key_points.append(point)

  new_key_points = np.array(new_key_points).astype(np.int32)

  new_mask = np.zeros_like(gt_label)
  new_mask = cv.fillPoly(new_mask, [new_key_points], 255)

  return new_mask
  