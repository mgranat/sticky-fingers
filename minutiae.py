# Minutiae extraction and matching for fingerprint matching
# Max Granat - mg46424

import numpy as np
import queue
import cv2
import pdb

# Returns the edges of an array as an array, wrapping aroudn
def gridEdges(grid):
  edges = np.concatenate([grid[0,:-1], grid[:-1,-1], \
    grid[-1,::-1], grid[-2:0:-1,0]])
  return np.append(edges, grid[0][0])

# Draw circles around minutiae
def visualize(img, coords):
  radius = 5

  for (j, i) in coords:
    cv2.circle(img, (i, j), radius, 255)

  return img

# Returns 1 if the point is a true ridge ending, zero otherwise
def identifyRidgeEnding(grid, x, y):
  # Initialize markings
  mark = np.zeros((len(grid), len(grid[0])))
  mark[x][y] = 1
  q = queue.Queue()
  q.put((x, y))

  while not q.empty():
    point = q.get()
    i = point[0]
    j = point[1]

    lowX = max(i - 1, 0)
    highX = min(i + 1, len(grid) - 1)
    lowY = max(j - 1, 0)
    highY = min(j + 1, len(grid[i]) - 1)

    for k in range(lowX, highX + 1):
      for l in range(lowY, highY + 1):
        if grid[k][l]:
          if not mark[k][l]:
            mark[k][l] = 1
            q.put((k, l))

  edges = gridEdges(mark)
  transitions = 0

  for i in range(len(edges) - 1):
    if edges[i + 1] == 1 and edges[i] == 0:
      transitions = transitions + 1

  if transitions == 1:
    return 1
  else:
    return 0

# Returns 2 if the point is a true bifurcation, zero otherwise
def identifyBifurcation(grid, x, y):
  # Initialize markings
  mark = np.zeros((len(grid), len(grid[0])))
  mark[x][y] = -1
  q = queue.Queue()
  q.put((x, y))

  while not q.empty():
    point = q.get()
    i = point[0]
    j = point[1]

    label = mark[i][j]
    if label == -1:
      label = 1

    lowX = max(i - 1, 0)
    highX = min(i + 1, len(grid) - 1)
    lowY = max(j - 1, 0)
    highY = min(j + 1, len(grid[i]) - 1)

    for k in range(lowX, highX + 1):
      for l in range(lowY, highY + 1):
        if grid[k][l]:
          if not mark[k][l]:
            mark[k][l] = label
            if mark[i][j] == -1:
              label = label + 1
            q.put((k, l))

  edges = gridEdges(mark)
  trans1 = 0
  trans2 = 0
  trans3 = 0

  for i in range(len(edges) - 1):
    if edges[i + 1] == 1 and edges[i] == 0:
      trans1 = trans1 + 1
    if edges[i + 1] == 2 and edges[i] == 0:
      trans2 = trans2 + 1
    if edges[i + 1] == 3 and edges[i] == 0:
      trans3 = trans3 + 1

  if trans1 == 1 and trans2 == 1 and trans3 == 1:
    return 2
  else:
    return 0

# Minutiae post-processing to reduce false positives
def cleanup(img, min_img):
  # Parameters
  filterSize = 23
  dist = int(filterSize / 2)

  for i in range(1, len(img) - 1):
    for j in range(1, len(img[i] - 1)):
      if not min_img[i][j]:
        continue

      lowX = max(i - dist, 0)
      highX = min(i + dist, len(img))
      lowY = max(j - dist, 0)
      highY = min(j + dist, len(img[i]))

      x = i - lowX
      y = j - lowY
      grid = img[lowX:highX, lowY:highY]

      # Ridge ending
      if (min_img[i][j] == 1):
        min_img[i][j] = identifyRidgeEnding(grid, x, y)
      # Bifurcation
      else:
        min_img[i][j] = identifyBifurcation(grid, x, y)

  return min_img

# Extracts minutiae from a binary image
# Returns binary image of minutiae and coordinates
def extract(img, seg):
  # Initialize output image
  out = np.zeros((len(img), len(img[0])))

  for i in range(1, len(img) - 1):
    for j in range(1, len(img[i]) - 1):
      # Must be edge point
      if not img[i][j]:
        continue

      # Must be part of foreground
      if not seg[i][j]:
        continue

      # Utility variable for counting minutiae in an image
      numPoints = 0

      # Compute the crossing number
      cn = 0
      cn = cn + abs(img[i + 1][j] - img[i + 1][j + 1])
      cn = cn + abs(img[i + 1][j + 1] - img[i][j + 1])
      cn = cn + abs(img[i][j + 1] - img[i - 1][j + 1])
      cn = cn + abs(img[i - 1][j + 1] - img[i - 1][j])
      cn = cn + abs(img[i - 1][j] - img[i - 1][j - 1])
      cn = cn + abs(img[i - 1][j - 1] - img[i][j - 1])
      cn = cn + abs(img[i][j - 1] - img[i + 1][j - 1])
      cn = cn + abs(img[i + 1][j - 1] - img[i + 1][j])
      cn = int(cn / 2)

      # cn == 1 => ridge ending
      if cn == 1:
        out[i][j] = 1
        numPoints = numPoints + 1
      # cn == 3 => bifurcation
      elif cn == 3:
        out[i][j] = 2
        numPoints = numPoints + 1

  # Cleanup minutiae
  clean = cleanup(img, out)
  coordsArray = clean.nonzero()

  coords = []
  for i in range(len(coordsArray[0])):
    # pdb.set_trace()
    coords.append((coordsArray[0][i], coordsArray[1][i]))

  # Return binarized image and coordinates
  return (clean >= 1) * 1, coords

# Wraps angles from [-2pi, 2pi] to [0, 2pi]
def wrapAngle(theta):
  if theta < 0:
    return theta + 2 * np.pi
  else:
    return theta

# Index scaling factor for accumulator array
def sIndex(s, sBins, sRange):
  if s < 1 - sRange:
    return -1
  elif s > 1 + sRange:
    return -1
  else:
    return int((s - 1 + sRange) / (2 * sRange) * sBins)

# Index angle for accumulator array
def thetaIndex(theta, thetaBins):
  if theta >= 2 * np.pi:
    return thetaBins - 1
  return int(theta / (2 * np.pi) * thetaBins)

# Find matching points support for p[i] <=> q[a]
def support(p, q, i, a, matchFlags):
  # Parameters
  sBins = 50
  # Limit scaling factor to 1 +/- sRange
  sRange = 0.2
  thetaBins = 50

  # Initialize accumulator array and distance/angle calculations
  accum = np.zeros((sBins, thetaBins))
  m = len(p)
  n = len(q)
  pI = []
  thetaI = []
  qA = []
  thetaA = []

  # Calculate length and angle of p[i] -> p[j]
  for j in range(m):
    if j == i:
      pI.append(0)
      thetaI.append(0)
      continue

    v = p[j] - p[i]
    pI.append(np.linalg.norm(v))
    thetaI.append(np.arctan2(v[0], v[1]))

  # Calculate length and angle of q[a] -> q[b]
  for b in range(n):
    if b == a:
      qA.append(0)
      thetaA.append(0)
      continue

    v = q[b] - q[a]
    qA.append(np.linalg.norm(v))
    thetaA.append(np.arctan2(v[0], v[1]))

  # Vote for transformations
  for j in range(m):
    if j == i:
      continue

    for b in range(n):
      if b == a:
        continue

      # Must be a possible match
      if not matchFlags[j][b]:
        continue

      if pI[j] == 0:
        continue

      # Calculate transformation parameters
      s = qA[b] / pI[j]
      theta = wrapAngle(thetaA[b] - thetaI[j])

      # Calculate indices to vote for
      sInd = sIndex(s, sBins, sRange)
      thetaInd = thetaIndex(theta, thetaBins)

      if sInd == -1:
        continue

      accum[sInd][thetaInd] = accum[sInd][thetaInd] + 1

  # Return the best transformation's number of matches
  return np.amax(accum)

# Find best match between two sets of points
def match(p_orig, q_orig):
  # Parameters
  lowThresh = 4
  highThresh = 1000

  # p is assumed to be the shorter array of points
  if len(p_orig) < len(q_orig):
    p = p_orig
    q = q_orig
  else:
    p = q_orig
    q = p_orig

  # Initialize
  m = len(p)
  n = len(q)
  matchFlags = np.ones((m, n))
  k = m
  pmax = 0
  found = False

  for i in range(m):
    print("Trying match: " + str(i) + " of " + str(m) + " (vs. " + str(n) + ")")
    for a in range(n):
      # Calculate support for p[i] <-> q[a]
      w = support(p, q, i, a, matchFlags)

      # Must exceed minimum threshold, otherwise mark as no match
      if w < lowThresh:
        matchFlags[i][a] = 0
      # Update new maximum
      if pmax < w:
        pmax = w
      # Patterns match if we exceed a match threshold
      if pmax > highThresh:
        found = True
        break
      # Patterns match if the maximum number of points match
      if pmax >= k - i:
        found = True
        break
    if found:
      break
    k = k - 1

  # Return number of points matched
  return pmax
