# Minutiae extraction and matching for fingerprint matching
# Max Granat - mg46424

import numpy as np

# Extracts minutiae from a binary image
# Returns binary image of minutiae and coordinates
def extract(img):
  # Initialize output image and coordinates
  out = np.zeros((len(img), len(img[0])))
  coords = []

  for i in range(len(img)):
    for j in range(len(img[i])):
      # Must be edge point
      if not img[i][j]:
        continue

      # Utility variable for counting minutiae in an image
      numPoints = 0

      # Establish bounds for crossing number filter
      lowX = max(i - 1, 0)
      highX = min(i + 1, len(img) - 1)
      lowY = max(j - 1, 0)
      highY = min(j + 1, len(img[i]) - 1)

      # Count the number of other edge points in a 3x3 grid
      count = 0

      for k in range(lowX, highX + 1):
        for l in range(lowY, highY + 1):
          count = count + img[k][l]

      # 1 other point == ridge ending
      if count == 2:
        out[i][j] = 1
        numPoints = numPoints + 1
        coords.append(np.array((i, j)))
      # 3 other points == bifurcation
      elif count == 4:
        out[i][j] = 2
        numPoints = numPoints + 1
        coords.append(np.array((i, j)))

  # print(numPoints)

  # Return binarized image and coordinates
  return (out >= 1) * 1, coords

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
