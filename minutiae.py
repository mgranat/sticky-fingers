import numpy as np
import pdb

# Input: binary image
# Output: image where 1 == ridge ending, 2 == bifurcation
def extract(img):
  out = np.zeros((len(img), len(img[0])))
  #for i in range(len(img)):
  #  out.append([0] * len(img[i]))

  numPoints = 0
  coords = []

  for i in range(len(img)):
    for j in range(len(img[i])):
      # Check if this is an edge
      if not img[i][j]:
        continue

      lowX = max(i - 1, 0)
      highX = min(i + 1, len(img) - 1)
      lowY = max(j - 1, 0)
      highY = min(j + 1, len(img[i]) - 1)

      count = 0

      for k in range(lowX, highX + 1):
        for l in range(lowY, highY + 1):
          count = count + img[k][l]

      # Ridge ending
      if count == 2:
        out[i][j] = 1
        numPoints = numPoints + 1
        coords.append(np.array((i, j)))
      # Bifurcation
      elif count == 4:
        out[i][j] = 2
        numPoints = numPoints + 1
        coords.append(np.array((i, j)))

  #print(numPoints)

  return (out >= 1) * 1, coords

# Wraps angles from [-2pi, 2pi] to [0, 2pi]
def wrapAngle(theta):
  if theta < 0:
    return theta + 2 * np.pi
  else:
    return theta

def sIndex(s, sBins, sRange):
  if s < 1 - sRange:
    return -1
  elif s > 1 + sRange:
    return -1
  else:
    return int((s - 1 + sRange) / (2 * sRange) * sBins)

def thetaIndex(theta, thetaBins):
  if theta >= 2 * np.pi:
    return thetaBins - 1
  return int(theta / (2 * np.pi) * thetaBins)

def support(p, q, i, a, matchFlags):
  # Parameters
  sBins = 50
  sRange = 0.2
  thetaBins = 50

  accum = np.zeros((sBins, thetaBins))
  m = len(p)
  n = len(q)
  pI = []
  thetaI = []
  qA = []
  thetaA = []

  for j in range(m):
    if j == i:
      pI.append(0)
      thetaI.append(0)
      continue

    v = p[j] - p[i]
    pI.append(np.linalg.norm(v))
    thetaI.append(np.arctan2(v[0], v[1]))

  for b in range(n):
    if b == a:
      qA.append(0)
      thetaA.append(0)
      continue

    v = q[b] - q[a]
    qA.append(np.linalg.norm(v))
    thetaA.append(np.arctan2(v[0], v[1]))


  for j in range(m):
    if j == i:
      continue

    for b in range(n):
      if b == a:
        continue

      if not matchFlags[j][b]:
        continue

      if pI[j] == 0:
        continue

      s = qA[b] / pI[j]
      theta = wrapAngle(thetaA[b] - thetaI[j])

      sInd = sIndex(s, sBins, sRange)
      thetaInd = thetaIndex(theta, thetaBins)

      if sInd == -1:
        continue

      accum[sInd][thetaInd] = accum[sInd][thetaInd] + 1

  return np.amax(accum)

def match(p, q):
  # Parameters
  lowThresh = 4
  highThresh = 1000

  m = len(p)
  n = len(q)
  matchFlags = np.ones((m, n))
  k = min(m, n)
  pmax = 0

  for i in range(m):
    for a in range(n):
      w = support(p, q, i, a, matchFlags)
      if w < lowThresh:
        matchFlags[i][a] = 0
      if pmax < w:
        pmax = w
      if pmax > highThresh:
        # PICK UP HERE


  return support(p, q, 0, 0, matchFlags)
