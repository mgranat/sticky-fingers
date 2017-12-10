# Image processing operations for fingerprint matching
# Max Granat - mg46424

# Requires OpenCV 3.3.1
import numpy as np
import cv2
from skimage.morphology import thin
import scipy

def writeOut(img, name):
  filename = ".\\imgs\\" + name + ".jpg"
  cv2.imwrite(filename, img)

# Utility function for displaying a grayscale image
def display(img):
  cv2.imshow("image", img.astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Utility function for displaying a binary image
def displayBinary(img):
  display(img * 255)

# Returns a binary image indicating fore/background
def segmentation(img):
  # Parameters
  # Image will be divided into xBlocks x yBlocks
  xBlocks = 32
  yBlocks = 32
  # Variance threshold for fingerprint segments
  varThreshold = 150

  # Calculate size of each block
  xSize = int(len(img) / xBlocks)
  ySize = int(len(img[0]) / yBlocks)

  # Initialize output array
  out = np.zeros((xBlocks, yBlocks))

  for i in range(xBlocks):
    for j in range(yBlocks):
      block = img[i*xSize:(i+1)*xSize, j*ySize:(j+1)*ySize]
      out[i][j] = np.var(block)

  # Threshold and binarize output array
  out = ((out > varThreshold) * 1).astype(np.uint8)

  # Close output array to fill holes
  kernel = np.ones((3, 3))
  out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

  dims = (len(img), len(img[0]))
  out_img = cv2.resize(out, dims, \
    interpolation=cv2.INTER_NEAREST)

  return out_img

# Visualize an orientation estimation
def visualizeOrientation(img, orientation):
  # Compute parameters
  xBlocks = len(orientation)
  yBlocks = len(orientation[0])
  xSize = int(len(img) / xBlocks)
  ySize = int(len(img[0]) / yBlocks)
  imgCopy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 0.6

  # Drawing coordinates are transpose of image coordinates
  orientation = np.transpose(orientation)

  for i in range(xBlocks):
    for j in range(yBlocks):
      theta = orientation[i, j] + np.pi / 2
      xStart = i * xSize
      yStart = j * ySize

      x0 = int(xSize / 2 * np.cos(theta))
      y0 = int(ySize / 2 * np.sin(theta))
      x1 = -x0
      y1 = -y0

      x0 = x0 + xStart + int(xSize / 2)
      y0 = y0 + yStart + int(ySize / 2)
      x1 = x1 + xStart + int(xSize / 2)
      y1 = y1 + yStart + int(ySize / 2)

      cv2.line(imgCopy, (x0, y0), (x1, y1), (0, 255, 255))

  display(imgCopy)

# Estimate the orientation of each image block
def estimateOrientation(img, xBlocks, yBlocks):
  # Parameters
  m = len(img)
  n = len(img[0])
  xSize = int(m / xBlocks)
  ySize = int(n / yBlocks)
  xRange = int(xSize / 2)
  yRange = int(ySize / 2)
  sobelSize = 3

  # Calculate orientation using Sobel operators
  sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobelSize)
  sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobelSize)

  # Compute dominant direction of the Fourier spectrum
  vxArray = 2 * np.multiply(sobelX, sobelY)

  sobelX2 = np.multiply(sobelX, sobelX)
  sobelY2 = np.multiply(sobelY, sobelY)
  vyArray = sobelX2 - sobelY2

  kSize = (xSize + 1, ySize + 1)
  vxSum = cv2.boxFilter(vxArray, -1, kSize, normalize = False)
  vySum = cv2.boxFilter(vyArray, -1, kSize, normalize = False)

  localOrientation = 0.5 * np.arctan2(vxSum, vySum)

  # Block orientation is value of center pixel
  blockOrientation = np.zeros((xBlocks, yBlocks))

  for i in range(xBlocks):
    for j in range(yBlocks):
      xInd = xRange + i * xSize
      yInd = yRange + j * ySize
      blockOrientation[i, j] = localOrientation[xInd, yInd]

  return blockOrientation

# Implements the ramp function
def ramp(x):
  if x <= 0:
    return 0
  else:
    return x

# Implements the unit step function
def step(x):
  if x <= 0:
    return 0
  else:
    return 1

# Estimate the ridge frequency of each image block
def estimateFrequency(img, orientation, estimator = None):
  m = len(img)
  n = len(img[0])
  xBlocks = len(orientation)
  yBlocks = len(orientation[0])
  xSize = int(m / xBlocks)
  ySize = int(n / yBlocks)
  l = 32
  w = 16
  wavelengthMin = 2 # Based on 350 DPI image
  wavelengthMax = 18 # Based on 350 DPI image
  # wavelengthMin = 5
  # wavelengthMax = 14
  gaussianSize = 7
  gaussianKSize = (gaussianSize, gaussianSize)
  gaussianStdDev = 3
  gRange = int(gaussianSize / 2)

  freqs = np.zeros((xBlocks, yBlocks))
  stats = []
  labels = np.ones((xBlocks, yBlocks)) * -1
  
  # Compute well-defined frequencies
  for i in range(xBlocks):
    for j in range(yBlocks):
      theta = orientation[i, j]
      cosTheta = np.cos(theta)
      sinTheta = np.sin(theta)

      # Extract x-signature
      xSig = np.zeros(l)
      for k in range(l):
        accum = 0
        divisor = w

        for d in range(w):
          u = int(np.rint(i+(d-w/2)*cosTheta+(k-l/2)*sinTheta))
          v = int(np.rint(j+(d-w/2)*sinTheta+(l/2-k)*cosTheta))
          
          if u >= 0 and v >= 0 and u < m and v < n:
            accum = accum + img[u, v]
          else:
            divisor = divisor - 1

        if divisor != 0:
          accum = accum / divisor

        xSig[k] = accum

      peaks = scipy.signal.find_peaks_cwt(xSig, \
        np.arange(wavelengthMin, wavelengthMax + 1))
      
      if len(peaks) >= 2:
        dists = peaks[1:] - peaks[:-1]
        avg = np.average(dists)

        # Compute statistics
        valleys = ((peaks[1:] + peaks[:-1]) / 2).astype(np.int)
        avgPeak = np.average([p for ind, p in enumerate(xSig) if ind in peaks])
        avgValley = np.average([v for ind, v in enumerate(xSig) if ind in valleys])
        alpha = avgPeak - avgValley
        beta = avg
        gamma = np.var(xSig)

        stats.extend([alpha, beta, gamma])

        if estimator is not None:
          label = estimator.predict([[alpha, beta, gamma]])
          labels[i, j] = label

        if avg >= wavelengthMin and avg <= wavelengthMax:
          freqs[i, j] = 1 / avg
        else:
          freqs[i, j] = -1
      else:
        freqs[i, j] = -1

  kernel1D = cv2.getGaussianKernel(gaussianSize, gaussianStdDev)
  kernel = np.matmul(kernel1D, np.transpose(kernel1D))

  newFreqs = freqs.copy()

  while (freqs == -1).any():
    for i in range(xBlocks):
      for j in range(yBlocks):
        if freqs[i, j] != -1:
          continue

        lowX = max(i - gRange, 0)
        highX = min(i + gRange, xBlocks - 1)
        lowY = max(j - gRange, 0)
        highY = min(j + gRange, yBlocks - 1)

        num = 0
        denom = 0

        for p in range(lowX, highX + 1):
          for q in range(lowY, highY + 1):
            kval = kernel[gRange + i - p, gRange + j - q]
            num = num + kval * ramp(freqs[p, q])
            denom = denom + kval * step(freqs[p, q] + 1)

        if denom == 0:
          newFreqs[i, j] = -1
        else:
          newFreqs[i, j] = num / denom

        freqs = newFreqs.copy()

  return cv2.GaussianBlur(freqs, gaussianKSize, 0), np.array(stats), labels

# Extract the frequency statistics from an image
def extractFreqStats(img):
  m = len(img)
  n = len(img[0])
  xBlocks = 32
  yBlocks = 32
  xSize = int(m / xBlocks)
  ySize = int(n / yBlocks)

  equalized = cv2.equalizeHist(img)
  blockOrientation = estimateOrientation(equalized, xBlocks, yBlocks)
  freqs, stats, labels = estimateFrequency(equalized, blockOrientation)

  return stats

# Convert a label image into a color image
def convertLabels(labels):
  colors = np.zeros((len(labels), len(labels[0]), 3))
  for i in range(len(labels)):
    for j in range(len(labels[i])):
      label = labels[i, j]
      if label == -1:
        colors[i, j] = [0, 0, 0]
      elif label == 0:
        colors[i, j] = [0, 0, 255]
      elif label == 1:
        colors[i, j] = [0, 255, 255]
      elif label == 2:
        colors[i, j] = [0, 255, 0]
      elif label == 3:
        colors[i, j] = [255, 0, 0]
      elif label == 4:
        colors[i, j] = [255, 102, 255]
      elif label == 5:
        colors[i, j] = [255, 255, 255]
  return colors.astype(np.uint8)

# Visualize the clusters of image blocks
def visualizeClusters(img, estimator):
  m = len(img)
  n = len(img[0])
  xBlocks = 32
  yBlocks = 32
  xSize = int(m / xBlocks)
  ySize = int(n / yBlocks)

  equalized = cv2.equalizeHist(img)
  blockOrientation = estimateOrientation(equalized, xBlocks, yBlocks)
  freqs, stats, labels = estimateFrequency(equalized, blockOrientation, estimator)
  colors = convertLabels(labels)
  colors = cv2.resize(colors, (m, n), interpolation = cv2.INTER_NEAREST)

  colorImg = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
  outImg = cv2.addWeighted(colors, 0.5, colorImg, 0.5, 0)
  display(outImg)

# Gabor filtering
def gabor(img):
  # Parameters
  m = len(img)
  n = len(img[0])
  xBlocks = 32
  yBlocks = 32
  xSize = int(m / xBlocks)
  ySize = int(n / yBlocks)

  # Initialize output image
  outImg = np.zeros((m, n))

  blockOrientation = estimateOrientation(img, xBlocks, yBlocks)
  freqs, stats, labels = estimateFrequency(img, blockOrientation)

  # Perform Gabor filtering by block
  for i in range(xBlocks):
    for j in range(yBlocks):
      # Orientation estimation
      ridgeOrientation = blockOrientation[i, j]

      # Ridge frequency estimation by block
      ridgeWavelength = 1 / freqs[i, j]

      # Compute Gabor kernel
      ksize = (11, 11)
      stdDev = 4
      orientation = ridgeOrientation
      wavelength = ridgeWavelength
      aspectRatio = 1

      # Get even-symmetric (phi = 0) Gabor kernel
      gaborKernel = cv2.getGaborKernel(ksize, stdDev, \
        orientation, wavelength, aspectRatio, 0)

      # Filter and transfer block
      filtered = cv2.filter2D(img, -1, gaborKernel)
      block = filtered[i*xSize:(i+1)*xSize, j*ySize:(j+1)*ySize]
      outImg[i*xSize:(i+1)*xSize, j*ySize:(j+1)*ySize] = block

  return (255 - outImg).astype(np.uint8)

# Implements region masking
def getRegionMask(img, xBlocks, yBlocks):
  r = xBlocks / 2
  factor = 0.8
  mask = np.zeros((xBlocks, yBlocks))
  for i in range(xBlocks):
    for j in range(yBlocks):
      mask[i, j] = (np.square(i - r) + \
        np.square(j - r) <= np.square(factor * r)) * 1

  dims = (len(img), len(img[0]))
  outMask = cv2.resize(mask, dims, \
    interpolation=cv2.INTER_NEAREST)
  return outMask

# Enhances a fingerprint image, returns image and segmentation
def enhance(img):
  # Parameters
  gaussianBlurSize = (5, 5)
  thresholdSize = 25
  thresholdShift = 8
  morphologySize = (3, 3)
  xBlocks = 32
  yBlocks = 32

  # Segment and equalize image
  segmented = segmentation(img)
  equalized = cv2.equalizeHist(img)

  # Perform Gabor filtering and region masking
  filtered = gabor(equalized)
  regionMask = getRegionMask(filtered, xBlocks, yBlocks)
  segmented = np.multiply(segmented, regionMask)

  # Enhance and binarize image
  blurred = cv2.GaussianBlur(filtered, gaussianBlurSize, 0)
  binarized = cv2.adaptiveThreshold(blurred, 255, \
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, \
    thresholdSize, thresholdShift)

  # Perform morphological operations on binarized image
  kernel = np.ones(morphologySize, np.uint8)
  opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
  closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

  # Invert, binarize, and thin image
  binary = (closed < 127) * 1
  # binary = (binarized < 127) * 1
  thinned = thin(binary) * 1

  return thinned, segmented
