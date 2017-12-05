# Image processing operations for fingerprint matching
# Max Granat - mg46424

# Requires OpenCV 3.3.1
import numpy as np
import cv2
from skimage.morphology import thin
import pdb
import scipy
import time

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
  imgCopy = np.copy(img)

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

      cv2.line(imgCopy, (x0, y0), (x1, y1), 255)

  displayBinary(imgCopy)

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

# Estimate the ridge frequency of each image block
def estimateFrequency(img, orientation):
  m = len(img)
  n = len(img[0])
  xBlocks = len(orientation)
  yBlocks = len(orientation[0])
  xSize = int(m / xBlocks)
  ySize = int(n / yBlocks)
  freqAvg = 9.2732
  freqRange = 1.8985
  devs = 1
  freqLo = freqAvg - devs * freqRange 
  freqHi = freqAvg + devs * freqRange
  blurSize = (3, 3)

  freqs = np.zeros((xBlocks, yBlocks))
  freqCounts = []

  for i in range(xBlocks):
    for j in range(yBlocks):
      block = img[i*xSize:(i+1)*xSize, j*ySize:(j+1)*ySize]
      theta = orientation[i, j]

      rotated = scipy.ndimage.interpolation.rotate(block, \
        np.degrees(-theta))

      signal = rotated[int(xSize / 2), :]

      # Find zero crossings, assume image has been normalized
      crossings = np.where((signal.astype(np.int16)[:-1] - 127) * \
        (signal.astype(np.int16)[1:] - 127) < 0)[0]

      if len(crossings) < 4:
        freqs[i, j] = 0
        continue

      distances = np.absolute(crossings[:-1] - crossings[1:])
      freq = np.median(distances)

      if freq < freqLo:
        freq = 0
      elif freq > freqHi:
        freq = 0
      else:
        freqCounts.append(freq)

      freqs[i, j] = freq

  if len(freqCounts) == 0:
    mid = freqAvg
  else:
    mid = np.average(freqCounts)

  freqLo = mid - devs * freqRange 
  freqHi = mid + devs * freqRange

  freqs = np.where((freqs < freqLo) | (freqs > freqHi), mid, freqs)

  return cv2.GaussianBlur(freqs, blurSize, 0)

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

  start = time.time()
  blockOrientation = estimateOrientation(img, xBlocks, yBlocks)  
  end = time.time()
  print("Orientation: " + str(end - start))

  start = time.time()
  freqs = estimateFrequency(img, blockOrientation)
  end = time.time()
  print("Frequency: " + str(end - start))

  # Perform Gabor filtering by block
  for i in range(xBlocks):
    for j in range(yBlocks):
      # Orientation estimation
      ridgeOrientation = blockOrientation[i, j]

      # Ridge frequency estimation by block
      ridgeWavelength = freqs[i, j]
      # ridgeWavelength = 9.2732

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

# Enhances a fingerprint image, returns image and segmentation
def enhance(img):
  # Parameters
  gaussianBlurSize = (5, 5)
  thresholdSize = 25
  thresholdShift = 8
  morphologySize = (3, 3)

  # Segment and equalize image
  segmented = segmentation(img)
  equalized = cv2.equalizeHist(img)

  # Perform Gabor filtering
  filtered = gabor(equalized)

  # Enhance and binarize image
  blurred = cv2.GaussianBlur(filtered, gaussianBlurSize, 0)
  # pdb.set_trace()
  binarized = cv2.adaptiveThreshold(blurred, 255, \
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, \
    thresholdSize, thresholdShift)

  # Perform morphological operations on binarized image
  kernel = np.ones(morphologySize, np.uint8)
  opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
  closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

  # Invert, binarize, and thin image
  binary = (closed < 127) * 1
  thinned = thin(binary) * 1

  return thinned, segmented
