# Image processing operations for fingerprint matching
# Max Granat - mg46424

# Requires OpenCV 3.3.1
import numpy as np
import cv2
from skimage.morphology import thin

# Utility function for displaying a binary image
def displayBinary(img):
  cv2.imshow("image", (img * 255).astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

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

  return out

def enhance(img):
  # Parameters
  gaussianBlurSize = (5, 5)
  thresholdSize = 25
  thresholdShift = 8
  morphologySize = (3, 3)

  # Segment, enhance, and binarize image
  segmented = segmentation(img)
  equalized = cv2.equalizeHist(img)
  blurred = cv2.GaussianBlur(equalized, gaussianBlurSize, 0)
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

  return thinned
