# Requires OpenCV 3.3.1
import numpy as np
import cv2
from skimage.morphology import thin

def enhance(img):
  # Parameters
  gaussianBlurSize = (5, 5)
  thresholdSize = 25
  thresholdShift = 8
  morphologySize = (3, 3)

  equalized = cv2.equalizeHist(img)
  blurred = cv2.GaussianBlur(equalized, gaussianBlurSize, 0)
  binarized = cv2.adaptiveThreshold(blurred, 255, \
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, \
    thresholdSize, thresholdShift)

  kernel = np.ones(morphologySize, np.uint8)
  opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
  closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

  binary = (closed < 127) * 1

  thinned = thin(binary) * 1

  return thinned