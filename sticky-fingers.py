# Requires OpenCV 3.3.1
import numpy as np
import cv2
from skimage.morphology import thin

def main():
  img = cv2.imread("figs/png_txt/figs_0/f0005_03.png", 0)
  equ = cv2.equalizeHist(img)
  blur = cv2.GaussianBlur(equ,(5,5),0)
  binarized = cv2.adaptiveThreshold(blur, 255, \
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)

  kernel = np.ones((3, 3), np.uint8)
  opened = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
  closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

  binary = (closed < 127) * 1

  thinned = thin(binary) * 1

  minutia = np.array(identifyMinutia(thinned)) * 123

  res = np.hstack((img, thinned * 255, minutia))

  cv2.imwrite("res.png", res)

def identifyMinutia(img):
  out = []
  for i in range(len(img)):
    out.append([0] * len(img[i]))

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
      # Bifurcation
      elif count == 4:
        out[i][j] = 2

  return out

main()
