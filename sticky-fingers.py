# Requires OpenCV 3.3.1
import numpy as np
import cv2
import imageProcessing
import minutiae

def main():
  img = cv2.imread("figs/png_txt/figs_0/f0031_02.png", 0)
  img2 = cv2.imread("figs/png_txt/figs_0/s0031_02.png", 0)

  thinned = imageProcessing.enhance(img)
  thinned2 = imageProcessing.enhance(img2)

  minutia, coords = minutiae.extract(thinned)
  minutia2, coords2 = minutiae.extract(thinned2)

  res = np.hstack((img, thinned * 255, minutia * 255, \
    img2, thinned2 * 255, minutia2 * 255))

  print(minutiae.match(coords, coords2))

  cv2.imwrite("res.png", res)

main()
