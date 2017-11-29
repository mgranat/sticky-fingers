# Fingerprint matching
# Max Granat - mg46424

# Requires OpenCV 3.3.1
import numpy as np
import cv2
import imageProcessing
import minutiae
import os

def main():
  path = "Max\\School\\sticky-fingers\\figs\\png_txt\\figs_0"
  directory = os.path.join("c:\\", path)
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".png"):
        img = cv2.imread(os.path.join(root, file), 0)
        enhanced = imageProcessing.enhance(img)
        minutiae_img, coords = minutiae.extract(enhanced)

        out = np.hstack((enhanced, minutiae_img))
        imageProcessing.displayBinary(out)

main()
