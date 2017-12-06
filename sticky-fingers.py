# Fingerprint matching
# Max Granat - mg46424

# Requires OpenCV 3.3.1
import numpy as np
import cv2
import imageProcessing
import minutiae
import os
import profile
import time

def main():
  path = "Max\\School\\sticky-fingers\\figs\\png_txt\\figs_0"
  directory = os.path.join("c:\\", path)
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".png"):
        img = cv2.imread(os.path.join(root, file), 0)
        enhanced, segmented = imageProcessing.enhance(img)

        start = time.time()
        min_img, coords = minutiae.extract(enhanced, segmented)
        end = time.time()
        print("Minutiae Extraction: " + str(end - start))

        min_img = minutiae.visualize(img, coords)
        # equalized = cv2.equalizeHist(img)
        # gabor = imageProcessing.gabor(equalized)

        # out = np.hstack((enhanced, min_img))
        # out = np.hstack((equalized, gabor))
        imageProcessing.display(min_img)
        # imageProcessing.displayBinary(out)
        # return

# profile.run('main()', sort = 'cumtime')
main()
