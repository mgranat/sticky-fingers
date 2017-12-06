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
import pdb

def main2():
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

def main():
  path = "Max\\School\\sticky-fingers\\figs\\png_txt\\figs_0"
  directory = os.path.join("c:\\", path)
  name = "f0019_10.png"
  img = cv2.imread(os.path.join(directory, name), 0)
  name2 = "s0248_02.png"
  img2 = cv2.imread(os.path.join(directory, name2), 0)

  start = time.time()
  enhanced, segmented = imageProcessing.enhance(img)
  enhanced2, segmented2 = imageProcessing.enhance(img2)
  end = time.time()
  print("Enhancement: " + str(end - start))

  start = time.time()
  minImg, coords = minutiae.extract(enhanced, segmented)
  minImg2, coords2 = minutiae.extract(enhanced2, segmented2)
  end = time.time()
  print("Minutiae Extraction: " + str(end - start))

  start = time.time()
  maxScore = min(len(coords), len(coords2))
  score = minutiae.match(np.array(coords), np.array(coords2))
  end = time.time()
  print("Minutiae Matching: " + str(end - start))

  print("Score: " + str(score / maxScore))

main()
