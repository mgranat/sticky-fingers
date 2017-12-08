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
import sys
import random
import pdb

def match(first, second):
  img = cv2.imread(first, 0)
  img2 = cv2.imread(second, 0)

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

  return score / maxScore

def test(numTrials):
  allImgs = ".\\figs\\png_txt\\"
  imgDirs = os.listdir(allImgs)
  firstFiles = []
  secondFiles = []
  for d in imgDirs:
    imgDir = os.path.join(allImgs, d)
    imgFiles = [os.path.join(imgDir, f) \
    for f in os.listdir(imgDir) \
    if f.endswith(".png") and f.startswith("s")]
    secondFiles.extend(imgFiles)
    imgFiles = [os.path.join(imgDir, f) \
    for f in os.listdir(imgDir) \
    if f.endswith(".png") and f.startswith("f")]
    firstFiles.extend(imgFiles)

  correctResults = []

  # Correct trials
  for i in range(numTrials):
    ind = int(random.random() * len(firstFiles))
    first = firstFiles[ind]
    second = secondFiles[ind]
    
    correctResults.append(match(first, second))

  incorrectResults = []

  # Incorrect trials
  for i in range(numTrials):
    firstInd = int(random.random() * len(firstFiles))
    secondInd = int(random.random() * len(secondFiles))
    first = firstFiles[firstInd]
    second = secondFiles[secondInd]
    
    incorrectResults.append(match(first, second))

  print("Final Results\n")
  print("Avg Correct Score: " + str(np.average(correctResults)))
  print("Correct variance: " + str(np.var(correctResults)))
  print("Avg Incorrect Score: " + str(np.average(incorrectResults)))
  print("Incorrect variance: " + str(np.var(incorrectResults)))

def main():
  if len(sys.argv) < 2:
    return

  if sys.argv[1] == 'test':
    if len(sys.argv) < 3:
      return

    numTrials = int(sys.argv[2])
    test(numTrials)
  elif sys.argv[1] == 'extract':
    extractStats()
  elif sys.argv[1] == 'cluster':
    cluster()

main()
