# Fingerprint matching
# Max Granat - mg46424

# Requires OpenCV 3.3.1
import numpy as np
import cv2
import imageProcessing
import minutiae
import os
import time
import sys
import random
import pickle
from sklearn.cluster import KMeans

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

def test2(numTrials):
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

  estimator = pickle.load(open("estimator.sav", 'rb'))

  # Correct trials
  for i in range(numTrials):
    ind = int(random.random() * len(firstFiles))
    first = firstFiles[ind]
    second = secondFiles[ind]

    img = cv2.imread(first, 0)
    img2 = cv2.imread(second, 0)

    imageProcessing.visualizeClusters(img, estimator)
    imageProcessing.visualizeClusters(img2, estimator)

# Extract block statistics
def extractStats():
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

  stats = np.empty(0)

  for i in range(len(firstFiles)):
    file = firstFiles[i]
    file2 = secondFiles[i]
    img = cv2.imread(file, 0)
    img2 = cv2.imread(file, 0)
    start = time.time()
    imgStats = imageProcessing.extractFreqStats(img)
    imgStats2 = imageProcessing.extractFreqStats(img2)
    stats = np.concatenate([stats, imgStats, imgStats2])
    end = time.time()
    print("Extracted stats for img pair " + str(i) + ": " + str(end - start))

  np.save("block_stats", stats)

def cluster():
  stats = np.load("block_stats.npy").reshape((-1, 3))
  kmeans = KMeans(n_clusters = 6, verbose = 1).fit(stats)
  pickle.dump(kmeans, open("estimator.sav", 'wb'))

def main():
  if len(sys.argv) < 2:
    print("use test or test2 with a number of tests, extract, or cluster")
    return

  if sys.argv[1] == 'test':
    if len(sys.argv) < 3:
      return

    numTrials = int(sys.argv[2])
    test(numTrials)
  elif sys.argv[1] == 'test2':
    if len(sys.argv) < 3:
      return

    numTrials = int(sys.argv[2])
    test2(numTrials)
  elif sys.argv[1] == 'extract':
    extractStats()
  elif sys.argv[1] == 'cluster':
    cluster()

main()
