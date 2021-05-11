import cv2 as cv
import numpy as np
import os

pionnenmap = "../../Datasets/pionnen"
totalFiles = 0
stack = cv.imread("../../Datasets/stack/stack6.png")
# aantal pionnen
for files in os.listdir(pionnenmap):
    totalFiles += 1

for i in range(1, totalFiles):
    needle = cv.imread("../../Datasets/pionnen/pion" + str(i) + ".png", 0)
    orb = cv.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(needle, None)
    kp2, des2 = orb.detectAndCompute(stack, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print(len(good))
    img3 = cv.drawMatchesKnn(needle, kp1, stack, kp2, good, None, flags=2)

    # cv2.imshow('Kp1',imgKp1)
    # cv2.imshow('Kp2',imgKp2)
    cv.imshow('needle', needle)
    cv.waitKey(0)

# "../../Datasets/stack")

# "../../Datasets/pionnen"
