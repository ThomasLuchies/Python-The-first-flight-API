import cv2 as cv
import numpy as np
import os

pionnenmap = "../../Datasets/pionnen"
totalFiles = 0
stack = cv.imread("../../Datasets/stack/stack6.png")
totalPionnen = len(pionnenmap)
# needle loop with result add
for files in os.listdir(pionnenmap):
    totalFiles += 1

print(totalFiles)
for i in range(1, totalFiles):
    needle = cv.imread("../../Datasets/pionnen/pion" + str(i) + ".png")
    print("dit is pion " + str(i))

    result = cv.matchTemplate(stack, needle, cv.TM_SQDIFF_NORMED)
    treshold = 0.05
    locations = np.where(result <= treshold)
    locations = list(zip(*locations[::-1]))
    print(locations)
    if locations:
        print('Found needle.')

        needle_w = needle.shape[1]
        needle_h = needle.shape[0]
        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        for loc in locations:
            top_left = loc
            bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
            cv.rectangle(stack, top_left, bottom_right, line_color, line_type)


# cv.imshow("result",result)
print(result)
# cv.waitKey()

cv.imshow('Matches', stack)
cv.waitKey()
