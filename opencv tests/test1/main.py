import cv2 as cv
import numpy as np
# import os

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

stack = cv.imread("stack/stack3.png",)
needle = cv.imread("PionSet/pion1.png")

result = cv.matchTemplate(stack,needle, cv.TM_SQDIFF_NORMED)
treshold = 0.10
locations = np.where(result <= treshold)
locations = list(zip(*locations[::-1]))

# cv.imshow("result",result)
# print(result)
# cv.waitKey()

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

    cv.imshow('Matches', stack)
    cv.waitKey()