import cv2
import socket

vid = cv2.VideoCapture(1)

check, frame = vid.read()
print(check)
print(frame)

cv2.imshow("window", frame)
vid.release()