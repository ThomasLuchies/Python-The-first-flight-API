import cv2
import socket

def MainVideo():
    capt = cv2.VideoCapture(1)

    while True:
        
        result, frame = capt.read() 
        cv2.imshow('Tello video', frame)
        key = cv2.waitKey(1) & 0xff 

if __name__ == '__main__':
    MainVideo()