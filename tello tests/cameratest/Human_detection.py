import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()
cap = cv2.VideoCapture("udp://0.0.0.0:11111")
while(True):
    ret, frame = cap.read()
    resize = cv2.resize(frame, (300, 300))
    gray = cv2.cvtColor(resize, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(resize, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(resize, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)

    cv2.imshow('Tello video', resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)