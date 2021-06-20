import cv2
import urllib.request
import numpy as np

req = urllib.request.urlopen('http://127.0.0.1:5000/Stream')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1) # 'Load it as it is'

cv2.imshow('stream', img)
if cv2.waitKey() & 0xff == 27: quit()