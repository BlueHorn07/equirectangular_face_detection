import sys
LIB_PATH = "..\\lib"
sys.path.append(LIB_PATH)

import numpy as np
from PIL import Image
import cv2

# load equirectangular image
TESTBENCH_PATH = "..\\testbench\\"
src_img = cv2.imread(TESTBENCH_PATH + '360_faces.jpg')


# face detection by 'haar cascade algorithm'
WEIGHT_FILE_PATH = "./haarcascade_frontalface_default.xml"
haar_cascade_face = cv2.CascadeClassifier(WEIGHT_FILE_PATH)

## make image gray
gray_src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

faces_rects = haar_cascade_face.detectMultiScale(gray_src_img, scaleFactor = 1.1, minNeighbors = 5);

## make boundary boxes
for (x,y,w,h) in faces_rects:
     cv2.rectangle(src_img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# save image
cv2.imwrite("detected_360_faces.jpg", src_img)



