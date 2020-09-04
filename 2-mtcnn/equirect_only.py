import sys
LIB_PATH = "..\\lib"
sys.path.append(LIB_PATH)

import numpy as np
from PIL import Image
import cv2

# load equirectangular image
TESTBENCH_PATH = "..\\testbench\\"
src_img = cv2.imread(TESTBENCH_PATH + '360_faces.jpg')


# face detection by 'MTCNN algorithm' - (c) ipazc
from mtcnn.mtcnn import MTCNN

detector = MTCNN() 

result = detector.detect_faces(src_img) 

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one. 
bounding_box = result[0]['box'] 
keypoints = result[0]['keypoints'] 
cv2.rectangle(src_img,(bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (0,155,255), 2) 
cv2.circle(src_img,(keypoints['left_eye']), 2, (0,155,255), 2) 
cv2.circle(src_img,(keypoints['right_eye']), 2, (0,155,255), 2) 
cv2.circle(src_img,(keypoints['nose']), 2, (0,155,255), 2) 
cv2.circle(src_img,(keypoints['mouth_left']), 2, (0,155,255), 2) 
cv2.circle(src_img,(keypoints['mouth_right']), 2, (0,155,255), 2)

# save image
cv2.imwrite("detected_360_faces.jpg", src_img)



