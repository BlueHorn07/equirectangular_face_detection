import sys
LIB_PATH = "..\\lib"
sys.path.append(LIB_PATH)

import numpy as np
from PIL import Image
import cv2

# load equirectangular image
TESTBENCH_PATH = "..\\testbench\\"
src_img = cv2.imread(TESTBENCH_PATH + '360_faces.jpg')

# face detection by 'tiny face algorithm' - (c) cydonia999
TINY_FACE_PATH = ".\\tiny_face"
sys.path.append(TINY_FACE_PATH)
WEIGHT_FILE_PATH = ".\\tiny_face\\hr_res101.pkl"

## To fit the version of tensorflow, use tensorflow 1.*
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tiny_face import evaluate

with tf.Graph().as_default():
  detected_img = evaluate(weight_file_path=WEIGHT_FILE_PATH, src_img=src_img, filename="360_faces.jpg")

cv2.imwrite("detected_360_faces.jpg", detected_img)



