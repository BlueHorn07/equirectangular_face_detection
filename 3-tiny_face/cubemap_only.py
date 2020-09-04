import sys
LIB_PATH = "..\\lib"
sys.path.append(LIB_PATH)

import numpy as np
from PIL import Image
import cv2

# load equirectangular image
TESTBENCH_PATH = "..\\testbench\\"
src_img = cv2.imread(TESTBENCH_PATH + '360_faces.jpg')

# convert equirect into cubemap
## I used py360convert package - (c) sunset1995
import py360convert
cubemap_img = py360convert.e2c(src_img)

# face detection by 'tiny face algorithm' - (c) cydonia999
TINY_FACE_PATH = ".\\tiny_face"
sys.path.append(TINY_FACE_PATH)
WEIGHT_FILE_PATH = ".\\tiny_face\\hr_res101.pkl"
TESTBENCH_PATH = "..\\testbench\\"

## To fit the version of tensorflow, use tensorflow 1.*
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tiny_face import evaluate

with tf.Graph().as_default():
  detected_cubemap_img = evaluate(weight_file_path=WEIGHT_FILE_PATH, src_img = cubemap_img, filename="360_faces_cubemap_only.jpg")

# convert cubemap into equirect
detected_img = py360convert.c2e(detected_cubemap_img, src_img.shape[0], src_img.shape[1], cube_format='dice')

# save image
cv2.imwrite("detected_360_faces_cubemap_only.jpg", detected_img)



