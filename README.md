# equirectangular_face_detection

## Face Detection
Only the tiny_face model can detect many faces on testbench image.

### Haar Cascade
This repository tested `cv2.CascadeClassifier()`.

### MTCNN
This repository tested `ipazc`'s mtcnn model. [link](https://github.com/ipazc/mtcnn)

### Tiny face
This repository tested `cydonia999`'s tiny face model. [link](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)


## Package Depedencies
### Common
``` bash
pip install Pillow
pip install tensorflow
pip install py360convert
```

py360convert - (c) sunset1995 [link](https://github.com/sunset1995/py360convert)

### MTCNN
``` bash
pip install mtcnn
```
mtcnn - (c) ipazc

## Testbench

`360_faces.jpg` image come from [this stackexchange link](https://photo.stackexchange.com/questions/73481/how-can-i-edit-equirectangular-images).