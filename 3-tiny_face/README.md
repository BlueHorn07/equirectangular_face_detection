
## Customization
At 2020.09, to use `cydonia999/Tiny_Faces_in_Tensorflow`, I've changed some codes.

1. downgrade version of `tensorflo 2.0` to `tensorflow 1.*`
   ``` python
     import tensorflow.compat.v1 as tf
     tf.disable_v2_behavior()
   ```
2. chnage `pl.frange()` into `np.arange()`
   In recent day, `pylab` removed `frange()`. They recommand us to use `np.arange()` instead. So, I've changed that.
3. Add `__init__.py` to use the advantages of package
   To use `evaluate()` in `tiny_face_eval.py`, I created `__init__.py` file. And link it into 
4. I also modify the `tiny_face_eval.py` source code that use **sinlge image**
   The original version read whole images in given directory, so I modified it to use only a single image and return it.

## Performance
| GPU Model | Compute Capability | time(sec) |
|:------:|:---:|:---:|
| GeFroce RTX 2070 | 7.5 | 3.7 ~ 4.5 |