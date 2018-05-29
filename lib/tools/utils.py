import numpy as np
import tensorflow as tf

def fast_hist(labels, pred, num_class=2):
    k = (labels >= 0) & (labels < num_class)
    res = np.bincount(num_class * labels[k].astype(int) + pred[k], minlength=num_class**2).reshape(num_class, num_class)
    return res.astype(np.float32)

def scales_to_255(a, min_, max_, type_):
    return tf.cast(((a - min_) / float(max_ - min_)) * 255, dtype=type_)
