import numpy as np
import tensorflow as tf

def normalize(x):
    """
    Normalizes numpy array to [0, 1].
    """
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)

def pad_2d(x, mode='constant'):
    """
    Pads 2d array x before FFT convolution.
    """
    _, _, h, w = x.shape
    padding = ((0, 0), (0, 0), 
               (int(np.ceil(h / 2)), int(np.floor(h / 2))),
               (int(np.ceil(w / 2)), int(np.floor(w / 2))))
    x = np.pad(x, pad_width=padding, mode=mode)
    return x

def pad_2d_tf(x, mode='CONSTANT'):
    _, _, h, w = x.shape
    padding = ((0, 0), (0, 0), 
               (int(np.ceil(h / 2)), int(np.floor(h / 2))),
               (int(np.ceil(w / 2)), int(np.floor(w / 2))))
    x = tf.pad(x, paddings=padding, mode=mode)
    return x

def crop_2d(v):
    """
    Crops 2d array x after FFT convolution. Inverse of pad2d.
    """
    h, w = v.shape
    h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
    w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
    return v[h1:h2, w1:w2]

def crop_2d_tf(v):
    """
    Crops 2d array x after FFT convolution. Inverse of pad2d.
    """
    h, w = v.shape
    h1, h2 = int(np.ceil(h / 4)), h - int(np.floor(h / 4))
    w1, w2 = int(np.ceil(w / 4)), w - int(np.floor(w / 4))
    return v[h1:h2, w1:w2]

