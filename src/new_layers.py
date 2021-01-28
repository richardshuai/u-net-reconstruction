import tensorflow as tf
from tensorflow.keras import layers
from src.utils import *

class MultiWienerDeconvolution(layers.Layer):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution, self).__init__()
        initial_psfs = tf.dtypes.cast(initial_psfs, dtype=tf.float32)
        initial_Ks = tf.dtypes.cast(initial_Ks, dtype=tf.float32)

        self.psfs = tf.Variable(initial_value=initial_psfs, trainable=True)
        self.Ks = tf.Variable(initial_value=initial_Ks, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
        _, h, w, _ = y.shape
        y = tf.dtypes.cast(y, dtype=tf.complex64)

        # Pad Y
        padding = ((0, 0), 
                   (int(tf.math.ceil(h / 2)), int(tf.math.floor(h / 2))),
                   (int(tf.math.ceil(w / 2)), int(tf.math.floor(w / 2))),
                    (0, 0))
        y = tf.pad(y, paddings=padding)

        # Temporarily transpose y since we cannot specify axes for fft2d
        y = tf.transpose(y, perm=[0, 3, 1, 2])   # Y is now shape (N, C, H, W)
        Y=tf.signal.fft2d(y)

        # Components preprocessing, psfs is shape (H, W, C)
        psf = tf.dtypes.cast(self.psfs, dtype=tf.complex64)
        h_psf, w_psf, _ = psf.shape

        # Pad psf
        padding_psf = (
                   (int(tf.math.ceil(h_psf / 2)), int(tf.math.floor(h_psf / 2))),
                   (int(tf.math.ceil(w_psf / 2)), int(tf.math.floor(w_psf / 2))),
                    (0, 0))

        H_sum = tf.pad(psf, paddings=padding_psf)

        H_sum = tf.transpose(H_sum, perm=[2, 0, 1])   # H_sum is now shape (C, H, W)
        H_sum = tf.signal.fft2d(H_sum)
        
        Ks = tf.transpose(self.Ks, [2, 0, 1]) # Ks is now shape (C, 1, 1)

        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*Ks, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        # x goes from shape (N, C, H, W) -> (N, H, W, C)
        x = tf.transpose(x, [0, 2, 3, 1])

        x = crop_2d_tf(x)

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config
        