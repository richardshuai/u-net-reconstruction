import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.utils import *

# Batchnorm epsilon
BN_EPS = 1e-4

# Use float16
# tf.keras.backend.set_floatx('float16')

class ConvBnRelu2d(layers.Layer):
    # Convolutional -> Batch norm -> ReLU
    
    def __init__(self, out_channels, kernel_size=(3, 3), padding='same', dilation_rate=1, separable_conv=False):
        super(ConvBnRelu2d, self).__init__()
        if separable_conv:
            self.conv = layers.SeparableConv2D(filters=out_channels, kernel_size=kernel_size, padding='same', 
                                  depth_multiplier=1, dilation_rate=dilation_rate, use_bias=False)
        else:
            self.conv = layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding='same', 
                                  dilation_rate=dilation_rate, use_bias=False)
        self.bn = layers.BatchNormalization(epsilon=BN_EPS)
        self.relu = layers.ReLU()
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StackEncoder(layers.Layer):
    # ConvBnRelu -> ConvBnRelu -> * -> max pool (2x2)
    #      * store tensor for concatenation with expansive path

    def __init__(self, y_channels, kernel_size=3, dilation_rate=1, separable_conv=False):
        super(StackEncoder, self).__init__()
        self.encode = keras.Sequential([
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)
        ])
        
#         self.max_pool = layers.MaxPool2D(pool_size=2, strides=2)
        self.max_pool = layers.AveragePooling2D(pool_size=2, strides=2)

        
    def call(self, x):
        x = self.encode(x)
        x_small = self.max_pool(x)
        return x, x_small


class StackDecoder(layers.Layer):
    # Upsample (2x) -> concatenation -> ConvBnRelu -> ConvBnRelu -> ConvBnRelu

    def __init__(self, y_channels, kernel_size=3, dilation_rate=1, separable_conv=False):
        super(StackDecoder, self).__init__()
        self.decode = keras.Sequential([
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ]
        )
        
        
    def call(self, x, down_tensor=None):
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        
        # Calculate cropping for down_tensor to concatenate with x
        if down_tensor is not None:
            _, h2, w2, _ = down_tensor.shape
            _, h1, w1, _ = x.shape
            h_diff, w_diff = h2 - h1, w2 - w1

            cropping = ((int(np.ceil(h_diff / 2)), int(np.floor(h_diff / 2))),
                        (int(np.ceil(w_diff / 2)), int(np.floor(w_diff / 2))))
            down_tensor = layers.Cropping2D(cropping=cropping)(down_tensor)        
            x = layers.concatenate([x, down_tensor], axis=3)
        
        x = self.decode(x)
        return x
    
class GaussianFourierFeatureTransform(layers.Layer):
    """
    An implementation of Gaussian Fourier feature mapping.
    
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
        https://arxiv.org/pdf/2006.10739.pdf
        
    Input: tensor of size [N, H, W, C]
    Return: tensor of size [N, H, W, mapping_size*2]
    """
    def __init__(self, mapping_size=128, scale=10):
        super(GaussianFourierFeatureTransform, self).__init__()
        self.mapping_size = scale
        self.scale = mapping_size
        
    def call(self, x):
        _, _, _, C = x.shape
        B = tf.random.normal([C, self.mapping_size], mean=0, stddev=self.scale)
        
        x = x @ B
        x = 2 * np.pi * x
        return tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1)
    
    
class WienerDeconvolutionOneStep(layers.Layer):
    """
    Performs Wiener Deconvolution in frequency domain.
    PSF, K are learnable parameters. K is enforced to be nonnegative everywhere.
    
    Input: initial_psf of shape (Y, X), initial_K is a scalar.
    """
    def __init__(self, initial_psf, initial_K):
#     def __init__(self):

        super(WienerDeconvolutionOneStep, self).__init__()
        initial_psf = tf.dtypes.cast(initial_psf, dtype=tf.float32)
        initial_K = tf.dtypes.cast(initial_K, dtype=tf.float32)
        
        self.psf = tf.Variable(initial_value=initial_psf, trainable=True)
        self.K = tf.Variable(initial_value=initial_K, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative

#         self.K = tf.Variable(initial_value=initial_K, trainable=True) # I want to test with inital K starting at 0 in a bit
#         psf_init = tf.random_normal_initializer()
#         self.psf = tf.Variable(initial_value=psf_init(shape=(648, 486), dtype='float32'), trainable=True)
        
#         K_init = tf.zeros_initializer()
#         self.K = tf.Variable(initial_value=K_init(shape=(), dtype='float32'), trainable=True)
        
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, C)
        _, h, w, _ = y.shape
        y = tf.squeeze(tf.dtypes.cast(y, dtype=tf.complex64), axis=-1) # Remove channel dimension
        
        # Pad Y
        padding = ((0, 0), 
                   (int(tf.math.ceil(h / 2)), int(tf.math.floor(h / 2))),
                   (int(tf.math.ceil(w / 2)), int(tf.math.floor(w / 2))))
        y = tf.pad(y, paddings=padding)
        Y=tf.signal.fft2d(y)

        # PSF preprocessing, psf is shape (H, W)
        psf = tf.dtypes.cast(self.psf, dtype=tf.complex64)
        h_psf, w_psf = psf.shape
        
        # Pad psf
        padding_psf = (
                   (int(tf.math.ceil(h_psf / 2)), int(tf.math.floor(h_psf / 2))),
                   (int(tf.math.ceil(w_psf / 2)), int(tf.math.floor(w_psf / 2))))
        H_sum = tf.pad(psf, paddings=padding_psf)
        H_sum=tf.signal.fft2d(H_sum)
        
        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+1000*self.K, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(1, 2))))
                
        x = crop_2d_tf(x)


        # Normalize
#         x_max = tf.reduce_max(x)
#         x_min = tf.reduce_min(x)
#         x = (x - x_min) / (x_max - x_min)  # Normalize values to [0, 1]
        return x[..., None] # Add channel dimension
    

class WienerDeconvolutionPerComponent(layers.Layer):
    """
    Performs Wiener Deconvolution in the frequency domain for each R components.
    
    Input: initial_comps of shape (Y, X, R), initial_K is a scalar to be used for all components.
    """
    def __init__(self, initial_comps, initial_K):
        super(WienerDeconvolutionPerComponent, self).__init__()
        initial_comps = tf.dtypes.cast(initial_comps, dtype=tf.float32)
        initial_K = tf.dtypes.cast(initial_K, dtype=tf.float32)
        
        self.comps = tf.Variable(initial_value=initial_comps, trainable=True)
        self.K = tf.Variable(initial_value=initial_K, constraint=tf.nn.relu, trainable=True) # K is constrained to be nonnegative
        
    def call(self, y):
        # Y preprocessing, Y is shape (N, H, W, R)
        _, h, w, _ = y.shape
        y = tf.dtypes.cast(y, dtype=tf.complex64)

        # Pad Y
        padding = ((0, 0), 
                   (int(tf.math.ceil(h / 2)), int(tf.math.floor(h / 2))),
                   (int(tf.math.ceil(w / 2)), int(tf.math.floor(w / 2))),
                    (0, 0))
        y = tf.pad(y, paddings=padding)
        
        # Temporarily transpose y since we cannot specify axes for fft2d
        y = tf.transpose(y, perm=[0, 3, 1, 2])   # Y is now shape (N, R, H, W)
        Y=tf.signal.fft2d(y)
        
      

        # Components preprocessing, comps is shape (H, W, R)
        psf = tf.dtypes.cast(self.comps, dtype=tf.complex64)
        h_psf, w_psf, _ = psf.shape
        
        # Pad psf
        padding_psf = (
                   (int(tf.math.ceil(h_psf / 2)), int(tf.math.floor(h_psf / 2))),
                   (int(tf.math.ceil(w_psf / 2)), int(tf.math.floor(w_psf / 2))),
                    (0, 0))

        H_sum = tf.pad(psf, paddings=padding_psf)
        
        H_sum = tf.transpose(H_sum, perm=[2, 0, 1])   # H_sum is now shape (R, H, W)
        H_sum = tf.signal.fft2d(H_sum)
        
      
        X=(tf.math.conj(H_sum)*Y) / tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+0.1*self.K, dtype=tf.complex64)
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X), axes=(2, 3))))
        
        # x goes from shape (N, R, H, W) -> (N, H, W, R)
        x = tf.transpose(x, [0, 2, 3, 1])
        
        x = crop_2d_tf(x)


        # Normalize
#         x_max = tf.reduce_max(x)
#         x_min = tf.reduce_min(x)
#         x = (x - x_min) / (x_max - x_min)  # Normalize values to [0, 1]
        return x