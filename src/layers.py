import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

# Batchnorm epsilon
BN_EPS = 1e-4

# Use float16
# tf.keras.backend.set_floatx('float16')

class ConvBnRelu2d(layers.Layer):
    # Convolutional -> Batch norm -> ReLU
    
    def __init__(self, out_channels, kernel_size=(3, 3), padding='same', dilation_rate=1):
        super(ConvBnRelu2d, self).__init__()
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

    def __init__(self, y_channels, kernel_size=3, dilation_rate=1):
        super(StackEncoder, self).__init__()
        self.encode = keras.Sequential([
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)
        ])
        
        self.max_pool = layers.MaxPool2D(pool_size=2, strides=2)
        
    def call(self, x):
        x = self.encode(x)
        x_small = self.max_pool(x)
        return x, x_small


class StackDecoder(layers.Layer):
    # Upsample (2x) -> concatenation -> ConvBnRelu -> ConvBnRelu -> ConvBnRelu

    def __init__(self, y_channels, kernel_size=3, dilation_rate=1):
        super(StackDecoder, self).__init__()
        self.decode = keras.Sequential([
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ConvBnRelu2d(y_channels, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate),
            ]
        )
        
        
    def call(self, x, down_tensor):
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        
        # Calculate cropping for down_tensor to concatenate with x
        _, h2, w2, _ = down_tensor.shape
        _, h1, w1, _ = x.shape
        h_diff, w_diff = h2 - h1, w2 - w1
        
        cropping = ((int(np.ceil(h_diff / 2)), int(np.floor(h_diff / 2))),
                    (int(np.ceil(w_diff / 2)), int(np.floor(w_diff / 2))))
        down_tensor = layers.Cropping2D(cropping=cropping)(down_tensor)        
        x = layers.concatenate([x, down_tensor], axis=3)
        x = self.decode(x)
        return x