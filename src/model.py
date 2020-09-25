import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from src.layers import *

class UNet_2d(keras.Model):
    def __init__(self):
        super(UNet_2d, self).__init__()
        self.down1 = StackEncoder(24, kernel_size=3, dilation_rate=4)
        self.down2 = StackEncoder(64, kernel_size=3, dilation_rate=4)
        self.down3 = StackEncoder(128, kernel_size=3, dilation_rate=4)
        self.down4 = StackEncoder(256, kernel_size=3, dilation_rate=4)
        self.down5 = StackEncoder(512, kernel_size=3, dilation_rate=4)
        self.down6 = StackEncoder(1024, kernel_size=3, dilation_rate=4)
        
        self.center = ConvBnRelu2d(1024, kernel_size=3, padding='same')

        self.up6 = StackDecoder(512, kernel_size=3, dilation_rate=4)
        self.up5 = StackDecoder(256, kernel_size=3, dilation_rate=4)
        self.up4 = StackDecoder(128, kernel_size=3, dilation_rate=4)
        self.up3 = StackDecoder(64, kernel_size=3, dilation_rate=4)
        self.up2 = StackDecoder(24, kernel_size=3, dilation_rate=4)
        self.up1 = StackDecoder(24, kernel_size=3, dilation_rate=4)
        
        # Final prediction uses a single feature channel (green)
        self.classify = layers.Conv2D(filters=1, kernel_size=1, use_bias=True)
        
        
        
    def call(self, x):
        out = x;
        down1_tensor, out = self.down1(out)
        down2_tensor, out = self.down2(out)
        down3_tensor, out = self.down3(out)
        down4_tensor, out = self.down4(out)
        down5_tensor, out = self.down5(out)
        down6_tensor, out = self.down6(out)

        out = self.center(out)

        out = self.up6(out, down6_tensor)
        out = self.up5(out, down5_tensor)
        out = self.up4(out, down4_tensor)
        out = self.up3(out, down3_tensor)
        out = self.up2(out, down2_tensor)
        out = self.up1(out, down1_tensor)

        out = self.classify(out)
        out = tf.squeeze(out, axis=3)
        
        return out