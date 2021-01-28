import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from src.layers import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam

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
        out = x
        
#         out = self.featurize(out)
        
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
    
    
    
class UNet_2d_wiener(keras.Model):
    def __init__(self, initial_psf, initial_K):
#     def __init__(self):

        super(UNet_2d_wiener, self).__init__()
        self.wiener = WienerDeconvolutionOneStep(initial_psf, initial_K)
#         self.wiener = WienerDeconvolutionOneStep()

        
        self.down1 = StackEncoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down2 = StackEncoder(64, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down3 = StackEncoder(128, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down4 = StackEncoder(256, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down5 = StackEncoder(512, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down6 = StackEncoder(1024, kernel_size=3, dilation_rate=1, separable_conv=False)
        
        self.center = ConvBnRelu2d(1024, kernel_size=3, dilation_rate=1, padding='same', separable_conv=False)
        
#         self.center = ConvBnRelu2d(512, kernel_size=3, padding='same', separable_conv=True)


        self.up6 = StackDecoder(512, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up5 = StackDecoder(256, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up4 = StackDecoder(128, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up3 = StackDecoder(64, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up2 = StackDecoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up1 = StackDecoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        
        # Final prediction uses a single feature channel (green)
        self.classify = layers.Conv2D(filters=1, kernel_size=1, use_bias=True)
        
#         cropping = ((216, 216),
#            (162, 162))
#         self.crop = layers.Cropping2D(cropping=cropping)
        

        
    def call(self, x):
        out = self.wiener(x)
        
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
        out = self.up1(out)

        out = self.classify(out)

#         out = self.crop(out)

        out = tf.squeeze(out, axis=3)
        
        return out
    
    
class UNet_2d_wiener_components(keras.Model):
    def __init__(self, initial_comps, initial_K):
        super(UNet_2d_wiener_components, self).__init__()
        self.wiener_comps = WienerDeconvolutionPerComponent(initial_comps, initial_K)

        self.down1 = StackEncoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down2 = StackEncoder(64, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down3 = StackEncoder(128, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down4 = StackEncoder(256, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down5 = StackEncoder(512, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down6 = StackEncoder(1024, kernel_size=3, dilation_rate=1, separable_conv=False)
        
        self.center = ConvBnRelu2d(1024, kernel_size=3, padding='same', separable_conv=False)

        self.up6 = StackDecoder(512, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up5 = StackDecoder(256, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up4 = StackDecoder(128, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up3 = StackDecoder(64, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up2 = StackDecoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up1 = StackDecoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        
        # Final prediction uses a single feature channel (green)
        self.classify = layers.Conv2D(filters=1, kernel_size=1, use_bias=True)
        

        
    def call(self, x):
        out = self.wiener_comps(x)
        
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
        out = self.up1(out)

        out = self.classify(out)

#         out = self.crop(out)

        out = tf.squeeze(out, axis=3)
        
        return out
    
    
    
class UNet_2d_multi_wiener(keras.Model):
    def __init__(self, initial_psfs, initial_Ks):
        super(UNet_2d_multi_wiener, self).__init__()
        self.multi_wiener = MultiWienerDeconvolution(initial_psfs, initial_Ks)

        self.down1 = StackEncoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down2 = StackEncoder(64, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down3 = StackEncoder(128, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.down4 = StackEncoder(256, kernel_size=3, dilation_rate=1, separable_conv=False)

#         self.down1 = StackEncoderMultiRes(24)
#         self.down2 = StackEncoderMultiRes(64)
#         self.down3 = StackEncoderMultiRes(128)
#         self.down4 = StackEncoderMultiRes(256)

#         self.down5 = StackEncoder(512, kernel_size=3, dilation_rate=1, separable_conv=False)
#         self.down6 = StackEncoder(1024, kernel_size=3, dilation_rate=1, separable_conv=False)
        
#         self.center = ConvBnRelu2d(1024, kernel_size=3, dilation_rate=1, padding='same', separable_conv=True) # Setting to true to align with model 15.2
        self.center = ConvBnRelu2d(256, kernel_size=3, dilation_rate=1, padding='same', separable_conv=False) # Setting to true to align with model 15.2

# #         self.up6 = StackDecoder(512, kernel_size=3, dilation_rate=1, separable_conv=False)
#         self.up5 = StackDecoder(256, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up4 = StackDecoder(128, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up3 = StackDecoder(64, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up2 = StackDecoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        self.up1 = StackDecoder(24, kernel_size=3, dilation_rate=1, separable_conv=False)
        
#         self.respath2 = ResPath(64, 3)
#         self.respath3 = ResPath(128, 2)
#         self.respath4 = ResPath(256, 1)
        
    
#         self.down1 = StackEncoderMultiRes(24)
#         self.down2 = StackEncoderMultiRes(64)
#         self.down3 = StackEncoderMultiRes(128)
#         self.down4 = StackEncoderMultiRes(256)
#         self.down5 = StackEncoderMultiRes(512)
#         self.down6 = StackEncoderMultiRes(1024)
        
#         self.center = ConvBnRelu2d(1024, kernel_size=3, dilation_rate=1, padding='same', separable_conv=True) # Setting to true to align with model 15.2
#         self.center = MultiResBlock(256)

#         self.up6 = StackDecoderMultiRes(512)
#         self.up5 = StackDecoderMultiRes(256)
#         self.up4 = StackDecoderMultiRes(128)
#         self.up3 = StackDecoderMultiRes(64)
#         self.up2 = StackDecoderMultiRes(24)
#         self.up1 = StackDecoderMultiRes(24)
        
        # Final prediction uses a single feature channel (green)
        self.classify = layers.Conv2D(filters=1, kernel_size=1, use_bias=True)
        
#         cropping = ((216, 216),
#            (162, 162))
#         self.crop = layers.Cropping2D(cropping=cropping)
        

        
    def call(self, x):
        out = self.multi_wiener(x)
                
        down1_tensor, out = self.down1(out)
        down2_tensor, out = self.down2(out)
        down3_tensor, out = self.down3(out)
        down4_tensor, out = self.down4(out)
#         down5_tensor, out = self.down5(out)
#         down6_tensor, out = self.down6(out)

        out = self.center(out)

#         out = self.up6(out, down6_tensor)
#         out = self.up5(out, down5_tensor)
#         out = self.up4(out, self.respath4(down4_tensor))
#         out = self.up3(out, self.respath3(down3_tensor))
#         out = self.up2(out, self.respath2(down2_tensor))
        out = self.up4(out, down4_tensor)
        out = self.up3(out, down3_tensor)
        out = self.up2(out, down2_tensor)
        out = self.up1(out)

        out = self.classify(out)
        out = tf.clip_by_value(out, 0, 1)
        
        out = tf.squeeze(out, axis=3)
        
        return out


class MultiWienerTrainable(keras.Model):
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerTrainable, self).__init__()
        self.multi_wiener = MultiWienerDeconvolution(initial_psfs, initial_Ks)
        self.classify = layers.Conv2D(filters=1, kernel_size=1, use_bias=True)

    def call(self, x):
        out = self.multi_wiener(x)
        
        out = self.classify(out)
        return out
    