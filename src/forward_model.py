import numpy as np
import tensorflow as tf

from src.utils import *

class ForwardModel():
#     def __init__(h):
#         self.Hconj = 
    
    def A_svd_2d(self, v, weights, h):
        """
        Same as A_svd_3d, but reshapes v of shape (y, x) into v of shape (1, y, x)
        Given a Fourier transformed H, calculates the convolution between H and alpha*v using FFT. 
        Sums up the results along the rank and all z-slices.
        
        Inputs:
        - v: Object of shape (y, x)
        - weights: Weights of shape (r, z, y, x)
        - h: Components of shape (r, z, y, x)
        
        Outputs:
        - b: Simulated image of shape (y, x)
        """
        v_pad = pad_2d(v[None, None, ...])
        weights_pad = pad_2d(weights)
        V = np.fft.fft2(np.fft.ifftshift(weights_pad*v_pad, axes=(2, 3)))
        H = np.fft.fft2(np.fft.ifftshift(pad_2d(h), axes=(2, 3)))
        Y = V*H
#         y = crop_2d(np.sum(np.real(np.fft.fftshift(np.fft.ifft2(Y), axes=(2, 3))), axis=(0, 1)))
        y = crop_2d(np.sum(np.real(np.fft.fftshift(np.fft.ifft2(Y), axes=(2, 3))), axis=(0, 1)))
    
        return y
    
    def A_svd_2d_tf(self, v, weights, h):
        v = tf.dtypes.cast(v, dtype=tf.complex64)
        weights = tf.dtypes.cast(weights, dtype=tf.complex64)
        h = tf.dtypes.cast(h, dtype=tf.complex64)

        v_pad = pad_2d_tf(v[None, None, ...])
        weights_pad = pad_2d_tf(weights)
        V = tf.signal.fft2d(tf.signal.ifftshift(weights_pad*v_pad, axes=(2, 3)))
        H = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(h), axes=(2, 3)))
        Y = V*H
        y = crop_2d_tf(tf.math.reduce_sum(tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(Y), axes=(2, 3))), axis=(0, 1)))

        return y
        
    def A_svd_2d_no_crop(self, v, weights, h):

        v_pad = pad_2d(v[None, None, ...])
        weights_pad = pad_2d(weights)
        V = np.fft.fft2(np.fft.ifftshift(weights_pad*v_pad, axes=(2, 3)))
        H = np.fft.fft2(np.fft.ifftshift(pad_2d(h), axes=(2, 3)))
        Y = V*H
#         y = crop_2d(np.sum(np.real(np.fft.fftshift(np.fft.ifft2(Y), axes=(2, 3))), axis=(0, 1)))
        y = np.sum(np.real(np.fft.fftshift(np.fft.ifft2(Y), axes=(2, 3))), axis=(0, 1))

        return y
    
    def A_adj_svd_2d(self, v, weights, h):
        v_pad = pad_2d(v[None, None, ...])
        weights_pad = pad_2d(weights)
        H = np.fft.fft2(np.fft.ifftshift(pad_2d(h), axes=(2, 3)))
        Hconj = np.conj(H)

        V = np.fft.fft2(np.fft.ifftshift(v_pad, axes=(2, 3)))

        Y = V*Hconj
        y = crop_2d(np.sum(weights_pad * np.real(np.fft.fftshift(np.fft.ifft2(Y), axes=(2, 3))), axis=(0, 1)))
        
        return y


    def A_adj_svd_2d_tf(self, v, weights, h):
        v = tf.dtypes.cast(v, dtype=tf.complex64)
#         weights = tf.dtypes.cast(weights, dtype=tf.complex64)
        h = tf.dtypes.cast(h, dtype=tf.complex64)
        
        v_pad = pad_2d_tf(v[None, None, ...])
        weights_pad = pad_2d_tf(weights)
        H = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(h), axes=(2, 3)))
        Hconj = tf.math.conj(H)

        V = tf.signal.fft2d(tf.signal.ifftshift(v_pad, axes=(2, 3)))

        Y = V*Hconj
        y = crop_2d_tf(tf.math.reduce_sum(weights_pad * tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(Y), axes=(2, 3))), axis=(0, 1)))
        
        return y
    
    
    def A_adj_svd_2d_tf(self, v, weights, h):
        """
        This function can be made even faster by computing H/Hconj outside. Create a new class to make all operations faster?
        """
        
        v = tf.dtypes.cast(v, dtype=tf.complex64)
#         weights = tf.dtypes.cast(weights, dtype=tf.complex64)
        h = tf.dtypes.cast(h, dtype=tf.complex64)
        
        v_pad = pad_2d_tf(v[None, None, ...])
        weights_pad = pad_2d_tf(weights)
        H = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(h), axes=(2, 3)))
        Hconj = tf.math.conj(H)

        V = tf.signal.fft2d(tf.signal.ifftshift(v_pad, axes=(2, 3)))

        Y = V*Hconj
        y = crop_2d_tf(tf.math.reduce_sum(weights_pad * tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(Y), axes=(2, 3))), axis=(0, 1)))
        
        return y
    
# def A_2d_svd(x,H,weights,pad,mode='shift_variant'): #NOTE, H is already padded outside to save memory
#     x=pad(x)
#     Y=np.zeros((np.shape(x)[0],np.shape(x)[1]))
        
#     if (mode =='shift_variant'):
#         for r in range (0,np.shape(weights)[2]):
#             X=np.fft.fft2((np.multiply(pad(weights[:,:,r]),x)))
#             Y=Y+ np.multiply(X,H[:,:,r])
    
#     return np.real((np.fft.ifftshift(np.fft.ifft2(Y))))
    
# def A_2d_adj_svd(Hconj,weights,y,pad):
#     y=pad(y)
#     x=np.zeros((np.shape(y)[0],np.shape(y)[1]))
#     for r in range (0, np.shape(weights)[2]):
#         x=x+np.multiply(pad(weights[:,:,r]),(np.real(np.fft.ifftshift(np.fft.ifft2(np.multiply(Hconj[:,:,r], np.fft.fft2((y))))))))
#     #note the weights are real so we dont take the complex conjugate of it, which is the adjoint of the diag 
#     return x
