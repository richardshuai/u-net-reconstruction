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
    
    def A_svd_2d_no_crop_tf(self, v, weights, h):
        v = tf.dtypes.cast(v, dtype=tf.complex64)
        weights = tf.dtypes.cast(weights, dtype=tf.complex64)
        h = tf.dtypes.cast(h, dtype=tf.complex64)

        v_pad = pad_2d_tf(v[None, None, ...])
        weights_pad = pad_2d_tf(weights)
        V = tf.signal.fft2d(tf.signal.ifftshift(weights_pad*v_pad, axes=(2, 3)))
        H = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(h), axes=(2, 3)))
        Y = V*H
        y = tf.math.reduce_sum(tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(Y), axes=(2, 3))), axis=(0, 1))

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
    
    def A2d_inv_cg_tf(self, x, weights, h, alpha_cg):
        return self.A_adj_svd_2d_tf(self.A_svd_2d_tf(x, weights, h), weights, h) + alpha_cg*x

    def cgsolve_tf(self, z, b, weights, h, Niter, alpha_cg):
        #b is A*y
        #solving z=M-1b
        r=b-self.A2d_inv_cg_tf(z, weights, h, alpha_cg)
        p=tf.identity(r)
        rsold=tf.math.conj(tf.math.reduce_sum(r*r))
    #     print('loss ', rsold)
        for i in range(0,Niter):
            Ap=self.A2d_inv_cg_tf(p, weights, h, alpha_cg)
            alpha=rsold/(tf.math.conj(tf.math.reduce_sum(p*Ap)))
            z=z+alpha*p
            r=r-alpha*Ap

            rsnew=tf.math.conj(tf.math.reduce_sum(r*r))
            if tf.math.sqrt(rsnew)<1e-10:
                break
                
            p=r+(rsnew/rsold)*p
            rsold=rsnew
    #         print('loss ',rsold)
        return z
    
    def wiener_deconvolve(self, x, weights, h, Niter):
        Ah_b_2d = self.A_adj_svd_2d_tf(x, weights, h)
        z_wiener=tf.zeros(tf.shape(x))
        x_wiener=self.cgsolve_tf(z_wiener, Ah_b_2d, weights, h, Niter, 1e-4)
        
        return x_wiener

    def wiener_deconvolve_one_step(self, y, psf, K):
        y = tf.dtypes.cast(y, dtype=tf.complex64)
        psf = tf.dtypes.cast(psf, dtype=tf.complex64)

        Y=tf.signal.fft2d((pad_2d_tf(y)))
        H_sum=tf.signal.fft2d((pad_2d_tf(psf)))#np.sum(H,2)

        X=(tf.math.conj(H_sum)*Y)/tf.dtypes.cast(tf.math.square(tf.math.abs(H_sum))+K, dtype=tf.complex64)
#         x=tf.math.real((tf.signal.ifftshift(np.fft.ifft2(X))))
        x=tf.math.real((tf.signal.ifftshift(tf.signal.ifft2d(X))))
        
        x = crop_2d_tf(x)
        
        return x
    
    def convolve_tf(self, x, psf):
        """
        Frequency domain convolution.

        Inputs:
          - x: input image of shape (y, x)
          - psf: psf of shape (y, x)
        """
        x = tf.dtypes.cast(x, dtype=tf.complex64)
        psf = tf.dtypes.cast(psf, dtype=tf.complex64)

        X = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(x)))
        H = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(psf)))

        X = X*H
        X = crop_2d_tf(tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(X))))

        return X    
    
    def A_svd_2d_tf_components(self, v, weights, h):
        v = tf.dtypes.cast(v, dtype=tf.complex64)
        weights = tf.dtypes.cast(weights, dtype=tf.complex64)
        h = tf.dtypes.cast(h, dtype=tf.complex64)

        v_pad = pad_2d_tf(v[None, None, ...])
        weights_pad = pad_2d_tf(weights)
        V = tf.signal.fft2d(tf.signal.ifftshift(weights_pad*v_pad, axes=(2, 3)))
        H = tf.signal.fft2d(tf.signal.ifftshift(pad_2d_tf(h), axes=(2, 3)))
        Y = V*H
        y = crop_2d_tf(tf.math.reduce_sum(tf.math.real(tf.signal.fftshift(tf.signal.ifft2d(Y), axes=(2, 3))), axis=1))

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
