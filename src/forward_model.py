import numpy as np

from src.utils import *

class ForwardModel():
    def A_svd_2d(self, x, weights, h):
        x_pad = pad_2d(x[None, None, ...])
        weights_pad = pad_2d(weights)
        X = np.fft.fft2(np.fft.ifftshift(weights_pad*x_pad, axes=(2, 3)))
        H = np.fft.fft2(np.fft.ifftshift(pad_2d(h), axes=(2, 3)))
        Y = X*H
        y = crop_2d(np.sum(np.real(np.fft.fftshift(np.fft.ifft2(Y), axes=(2, 3))), axis=(0, 1)))
        return y