function b = A_svd_3d(v, alpha, H)
b = real(ifft2(sum(sum(H.*fft2(v.*alpha),3),4)));