function y = A_svd_2d(H, weights, x,pad,padweights,crop)
x=pad(x) %pad the input
Y = zeros(size(x,1),size(x,2)); %makes a variable of the shape of the padded input (in X and Y)
for r = 1:size(H,3)
    X = fft2(padweights(weights(:,:,r)).*x); %pad weights to be multiplied by padded input. Note, to pad the weights, make sure the padding function is padded by extension not zeros
    Y = Y + (X.*H(:,:,r));  %here, H is the padded fourier transform of the compomnents. The reason we pad before passing H in is just to save time.
end
return y = crop(ifft2(Y));