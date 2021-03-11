function [comps, weights_interp, weights,shifts,yi_reg_out] = Miniscope_svd_xy(stack,ref_im,rnk,varargin)
% [comps, weights,shifts] = Miniscope_svd_xy(stack,ref_im,rnk)
% Takes in a background subtracted stack. Stack does not need to be on a
% grid and will be blindly registerd with cross correlation. Registration
% is performed relative to ref_im. 
% SVD up to rank rnk is done after registration.
% 'boundary_condition' : 'circular' (defualt) or 'zero'. 'circular' uses no
% padding/cropping, 'zero' pads and crops to avoid wrapping (good if
% cropping is happening'). Default is 'circular'
% Returns 3d arrays comps and weights. comps
% contains the psf components, and weights contains the spatially-varying
% weights_iterp, upsampled to the same grid size as the inpus stack.
% Returns weights, the SVD weighting for each measurement (not on a
% grid--this is index-aligned with shifts)
% Also returns cell array of shifts found in registration
% Returns yi_reg_out, an 8-bit image of each psf after registration. This
% is just for visual inspection!

[Ny, Nx] = size(stack(:,:,1));
vec = @(x)x(:);
p = inputParser;
addParameter(p,'boundary_condition','circular')
addParameter(p,'roi',zeros(Ny,Nx))
parse(p,varargin{:})
params = p.Results;



pad2d = @(x)padarray(x,[Ny/2,Nx/2],'both');
fftcorr = @(x,y)(ifft2(fft2(pad2d(x)).*conj(fft2(ifftshift(pad2d(y))))));

findpeak_id = @(x)find(x == max(x(:)));

M = size(stack,3);
Si = @(x,si)circshift(x,si);

pr = Ny + 1;
pc = Nx + 1; % Relative centers of all correlations


yi_reg = 0*stack;   %Registered stack

switch lower(params.boundary_condition)
    case('circular')
        pad = @(x)x;
        crop = @(x)x;
        pad2d = @(x)padarray(x,[Ny/2,Nx/2],'both');
        crop2d = @(x)x(Ny/2+1:3*Ny/2,Nx/2+1:3*Nx/2,:);
    case('zero')
        pad = @(x)padarray(x,[Ny/2,Nx/2],'both');
        crop = @(x)x(Ny/2+1:3*Ny/2,Nx/2+1:3*Nx/2,:);
end

% Normalize the stack first
stack_norm = zeros(1,M);
stack_dct = stack;
ref_norm = norm(ref_im,'fro');
for m = 1:M
    stack_norm(m) = norm(stack_dct(:,:,m),'fro');
    stack_dct(:,:,m) = stack_dct(:,:,m)/stack_norm(m);
    stack(:,:,m) = stack(:,:,m)/ref_norm;
end
ref_im = ref_im/ref_norm;
si = cell(1,M);
% Do fft registration
fprintf('Removing background and hot pixels...\n')




for n = 1:size(stack_dct,3)
    im = stack_dct(:,:,n);
    %bg_dct = dct2(remove_hot_pixels(im,3,.0001));
    bg_dct = dct2(im);
    bg_dct(1:20,1:20) = 0;

    stack_dct(:,:,n) = idct2(reshape(bg_dct,size(im)));

    
end

fprintf('done\n')

fprintf('registering...\n')
good_count = 0;
for m = 1:M
    
    %[r,c] = ind2sub(2*[Ny, Nx],findpeak_id(fftcorr(remove_hot_pixels(stack(:,:,m),3,.0001),stack(:,:,icenter))));
    corr_im = fftcorr(stack_dct(:,:,m),ref_im);
    
    if max(corr_im(:)) < .01
        fprintf('image %i has poor match. Skipping\n',m);
    else
        good_count = good_count + 1;
        [r,c] = ind2sub(2*[Ny, Nx],findpeak_id(corr_im));
        si{good_count} = [-(r-pr),-(c-pc)];
        W = crop2d(Si(~pad2d(~params.roi),-si{good_count}));
        bg_estimate = sum(sum(W.*stack(:,:,m)))/max(nnz(params.roi),1);
        im_reg = ref_norm*crop(Si(pad(stack(:,:,m)-bg_estimate),si{good_count}));
        %bg_estimate = sum(sum(params.roi .* im_reg))/max(nnz(params.roi),1);
        yi_reg(:,:,good_count) = im_reg;
        %yi_reg_out(:,:,good_count) = uint8(yi_reg(:,:,good_count)/max(max(yi_reg(:,:,good_count)))*255);
    end
    
end

yi_reg = yi_reg(:,:,1:good_count);
%yi_reg_out = yi_reg_out(:,:,1:good_count);
fprintf('done registering\n')

fprintf('creating matrix\n')
Mgood = size(yi_reg,3);
ymat = zeros(Ny*Nx,Mgood);
for m = 1:Mgood
    ymat(:,m) = vec(yi_reg(:,:,m));
    
end
fprintf('done\n')

fprintf('starting svd...\n')
tic
[u,s,v] = svds(ymat,rnk);
t_svd = toc;
fprintf('svd took %.2f seconds \n',t_svd)

comps = reshape(u,[Ny, Nx,rnk]);
vt = v';

weights = zeros(Mgood,rnk);
for m = 1:Mgood
    for r = 1:rnk
        weights(m,r) = s(r,r)*vt(r,m);
    end
end
si_mat = reshape(cell2mat(si)',[2,Mgood]);
xq = -Nx/2+1:Nx/2;
yq = -Ny/2+1:Ny/2;
[Xq, Yq] = meshgrid(xq,yq);

weights_interp = zeros(Ny, Nx,rnk);
fprintf('interpolating...\n')
for r = 1:rnk
    interpolant_r = scatteredInterpolant(si_mat(2,:)', si_mat(1,:)', weights(:,r),'natural','nearest');
    weights_interp(:,:,r) = rot90(interpolant_r(Xq,Yq),2);
end
fprintf('done\n\n')

shifts = si;

yi_reg_out = yi_reg;
return





