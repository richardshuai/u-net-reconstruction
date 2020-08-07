% Input filepaths
comps_path = ['./data/SVD_2_5um_PSF_5um_1_ds2_components_green_SubAvg.mat'];
weights_path = ['./data/SVD_2_5um_PSF_5um_1_ds2_weights_interp_green_SubAvg.mat'];
v_path = ['./data/sample3.mat'];

params.ds_psf = 2;   %PSf downsample ratio (how much to further downsample -- if preprocessing included downsampling, use 1)
params.z_range = 1:24; %Must be even number!! Range of z slices to be solved for. If this is a scalar, 2D. Use this for subsampling z also (e.g. 1:4:... to do every 4th image)


% Load in weights and components
fprintf('loading components\n')
h_in = load(comps_path);
fprintf('done.\nLoading weights\n')
weights_in = load(weights_path);
fprintf('done loading PSF data\n')

% Make sure h and weights are in order y,x,z,rank
fprintf('permuting PSF data\n')
h = permute(h_in.comps_out(:,:,1:params.rank,params.z_range),[1,2,4,3]);
weights = permute(weights_in.weights_out(:,:,1:params.rank,params.z_range),[1,2,4,3]);
fprintf('Done permuting. Resampling PSF\n');

% Downsampling 
h = single(imresize(squeeze(h),1/params.ds_psf,'box'));
weights = single(imresize(squeeze(weights),1/params.ds_psf,'box'));

% Normalize weights to have maximum sum through rank of 1
weights_norm = max(sum(weights(size(weights,1)/2,size(weights,2)/2,:,:),4),[],3);  
weights = weights/weights_norm;
h = h/norm(vec(h));

% Load sample
v = load(v_path);
v = permute(v.dset(:, :, :), [3, 2, 1]);

fprintf('Done. PSF ready!\n');

pad2d = @(x)padarray(x,[size(h,1)/2,size(h,2)/2],0,'both');
pad2d_weights = @(x)padarray(x,[size(h,1)/2,size(h,2)/2],'replicate', 'both');
ccL = size(h,2)/2+1;
ccU = 3*size(h,2)/2;
rcL = size(h,1)/2+1;
rcU = 3*size(h,1)/2;

crop2d = @(x)x(rcL:rcU,ccL:ccU);

H = fft2(ifftshift(ifftshift(pad2d(h),1),2));

sim_image = double(A_svd_2d(H,weights,x,pad2d,pad2d_weights,crop2d));

save('sim_image.mat', 'sim_image')