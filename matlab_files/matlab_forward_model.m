% Paths to PSF and sample
comps_path = ['./data/SVD_2_5um_PSF_5um_1_ds2_components_green_SubAvg.mat']
weights_path = ['./data/SVD_2_5um_PSF_5um_1_ds2_weights_interp_green_SubAvg.mat']
v_path = ['./data/sample3.mat']

% Parameters
params.ds_psf = 2;   %PSf downsample ratio (how much to further downsample -- if preprocessing included downsampling, use 1)
params.rank = 10;

% Load h, reorder axes, and downsample
h = load(comps_path);
h = permute(h.comps_out(:,:,1:params.rank, :),[1,2,4,3]);
h = single(imresize(squeeze(h),1/params.ds_psf,'box'));

% Load weights, reorder axes, and downsample
weights = load(weights_path);
weights = permute(weights.weights_out(:,:,1:params.rank, :),[1,2,4,3]);
weights = single(imresize(squeeze(weights),1/params.ds_psf,'box'));

% Load sample
v = load(v_path);
v = permute(v.dset(:, :, :), [3, 2, 1]);

fprintf('Done. PSF ready!\n');

H = fft2(ifftshift(ifftshift(h,1),2));

output = A_svd_3d(v, weights, H);
save('output.mat', 'output')
