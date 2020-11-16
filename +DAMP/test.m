% test.m
%
% Test out the main method of the framework.
%

% run set_up
run DAMP.set_up.m;

% reconstruction parameters
reconstruction_method = 'DAMP.reconstruction_damp';
specifics = struct;
specifics.denoiser = 'CBM3D';
specifics.iters = 30;

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
img_path = '~/Downloads/color_512/lena_color_512.tif';
input_channel = 3;
input_width = 512;
input_height = 512;

% sensing parameters
sensing_method = 'DAMP.sensing_guassian_random_columnwise';
ratio = 0.2;
n = input_width * input_height * input_channel;
m = round(n * ratio);

% slicing parameters
specifics.slice_size = 128;

% if slicing, recalculate number of measurements
if isfield(specifics, 'slice_size')
    m = round(specifics.slice_size * specifics.slice_size * input_channel * ratio);
end

% main execution
[x,x_hat,metrics] = DAMP.main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics);
imshow([x, x_hat]); % display images side by side
img_dif = x - x_hat;
fprintf("Reconstruction Percent Error: %.2f %%\n", 100 * norm(img_dif(:), 1) / norm(x(:), 1));