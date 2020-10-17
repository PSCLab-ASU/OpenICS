% test.m
%
% Test out the main method of the framework.
%

% run set_up
run set_up

% reconstruction parameters
reconstruction_method = 'reconstruction_damp';
specifics = struct;
specifics.denoiser = 'BM3D';
specifics.iters = 30;

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% sensing parameters
sensing_method = 'sensing_guassian_random_columnwise';
n = input_width * input_height * input_channel;
m = round(128 * 128 * 0.2);

% slicing parameters
slice = true;
slice_size = 128;

% main execution
[x,x_hat] = main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics,slice,slice_size);
imshow([x, x_hat]); % display images side by side
disp("Reconstruction error: " + norm(x - x_hat, 1));