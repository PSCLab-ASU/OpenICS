% test.m
%
% Test out the main method of the framework.
%

% run set_up
run set_up;

% reconstruction parameters
reconstruction_method = 'reconstruction_damp';
specifics = struct;
specifics.denoiser = 'fast-BM3D';
specifics.iters = 30;

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% sensing parameters
sensing_method = 'sensing_guassian_random_columnwise';
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
[x,x_hat] = main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics);
imshow([x, x_hat]); % display images side by side
disp("Reconstruction Error: " + norm(x - x_hat, 1));