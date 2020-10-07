% test.m
%
% Test out the main method of the framework.
%

% reconstruction parameters
reconstruction_method = 'reconstruction_l1';
specifics = struct;
specifics.lbtol = 1e-3;
specifics.mu = 5;
specifics.lintol = 1e-8;
specifics.linmaxiter = 200;
specifics.constraint = 'eq';
specifics.normalization = true;

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% sensing parameters
sensing_method = 'sensing_scrambled_fourier';
m = 300;
n = input_width * input_height * input_channel;

% slicing parameters
slice = true;
slice_size = 32;

% main execution
[x,x_hat] = main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics,slice,slice_size);
imshow([x, x_hat]); % display images side by side
disp("Reconstruction Error: " + norm(x - x_hat, 1));