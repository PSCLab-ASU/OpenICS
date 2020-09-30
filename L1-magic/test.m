% test.m
%
% Test out the main method of the framework.
%

% reconstruction parameters
specifics = containers.Map();
reconstruction_method = 'reconstruction_tv';
specifics('lbtol') = 1e-3;
specifics('mu') = 5;
specifics('lintol') = 1e-8;
specifics('linmaxiter') = 200;
specifics('constraint') = 'eq';
specifics('normalization') = true;

% sensing parameters
sensing_method = 'sensing_scrambled_fourier';
m = 300;
n = input_width * input_height * input_channel;

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% slicing parameters
slice = true;
slice_size = 32;

% main execution
[x,x_hat] = main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics,slice,slice_size);
imshow([x, x_hat]); % display images side by side
disp("Error: " + norm(x - x_hat, 1));