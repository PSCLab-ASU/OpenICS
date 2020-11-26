% test.m
%
% Test out the main method of the framework.
%

% run set_up
run TVAL3.set_up.m;

% reconstruction parameters
reconstruction_method = 'TVAL3.reconstruction_tval3';
specifics = struct;

specifics.mu0 = 2^4;       % trigger continuation shceme
specifics.beta0 = 2^-2;    % trigger continuation shceme

% image parameters and information
img_path = '~/Downloads/color_512';
input_channel = 3;
input_width = 512;
input_height = 512;

% img_path = fullfile(matlabroot, '/toolbox/images/imdata/autumn.tif');
% [input_height, input_width, input_channel] = size(imread(img_path));

% sensing parameters
sensing_method = 'TVAL3.sensing_walsh_hadamard';
ratio = 0.2;
n = input_width * input_height * input_channel;
m = round(n * ratio);

% slicing parameters
% specifics.slice_size = 64;

% specifics.colored_reconstruction_mode = 'vectorized';

% if slicing, recalculate number of measurements
if isfield(specifics, 'slice_size')
    m = round(specifics.slice_size * specifics.slice_size * input_channel * ratio);
end

% main execution
[x,x_hat,metrics] = TVAL3.main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics);
imshow([x, x_hat]); % display images side by side
img_dif = x - x_hat;
fprintf("Reconstruction Percent Error: %.2f %%\n", 100 * norm(img_dif(:), 1) / norm(x(:), 1));