% test.m
%
% Test out the main method of the framework.
%

% run set_up
run set_up

% reconstruction parameters
reconstruction_method = 'reconstruction_tval3';
specifics = struct;
specifics.normalization = true;
specifics.TVnorm = 2;
specifics.nonneg = false;
specifics.mu = 2^12;
specifics.beta = 2^6;
specifics.maxcnt = 10;
specifics.tol_inn = 1e-3;
specifics.tol = 1E-6;
specifics.maxit = 300;

specifics.mu0 = 2^4;       % trigger continuation shceme
specifics.beta0 = 2^-2;    % trigger continuation shceme

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% sensing parameters
sensing_method = 'sensing_walsh_hadamard';
n = input_width * input_height * input_channel;
m = round(65536 * 0.6);

% slicing parameters
slice = false;
slice_size = 64;

% main execution
[x,x_hat] = main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics,slice,slice_size);
imshow([x, x_hat]); % display images side by side
disp("Reconstruction error: " + norm(x - x_hat, 1));