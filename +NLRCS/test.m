% test.m
%
% Test out the main method of the framework.
%

% run setup
run NLRCS.set_up.m;

% reconstruction parameters
reconstruction_method = 'NLRCS.reconstruction_nlr_cs';
specifics = struct;

% image parameters and information
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% sensing parameters
sensing_method = 'NLRCS.sensing_rectmasked_uhp_fourier';
ratio = 0.2;
n = input_width * input_height * input_channel;
m = round(n * ratio);

% slicing parameters
% specifics.slice_size = 64;

% if slicing, recalculate number of measurements
if isfield(specifics, 'slice_size')
    m = round(specifics.slice_size * specifics.slice_size * input_channel * ratio);
end

% main execution
[x,x_hat,metrics] = NLRCS.main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics);
imshow([x, x_hat]); % display images side by side
img_dif = x - x_hat;
fprintf("Reconstruction Percent Error: %.2f %%\n", 100 * norm(img_dif(:), 1) / norm(x(:), 1));