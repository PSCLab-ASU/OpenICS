% demo.m
%
% Script to quickly run the MATLAB portion of the framework.
% Modify this script to modify reconstruction parameters.
%

% Run setup
run set_up.m;

% To modify which methods are used for reconstruction/sensing, edit the two
% variables below. The name of the method must match its file within the
% corresponding package directories, i.e. to run reconstruction_tval3 from
% the +TVAL3 directory, reconstruction_method should be 'TVAL3.reconstruction_tval3'
% Methods implemented in Python CANNOT be called from here. MATLAB methods
% have a '+' character appended to the beginning of their directory names.
reconstruction_method = 'TVAL3_reconstruction_tval3';
sensing_method = 'TVAL3_sensing_walsh_hadamard';

% Reconstruction parameters
specifics = struct;

% To modify parameters, simply specify the property within specifics. i.e.:
specifics.colored_reconstruction_mode = 'channelwise';

% Image path and details
% To modify which images are reconstructed, change img_path to the path
% leading directly to either the single image to reconstruct, or the
% directory of images to reconstruct.
% The size of the images within the directory should all be the same.
% Change input_channel, input_width, and input_height to the dimensions of
% the images to reconstruct.
img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
input_channel = 1;
input_width = 256;
input_height = 256;

% Sensing parameters
% To modify the compression ratio, change ratio to the desired compression
% ratio. It should usually not exceed 1.
ratio = 1 / 8;

n = input_width * input_height * input_channel;
m = round(n * ratio);

% If slice_size is specified, recalculate number of measurements
if isfield(specifics, 'slice_size')
    m = round(specifics.slice_size * specifics.slice_size * input_channel * ratio);
end

% Main execution
[x,x_hat,metrics] = main(sensing_method,reconstruction_method,false,img_path,input_channel,input_width,input_height,m,n,specifics);
imshow([x, x_hat]); % Display images side by side
fprintf('Reconstruction PSNR: %.2f, SSIM: %.4f\n', metrics.psnr(end), metrics.ssim(end));