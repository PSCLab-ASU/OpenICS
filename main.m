% main.m
% 
% Entry point for the MATLAB portion of the CS_Framework.
% Calls the main functions from each package, which may differ in the future.
% 
% Usage: [x,x_hat,metrics] = main(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics)
%
% sensing - string, the sensing method to use.
%
% reconstruction - string, the reconstruction method to use.
%
% default - boolean, whether to use default parameters.
%
% img_path - string, the path to the image or directory to use for
%            compressed sensing.
%
% input_channel - The number of input channels in the original image.
%
% input_width - The width of the original image.
%
% input_height - The height of the original image.
%
% m - The number of observations to make.
%
% n - The size of the original signal.
%
% specifics - struct, any specific parameters for reconstruction.
%

function [x,x_hat,metrics] = main(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics)
    package_name = strsplit(reconstruction, '_');
    package_name = package_name{1};
    main_handle = str2func(char([package_name, '_', 'main']));
    [x,x_hat,metrics] = main_handle(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics);
end
