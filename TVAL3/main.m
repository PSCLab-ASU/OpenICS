% main.m
% 
% Main method for the CS_Framework.
% 
% Usage: [x,x_hat] = main(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics)
%
% sensing - string, the sensing method to use.
%
% reconstruction - string, the reconstruction method to use.
%
% default - boolean, whether to use default parameters.
%
% img_path - string, the path to the image to use for compressed sensing.
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
% specifics - struct, any specific parameters for the reconstruction method.
%
% slice - boolean, whether to slice the image into submatrices.
%
% slice_size - scalar or 2-element vector, the size of each slice.
%

function [x,x_hat] = main(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics,slice,slice_size)

    if default
        % set all parameters with default values.
        img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
        input_channel = 1;
        input_width = 256;
        input_height = 256;
        m = 25000;
        n = input_height * input_width * input_channel;
        specifics = struct;
        slice = false;
    else
        % check parameters are valid
        if input_channel > 1
            error('ERROR: Multi-channel images not yet supported!');
        end
        
        if input_channel * input_height * input_width ~= n
            error('ERROR: Input dimensions do not match n!');
        end
        
        if ~exist('specifics', 'var')
            specifics = struct;
        end
        
        if numel(slice_size) == 1
            slice_size = repelem(slice_size, 2);
        end
    end
    
    sensing_method=str2func(sensing); % convert to function handle
    reconstruction_method=str2func(reconstruction); % convert to function handle
    x=im2double(imread(img_path)); % read image
    
    if ~slice
        [A,At]=sensing_method(n, m); % get sensing method function handles
        y=A(x(:)); % apply sensing to x
        x_hat=reconstruction_method(x, y, input_width, input_height, A, At, specifics); % apply reconstruction method
    else
        n=prod(slice_size); % calculate new n
        [A,At]=sensing_method(n, m); % get sensing method function handles
        x=imslice(x, input_channel, input_width, input_height, slice_size); % slice image into cell array
        x_hat=cell(size(x)); % create empty cell array for x_hat
        
        % iterate over each slice in cell array
        for i = 1:numel(x)
            disp("On slice " + i);
            temp_x=cell2mat(x(i)); % turn slice from x into matrix
            y=A(temp_x(:)); % apply sensing to temp_x
            temp_x_hat=reconstruction_method(temp_x, y, slice_size(1), slice_size(2), A, At, specifics); % apply reconstruction method
            temp_x_hat=reshape(temp_x_hat, slice_size); % reshape into original shape
            x_hat(i)=num2cell(temp_x_hat,[1,2]); % add cell into x_hat
        end
        
        % convert cell arrays to matrices
        x=cell2mat(x);
        x_hat=cell2mat(x_hat);
    end
    
    x_hat=reshape(x_hat, input_height, input_width, input_channel); % reshape to match original
end
