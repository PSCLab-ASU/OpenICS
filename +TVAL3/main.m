% main.m
% 
% Main method for the CS_Framework.
% 
% Usage: [x,x_hat,metrics] = main(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics)
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
% specifics - struct, any specific parameters for reconstruction.
%

function [x,x_hat,metrics] = main(sensing,reconstruction,default,img_path,input_channel,input_width,input_height,m,n,specifics)

    if default
        % set all parameters with default values.
        img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
        input_channel = 1;
        input_width = 256;
        input_height = 256;
        m = 25000;
        n = input_height * input_width * input_channel;
        specifics = struct;
        sensing = default_sensing(reconstruction);
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
        
    end
    
    if isfield(specifics, 'slice_size')
        slice = true;
        
        if numel(specifics.slice_size) == 1
            slice_size = repelem(specifics.slice_size, 2);
        else
            slice_size = specifics.slice_size;
        end
        
    else
        slice = false;
    end
    
    % get current package name
    file_path = fileparts(mfilename('fullpath'));
    folders = strsplit(file_path,'/');
    pkg_name = folders{end};
    pkg_name = [pkg_name(2:end),'.'];
    
    % append package names to string function handles
    sensing = [pkg_name, sensing];
    reconstruction = [pkg_name, reconstruction];
    sensing_method=str2func(sensing); % convert to function handle
    reconstruction_method=str2func(reconstruction); % convert to function handle
    x=im2double(imread(img_path)); % read image
    
    if ~slice
        img_size=[input_channel,input_width,input_height]; % size vector ordered [c,w,h]
        [A,At]=sensing_method(img_size, m); % get sensing method function handles
        y=A(x(:)); % apply sensing to x
        time0 = clock;
        x_hat=reconstruction_method(x, y, img_size, A, At, specifics); % apply reconstruction method
        metrics.runtime = etime(clock, time0);
    else
        disp("Slicing");
        metrics.runtime = 0;
        n=prod(slice_size); % calculate new n
        img_size=[input_channel,slice_size(1),slice_size(2)]; % size vector ordered [c,w,h]
        [A,At]=sensing_method(img_size, m); % get sensing method function handles
        x=imslice(x, input_channel, input_width, input_height, slice_size); % slice image into cell array
        x_hat=cell(size(x)); % create empty cell array for x_hat
        
        % iterate over each slice in cell array
        for i = 1:numel(x)
            disp("On slice " + i);
            temp_x=cell2mat(x(i)); % turn slice from x into matrix
            y=A(temp_x(:)); % apply sensing to temp_x
            time0 = clock;
            temp_x_hat=reconstruction_method(temp_x, y, img_size, A, At, specifics); % apply reconstruction method
            metrics.runtime = metrics.runtime + etime(clock, time0);
            temp_x_hat=reshape(temp_x_hat, slice_size); % reshape into original shape
            x_hat(i)=num2cell(temp_x_hat,[1,2]); % add cell into x_hat
        end
        
        % convert cell arrays to matrices
        x=cell2mat(x);
        x_hat=cell2mat(x_hat);
    end
    
    x_hat=reshape(x_hat, input_height, input_width, input_channel); % reshape to match original
    metrics.psnr = psnr(x_hat, x);
    metrics.ssim = ssim(x_hat, x);
end

function slices = imslice(img,img_channel,img_width,img_height,slice_size)
% Slices an image into submatrices, ordered into a cell array.
% 
% Usage: slices = imslice(img,img_channel,img_width,img_height,size)
%
% img - The image to slice into submatrices.
%
% img_channel - The number of channels in the image.
%
% img_width - The width of the image.
%
% img_height - The height of the image.
%
% slice_size - The size of each slice. Can be a vector with 2 elements or
%              an integer. Vector is ordered [width, height].
%

    if img_channel > 1
        error('ERROR: Multi-channel images not yet supported!');
    end
    
    % check that img_width and img_height are divisible by slice_size
    if any(mod([img_width, img_height], slice_size))
        error('ERROR: Slice dimensions do not match image dimensions!');
    end

    if numel(slice_size) == 1
        slices = mat2cell(img, repelem(slice_size, img_height / slice_size), repelem(slice_size, img_width / slice_size));
    elseif numel(slice_size) == 2
        slices = mat2cell(img, repelem(slice_size(2), img_height / slice_size(2)), repelem(slice_size(1), img_width / slice_size(1)));
    else
        error('ERROR: Size vector larger than 2 dimensions!');
    end

end

function sensing = default_sensing(reconstruction_method)
% Returns the name of the preferred sensing method for the specified
% reconstruction method.
%
% Usage: sensing = default_sensing(reconstruction_method)
%
% reconstruction_method - The specified reconstruction method.
%

    switch reconstruction_method
        case 'reconstruction_tval3'
            sensing = 'sensing_walsh_hadamard';
        case 'reconstruction_nlr_cs'
            sensing = 'sensing_scrambled_fourier';
        case 'reconstruction_tv'
            sensing = 'sensing_scrambled_fourier';
        case 'reconstruction_l1'
            sensing = 'sensing_uhp_fourier';
        case 'reconstruction_damp'
            sensing = 'sensing_guassian_random_columnwise';
        otherwise
            sensing = 'sensing_guassian_random';
    end

end
