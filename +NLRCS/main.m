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
        
        if input_channel * input_height * input_width ~= n
            error('ERROR: Input dimensions do not match n!');
        end
        
        if ~exist('specifics', 'var')
            specifics = struct;
        end
        
    end
    
    method_name = strsplit(reconstruction,'.');
    method_name = method_name{end};
    [~,data_name,extension_name] = fileparts(img_path);
    
    if isfolder(img_path)
        folder = dir(img_path);
        % Remove any files that start with '.', usually invisible files
        dot_files = regexp({folder.name},'^\.');
        folder = folder(cellfun(@isempty,dot_files));
        folder_size = numel(folder);
        
        % Pick total number of sample images to include in .bmp
        if folder_size < 4
            num_picks = 1;
        elseif folder_size < 9
            num_picks = 4;
        elseif folder_size < 16
            num_picks = 9;
        else
            num_picks = 16;
        end
        
        picks = randperm(folder_size, num_picks);
        picks_saved = 0;
        x = cell([sqrt(num_picks), sqrt(num_picks)]);
        x_hat = cell([sqrt(num_picks), sqrt(num_picks)]);
        metrics.psnr = zeros(folder_size+1,1);
        metrics.ssim = zeros(folder_size+1,1);
        metrics.runtime = zeros(folder_size+1,1);
        metrics.meta = strings(folder_size+1,1);
        
        % Loop through directory
        for i = 1:folder_size
            file = folder(i);
            file_path = fullfile(file.folder, file.name);
            
            try
                imread(file_path);
            catch
                % Note that failed to read image
                metrics.psnr(i) = NaN;
                metrics.ssim(i) = NaN;
                metrics.runtime(i) = NaN;
                metrics.meta{i} = file.name;
                
                % If picked as a sample image, fill with black square
                if any(picks == i)
                    picks_saved = picks_saved + 1;
                    x{picks_saved} = zeros(input_height,input_width,input_channel);
                    x_hat{picks_saved} = zeros(input_height,input_width,input_channel);
                end
                
                continue;
            end
            
            fprintf('Reconstructing %s\n', file.name);
            file_x=im2double(imread(file_path));
            
            % Reconstruct colored images as 3 grayscale images
            % Exception for DAMP, which supports rgb images
            if input_channel > 1 && ~strcmp(reconstruction,'DAMP.reconstruction_damp')
                file_runtime = 0;
                file_x_hat = zeros(input_height,input_width,input_channel);

                for j = 1:input_channel
                    [channel_x_hat,channel_metrics] = reconstruct(sensing,reconstruction,file_x(:,:,j),1,input_width,input_height,m,n,specifics);
                    file_runtime = file_runtime + channel_metrics.runtime;
                    file_x_hat(:,:,j) = channel_x_hat;
                end

                file_metrics.psnr = psnr(file_x_hat, file_x);
                file_metrics.ssim = ssim(file_x_hat, file_x);
                file_metrics.runtime = file_runtime;
            else
                [file_x_hat,file_metrics] = reconstruct(sensing,reconstruction,file_x,input_channel,input_width,input_height,m,n,specifics);
            end
            
            metrics.psnr(i) = file_metrics.psnr;
            metrics.ssim(i) = file_metrics.ssim;
            metrics.runtime(i) = file_metrics.runtime;
            metrics.meta(i) = file.name;
            
            if any(picks == i)
                % save into array
                picks_saved = picks_saved + 1;
                x{picks_saved} = file_x;
                x_hat{picks_saved} = file_x_hat;
            end
            
        end
        
        % Convert cells back to tensors
        x = cell2mat(x);
        x_hat = cell2mat(x_hat);
        
        % Calculate averages, last entry in all metrics
        metrics.psnr(end) = sum(metrics.psnr(~isnan(metrics.psnr))) / (numel(~isnan(metrics.psnr)) - 1);
        metrics.ssim(end) = sum(metrics.ssim(~isnan(metrics.ssim))) / (numel(~isnan(metrics.ssim)) - 1);
        metrics.runtime(end) = sum(metrics.runtime(~isnan(metrics.runtime))) / (numel(~isnan(metrics.runtime)) - 1);
        metrics.meta(end) = 'Average';
        keyword = random_string(16);
    else
        x=im2double(imread(img_path)); % read image
        
        % Reconstruct colored images as 3 grayscale images
        % Exception for DAMP, which supports rgb images
        if input_channel > 1 && ~strcmp(reconstruction,'DAMP.reconstruction_damp')
            total_runtime = 0;
            x_hat = zeros(input_height,input_width,input_channel);

            for j = 1:input_channel
                [channel_x_hat,channel_metrics] = reconstruct(sensing,reconstruction,x(:,:,j),1,input_width,input_height,m,n,specifics);
                total_runtime = total_runtime + channel_metrics.runtime;
                x_hat(:,:,j) = channel_x_hat;
            end

            metrics.psnr = psnr(x_hat, x);
            metrics.ssim = ssim(x_hat, x);
            metrics.runtime = total_runtime;
        else
            [x_hat,metrics] = reconstruct(sensing,reconstruction,x,input_channel,input_width,input_height,m,n,specifics);
        end
        
        metrics.meta = string([data_name, extension_name]);
        keyword = data_name;
    end
    
    % Save results
    [log_file, bmp_file] = generate_results(method_name, data_name, keyword);
    imwrite([x, x_hat], bmp_file);
    log_file = fopen(log_file, 'a');
    
    for i = 1:numel(metrics.meta)
        fprintf(log_file, '%s: PSNR: %.3f, SSIM: %.3f, Runtime: %.3f\n', metrics.meta(i), metrics.psnr(i), metrics.ssim(i), metrics.runtime(i));
    end
end

function [x_hat,metrics] = reconstruct(sensing,reconstruction,x,channel,width,height,m,n,specifics)
% Reconstruct a single tensor.
%
% Usage: [x_hat,metrics] = reconstruct(sensing,reconstruction,x,channel,width,height,m,n,specifics)
%
% sensing - The sensing method to use.
%
% reconstuction - The reconstruction method to use.
%
% x - The tensor to use for reconstruction. Should represent an image.
%
% channel - The number of channels in the tensor.
%
% width - The width of the tensor.
%
% height - The height of the tensor.
%
% m - The measurement size.
%
% n - The total size of the tensor.
%
% specifics - Any specific parameters for reconstruction.
%
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
    
    sensing_method=str2func(sensing); % convert to function handle
    reconstruction_method=str2func(reconstruction); % convert to function handle
    
    if ~slice
        img_size=[channel,width,height]; % size vector ordered [c,w,h]
        [A,At]=sensing_method(img_size, m); % get sensing method function handles
        y=A(x(:)); % apply sensing to x
        time0 = clock;
        x_hat=reconstruction_method(x, y, img_size, A, At, specifics); % apply reconstruction method
        metrics.runtime = etime(clock, time0);
    else
        disp("Slicing");
        metrics.runtime = 0;
        img_size=[channel,slice_size(1),slice_size(2)]; % size vector ordered [c,w,h]
        n=prod(img_size); % calculate new n
        [A,At]=sensing_method(img_size, m); % get sensing method function handles
        x_sliced=imslice(x, channel, width, height, slice_size); % slice image into cell array
        x_hat=cell(size(x_sliced)); % create empty cell array for x_hat
        
        % iterate over each slice in cell array
        for i = 1:numel(x_sliced)
            disp("On slice " + i);
            temp_x=x_sliced{i}; % turn slice from x into matrix
            y=A(temp_x(:)); % apply sensing to temp_x
            time0 = clock;
            temp_x_hat=reconstruction_method(temp_x, y, img_size, A, At, specifics); % apply reconstruction method
            metrics.runtime = metrics.runtime + etime(clock, time0);
            temp_x_hat=reshape(temp_x_hat, flip(img_size)); % reshape into original shape
            x_hat{i}=temp_x_hat; % add cell into x_hat
        end
        
        % convert cell arrays to matrices
        x_hat=cell2mat(x_hat);
    end
    
    x_hat=reshape(x_hat, height, width, channel); % reshape to match original
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
    % check that img_width and img_height are divisible by slice_size
    if any(mod([img_width, img_height], slice_size))
        error('ERROR: Slice dimensions do not match image dimensions!');
    end

    if numel(slice_size) == 1
        slices = mat2cell(img, repelem(slice_size, img_height / slice_size), repelem(slice_size, img_width / slice_size),img_channel);
    elseif numel(slice_size) == 2
        slices = mat2cell(img, repelem(slice_size(2), img_height / slice_size(2)), repelem(slice_size(1), img_width / slice_size(1)),img_channel);
    else
        error('ERROR: Size vector larger than 2 dimensions!');
    end

end

function str = random_string(len)
% Generate random string of length len
%
% Usage: str = random_string(len)
%
% len - The length of the random string
%
    alphabet = ['A':'Z', 'a':'z', '0':'9'];
    perm = randi(numel(alphabet), [1, len]);
    str = alphabet(perm);
end

function [log_file, bmp_file] = generate_results(method_name, data_name, keyword)
% Creates the result directory
%
% Usage: [log_file, bmp_file] = generate_results(method_name, data_name, keyword)
%
% method_name - The name of the reconstruction method
%
% data_name - The name of the dataset/image reconstructed
%
% keyword - The name of the files in results
%
    result_dir = ['./results/', method_name, '/', data_name, '/'];
    
    if ~isfolder(result_dir)
        mkdir(result_dir);
    end
    
    log_file = [result_dir, keyword, '.txt'];
    bmp_file = [result_dir, keyword, '.bmp'];
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
        case 'TVAL3.reconstruction_tval3'
            sensing = 'TVAL3.sensing_walsh_hadamard';
        case 'NLRCS.reconstruction_nlr_cs'
            sensing = 'NLRCS.sensing_scrambled_fourier';
        case 'L1_magic.reconstruction_tv'
            sensing = 'L1_magic.sensing_scrambled_fourier';
        case 'L1_magic.reconstruction_l1'
            sensing = 'L1_magic.sensing_uhp_fourier';
        case 'DAMP.reconstruction_damp'
            sensing = 'DAMP.sensing_guassian_random_columnwise';
        otherwise
            sensing = 'L1_magic.sensing_guassian_random';
    end

end
