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
    total_time0 = clock;

    if default
        % Set all parameters with default values.
        img_path = fullfile(matlabroot, '/toolbox/images/imdata/cameraman.tif');
        input_channel = 1;
        input_width = 256;
        input_height = 256;
        m = 25000;
        n = input_height * input_width * input_channel;
        specifics = struct;
        sensing = default_sensing(reconstruction);
    else
        % Check parameters are valid
        
        % Ensure specific sizes match n
        if input_channel * input_height * input_width ~= n
            error('ERROR: Input dimensions do not match n!');
        end
        
        % Ensure specifics exists
        if ~exist('specifics', 'var')
            specifics = struct;
        end
        
        % Ensure slice_size has 2 elements
        if isfield(specifics, 'slice_size') && numel(specifics.slice_size) == 1
            specifics.slice_size = repelem(specifics.slice_size, 2);
        end
        
    end
    
    % Set default colored reconstruction mode
    if ~isfield(specifics, 'colored_reconstruction_mode')
        specifics.colored_reconstruction_mode = 'channelwise';
    end
    
    if strcmp(reconstruction, 'DAMP.reconstruction_damp')
        specifics.colored_reconstruction_mode = 'vectorized';
    end
    
    method_name = strsplit(reconstruction,'.');
    method_name = method_name{end};
    [~,data_name,extension_name] = fileparts(img_path);
    [A,At] = get_sensing_handles(sensing,input_channel,input_width,input_height,m,specifics);
    
    if isdir(img_path)
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
        
        % Variable initialization
        picks = randperm(folder_size, num_picks);
        picks_saved = zeros(folder_size, 1);
        picks_saved(picks) = 1:num_picks;
        x = cell(folder_size,1);
        x_hat = cell(folder_size,1);
        psnr = zeros(folder_size+1,1);
        ssim = zeros(folder_size+1,1);
        runtime = zeros(folder_size+1,1);
        meta = repmat({''},folder_size+1,1);
        specifics_history = cell(folder_size,1);
        
        % Set number of workers to 0 (sequential) if unspecified
        if ~isfield(specifics, 'workers')
            specifics.workers = 0;
        end
        
        % Loop through directory
        for i = 1:folder_size
            file = folder(i);
            file_path = fullfile(img_path, file.name);
            
            try
                file_x=im2double(imread(file_path));
                assert(numel(file_x) == input_height * input_width * input_channel);
            catch
                % Note that failed to read image
                psnr(i) = NaN;
                ssim(i) = NaN;
                runtime(i) = NaN;
                meta{i} = file.name;
                
                % If picked as a sample image, fill with black square
                if picks_saved(i) > 0
                    x{i} = zeros(input_height,input_width,input_channel);
                    x_hat{i} = zeros(input_height,input_width,input_channel);
                end
                
                continue;
            end
            
            fprintf('Reconstructing %s\n', file.name);
            
            if size(file_x,1) ~= input_height || size(file_x,2) ~= input_width || size(file_x,3) ~= input_channel
                error('ERROR: Image dimensions do not match input size!');
            end
            
            [file_x_hat,file_metrics,specifics_history{i}] = reconstruct(reconstruction,A,At,file_x,input_channel,input_width,input_height,specifics);
            psnr(i) = file_metrics.psnr;
            ssim(i) = file_metrics.ssim;
            runtime(i) = file_metrics.runtime;
            meta(i) = {file.name};
            
            if picks_saved(i) > 0
                % save into array
                x{i} = file_x;
                x_hat{i} = file_x_hat;
            end
            
        end
        
        % Convert cells
        x = cell2mat(reshape(x(~cellfun(@isempty,x)),[sqrt(num_picks),sqrt(num_picks)]));
        x_hat = cell2mat(reshape(x_hat(~cellfun(@isempty,x_hat)),[sqrt(num_picks),sqrt(num_picks)]));
        specifics = specifics_history{find(~cellfun(@isempty, specifics_history), 1)};
        
        % Calculate averages, last entry in all metrics
        avg_entries = and(~isnan(psnr), ~isinf(psnr));
        psnr(end) = sum(psnr(avg_entries)) / (nnz(avg_entries) - 1);
        ssim(end) = sum(ssim(avg_entries)) / (nnz(avg_entries) - 1);
        runtime(end) = sum(runtime(avg_entries)) / (nnz(avg_entries) - 1);
        meta(end) = {'Average'};
        
        % Save to log file
        keyword = random_string(16);
        [log_file, bmp_file] = generate_results(method_name, data_name, keyword);
        log_file = fopen(log_file, 'a');
        
        % Log all results in a loop
        for i = 1:folder_size+1
            fprintf(log_file, '%s: PSNR: %.3f, SSIM: %.3f, Runtime: %.3f\n', char(meta(i)), psnr(i), ssim(i), runtime(i));
        end
        
        metrics.psnr = psnr;
        metrics.ssim = ssim;
        metrics.runtime = runtime;
        metrics.meta = meta;
    else
        x=im2double(imread(img_path)); % read image
        
        if size(x,1) ~= input_height || size(x,2) ~= input_width || size(x,3) ~= input_channel
            error('ERROR: Image dimensions do not match input size!');
        end
        
        [x_hat,metrics,specifics] = reconstruct(reconstruction,A,At,x,input_channel,input_width,input_height,specifics);
        metrics.meta = strcat(data_name, extension_name);
        
        % Save to log file
        keyword = data_name;
        [log_file, bmp_file] = generate_results(method_name, data_name, keyword);
        log_file = fopen(log_file, 'a');
        fprintf(log_file, '%s: PSNR: %.3f, SSIM: %.3f, Runtime: %.3f\n', metrics.meta, metrics.psnr, metrics.ssim, metrics.runtime);
    end
    
    % Save further results
    imwrite([x, x_hat], bmp_file);
    
    f = fieldnames(specifics);
    
    fprintf(log_file, '\nParameters:\n');
    fprintf(log_file, 'Image dimensions: %d x %d x %d\n', input_channel, input_width, input_height);
    fprintf(log_file, 'Sensing method: %s\n', sensing);
    fprintf(log_file, 'Measurements: %d\n', m);
    fprintf(log_file, 'Compression ratio: %f\n', m / n);
    fprintf(log_file, 'Specifics:\n');
    
    for i = 1:length(f)
        fprintf(log_file, '\t%s: %s\n', f{i}, num2str(specifics.(f{i})));
    end
        
    fprintf(log_file, '\nTotal time elapsed: %.3f\n', etime(clock, total_time0));
    fclose(log_file);
end

function [A, At] = get_sensing_handles(sensing,channel,width,height,m,specifics)
% Gets A, At handles from sensing method. Implemented to reduce time spent
% regenerating already-existing sensing handles.
%
% Usage: [A,At] = get_sensing_handles(sensing,channel,width,height,m,specifics)
%
% sensing - The sensing method to use
%
% channel - The number of channels in the image
%
% width - The width of the image
%
% height - The height of the image
%
% m - The number of measurements
%
% specifics - Specific parameters for reconstruction
%
    slicing = isfield(specifics, 'slice_size');
    sensing = str2func(char(sensing));

    if isfield(specifics, 'colored_reconstruction_mode') && strcmp(specifics.colored_reconstruction_mode,'channelwise')
        m = round(m / channel);
        
        if slicing
            img_dims = [1, specifics.slice_size];
        else
            img_dims = [1, width, height];
        end
        
    else
        
        if slicing
            img_dims = [channel, specifics.slice_size];
        else
            img_dims = [channel, width, height];
        end
        
    end
    
    [A, At] = sensing(img_dims, m);
end

function [x_hat,metrics,specifics] = reconstruct(reconstruction,A,At,x,channel,width,height,specifics)
% Selects a reconstruction mode before actually reconstructing a tensor.
%
% Usage: [x_hat,metrics,specifics] = reconstruct(reconstruction_method,A,At,x,channel,width,height,m,n,specifics)
%
% reconstruction - The name of the reconstruction method.
%
% A - The sensing method handle.
%
% At - The transposed sensing method handle.
%
% x - The tensor to use for reconstruction. Should represent an image.
%
% channel - The number of channels in the tensor.
%
% width - The width of the tensor.
%
% height - The height of the tensor.
%
% specifics - Any specific parameters for reconstruction.
%
    reconstruction_method = str2func(char(reconstruction));

    % Reconstruction modes
    %   Regular - For DAMP reconstruction or single-channel images,
    %             reconstructs image as a whole.
    %   Channelwise - For colored images, reconstructs each channel
    %                 as a separate grayscale image.
    %   Vectorized - For colored images, reconstructs whole image
    %                by flattening all channels into one 2D tensor.
    if channel == 1 || ~isfield(specifics, 'colored_reconstruction_mode') || strcmp(reconstruction, 'DAMP.reconstruction_damp')
        [x_hat,metrics,specifics] = reconstruct_tensor(reconstruction_method,A,At,x,channel,width,height,specifics);
    elseif strcmp(specifics.colored_reconstruction_mode,'channelwise')
        total_runtime = 0;
        x_hat = zeros(height,width,channel);

        for j = 1:channel
            [channel_x_hat,channel_metrics,specifics] = reconstruct_tensor(reconstruction_method,A,At,x(:,:,j),1,width,height,specifics);
            total_runtime = total_runtime + channel_metrics.runtime;
            x_hat(:,:,j) = channel_x_hat;
        end

        metrics.psnr = psnr(x_hat, x);
        metrics.ssim = ssim(x_hat, x);
        metrics.runtime = total_runtime;
    elseif strcmp(specifics.colored_reconstruction_mode,'vectorized')
        flattened_x = reshape(x,[height,width*channel]);
        [x_hat,metrics,specifics] = reconstruct_tensor(reconstruction_method,A,At,flattened_x,1,width*channel,height,specifics);
        x_hat = reshape(x_hat, size(x));
    else
        error('ERROR: Invalid reconstruction mode!');
    end
    
    % Ensure psnr <= 48
    metrics.psnr = min(48, metrics.psnr);
end

function [x_hat,metrics,specifics] = reconstruct_tensor(reconstruction_method,A,At,x,channel,width,height,specifics)
% Reconstruct a single tensor.
%
% Usage: [x_hat,metrics,specifics] = reconstruct_tensor(reconstruction_method,A,At,x,channel,width,height,m,n,specifics)
%
% reconstruction_method - The reconstruction method handle.
%
% A - The sensing method handle.
%
% At - The transposed sensing method handle.
%
% x - The tensor to use for reconstruction. Should represent an image.
%
% channel - The number of channels in the tensor.
%
% width - The width of the tensor.
%
% height - The height of the tensor.
%
% specifics - Any specific parameters for reconstruction.
%    
    if ~isfield(specifics, 'slice_size')
        img_size=[channel,width,height]; % size vector ordered [c,w,h]
        y=A(x(:)); % apply sensing to x
        [x_hat,specifics,runtime]=reconstruction_method(x, y, img_size, A, At, specifics); % apply reconstruction method
        metrics.runtime = runtime;
    else
        disp('Slicing');
        metrics.runtime = 0;
        img_size=[channel,specifics.slice_size]; % size vector ordered [c,w,h]
        x_sliced=imslice(x, channel, width, height, specifics.slice_size); % slice image into cell array
        x_hat=cell(size(x_sliced)); % create empty cell array for x_hat
        
        % iterate over each slice in cell array
        for i = 1:numel(x_sliced)
            disp('On slice ' + i);
            temp_x=x_sliced{i}; % turn slice from x into matrix
            y=A(temp_x(:)); % apply sensing to temp_x
            [temp_x_hat,specifics,runtime]=reconstruction_method(temp_x, y, img_size, A, At, specifics); % apply reconstruction method
            metrics.runtime = metrics.runtime + runtime;
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
    result_dir = strcat('./results/', method_name, '/', data_name, '/');
    
    if ~isdir(result_dir)
        mkdir(result_dir);
    end
    
    log_file = strcat(result_dir, keyword, '.txt');
    bmp_file = strcat(result_dir, keyword, '.bmp');
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
        case 'TVAL3_reconstruction_tval3'
            sensing = 'TVAL3_sensing_walsh_hadamard';
        case 'NLRCS_reconstruction_nlr_cs'
            sensing = 'NLRCS_sensing_scrambled_fourier';
        case 'L1magic_reconstruction_tv'
            sensing = 'L1magic_sensing_scrambled_fourier';
        case 'L1magic_reconstruction_l1'
            sensing = 'L1magic_sensing_uhp_fourier';
        case 'DAMP_reconstruction_damp'
            sensing = 'DAMP_sensing_gaussian_random_columnwise';
        otherwise
            sensing = 'L1magic_sensing_gaussian_random';
    end

end
