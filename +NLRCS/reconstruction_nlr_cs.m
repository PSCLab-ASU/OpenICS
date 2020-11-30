% reconstruction_tv.m
% 
% Uses NLR-CS method from original source code.
% 
% Usage: [x_hat,specifics,runtime] = reconstruction_nlr_cs(x,y,img_dims,A,At,specifics)
%
% x - nx1 vector, original signal
%
% y - mx1 vector, observations
%
% img_dims - Vector denoting the size of the original image. [c,w,h]
%
% A - Function handle to sensing method
%
% At - Function handle to transposed sensing method
%
% specifics - struct that may contain the following arguments:
%
%   win - The size of each patch.
%       Default: 6
%
%   nblk - The number of similar patches in each grouping.
%       Default: 45
%
%   step - The steps between patches.
%       Default: 5
%
%   K0 - Iterations of NLR-CS after which to apply adaptive weights to SVT.
%       Default: 3
%
%   K - Iterations of NLR-CS to run.
%       Default: 18
%
%   t0 - Threshold for DCT-thresholding.
%       Default: Varies based on sampling ratio
%
%   nSig - Threshold modifier for DCT-thresholding.
%       Default: Varies based on sampling ratio
%
%   c0 - Threshold for non-weighted SVT.
%       Default: Varies based on sampling ratio
%
%   c1 - Threshold for weighted SVT.
%       Default: Varies based on sampling ratio
%
% More details about their default values may be found at the bottom of this file.
%

function [x_hat,specifics,runtime] = reconstruction_nlr_cs(x, y, img_dims, A, At, specifics)

    % set default values based on Set_parameters method
    % to define the ratio to use for parameter setting as we do in our
    % framework, divide it by two
    defaults = Set_parameters(numel(y) / numel(x) / 2, 0, 1);
    
    f = fieldnames(defaults);
    for i = 1:length(f)
        
        % if field does not exist, set to default value
        if ~isfield(specifics, f{i})
            specifics.(f{i}) = defaults.(f{i});
        end
        
    end
    
    % set code-defined properties
    % note, NLR-CS uses 0-255 instead of 0-1, so multiply image data by 255
    specifics.y = y * 255;
    specifics.ori_im = x .* 255;
    specifics.s_model = 1;
    q = randperm(round(numel(x)/2)-1)+1;
    specifics.picks = q(1:numel(y) / 2)';

    time0 = clock;
    % divide image data by 255 to get back to 0-1 range
    x_hat = NLR_CS_Reconstruction(specifics, A, At) ./ 255;
    runtime = etime(clock, time0);
    fprintf('Total elapsed time = %f secs\n\n', runtime);
    
    % remove unnecessary fields
    specifics = rmfield(specifics, 'y');
    specifics = rmfield(specifics, 'ori_im');
    specifics = rmfield(specifics, 's_model');
    specifics = rmfield(specifics, 'picks');
end

% If ratio <= 0.1:
%   t0 = 3.8
%   nSig = 4.66
%   c0 = 0.6
%   c1 = 2.2
%
% Else if ratio <= 0.2:
%   t0 = 2.4
%   nSig = 3.25
%   c0 = 0.49
%   c1 = 1.55
%
% Else if ratio <= 0.3:
%   t0 = 1.8
%   nSig = 2.65
%   c0 = 0.49
%   c1 = 1.35
%
% Else if ratio <= 0.4:
%   t0 = 1.4
%   nSig = 2.35
%   c0 = 0.49
%   c1 = 1.32
%
% Else if ratio <= 0.5:
%   t0 = 1.0
%   nSig = 2.1
%   c0 = 0.49
%   c1 = 1.15
%
% Else if ratio <= 0.6:
%   t0 = 0.8
%   nSig = 1.8
%   c0 = 0.49
%   c1 = 0.9
%
% Otherwise:
%   t0 = 0.8
%   nSig = 1.4
%   c0 = 0.49
%   c1 = 0.75
%